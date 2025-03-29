import numpy as np
import pandas as pd
import shap
import matplotlib
# Set non-interactive Agg backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import logging


logging.getLogger().setLevel(logging.WARNING)


# shap.initjs()
plt.rc('font', family = 'Times New Roman')


"""For classification tasks:
sklearn.gbdt, xgboost, lightgbm, catboost all work in log-odds space, and convert the final predicted values back to probabilities.
Therefore, TreeExplainer transforms their output probability values p through logit conversion, returning to the model's original log-odds space for explanation.
That is: ln(p/(1 - p))
A value >0 represents prediction as positive class (p>0.5), <0 represents prediction as negative class (p<0.5), =0 represents a neutral prediction (p=0.5).

- Log-odds is the ideal space for SHAP value calculation, ensuring additivity, unboundedness, and consistent interpretation of feature contributions.
- SHAP additivity: The sum of shap_values across all features for a sample, plus the base_value (expected_value), equals the predicted output value for that sample.
- For binary classification tasks, the output shap_values have dimensions (n_samples, n_features), where values represent contributions to the positive class, explained in log-odds space.
- For multi-class tasks, the output shap_values have dimensions (n_samples, n_features, n_targets), where values represent contributions to each class, explained in log-odds space.

Taking binary classification as an example:
Get prediction probability values
>> proba = self.model_obj.predict_proba(self.shap_data)[0]
The first sample's model prediction probability values [negative, positive]
>> print(proba[0], proba[1])

Try to convert the output probabilities of two classes to log-odds
For binary classification problems, the log-odds of the positive class and negative class have opposite signs but the same absolute value
Specifically: log-odds(p) = -log-odds(1 - p)
Therefore define the conversion formula:
>> def log_odds(x):
>>     return np.log(x / (1 - x))
The first sample's output probability after log-odds transformation [negative, positive]
>> print(log_odds(proba[0]), log_odds(proba[1]))

Sum up the features:
>> feature_sum_i = np.sum(self.shap_values[0, :])

Add the base_value:
>> base_value_i = self.shap_base_values[0]

This will be the same as the model's prediction:
>> sum_i = feature_sum_i + base_value_i
"""


"""SHAP visualization functions comparison and selection guide.

Attributes:
    Supported visualization methods:
    1. partial_dependence_plot:
        - Shows average marginal effect of a feature on model output
        - Similar to traditional Partial Dependence Plot (PDP)
        - X-axis: Feature values, Y-axis: Model predictions

    2. dependence_plot:
        - Visualizes feature value vs SHAP value relationship
        - Automatically detects interactions (color-codes 2nd influential feature)
        - X-axis: Feature values, Y-axis: SHAP values

    3. summary_plot:
        - Displays feature importance and value impacts
        - Combines feature importance with SHAP value distributions
        - Y-axis: Feature names, X-axis: SHAP values

Key Differences Table:
    | Function                | Data Used       | SHAP Values | Interactions | Output Scale   |
    |-------------------------|-----------------|-------------|--------------|----------------|
    | partial_dependence_plot | Raw features    | ❌          | ❌          | Model output   |
    | dependence_plot         | Features+SHAP   | ✅          | ✅          | SHAP values    |
    | summary_plot            | SHAP values     | ✅          | ❌          | SHAP magnitude |
"""



class MyExplainer:
    def __init__(
            self,
            results_dir: str | pathlib.Path,
            model_object,
            model_name: str,
            background_data: pd.DataFrame,
            shap_data: pd.DataFrame,
            sample_background_data_k: int | float | None = None,
            sample_shap_data_k:  int | float | None = None,       
        ):
        """Initialize the MyExplainer with model and data.

        Args:
            results_dir (str|pathlib.Path): Directory to save explanation results.
            model_object: Trained model object to be explained.
            model_name (str): Identifier for the model type (e.g., 'rfr', 'xgbc').
            background_data (pd.DataFrame): Reference data for SHAP explainer.
            shap_data (pd.DataFrame): Data samples to be explained.
            sample_background_data_k (int|float|None): If int, the number of background samples
                to use; if float, the fraction of background data to sample; if None, use all data.
            sample_shap_data_k (int|float|None): If int, the number of samples to explain;
                if float, the fraction of data to explain; if None, explain all data.
        """
        self.results_dir = pathlib.Path(results_dir)
        self.model_obj = model_object
        self.model_name = model_name
        self.background_data = background_data
        self.shap_data = shap_data
        self.sample_background_data_k = sample_background_data_k
        self.sample_shap_data_k = sample_shap_data_k

        # After checking input
        self.classes_ = None

        # After explain()
        self.shap_values = None
        self.shap_base_values = None
        self.feature_names = None
        self.shap_values_dataframe = None
        self.numeric_features = None

        # After plot_results()
        self.show = None
        self.plot_format = None
        self.plot_dpi = None

        self._check_input()

    
    def _check_input(self):
        # Validate input paths and directories
        assert isinstance(self.results_dir, pathlib.Path), "results_dir must be a Path object"
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

        # Validate dataframes
        assert isinstance(self.background_data, pd.DataFrame), "background_data must be a pandas DataFrame"
        assert isinstance(self.shap_data, pd.DataFrame), "shap_data must be a pandas DataFrame"
            
        # Validate sample sizes
        if self.sample_background_data_k:
            if not isinstance(self.sample_background_data_k, (int, float)) or self.sample_background_data_k < 0:
                raise ValueError("sample_background_data_k must be a positive integer or float")
                
        if self.sample_shap_data_k:
            if not isinstance(self.sample_shap_data_k, (int, float)) or self.sample_shap_data_k < 0:
                raise ValueError("sample_shap_data_k must be a positive integer or float")
            if self.sample_shap_data_k > len(self.shap_data):
                raise ValueError("sample_shap_data_k cannot be larger than shap_data set size")

        # For classification tasks, the model's classes_ attribute contains names of all classes,
        # SHAP will output shap_values in corresponding order
        if hasattr(self.model_obj, "classes_"):
            self.classes_ = self.model_obj.classes_
        
        return None

    

    def explain(
            self,
            numeric_features: list[str] | tuple[str],
            plot: bool = True,
            show: bool = False,
            plot_format: str = "jpg",
            plot_dpi: int = 500,
            output_raw_data: bool = False
        ):
        """Transform categorical data to numerical data
        'Cause the input data for calculating SHAP values must be consistent with the training data,
        so we need to convert the categorical variables in the test data to numerical variables.
        """
        self.numeric_features = numeric_features
        self.show = show
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi

        # Check if the model is a multi-class GBDT model
        if self.model_name == "gbdtc" and len(self.classes_) > 2:
            logging.error("SHAP currently does not support explanation for multi-class GBDT models")
            return None

        ###########################################################################################
        # Sampling for reducing the size of the background data and shap data
        if self.sample_background_data_k:
            if isinstance(self.sample_background_data_k, float):
                self.background_data = shap.sample(self.background_data,
                                                   int(self.sample_background_data_k * len(self.background_data)))
            elif isinstance(self.sample_background_data_k, int):
                self.background_data = shap.sample(self.background_data, 
                                                   self.sample_background_data_k)

        if self.sample_shap_data_k:
            if isinstance(self.sample_shap_data_k, float):
                self.shap_data = shap.sample(self.shap_data,
                                             int(self.sample_shap_data_k * len(self.shap_data)))
            elif isinstance(self.sample_shap_data_k, int):
                self.shap_data = shap.sample(self.shap_data,
                                             self.sample_shap_data_k)
        ###########################################################################################


        ###########################################################################################
        # Set the explainer
        # 这里没有使用shap.Explainer，因为对于xgboost和random forest, 它没有选择TreeExplainer
        if self.model_name in ["svr", "knr", "mlpr", "adar"]:
            _explainer = shap.KernelExplainer(self.model_obj.predict, self.background_data)
        elif self.model_name in ["svc", "adac"]:
            # `decision_function` 本身的含义：
            # 返回样本到决策边界的带符号距离（或决策分数）
            # 在二分类任务中：
            # 正值：模型倾向于将样本分类为正类（类别 1）
            # 负值：模型倾向于将样本分类为负类（类别 0）
            # 数值大小：模型确信度的度量
            # 此时输出的shap_values的维度都是(n_samples, n_features), shap_values的值是指对正样本的贡献
            _explainer = shap.KernelExplainer(self.model_obj.decision_function, self.background_data)
        elif self.model_name in ["knc", "mlpc"]:
            # 对于sklearn的knc和mlpc, 因为其模型内部的决策机制是基于概率
            # 所以使用KernelExplainer对它们进行解释的时候，输出的shap_values是概率值
            # 同时注意，输出的shap_values的维度是(n_samples, n_features, n_targets)
            _explainer = shap.KernelExplainer(self.model_obj.predict_proba, self.background_data)
        elif self.model_name in ["dtr", "rfr", "gbdtr", "xgbr", "lgbr", "catr",
                                 "dtc", "rfc", "gbdtc", "xgbc", "lgbc", "catc"]:
            # 对sklearn的decision tree和random forest, 因为其模型内部的决策机制是基于概率
            # 所以使用TreeExplainer对它们进行解释的时候，输出的shap_values是概率值
            # 而对于sklearn的gbdt, 以及下面提到的xgboost, lightgbm, catboost, 因为其模型内部的决策机制是基于log-odds空间
            # 所以使用TreeExplainer对它们进行解释的时候，输出的shap_values是log-odds值
            _explainer = shap.TreeExplainer(self.model_obj)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        ###########################################################################################
        
        # Calculate SHAP values
        _explanation = _explainer(self.shap_data)
        self.shap_values = _explanation.values
        self.shap_base_values = _explanation.base_values
        self.feature_names = _explanation.feature_names


        # Plot the results
        if plot:
            self._plot_results()
        
        # Output the raw data
        if output_raw_data:
            self._output_shap_values()
    
        return None



    def _plot_results(self):
        if self.shap_values.ndim == 2:
            # 回归任务中使用的所有模型、
            # 二分类任务中使用的SVC, adaboost, gbdt, xgboost, lightgbm, catboost模型,
            # 输出的shap_values的维度都是(n_samples, n_features)

            # Summary plot for demonstrating feature importance
            self._plot_summary(
                shap_values = self.shap_values,
                save_dir = self.results_dir,
                file_name = "shap_summary",
                title = "SHAP Summary Plot"
            )

            # Dependence plot for demonstrating relationship between feature values and SHAP values
            self._plot_dependence(
                shap_values = self.shap_values,
                save_dir = self.results_dir.joinpath("dependence_plots/")
            )

            """Partial Dependence Plot 
            is supported for regression task only.
            is not supported for categorical features.
            """
            if self.model_name in ["svr", "knr", "mlpr", "adar", "dtr", "rfr", "gbdtr", "xgbr", "lgbr", "catr"]:
                self._plot_partial_dependence(
                    save_dir = self.results_dir.joinpath("partial_dependence_plots/")
                )
    
        elif self.shap_values.ndim == 3:
            # 二分类任务中使用sklearn的决策树、随机森林模型,
            # 以及多分类任务使用的所有模型，
            # 输出shap_values的维度都是(n_samples, n_features, n_targets)
            # 其中
            # 二分类任务中sklearn的决策树、随机森林，shap values代表的是每个特征对样本被分为正类和负类的概率贡献
            # 因此对每个类别都输出结果，
            # 保存在shap_summary目录下，并根据类别命名
            # 同时，在dependence_plots目录下，根据类别创建子目录，并根据类别命名
            summary_plot_dir = self.results_dir.joinpath("shap_summary")
            summary_plot_dir.mkdir(parents = True, exist_ok = True)
            for i in range(0, len(self.classes_)):
                # Summary plot for ranking features' importance
                self._plot_summary(
                    shap_values = self.shap_values[:, :, i],
                    save_dir = summary_plot_dir,
                    file_name = f"class_{self.classes_[i]}",
                    title = f"SHAP Summary Plot for Class: {self.classes_[i]}"
                )

                # Dependence plot for demonstrating relationship between feature values and SHAP values
                dependence_plot_dir = self.results_dir.joinpath(f"dependence_plots/class_{self.classes_[i]}/")
                dependence_plot_dir.mkdir(parents = True, exist_ok = True)
                self._plot_dependence(
                    shap_values = self.shap_values[:, :, i],
                    save_dir = dependence_plot_dir
                )

        else:
            raise ValueError(f"Invalid SHAP values dimension: {self.shap_values.ndim}")
        
        return None



    def _plot_summary(self, shap_values, save_dir: pathlib.Path, file_name: str, title: str):
        """Summary Plot
        https://shap.readthedocs.io/en/latest/release_notes.html#release-v0-36-0
        """
        fig = plt.figure()
        shap.summary_plot(shap_values, self.shap_data, show = False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_dir.joinpath(file_name + '.' + self.plot_format), dpi = self.plot_dpi)
        if self.show:
            plt.show()
        plt.close("all")
        return None



    def _plot_dependence(self, shap_values, save_dir: pathlib.Path):
        """Dependence Plot
        https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html#Using-color-to-highlight-interaction-effects
        """
        _results_dir = save_dir
        _results_dir.mkdir(parents = True, exist_ok = True)

        def _dp(_feature_name):
            # Close any existing figures before creating a new one
            # shap.dependence_plot creates its own figure internally
            shap.dependence_plot(_feature_name, shap_values, self.shap_data, show = False)
            plt.tight_layout()
            plt.savefig(_results_dir.joinpath(_feature_name + '.' + self.plot_format), dpi = self.plot_dpi)
            if self.show:
                plt.show()
            plt.close("all")
            return None

        for i in self.feature_names:
            _dp(str(i))
            
        return None
    


    def _plot_partial_dependence(self, save_dir: pathlib.Path):
        _results_dir = save_dir
        _results_dir.mkdir(parents = True, exist_ok = True)
        
        def _pdp(_feature_name):
            _fig, _ax = shap.partial_dependence_plot(
                _feature_name,
                self.model_obj.predict,
                self.shap_data,
                model_expected_value = True,
                feature_expected_value = True,
                ice = False,
                show = False
            )
            if self.show:
                _fig.show()
            _fig.tight_layout()
            _fig.savefig(_results_dir.joinpath(_feature_name + '.' + self.plot_format), dpi = self.plot_dpi)
            plt.close("all")
        
        for r in self.numeric_features:
            _pdp(str(r))

        return None 



    def _output_shap_values(self):
        # Create a DataFrame from SHAP values with feature names as columns
        if self.shap_values.ndim == 2:
            # For regression and binary classification models with 2D SHAP values
            self.shap_values_dataframe = pd.DataFrame(
                data=self.shap_values,
                columns=self.feature_names,
                index=self.shap_data.index
            )
            # Output the raw data
            self.shap_data.to_csv(self.results_dir.joinpath("shap_data.csv"), index = True)
            self.shap_values_dataframe.to_csv(self.results_dir.joinpath("shap_values.csv"), index = True)

        elif self.shap_values.ndim == 3:
            # For multi-class classification models with 3D SHAP values, 
            # or 2D SHAP values for binary classification models like SVC, KNC, MLPC, DTC, RFC, GBDTC
            # Create a dictionary of DataFrames, one for each class
            _shap_values_dir = self.results_dir.joinpath("shap_values/")
            _shap_values_dir.mkdir(parents = True, exist_ok = True)
            self.shap_data.to_csv(_shap_values_dir.joinpath("shap_data.csv"), index = True)
            self.shap_values_dataframe = {}
            for i, class_name in enumerate(self.classes_):
                self.shap_values_dataframe[class_name] = pd.DataFrame(
                    data=self.shap_values[:, :, i],
                    columns=self.feature_names,
                    index=self.shap_data.index
                )
            # Output the raw data
            for _class_name, _df in self.shap_values_dataframe.items():
                # print(_df.head(30))
                _df.to_csv(_shap_values_dir.joinpath(f"shap_values_{_class_name}.csv"), index = True)
        
        return None
    