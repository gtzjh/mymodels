import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import logging


logging.getLogger().setLevel(logging.WARNING)


# shap.initjs()
# matplotlib.use('Agg')
plt.rc('font', family = 'Times New Roman')


"""
SHAP 的可加性: 样本在所有特征的shap_values的和, 再加上base_value(expected_value)即为该样本的预测输出值。

xgboost, lightgbm, catboost都是在log-odds空间中工作, 并将最后的预测值转换回概率值
因此, TreeExplainer会对它们输出的概率进行对数几率转换, 会回到模型原始的log-odds空间中解释
log-odds 是 SHAP 值计算的理想空间，能够保证特征贡献的可加性、无界性和一致的解释。
对数几率转换: ln(p/(1 - p))
该值>0代表预测为正类(p>0.5), <0代表预测为负类(p<0.5), =0代表预测为中性(p=0.5),

对于二分类任务, 输出的shap_values的维度是(n_samples, n_features), shap_values的值是指对正类的贡献, 是在 log-odds 空间中解释
对于多分类任务, 输出的shap_values的维度是(n_samples, n_features, n_targets), shap_values的值是指对每个类别的贡献

举例
二分类模型预测的概率值
使用 `.predict_proba()` 方法获取预测概率值

>> proba = self.model_obj.predict_proba(self.shap_data)[0]
>> print("\n第一个样本的模型预测概率值[阴性, 阳性]:")
>> print(proba[0], proba[1])

尝试对两个类别的输出概率进行log-odds转换
对于二分类问题, 正类的对数几率与负类的对数几率符号相反但绝对值相同
即: log-odds(p) = -log-odds(1 - p)

>> def log_odds(x):
>>     return np.log(x / (1 - x))
>> print("\n第一个样本的输出概率的log-odds转换后的值[阴性, 阳性]:")
>> print(log_odds(proba[0]), log_odds(proba[1]))

>> feature_sum_i = np.sum(self.shap_values[0, :])
>> base_value_i = self.shap_base_values[0]
>> sum_i = feature_sum_i + base_value_i
>> print("\n第一个样本的shap预测值(shap_values + base_value) | shap_values的和 | base_value")
>> print(sum_i, feature_sum_i, base_value_i)
"""


"""SHAP visualization functions comparison and selection guide.

Attributes:
    Supported visualization methods:
    1. partial_dependence_plot (currently unused):
        - Shows average marginal effect of a feature on model output
        - Similar to traditional Partial Dependence Plot (PDP)
        - X-axis: Feature values, Y-axis: Model predictions

    2. dependence_plot (implemented):
        - Visualizes feature value vs SHAP value relationship
        - Automatically detects interactions (color-codes 2nd influential feature)
        - X-axis: Feature values, Y-axis: SHAP values

    3. summary_plot (implemented):
        - Displays feature importance and value impacts
        - Combines feature importance with SHAP value distributions
        - Y-axis: Feature names, X-axis: SHAP values

Key Differences Table:
    | Function                | Data Used       | SHAP Values | Interactions | Output Scale   |
    |-------------------------|-----------------|-------------|--------------|----------------|
    | partial_dependence_plot | Raw features    | ❌          | ❌          | Model output   |
    | dependence_plot         | Features+SHAP   | ✅          | ✅          | SHAP values    |
    | summary_plot            | SHAP values     | ✅          | ❌          | SHAP magnitude |

Implementation Note:
    Current implementation focuses on:
    - summary_plot for global feature importance
    - dependence_plot for detailed feature analysis
    - Partial dependence plots are deprecated in this implementation due to:
        * Lack of categorical feature support
        * Redundancy with dependence_plot functionality
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
            cat_features: list[str] | tuple[str] | None = None,           
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
            cat_features (list[str]|tuple[str]|None): Names of categorical features in the dataset.
        """
        self.results_dir = pathlib.Path(results_dir)
        self.model_obj = model_object
        self.model_name = model_name
        self.background_data = background_data
        self.shap_data = shap_data
        self.sample_background_data_k = sample_background_data_k
        self.sample_shap_data_k = sample_shap_data_k
        self.cat_features = cat_features

        # After checking input
        self.classes_ = None

        # After explain()
        self.shap_values = None
        self.shap_base_values = None
        self.feature_names = None
        self.shap_values_dataframe = None

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

        # Validate categorical features if provided
        if self.cat_features:
            if not isinstance(self.cat_features, (list, tuple)):
                raise TypeError("cat_features must be a list or tuple")

        # For classification tasks, the model's classes_ attribute contains names of all classes,
        # SHAP will output shap_values in corresponding order
        if hasattr(self.model_obj, "classes_"):
            self.classes_ = self.model_obj.classes_
        
        return None

    
    def explain(
            self,
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
        self.show = show
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi


        if self.model_name == "gbdtc" and len(self.classes_) > 2:
            logging.error("SHAP currently does not support explanation for multi-class GBDT models")
            return None

        # Sampling or not for reducing the size of the background data and shap data
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
        
        # Calculate SHAP values
        _explanation = _explainer(self.shap_data)
        self.shap_values = _explanation.values
        self.shap_base_values = _explanation.base_values
        self.feature_names = _explanation.feature_names


        # Plot the results
        if plot:
            self.plot_results()
        
        # Output the raw data
        if output_raw_data:
            self.output_shap_values()
    
        return None


    def plot_results(self):

        if self.shap_values.ndim == 2:
            # 回归任务中使用的所有模型、
            # 二分类任务中使用的SVC, adaboost, gbdt, xgboost, lightgbm, catboost模型,
            # 输出的shap_values的维度都是(n_samples, n_features)

            # Summary plot for demonstrating feature importance
            self.summary_plot(
                shap_values = self.shap_values,
                save_dir = self.results_dir,
                file_name = "shap_summary",
                title = "SHAP Summary Plot"
            )

            # Dependence plot for demonstrating relationship between feature values and SHAP values
            self.dependence_plot(
                shap_values = self.shap_values,
                save_dir = self.results_dir.joinpath("dependence_plots/")
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
                # Summary plot for demonstrating feature importance
                self.summary_plot(
                    shap_values = self.shap_values[:, :, i],
                    save_dir = summary_plot_dir,
                    file_name = f"class_{self.classes_[i]}",
                    title = f"SHAP Summary Plot for Class: {self.classes_[i]}"
                )

                # Dependence plot for demonstrating relationship between feature values and SHAP values
                dependence_plot_dir = self.results_dir.joinpath(f"dependence_plots/class_{self.classes_[i]}/")
                dependence_plot_dir.mkdir(parents = True, exist_ok = True)
                self.dependence_plot(
                    shap_values = self.shap_values[:, :, i],
                    save_dir = dependence_plot_dir
                )
        else:
            raise ValueError(f"Invalid SHAP values dimension: {self.shap_values.ndim}")
        
        return None


    def summary_plot(self, shap_values, save_dir: pathlib.Path, file_name: str, title: str):
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
        plt.close()
        return None


    def dependence_plot(self, shap_values, save_dir: pathlib.Path):
        """Dependence Plot
        https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html#Using-color-to-highlight-interaction-effects
        """
        results_dir = save_dir
        results_dir.mkdir(parents = True, exist_ok = True)

        def _plot_dependence_plot(_feature_name):
            # Close any existing figures before creating a new one
            plt.close('all')
            # shap.dependence_plot creates its own figure internally
            shap.dependence_plot(_feature_name, shap_values, self.shap_data, show = False)
            plt.tight_layout()
            plt.savefig(results_dir.joinpath(_feature_name + '.' + self.plot_format), dpi = self.plot_dpi)
            if self.show:
                plt.show()
            # Make sure to close all figures
            plt.close('all')
            return None

        for i in self.feature_names:
            _plot_dependence_plot(str(i))
            
        return None
    

    # 这个功能暂时不用
    """
    def partial_dependence_plot(self):
        # Partial Dependence Plot
        # Partial Dependence Plot is not supported for categorical features.
        _results_dir = self.results_dir.joinpath("partial_dependence_plots")
        _results_dir.mkdir(parents = True, exist_ok = True)
        for _feature_name in self.feature_names:
            if _feature_name not in self.cat_features:
                shap.partial_dependence_plot(
                    _feature_name,
                    self.model_obj.predict,
                    self.shap_data,
                    model_expected_value = True,
                    feature_expected_value = False,
                    ice = False,
                    show = False
                )
                plt.tight_layout()
                plt.savefig(_results_dir.joinpath(_feature_name + '.' + self.plot_format), dpi = self.plot_dpi)
                plt.close()
        return None 
    """

    def output_shap_values(self):
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
    