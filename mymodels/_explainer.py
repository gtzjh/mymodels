import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import logging


# logging.getLogger().setLevel(logging.WARNING)
# shap.initjs()
matplotlib.use('Agg')


class MyExplainer:
    def __init__(
            self,
            model_object,
            model_name: str,
            background_data: pd.DataFrame,
            shap_data: pd.DataFrame,
            sample_background_data_k: int | float | None = None,
            sample_shap_data_k:  int | float | None = None,
        ):
        """Initialize the MyExplainer with model and data.

        Args:
            model_object: Trained model object to be explained.
            model_name (str): Identifier for the model type (e.g., 'rfr', 'xgbc').
            background_data (pd.DataFrame): Reference data for SHAP explainer.
            shap_data (pd.DataFrame): Data samples to be explained.
            sample_background_data_k (int|float|None): If int, the number of background samples
                to use; if float, the fraction of background data to sample; if None, use all data.
            sample_shap_data_k (int|float|None): If int, the number of samples to explain;
                if float, the fraction of data to explain; if None, explain all data.
        """
        if not callable(self.model_obj):
            raise ValueError("model_object must be callable")
        
        self.model_obj = model_object
        self.model_name = model_name
        self.background_data = background_data
        self.shap_data = shap_data
        self.sample_background_data_k = sample_background_data_k
        self.sample_shap_data_k = sample_shap_data_k

        # After checking input
        self.classes_ = None
        # After explain()
        self.results_dir = None
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

    

    def explain(self,
            results_dir: str | pathlib.Path,
            numeric_features: list[str] | tuple[str],
            plot: bool = True,
            show: bool = False,
            plot_format: str = "jpg",
            plot_dpi: int = 500,
            output_raw_data: bool = False
        ):
        """Calculate SHAP values and generate explanations.
        
        Args:
            numeric_features: List of feature names considered numerical
            plot: Whether to generate visualization plots
            show: Directly display plots when True
            plot_format: Image format for saving plots (jpg/png/pdf)
            plot_dpi: Image resolution in dots per inch
            output_raw_data: Export raw SHAP values to CSV when True
        """
        # Convert to list if input is tuple
        self.results_dir = pathlib.Path(results_dir)
        self.numeric_features = list(numeric_features) if isinstance(numeric_features, tuple) else numeric_features
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
        # Here we do not use shap.Explainer, because for xgboost and random forest, it does not choose TreeExplainer by default
        if self.model_name in ["lr", "svr", "knr", "mlpr", "adar"]:
            _explainer = shap.KernelExplainer(self.model_obj.predict, self.background_data)
        elif self.model_name in ["lc", "svc", "knc", "mlpc", "adac"]:
            _explainer = shap.KernelExplainer(self.model_obj.predict_proba, self.background_data)
        elif self.model_name in ["dtr", "rfr", "gbdtr", "xgbr", "lgbr", "catr",
                                 "dtc", "rfc", "gbdtc", "xgbc", "lgbc", "catc"]:
            # For sklearn's decision tree and random forest, since their internal decision mechanisms are probability-based
            # when using TreeExplainer to explain them, the output shap_values are probability values
            # For sklearn's gbdt, as well as xgboost, lightgbm, catboost mentioned below, since their internal decision mechanisms are based on log-odds space
            # when using TreeExplainer to explain them, the output shap_values are log-odds values
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
            # All models used in regression tasks,
            # and all models used in binary classification tasks: SVC, adaboost, gbdt, xgboost, lightgbm, catboost,
            # The dimensions of the output shap_values are (n_samples, n_features)

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
            if self.model_name in ["lr", "svr", "knr", "mlpr", "adar", "dtr", "rfr", "gbdtr", "xgbr", "lgbr", "catr"]:
                self._plot_partial_dependence(
                    save_dir = self.results_dir.joinpath("partial_dependence_plots/")
                )
    
        elif self.shap_values.ndim == 3:
            # For binary classification tasks using sklearn's decision tree and random forest models,
            # as well as all models used in multi-classification tasks,
            # the dimensions of the output shap_values are (n_samples, n_features, n_targets)
            # Where:
            # In binary classification tasks with sklearn's decision tree and random forest, shap values represent 
            # each feature's contribution to the probability of a sample being classified as positive or negative
            # Therefore, results are output for each class,
            # saved in the shap_summary directory, and named according to the class
            # Similarly, in the dependence_plots directory, subdirectories are created and named according to the class
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
    