import numpy as np
import pandas as pd
import shap
import pathlib
import logging



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
    
        self.model_object = model_object
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

        self._check_input()

    
    def _check_input(self):
        """Check the input data and model object."""

        if not callable(self.model_object):
            raise ValueError("model_object must be callable")
        
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
        ):
        """Calculate SHAP values and generate explanations.
        
        Args:
            numeric_features: List of feature names considered numerical
        """
        # Convert to list if input is tuple
        self.numeric_features = list(numeric_features) if isinstance(numeric_features, tuple) else numeric_features

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
    
        return None
