import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import shap
import logging


from ._data_loader import MyDataLoader
from ._estimator import MyEstimator
from ..plotting import Plotter
from ..output import Output


class MyExplainer:
    def __init__(
        self,
        optimized_estimator: MyEstimator,
        optimized_dataset: MyDataLoader,
        optimized_data_engineer_pipeline: Pipeline | None = None,
        plotter: Plotter | None = None,
        output: Output | None = None
    ):
    
        """A class for evaluating machine learning models.

        This class handles the evaluation of machine learning models, computing various 
        accuracy metrics (R², RMSE, MAE, F1, Kappa, etc.), visualizing actual vs predicted values, 
        and saving/printing results.
        
        Args:
            optimized_dataset: Dataset containing the train and test data.
            optimized_estimator: Trained estimator to evaluate.
            optimized_data_engineer_pipeline: Optional data engineering pipeline to transform data.
            plotter: The plotter to use.
            output: The output object.
        """

        # Validate input
        assert isinstance(optimized_dataset, MyDataLoader), \
            "optimized_dataset must be a mymodels.MyDataLoader object"
        assert isinstance(optimized_estimator, MyEstimator), \
            "optimized_estimator must be a mymodels.MyEstimator object"
        # Check data_engineer_pipeline validity
        if optimized_data_engineer_pipeline is not None:
            assert isinstance(optimized_data_engineer_pipeline, Pipeline), \
                "optimized_data_engineer_pipeline must be a `sklearn.pipeline.Pipeline` object"
        else:
            logging.warning("No data engineering will be implemented, the raw data will be used.")

        # Initialize attributes
        self.optimized_dataset = optimized_dataset
        self.optimized_estimator = optimized_estimator
        self.optimized_data_engineer_pipeline = optimized_data_engineer_pipeline
        self.optimal_model_object = self.optimized_estimator.optimal_model_object

        self.plotter = plotter
        self.output = output

        self._shap_explanation = None
    

    def explain(
            self,
            select_background_data: str = "train",
            select_shap_data: str = "test",
            sample_background_data_k: int | float | None = None,
            sample_shap_data_k:  int | float | None = None,
        ):
        """Use training set to build the explainer, use test set to calculate SHAP values.

        Args:
            select_background_data (str): The data to use to build the explainer.
            select_shap_data (str): The data to use to calculate SHAP values.
            sample_background_data_k (int | float | None): The number of samples to use to build the explainer.
            sample_shap_data_k (int | float | None): The number of samples to use to calculate SHAP values.
        """

        # Transform X data
        if self.optimized_data_engineer_pipeline:
            _transformed_x_train = self.optimized_data_engineer_pipeline.transform(self.optimized_dataset.x_train)
            _transformed_x_test = self.optimized_data_engineer_pipeline.transform(self.optimized_dataset.x_test)
        else:
            _transformed_x_train = self.optimized_dataset.x_train
            _transformed_x_test = self.optimized_dataset.x_test


        ###########################################################################################
        # Background data for building the explainer
        if select_background_data == "train":
            _background_data = _transformed_x_train
        elif select_background_data == "test":
            _background_data = _transformed_x_test
        elif select_background_data == "all":
            _background_data = pd.concat([_transformed_x_train, _transformed_x_test]).sort_index()

        # SHAP data for calculating SHAP values
        if select_shap_data == "train":
            _shap_data = _transformed_x_train
        elif select_shap_data == "test":
            _shap_data = _transformed_x_test
        elif select_shap_data == "all":
            _shap_data = pd.concat([_transformed_x_train, _transformed_x_test]).sort_index()
        ###########################################################################################

        ###########################################################################################
        # Sampling the background data and shap data
        if sample_background_data_k:
            if isinstance(sample_background_data_k, float):
                _background_data = shap.sample(_background_data,
                                               int(sample_background_data_k * len(_background_data)))
            elif isinstance(sample_background_data_k, int):
                _background_data = shap.sample(_background_data,
                                               sample_background_data_k)

        if sample_shap_data_k:
            if isinstance(sample_shap_data_k, float):
                _shap_data = shap.sample(_shap_data,
                                         int(sample_shap_data_k * len(_shap_data)))
            elif isinstance(sample_shap_data_k, int):
                _shap_data = shap.sample(_shap_data,
                                         sample_shap_data_k)
        ###########################################################################################

        ###########################################################################################
        # Build the explainer
        if self.optimized_estimator.shap_explainer_type == "kernel":
            _explainer = shap.KernelExplainer(self.optimized_estimator.optimal_model_object.predict,
                                              _background_data)
        elif self.optimized_estimator.shap_explainer_type == "tree":
            _explainer = shap.TreeExplainer(self.optimized_estimator.optimal_model_object)
        else:
            raise ValueError(f"Unregistered SHAP explainer type: {self.optimized_estimator.shap_explainer_type}")

        # Calculate
        _shap_explanation = _explainer(_shap_data)

        ###########################################################################################

        ###########################################################################################
        # Plot
        self.plotter.plot_shap_summary(_shap_explanation,
                                       self.optimized_dataset.y_mapping_dict)
        self.plotter.plot_shap_dependence(_shap_explanation,
                                          self.optimized_dataset.y_mapping_dict)
        
        # Output
        self.output.output_shap_values(_shap_explanation,
                                       _shap_data,
                                       self.optimized_dataset.y_mapping_dict)
        ###########################################################################################

        return None

