import pandas as pd

from ._pdp_explainer import pdp_explainer
from ._shap_explainer import shap_explainer


class MyExplainer:
    def __init__(
            self,
            optimized_estimator,
            optimized_dataset,
            optimized_data_engineer_pipeline,
            plotter
        ):
        self.estimator = optimized_estimator
        self.dataset = optimized_dataset
        self.data_engineer = optimized_data_engineer_pipeline
        
        # The format parameters
        self.results_dir = plotter.results_dir
        self.dpi = plotter.plot_dpi
        self.format = plotter.plot_format


    def explain(
            self,
            select_background_data: str = "train",
            select_shap_data: str = "test",
            sample_background_data_k: int | float | None = None,
            sample_shap_data_k: int | float | None = None,
        ):
        """Use training set to build the explainer,
           use test set to calculate SHAP values by default.
           Use the SHAP data to calculate PDP values as well.

        Args:
            select_background_data (str):
                The data to use to build the explainer.
            select_shap_data (str):
                The data to use to calculate SHAP values.
            sample_background_data_k (int | float | None):
                The number of samples to use to build the explainer.
            sample_shap_data_k (int | float | None):
                The number of samples to use to calculate SHAP values.
        """
        ###########################################################################################
        # Transform X data
        if self.data_engineer:
            _transformed_x_train = self.data_engineer.transform(
                self.dataset.x_train
            )
            _transformed_x_test = self.data_engineer.transform(
                self.dataset.x_test
            )
        else:
            _transformed_x_train = self.dataset.x_train
            _transformed_x_test = self.dataset.x_test
        ###########################################################################################

        ###########################################################################################
        # Background data for building the explainer
        if select_background_data == "train":
            background_data = _transformed_x_train
        elif select_background_data == "test":
            background_data = _transformed_x_test
        elif select_background_data == "all":
            background_data = pd.concat([_transformed_x_train, _transformed_x_test]).sort_index()

        # SHAP data for calculating SHAP values
        if select_shap_data == "train":
            shap_data = _transformed_x_train
        elif select_shap_data == "test":
            shap_data = _transformed_x_test
        elif select_shap_data == "all":
            shap_data = pd.concat([_transformed_x_train, _transformed_x_test]).sort_index()

        # Sample the background data
        if sample_background_data_k:
            if isinstance(sample_background_data_k, float):
                background_data = background_data.sample(
                    int(sample_background_data_k * len(background_data))
                )
            elif isinstance(sample_background_data_k, int):
                background_data = background_data.sample(
                    sample_background_data_k
                )

        # Sample the SHAP data
        if sample_shap_data_k:
            if isinstance(sample_shap_data_k, float):
                shap_data = shap_data.sample(
                    int(sample_shap_data_k * len(shap_data))
                )
            elif isinstance(sample_shap_data_k, int):
                shap_data = shap_data.sample(
                    sample_shap_data_k
                )
        ###########################################################################################
        
        ###########################################################################################
        # SHAP explainer
        shap_explainer(
            model = self.estimator.optimal_model_object,
            background_data = background_data,
            shap_data = shap_data,
            explainer_type = self.estimator.shap_explainer_type,
            results_dir = self.results_dir,
            dpi = self.dpi,
            format = self.format,
            y_mapping_dict = self.dataset.y_mapping_dict
        )

        # PDP explainer
        pdp_explainer(
            model = self.estimator.optimal_model_object,
            explain_data = shap_data,
            results_dir = self.results_dir,
            dpi = self.dpi,
            format = self.format,
            y_mapping_dict = self.dataset.y_mapping_dict
        )
        ###########################################################################################
