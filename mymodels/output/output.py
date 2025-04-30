from pathlib import Path
import numpy as np
import pandas as pd

from ._output_evaluation import _output_evaluation
from ._output_optimal_params import _output_optimal_params
from ._output_optimal_model import _output_optimal_model
from ._output_raw_data import _output_raw_data
from ._output_shap_values import _output_shap_values


class Output:
    def __init__(
        self,
        results_dir: str | Path,
        save_optimal_model: bool = False,
        save_raw_data: bool = False,
        save_shap_values: bool = False
    ):
        """Initialize the Output object.
        
        Args:
            results_dir (str or Path, optional): Directory to save results. Defaults to None.
            save_optimal_model (bool, optional): Whether to save the optimal model. Defaults to False.
            save_raw_data (bool, optional): Whether to save the raw data. Defaults to False.
            output_shap_values (bool, optional): Whether to output the SHAP values. Defaults to False.
        """

        assert isinstance(results_dir, Path) \
            or isinstance(results_dir, str), \
            "results_dir must be a valid directory path"
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        results_dir.mkdir(parents = True, exist_ok = True)

        self.results_dir = results_dir

        assert isinstance(save_optimal_model, bool), \
            "save_optimal_model must be a boolean"
        assert isinstance(save_raw_data, bool), \
            "save_raw_data must be a boolean"
        assert isinstance(save_shap_values, bool), \
            "save_shap_values must be a boolean"
        
        self.save_optimal_model = save_optimal_model
        self.save_raw_data = save_raw_data
        self.save_shap_values = save_shap_values
    

    def output_evaluation(
        self,
        accuracy_dict: dict
    ):
        _results_dir = self.results_dir.joinpath("evaluation/")
        _output_evaluation(_results_dir, accuracy_dict)


    def output_optimal_params(
        self,
        optimal_params: dict
    ):
        _results_dir = self.results_dir.joinpath("optimization/")
        _output_optimal_params(_results_dir, optimal_params)

    
    def output_optimal_model(
        self,
        optimal_model: object,
        model_name: str
    ):
        _results_dir = self.results_dir.joinpath("optimization/")
        if self.save_optimal_model:
            _output_optimal_model(_results_dir, optimal_model, model_name)

    
    def output_raw_data(
        self,
        y_test: pd.Series | pd.DataFrame, 
        y_test_pred: pd.Series | pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        y_train_pred: pd.Series | pd.DataFrame,
    ):
        _results_dir = self.results_dir.joinpath("evaluation/raw_data/")
        if self.save_raw_data:
            _output_raw_data(_results_dir, y_test, y_test_pred, y_train, y_train_pred)


    def output_shap_values(
        self,
        shap_explanation: object,
        data: pd.DataFrame,
        _y_mapping_dict: dict | None = None
    ):
        _results_dir = self.results_dir.joinpath("explanation/SHAP/")
        if self.save_shap_values:
            _output_shap_values(_results_dir, shap_explanation, data, _y_mapping_dict)
        
        

