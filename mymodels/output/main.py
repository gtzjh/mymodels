import numpy as np
import pandas as pd
from pathlib import Path
import yaml, pickle
from joblib import dump
import shap



class Output:
    def __init__(
        self,
        results_dir: str | Path
    ):
        """Initialize the Output object.
        
        Args:
            results_dir (str or Path, optional): Directory to save results. Defaults to None.
        """

        assert isinstance(results_dir, Path) \
            or isinstance(results_dir, str), \
            "results_dir must be a valid directory path"
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        results_dir.mkdir(parents = True, exist_ok = True)

        self.results_dir = results_dir



    ###########################################################################################
    # Save optimal parameters
    ###########################################################################################
    def save_optimal_params(self, optimal_params):
        """Save the optimal parameters to a YAML file.
        """
        assert isinstance(optimal_params, dict), \
            "optimal_params must be a dictionary"
        
        with open(self.results_dir.joinpath("params.yml"), 'w', encoding="utf-8") as file:
            yaml.dump(optimal_params, file)
        return None
    ###########################################################################################



    ###########################################################################################
    # Save optimal model
    ###########################################################################################
    def save_optimal_model(self, optimal_model, model_name):
        """Save the optimal model using the recommended export method based on model type.
        
        Different models have different recommended export methods:
        - CatBoost: save_model() method to save in binary format
        - XGBoost: save_model() method to save in binary format  
        - LightGBM: booster_.save_model() method to save in text format
        - Other scikit-learn models: joblib is recommended over pickle
        
        Args:
            optimal_model: The trained model to save
            model_name: String identifier of the model type (e.g., "xgbr", "lgbc")
        """
        assert optimal_model is callable, \
            "optimal_model must be a callable model object"

        model_path = self.results_dir.joinpath("optimal_model")
        
        # XGBoost models
        if model_name in ["xgbr", "xgbc"]:
            optimal_model.save_model(f"{model_path}.json")
            
        # LightGBM models
        elif model_name in ["lgbr", "lgbc"]:
            optimal_model.booster_.save_model(f"{model_path}.txt")
            
        # CatBoost models
        elif model_name in ["catr", "catc"]:
            optimal_model.save_model(f"{model_path}.cbm")
        
        # For scikit-learn based models, use joblib which is more efficient for numpy arrays
        else:
            dump(optimal_model, f"{model_path}.joblib")
            
        # Also save a pickle version for backward compatibility
        with open(self.results_dir.joinpath("optimal_model.pkl"), 'wb') as file:
            pickle.dump(optimal_model, file)
            
        return None
    ###########################################################################################



    ###########################################################################################
    # Output evaluation
    ###########################################################################################
    def output_evaluation(
            self,
            accuracy_dict, 
            save_raw_data=False,
            y_test=None, 
            y_test_pred=None,
            y_train=None,
            y_train_pred=None,
        ):
        """
        Handles saving results to files, printing to console, generating plots,
        and saving raw prediction data based on the configuration settings.
        """

        assert isinstance(accuracy_dict, dict), \
            "accuracy_dict must be a dictionary"
        assert isinstance(save_raw_data, bool), \
            "save_raw_data must be a boolean"
        assert isinstance(y_test, (pd.Series, pd.DataFrame)) or y_test is None, \
            "y_test must be a pandas Series or DataFrame or None"
        assert isinstance(y_test_pred, (pd.Series, pd.DataFrame)) or y_test_pred is None, \
            "y_test_pred must be a pandas Series or DataFrame or None"
        assert isinstance(y_train, (pd.Series, pd.DataFrame)) or y_train is None, \
            "y_train must be a pandas Series or DataFrame or None"
        assert isinstance(y_train_pred, (pd.Series, pd.DataFrame)) or y_train_pred is None, \
            "y_train_pred must be a pandas Series or DataFrame or None"
        
        
        # Save results to files
        with open(self.results_dir.joinpath("accuracy.yml"), 'w', encoding = "utf-8") as file:
            yaml.dump(accuracy_dict, file)


        # Output train and test results
        if save_raw_data:
            y_test_pred_1d = y_test_pred
            y_train_pred_1d = y_train_pred
            
            # Flatten predictions if they are 2D with second dimension of 1
            if len(y_test_pred_1d.shape) > 1 and y_test_pred_1d.shape[1] == 1:
                y_test_pred_1d = y_test_pred_1d.flatten()
            if len(y_train_pred_1d.shape) > 1 and y_train_pred_1d.shape[1] == 1:
                y_train_pred_1d = y_train_pred_1d.flatten()
            
            test_results = pd.DataFrame(data={"y_test": y_test,
                                              "y_test_pred": y_test_pred_1d})
            train_results = pd.DataFrame(data={"y_train": y_train,
                                               "y_train_pred": y_train_pred_1d})
            test_results.to_csv(self.results_dir.joinpath("test_results.csv"), index = True)
            train_results.to_csv(self.results_dir.joinpath("train_results.csv"), index = True)

        return None
    ###########################################################################################



    ###########################################################################################
    # Output SHAP values
    ###########################################################################################
    def output_shap_values(self, shap_explanation):
        """
        Output SHAP values to files and console.
        
        Args:
            shap_explanation: SHAP explanation object
            classes_: List of class names for multi-class classification models
        """

        assert isinstance(shap_explanation, shap.Explanation), \
            "shap_explanation must be a shap.Explanation object"
        
        shap_values = shap_explanation.values
        feature_names = shap_explanation.feature_names
        shap_data = shap_explanation.data
        
        # Create a DataFrame from SHAP values with feature names as columns
        if shap_values.ndim == 2:
            # For regression and binary classification models with 2D SHAP values
            shap_values_dataframe = pd.DataFrame(
                data=shap_values,
                columns=feature_names,
                index=shap_data.index
            )
            # Output the raw data
            shap_data.to_csv(self.results_dir.joinpath("shap_data.csv"), index = True)
            shap_values_dataframe.to_csv(self.results_dir.joinpath("shap_values.csv"), index = True)

        elif shap_values.ndim == 3:
            # For multi-class classification models with 3D SHAP values, 
            # or 2D SHAP values for binary classification models like SVC, KNC, MLPC, DTC, RFC, GBDTC
            # Create a dictionary of DataFrames, one for each class
            _shap_values_dir = self.results_dir.joinpath("shap_values/")
            _shap_values_dir.mkdir(parents = True, exist_ok = True)
            shap_data.to_csv(_shap_values_dir.joinpath("shap_data.csv"), index = True)
            shap_values_dataframe = {}
            for i, class_name in enumerate(classes_):
                shap_values_dataframe[class_name] = pd.DataFrame(
                    data=shap_values[:, :, i],
                    columns=feature_names,
                    index=shap_data.index
                )
            # Output the raw data
            for _class_name, _df in shap_values_dataframe.items():
                _df.to_csv(_shap_values_dir.joinpath(f"shap_values_{_class_name}.csv"), index = True)
        
        return shap_values_dataframe
    ###########################################################################################
