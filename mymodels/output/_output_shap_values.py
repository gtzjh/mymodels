from pathlib import Path

import pandas as pd
import shap


def _output_shap_values(results_dir, shap_explanation, data, _y_mapping_dict = None):
    """
    Output SHAP values to files and console.
    
    Args:
        results_dir: The directory to save the SHAP values
        shap_explanation: SHAP explanation object
        data: The data used to calculate the SHAP values
        _y_mapping_dict: The mapping dictionary of the target variable
    """
    # Check if the results_dir is a valid directory
    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    assert isinstance(shap_explanation, shap.Explanation), \
        "shap_explanation must be a shap.Explanation object"
    assert isinstance(data, pd.DataFrame), \
        "data must be a pandas DataFrame"
    assert isinstance(_y_mapping_dict, dict) or _y_mapping_dict is None, \
        "_y_mapping_dict must be a dictionary or None"
    
    shap_values = shap_explanation.values
    feature_names = shap_explanation.feature_names


    # Create a DataFrame from SHAP values with feature names as columns
    if shap_values.ndim == 2:
        # For regression and binary classification models with 2D SHAP values
        shap_values_dataframe = pd.DataFrame(
            data=shap_values,
            columns=feature_names,
            index=data.index
        )
        # Output the shap values
        shap_values_dataframe.to_csv(results_dir.joinpath("shap_values.csv"),
                                     encoding = "utf-8",
                                     index = True)

    elif shap_values.ndim == 3:
        # For multi-class classification models with 3D SHAP values, 

        # For 2D SHAP values for binary classification models like SVC, KNC, MLPC, DTC, RFC, GBDTC
        if _y_mapping_dict is None:
            _y_mapping_dict = {i: i for i in range(shap_values.shape[2])}

        # Create a dictionary for storing DataFrames, one for each class
        shap_values_dataframe = dict()

        for class_name, i in _y_mapping_dict.items():
            shap_values_dataframe[class_name] = pd.DataFrame(
                data=shap_values[:, :, i],
                columns=feature_names,
                index=data.index
            )
        # Output the raw data
        for _class_name, _df in shap_values_dataframe.items():
            _df.to_csv(results_dir.joinpath(f"shap_values_{_class_name}.csv"),
                       encoding = "utf-8",
                       index = True)
