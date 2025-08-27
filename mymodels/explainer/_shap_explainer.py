import pathlib
from typing import Dict, Optional, Union

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.base import is_classifier, is_regressor


def shap_explainer(
    model,
    background_data,
    shap_data,
    explainer_type,
    results_dir,
    dpi=300,
    format="jpg",
    y_mapping_dict: Optional[Dict] = None
):
    """Generate SHAP explanations for a model.

    Args:
        model: The trained model to explain
        background_data: Background data for the explainer
        shap_data: Data to generate SHAP values for
        explainer_type: Type of explainer to use
        results_dir: Directory to save results
        dpi: DPI for saved figures
        format: Format for saved figures
        y_mapping_dict: Optional mapping of class indices to names
    """
    assert isinstance(background_data, pd.DataFrame), \
        "background_data must be a pandas DataFrame"
    assert isinstance(shap_data, pd.DataFrame), \
        "shap_data must be a pandas DataFrame"
    assert isinstance(explainer_type, str) \
        or explainer_type is None, \
        "explainer_type must be a str or None"
    assert isinstance(results_dir, str) \
        or isinstance(results_dir, pathlib.Path), \
        "results_dir must be a str or pathlib.Path"
    assert isinstance(dpi, int), \
        "dpi must be an int"
    assert isinstance(format, str), \
        "format must be a str"
    assert isinstance(y_mapping_dict, dict) \
        or y_mapping_dict is None, \
        "y_mapping_dict must be a dict or None"

    # Select the explainer
    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        if is_regressor(model):
            explainer = shap.Explainer(model.predict, background_data)
        elif is_classifier(model):
            explainer = shap.Explainer(model.predict_proba, background_data)
        else:
            raise ValueError("Model must be either a classifier or regressor")
    
    # Calculate the SHAP values
    shap_explanation = explainer(shap_data)

    # Validate the results directory, and create a SHAP directory if it does not exist
    results_dir = pathlib.Path(results_dir)
    results_dir = results_dir.joinpath("explanation/SHAP/")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save the SHAP summary
    _plot_shap_summary(
        shap_explanation,
        results_dir,
        dpi,
        format,
        y_mapping_dict
    )

    # Plot and save the SHAP dependence plots
    _plot_shap_dependence(
        shap_explanation,
        results_dir,
        dpi,
        format,
        y_mapping_dict
    )

    # Output the SHAP values
    # Only for binary classification or regression
    # But also except for random forest classifier and decision tree classifier in sklearn package
    if shap_explanation.values.ndim == 2:
        shap_values_df = pd.DataFrame(
            shap_explanation.values,
            columns = shap_explanation.feature_names,
            index = shap_data.index
        )
        shap_values_df.to_csv(results_dir.joinpath("shap_values.csv"), encoding="utf-8")

def _plot_shap_summary(
    shap_explanation: shap.Explanation,
    results_dir: Union[str, pathlib.Path],
    dpi: int = 300,
    format: str = "jpg",
    y_mapping_dict: Optional[Dict] = None
):
    """Plot and save SHAP summary.

    Args:
        shap_explanation: SHAP explanation object
        results_dir: Directory to save results
        dpi: DPI for saved figures
        format: Format for saved figures
        y_mapping_dict: Dictionary mapping class names to their indices
    """
    # Sometimes the SHAP explanation object has no feature names
    if shap_explanation.feature_names is None:
        shap_explanation.feature_names = [
            f"x_{i}" for i in range(shap_explanation.values.shape[1])
        ]

    if shap_explanation.values.ndim == 2:
        _fig = plt.figure()
        _ax = _fig.gca()
        shap.plots.beeswarm(shap_explanation, show=False)
        plt.tight_layout()
        
        _fig.savefig(
            results_dir.joinpath(f"shap_summary.{format}"),
            dpi=dpi,
            bbox_inches='tight'
        )
        plt.close()

    elif shap_explanation.values.ndim == 3:
        if y_mapping_dict is not None:
            # Inverse the mapping dict (value → key)
            _inverse_mapping = {v: k for k, v in y_mapping_dict.items()}
        else:
            # If y_mapping_dict is not provided, use the index of the class
            _num_classes = shap_explanation.values.shape[2]
            _inverse_mapping = {i: str(i) for i in range(_num_classes)}
        
        # Output the SHAP summary of different classes
        for i in range(shap_explanation.values.shape[2]):
            _fig = plt.figure()
            _ax = _fig.gca()

            shap.summary_plot(
                shap_explanation.values[:, :, i],
                shap_explanation.data,
                feature_names=shap_explanation.feature_names,
                show=False
            )
            
            plt.tight_layout()
            
            _fig.savefig(
                results_dir.joinpath(
                    f"shap_summary_class_{str(_inverse_mapping[i])}.{format}"
                ),
                dpi=dpi,
                bbox_inches='tight'
            )
            plt.close()


def _plot_shap_dependence(
    shap_explanation: shap.Explanation, 
    results_dir: Union[str, pathlib.Path],
    dpi: int = 300,
    format: str = "jpg",
    y_mapping_dict: Optional[Dict] = None
):
    """Plot and save SHAP dependence plots.
    
    Args:
        shap_explanation: SHAP explanation object
        results_dir: Directory to save results
        dpi: DPI for saved figures
        format: Format for saved figures
        y_mapping_dict: Dictionary mapping class names to their indices
    """
    # Create a SHAP dependence directory
    _dp_results_dir = results_dir.joinpath("dependence/")
    _dp_results_dir.mkdir(parents=True, exist_ok=True)

    # For binary classification or regression
    if shap_explanation.values.ndim == 2:
        for i, feature_name in enumerate(shap_explanation.feature_names):
            # Use feature index instead of name for direct indexing
            shap.dependence_plot(
                i,
                shap_explanation.values,
                shap_explanation.data,
                shap_explanation.feature_names,
                interaction_index="auto",
                show=False
            )
            plt.tight_layout()
            fig = plt.gcf()
            
            fig.savefig(
                _dp_results_dir.joinpath(f"{feature_name}.{format}"),
                dpi=dpi,
                bbox_inches='tight'
            )
            plt.close()

    # For multi-class, or decision tree and random forest in sklearn for binary classification
    elif shap_explanation.values.ndim == 3:
        if y_mapping_dict is not None:
            # Inverse the mapping dict (value → key)
            _inverse_mapping = {v: k for k, v in y_mapping_dict.items()}
        else:
            # If y_mapping_dict is None, use the index of the class
            _num_classes = shap_explanation.values.shape[2]
            _inverse_mapping = {i: str(i) for i in range(_num_classes)}
        
        # Output each feature's dependence plots of different classes
        for i, feature_name in enumerate(shap_explanation.feature_names):

            # Create a sub directory for each feature
            _sub_dir = _dp_results_dir.joinpath(feature_name)
            _sub_dir.mkdir(parents=True, exist_ok=True)

            # Output the dependence plots of different classes
            for class_idx in range(shap_explanation.values.shape[2]):
                shap.dependence_plot(
                    i,
                    shap_explanation.values[:, :, class_idx],
                    shap_explanation.data,
                    shap_explanation.feature_names,
                    interaction_index="auto",
                    show=False
                )
                plt.tight_layout()
                fig = plt.gcf()
                
                fig.savefig(
                    _sub_dir.joinpath(
                        f"class_{_inverse_mapping[class_idx]}_{feature_name}.{format}"
                    ),
                    dpi=dpi,
                    bbox_inches='tight'
                )
                plt.close()
