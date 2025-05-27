import pathlib
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
    dpi = 300,
    format = "jpg",
    y_mapping_dict: dict | None = None
):
    assert isinstance(background_data, pd.DataFrame), \
        "background_data must be a pandas DataFrame"
    assert isinstance(shap_data, pd.DataFrame), \
        "shap_data must be a pandas DataFrame"
    assert isinstance(explainer_type, str) or explainer_type is None, \
        "explainer_type must be a str or None"
    assert isinstance(results_dir, str) or isinstance(results_dir, pathlib.Path), \
        "results_dir must be a str or pathlib.Path"
    assert isinstance(dpi, int), \
        "dpi must be an int"
    assert isinstance(format, str), \
        "format must be a str"
    assert isinstance(y_mapping_dict, dict) or y_mapping_dict is None, \
        "y_mapping_dict must be a dict or None"

    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        if is_regressor(model):
            explainer = shap.Explainer(model.predict, background_data)
        elif is_classifier(model):
            explainer = shap.Explainer(model.predict_proba, background_data)
    shap_explanation = explainer(shap_data)

    # Validate the results directory, and create a SHAP directory if it does not exist
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents = True, exist_ok = True)
    results_dir = results_dir.joinpath("explanation/SHAP/")
    results_dir.mkdir(parents = True, exist_ok = True)

    plot_shap_summary(
        shap_explanation,
        results_dir,
        dpi,
        format,
        y_mapping_dict
    )

    plot_shap_dependence(
        shap_explanation,
        results_dir,
        dpi,
        format,
        y_mapping_dict
    )


def plot_shap_summary(
        shap_explanation: shap.Explanation,
        results_dir: str | pathlib.Path | None = None,
        dpi: int = 300,
        format: str = "jpg",
        y_mapping_dict: dict | None = None
    ):
    """Plot SHAP summary

    Args:
        shap_explanation: SHAP explanation object
        y_mapping_dict: Dictionary mapping class names to their indices
    """
    
    if shap_explanation.values.ndim == 2:
        fig, _ = _get_shap_summary(shap_explanation)
        fig.savefig(
            results_dir.joinpath(f"shap_summary.{format}"),
            dpi=dpi,
            bbox_inches = 'tight'
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

        _fig_ax_dict = _get_shap_summary(shap_explanation)
        for class_idx, (fig, _) in _fig_ax_dict.items():
            fig.savefig(
                results_dir.joinpath(f"shap_summary_class_{str(_inverse_mapping[class_idx])}.{format}"),
                dpi=dpi,
                bbox_inches = 'tight'
            )
            plt.close()

def _get_shap_summary(shap_explanation):
    """SHAP summary plot

    Args:
        shap_explanation: SHAP explanation object
    
    Returns:
        tuple or dict: Either (fig, ax) tuple for 2D values or dict mapping class indices to (fig, ax) tuples for 3D values
    """

    assert isinstance(shap_explanation, shap.Explanation), \
        "shap_explanation must be a shap.Explanation object"
    
    # Sometimes the SHAP explanation object has no feature names
    if shap_explanation.feature_names is None:
        shap_explanation.feature_names = [f"x_{i}" for i in range(shap_explanation.values.shape[1])]

    if shap_explanation.values.ndim == 2:
        _fig = plt.figure()
        _ax = _fig.gca()
        shap.plots.beeswarm(shap_explanation, show = False)
        plt.tight_layout()
        return _fig, _ax

    # For 3D values (multi-class)
    _fig_ax_dict = {}
    for i in range(shap_explanation.values.shape[2]):
        _fig = plt.figure()
        _ax = _fig.gca()

        shap.summary_plot(shap_explanation.values[:, :, i],
                          shap_explanation.data,
                          feature_names = shap_explanation.feature_names,
                          show = False)
        
        plt.tight_layout()
        _fig_ax_dict[i] = (_fig, _ax)

    return _fig_ax_dict


def plot_shap_dependence(
        shap_explanation: shap.Explanation, 
        results_dir: str | pathlib.Path | None = None,
        dpi: int = 300,
        format: str = "jpg",
        y_mapping_dict: dict | None = None
    ):
    """Plot SHAP dependence
    
    Args:
        shap_explanation: SHAP explanation object
        y_mapping_dict: Dictionary mapping class names to their indices
    """

    # Create a SHAP dependence directory
    _dp_results_dir = results_dir.joinpath("dependence/")
    _dp_results_dir.mkdir(parents = True, exist_ok = True)

    shap_dp_plots = _get_shap_dependence(shap_explanation)
    if shap_explanation.values.ndim == 2:
        for fig, _, feature_name in shap_dp_plots:
            fig.savefig(
                _dp_results_dir.joinpath(f"{feature_name}.{format}"),
                dpi=dpi,
                bbox_inches = 'tight'
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
        
        for class_idx, shap_dp_plots in shap_dp_plots.items():
            for fig, _, feature_name in shap_dp_plots:
                fig.savefig(
                    _dp_results_dir.joinpath(f"class_{_inverse_mapping[class_idx]}_{feature_name}.{format}"),
                    dpi=dpi,
                    bbox_inches = 'tight'
                )
                plt.close()

def _get_shap_dependence(shap_explanation):
    """SHAP dependence plot

    Args:
        shap_explanation: SHAP explanation object
    
    Returns:
        list or dict: 
            Either list of (fig, ax, feature_name) tuples for 2D values or 
            dict mapping class indices to lists of (fig, ax, feature_name) tuples for 3D values
    """
    
    assert isinstance(shap_explanation, shap.Explanation), \
        "shap_explanation must be a shap.Explanation object"

    # Sometimes the SHAP explanation object has no feature names
    # We need to specify the feature names manually
    if shap_explanation.feature_names is None:
        shap_explanation.feature_names = [f"x_{i}" for i in range(shap_explanation.values.shape[1])]

    # For binary classification (except for decision tree and random forest of sklearn)
    if shap_explanation.values.ndim == 2:
        shap_dp_plots = []
        for i, feature_name in enumerate(shap_explanation.feature_names):
            # Use feature index instead of name for direct indexing
            shap.dependence_plot(
                i,
                shap_explanation.values,
                shap_explanation.data,
                shap_explanation.feature_names,
                # display_features = None,
                interaction_index = "auto",
                show=False
            )
            plt.tight_layout()
            fig = plt.gcf()
            ax = plt.gca()
            shap_dp_plots.append((fig, ax, feature_name))
        return shap_dp_plots

    # For multi-class, or decision tree and random forest in sklearn for binary classification
    shap_dp_plots = {}
    for class_idx in range(shap_explanation.values.shape[2]):
        shap_dp_plots[class_idx] = []
        for i, feature_name in enumerate(shap_explanation.feature_names):
            # Use feature index instead of name for direct indexing
            shap.dependence_plot(
                i,
                shap_explanation.values[:, :, class_idx],
                shap_explanation.data,
                shap_explanation.feature_names,
                # display_features = None,
                interaction_index = "auto",
                show=False
            )
            plt.tight_layout()
            fig = plt.gcf()
            ax = plt.gca()
            shap_dp_plots[class_idx].append((fig, ax, feature_name))

    return shap_dp_plots

