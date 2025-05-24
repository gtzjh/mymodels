"""Model explainability visualization module.

This module provides functions for visualizing SHAP (SHapley Additive exPlanations) values.
"""
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def _plot_shap_summary(shap_explanation):
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
    _fig_ax_dict = dict()
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



def _plot_shap_dependence(shap_explanation):
    """SHAP dependence plot

    Args:
        shap_explanation: SHAP explanation object
    
    Returns:
        list or dict: Either list of (fig, ax, feature_name) tuples for 2D values or 
                      dict mapping class indices to lists of (fig, ax, feature_name) tuples for 3D values
    """
    
    assert isinstance(shap_explanation, shap.Explanation), \
        "shap_explanation must be a shap.Explanation object"


    # Sometimes the SHAP explanation object has no feature names
    if shap_explanation.feature_names is None:
        shap_explanation.feature_names = [f"x_{i}" for i in range(shap_explanation.values.shape[1])]

    # For binary classification (except for decision tree and random forest of sklearn)
    if shap_explanation.values.ndim == 2:
        shap_dp_plots = []
        for i, feature_name in enumerate(shap_explanation.feature_names):
            # Use feature index instead of name for direct indexing
            shap.plots.scatter(shap_explanation[:, i], show=False, color=shap_explanation)
            plt.tight_layout()
            fig = plt.gcf()
            ax = plt.gca()
            shap_dp_plots.append((fig, ax, feature_name))
        return shap_dp_plots

    # For multi-class
    shap_dp_plots = dict()
    for class_idx in range(shap_explanation.values.shape[2]):
        shap_dp_plots[class_idx] = []
        for i, feature_name in enumerate(shap_explanation.feature_names):
            # Use feature index instead of name for direct indexing
            shap.plots.scatter(shap_explanation[:, i, class_idx],
                              show=False, color=shap_explanation[:, :, class_idx])
            plt.tight_layout()
            fig = plt.gcf()
            ax = plt.gca()
            shap_dp_plots[class_idx].append((fig, ax, feature_name))

    return shap_dp_plots