import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence



def _plot_shap_summary(shap_values):
    """SHAP summary plot

    Args:
        shap_values: SHAP values
    
    Returns:
        fig_ax_list: list of tuple (fig, ax)
    """
    assert isinstance(shap_values, (np.ndarray, pd.DataFrame)), \
        "shap_values must be a ndarray or pd.DataFrame"
    assert hasattr(shap_values, 'ndim'), "shap_values must have the attribute 'ndim'"
    assert shap_values.ndim == 2

    fig = plt.figure()
    ax = fig.gca()
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()

    return fig, ax



def _plot_shap_dependence(shap_values):
    """SHAP dependence plot

    Args:
        shap_values: SHAP values
    
    Returns:
        shap_dp_plot_list: list of tuple (fig, ax)
    """
    
    assert isinstance(shap_values, (np.ndarray, pd.DataFrame)), \
        "shap_values must be a ndarray or pd.DataFrame"
    assert hasattr(shap_values, 'ndim'), "shap_values must have the attribute 'ndim'"
    assert shap_values.ndim == 2

    shap_dp_plot_list = []
    for i in range(0, shap_values.shape[1]):
        fig = plt.figure()
        # shap.dependence_plot creates its own plot
        shap.dependence_plot(i, shap_values, show=False)
        plt.tight_layout()
        ax = plt.gca()
        shap_dp_plot_list.append((fig, ax))

    return shap_dp_plot_list