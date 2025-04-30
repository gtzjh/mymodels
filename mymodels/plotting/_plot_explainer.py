import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence



def _plot_shap_summary(shap_explanation):
    """SHAP summary plot

    Args:
        shap_values: SHAP values
    
    Returns:
        fig_ax_list: list of tuple (fig, ax)
    """

    """
    assert isinstance(shap_values, (np.ndarray, pd.DataFrame)), \
        "shap_values must be a ndarray or pd.DataFrame"
    assert hasattr(shap_values, 'ndim'), "shap_values must have the attribute 'ndim'"
    assert shap_values.ndim == 2
    """

    assert isinstance(shap_explanation, shap.Explanation), \
        "shap_explanation must be a shap.Explanation object"
    
    if shap_explanation.values.ndim == 2:
        _fig = plt.figure()
        _ax = _fig.gca()
        shap.plots.beeswarm(shap_explanation, show = False)
        plt.tight_layout()
        return _fig, _ax

    elif shap_explanation.values.ndim == 3:
        _fig_ax_dict = dict()
        for i in range(shap_explanation.values.shape[2]):
            _fig = plt.figure()
            _ax = _fig.gca()

            shap.summary_plot(shap_explanation.values[:, :, i],
                              shap_explanation.data,
                              feature_names = shap_explanation.feature_names,
                              show = False)

            """
            _shap_dataframe = pd.DataFrame(shap_explanation.values[:, :, i],
                                           columns = shap_explanation.feature_names)
            shap.summary_plot(_shap_dataframe, show = False)
            """

            
            plt.tight_layout()
            _fig_ax_dict[i] = (_fig, _ax)

        return _fig_ax_dict



def _plot_shap_dependence(shap_explanation):
    """SHAP dependence plot

    Args:
        shap_values: SHAP values
    
    Returns:
        shap_dp_plot_list: list of tuple (fig, ax)
    """
    
    assert isinstance(shap_explanation, shap.Explanation), \
        "shap_explanation must be a shap.Explanation object"
    
    # For binary classification (except for decision tree and random forest of sklearn)
    if shap_explanation.values.ndim == 2:
        shap_dp_plots = []
        for feature_name in shap_explanation.feature_names:
            shap.plots.scatter(shap_explanation[:, feature_name], show=False, color=shap_explanation)
            plt.tight_layout()
            fig = plt.gcf()
            ax = plt.gca()
            shap_dp_plots.append((fig, ax, feature_name))

    elif shap_explanation.values.ndim == 3:
        shap_dp_plots = dict()
        for class_idx in range(shap_explanation.values.shape[2]):
            shap_dp_plots[class_idx] = []
            for feature_name in shap_explanation.feature_names:
                ax = shap.plots.scatter(shap_explanation[:, feature_name, class_idx],
                                        show=False, color=shap_explanation[:, :, class_idx])
                plt.tight_layout()
                fig = plt.gcf()
                ax = plt.gca()
                shap_dp_plots[class_idx].append((fig, ax, feature_name))

    return shap_dp_plots