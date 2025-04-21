import shap
import matplotlib.pyplot as plt



def _plot_summary(shap_values, shap_data, title: str = None):
    """Summary Plot
    https://shap.readthedocs.io/en/latest/release_notes.html#release-v0-36-0
    """
    fig = plt.figure()
    ax = fig.gca()
    shap.summary_plot(shap_values, shap_data, show=False)
    plt.title(title)
    plt.tight_layout()
    return fig, ax



def _plot_dependence(shap_values, shap_data, feature_name=None):
    """Dependence Plot
    https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html#Using-color-to-highlight-interaction-effects
    """
    fig = plt.figure()
    # shap.dependence_plot creates its own plot
    shap.dependence_plot(feature_name, shap_values, shap_data, show=False)
    plt.tight_layout()
    ax = plt.gca()
    return fig, ax



def _plot_partial_dependence(model_obj, shap_data, numeric_features, feature_name=None):
    if feature_name is not None:
        fig, ax = shap.partial_dependence_plot(
            feature_name,
            model_obj.predict,
            shap_data,
            model_expected_value=True,
            feature_expected_value=True,
            ice=False,
            show=False
        )
        fig.tight_layout()
        return fig, ax
    
    results = {}
    for r in numeric_features:
        fig, ax = shap.partial_dependence_plot(
            str(r),
            model_obj.predict,
            shap_data,
            model_expected_value=True,
            feature_expected_value=True,
            ice=False,
            show=False
        )
        fig.tight_layout()
        results[str(r)] = (fig, ax)
        plt.close(fig)
    
    return results
