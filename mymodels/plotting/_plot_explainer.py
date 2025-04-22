import shap
import matplotlib.pyplot as plt



def _plot_shap_results(shap_values, results_dir, model_name=None, classes_=None):
    if shap_values.ndim == 2:
        # All models used in regression tasks,
        # and all models used in binary classification tasks: SVC, adaboost, gbdt, xgboost, lightgbm, catboost,
        # The dimensions of the output shap_values are (n_samples, n_features)

        # Summary plot for demonstrating feature importance
        _plot_summary(
            shap_values = shap_values,
            save_dir = results_dir,
            file_name = "shap_summary",
            title = "SHAP Summary Plot"
        )

        # Dependence plot for demonstrating relationship between feature values and SHAP values
        _plot_dependence(
            shap_values = shap_values,
            save_dir = results_dir.joinpath("dependence_plots/")
        )

        # Partial Dependence Plot 
        # is supported for regression task only.
        # is not supported for categorical features.
        if model_name in ["lr", "svr", "knr", "mlpr", "adar", "dtr", "rfr", "gbdtr", "xgbr", "lgbr", "catr"]:
            _plot_partial_dependence(
                save_dir = results_dir.joinpath("partial_dependence_plots/")
            )

    elif shap_values.ndim == 3:
        # For binary classification tasks using sklearn's decision tree and random forest models,
        # as well as all models used in multi-classification tasks,
        # the dimensions of the output shap_values are (n_samples, n_features, n_targets)
        # Where:
        # In binary classification tasks with sklearn's decision tree and random forest, shap values represent 
        # each feature's contribution to the probability of a sample being classified as positive or negative
        # Therefore, results are output for each class,
        # saved in the shap_summary directory, and named according to the class
        # Similarly, in the dependence_plots directory, subdirectories are created and named according to the class
        summary_plot_dir = results_dir.joinpath("shap_summary")
        summary_plot_dir.mkdir(parents = True, exist_ok = True)
        for i in range(0, len(classes_)):
            # Summary plot for ranking features' importance
            _plot_summary(
                shap_values = shap_values[:, :, i],
                save_dir = summary_plot_dir,
                file_name = f"class_{classes_[i]}",
                title = f"SHAP Summary Plot for Class: {classes_[i]}"
            )

            # Dependence plot for demonstrating relationship between feature values and SHAP values
            dependence_plot_dir = results_dir.joinpath(f"dependence_plots/class_{classes_[i]}/")
            dependence_plot_dir.mkdir(parents = True, exist_ok = True)
            _plot_dependence(
                shap_values = shap_values[:, :, i],
                save_dir = dependence_plot_dir
            )

    else:
        raise ValueError(f"Invalid SHAP values dimension: {shap_values.ndim}")
    
    return None




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
