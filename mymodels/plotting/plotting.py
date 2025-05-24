"""Module for plotting functionalities in mymodels.

This module provides plotting utilities for data visualization,
model evaluation, and explainability through the Plotter class.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap


from ._plot_diagnosed_data import _plot_category, _plot_data_distribution, _plot_correlation
from ._plot_optimizer import _plot_optimize_history
from ._plot_evaluated_classifier import _plot_roc_curve, _plot_pr_curve, _plot_confusion_matrix
from ._plot_evaluated_regressor import _plot_regression_scatter
from ._plot_explainer import _plot_shap_summary, _plot_shap_dependence


class Plotter:
    """Base class for plotting functionality in mymodels.
    
    This class serves as a foundation for all plotting modules, providing
    common settings and functionality.
    
    Attributes:
        show (bool): Whether to display plots when generated.
        plot_format (str): Output format of the saved plots (e.g., 'jpg', 'png', 'svg', 'pdf').
        plot_dpi (int): DPI (dots per inch) for the saved plots.
        results_dir (Path): Directory path where plots will be saved.
    """
    
    def __init__(
        self,
        results_dir: str | Path,
        show: bool = False,
        plot_format: str = "jpg",
        plot_dpi: int = 500,
    ):
        """Initialize the Plotting object.
        
        Args:
            results_dir (str or Path): Directory to save plots.
            show (bool, optional): Whether to display plots. Defaults to False.
            plot_format (str, optional): Format for saved plots. Defaults to "jpg".
            plot_dpi (int, optional): DPI for saved plots. Defaults to 500.
        """

        # Check input parameters
        assert isinstance(show, bool), "show must be a boolean"
        # assert plot_format in ["jpg", "png", "jpeg", "tiff", "pdf", "svg", "eps"], \
        #     "plot_format must be one of the following: jpg, png, jpeg, tiff, pdf, svg, eps"
        assert isinstance(plot_dpi, int), "plot_dpi must be an integer"

        assert isinstance(results_dir, (Path, str)), \
            "results_dir must be a valid directory path"
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        results_dir.mkdir(parents = True, exist_ok = True)

        # Initialize attributes
        self.show = show
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.results_dir = results_dir


    ###########################################################################################
    # Utility functions (private)
    ###########################################################################################
    def _finalize_plot(self, fig, sub_dir = None, saved_file_name = None):
        """Finalize a plot by optionally saving and showing it.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to finalize.
            sub_dir (str, optional): Directory for the saved file.
                If None, the figure will not be saved.
            saved_file_name (str, optional): Name for the saved file (without extension).
                If None, the figure will not be saved.
                
        Returns:
            str or None: Path to the saved file if saved, otherwise None.
        """
        assert isinstance(fig, plt.Figure), "fig must be a matplotlib.figure.Figure"
        assert isinstance(sub_dir, str) or sub_dir is None, "sub_dir must be a string or None"
        assert isinstance(saved_file_name, str) or saved_file_name is None, "saved_file_name must be a string or None"

        saved_path = None
        # Save figure if filename is provided
        if saved_file_name is not None:
            saved_path = self._save_figure(fig, sub_dir, saved_file_name)
        
        # Show figure if requested
        if self.show:
            plt.show()
        else:
            plt.close(fig)
            
        return saved_path
    

    def _save_figure(self, fig, sub_dir = None, saved_file_name = None):
        """Save the provided figure to the results directory.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            saved_dir (str, optional): Directory for the saved file.
                If None, the figure will not be saved.
            saved_file_name (str, optional): Name for the saved file (without extension).
                If None, the figure will not be saved.
            
        Returns:
            str or None: Path to the saved file if results_dir is set, otherwise None.
        """
        
        if self.results_dir is None:
            logging.warning("Cannot save figure: results_dir is not set.")
            return None
        
        saved_dir = self.results_dir
        # Create the sub_dir if it does not exist
        if sub_dir is not None:
            saved_dir = saved_dir.joinpath(sub_dir)
            saved_dir.mkdir(parents = True, exist_ok = True)
        
        # Ensure filename has the correct extension
        if not saved_file_name.endswith(f".{self.plot_format}"):
            saved_dir = saved_dir.joinpath(f"{saved_file_name}.{self.plot_format}")
        else:
            saved_dir = saved_dir.joinpath(saved_file_name)
        
        # Save the figure
        fig.savefig(saved_dir, dpi = self.plot_dpi, bbox_inches = 'tight')
        
        return str(saved_dir)
    ###########################################################################################



    ###########################################################################################
    # Plotting functions for data diagnosis
    ###########################################################################################
    def plot_category(self, data, name=None):
        """Plot categorical data.
        
        Args:
            data: Data to plot
            name: Name of the data
        """
        fig, _ = _plot_category(data, name=name, show=self.show)
        self._finalize_plot(fig, sub_dir = "diagnosis/category/", saved_file_name = f"{name}")
    

    def plot_data_distribution(self, data, name=None):
        """Plot the data distribution.
        
        Args:
            data: Data to plot
            name: Name of the data
        """
        fig, _ = _plot_data_distribution(data, name=name, show=self.show)
        self._finalize_plot(fig, sub_dir = "diagnosis/distribution/", saved_file_name = f"{name}")
    

    def plot_correlation(self, data, corr_threshold=0.8, name=None):
        """Plot the correlation matrix.
        
        Args:
            data: Data to plot
            name: Name of the data
        """
        result = _plot_correlation(data, name=name, corr_threshold=corr_threshold, show=self.show)
        pearson_fig, _ = result["pearson"]
        spearman_fig, _ = result["spearman"]

        self._finalize_plot(pearson_fig, sub_dir = "diagnosis/correlation", saved_file_name = "pearson")
        self._finalize_plot(spearman_fig, sub_dir = "diagnosis/correlation", saved_file_name = "spearman")
    ###########################################################################################



    ###########################################################################################
    # Plotting functions for optimizer
    ###########################################################################################
    def plot_optimize_history(self, optuna_study_object):
        """Plot the optimization history.
        
        Args:
            optuna_study_object: The completed Optuna study containing trial results.
        """
        fig, _ = _plot_optimize_history(optuna_study_object)
        self._finalize_plot(fig, sub_dir = "optimization", saved_file_name = "optimization_history")
    ###########################################################################################



    ###########################################################################################
    # Plotting functions for evaluated classifier
    ###########################################################################################
    def plot_roc_curve(self, y_test, x_test, optimal_model_object):
        """Plot ROC curve for classification model evaluation.
        
        Args:
            y_test: Actual test target values
            x_test: Test feature data
            optimal_model_object: Trained model object
        """
        fig, _ = _plot_roc_curve(y_test, x_test, optimal_model_object)
        self._finalize_plot(fig, sub_dir = "evaluation/", saved_file_name = "roc_curve")


    def plot_pr_curve(self, y_test, x_test, optimal_model_object):
        """Plot PR curve for classification model evaluation.
        
        Args:
            y_test: Actual test target values
            x_test: Test feature data
            optimal_model_object: Trained model object
        """
        fig, _ = _plot_pr_curve(y_test, x_test, optimal_model_object)
        self._finalize_plot(fig, sub_dir = "evaluation/", saved_file_name = "pr_curve")
    

    def plot_confusion_matrix(self, y_test, y_test_pred):
        """Plot confusion matrix for classification model evaluation.
        
        Args:
            y_test: Actual test target values
            y_test_pred: Predicted test target values
        """
        fig, _ = _plot_confusion_matrix(y_test, y_test_pred)
        self._finalize_plot(fig, sub_dir = "evaluation/", saved_file_name = "confusion_matrix")
    ###########################################################################################



    ###########################################################################################
    # Plotting functions for evaluated regressor
    ###########################################################################################
    def plot_regression_scatter(self, y_test, y_test_pred):
        """Plot regression scatter plot.
        
        Args:
            y_test: Actual test target values
            y_test_pred: Predicted test target values
        """
        fig, _ = _plot_regression_scatter(y_test, y_test_pred)
        self._finalize_plot(fig, sub_dir = "evaluation/", saved_file_name = "scatter")
    ###########################################################################################



    ###########################################################################################
    # Plotting for explainer
    ###########################################################################################
    # For binary classification tasks using sklearn's decision tree and random forest models,
    # as well as all models used in multi-classification tasks,
    # the dimensions of the output shap_values are (n_samples, n_features, n_targets)
    # In binary classification tasks with sklearn's decision tree and random forest, shap values represent 
    # each feature's contribution to the probability of a sample being classified as positive or negative
    # Therefore, results are output for each class,
    # saved in the shap_summary directory, and named according to the class
    # Similarly, in the dependence_plots directory, subdirectories are created and named according to the class

    def plot_shap_summary(
            self,
            shap_explanation: shap.Explanation,
            y_mapping_dict: dict | None = None
        ):
        """Plot SHAP summary

        Args:
            shap_explanation: SHAP explanation object
            y_mapping_dict: Dictionary mapping class names to their indices
        """

        assert isinstance(shap_explanation, shap.Explanation), \
            "shap_explanation must be a shap.Explanation object"
        assert isinstance(y_mapping_dict, dict) or y_mapping_dict is None, \
            "y_mapping_dict must be a dictionary or None"
        
        if shap_explanation.values.ndim == 2:
            fig, _ = _plot_shap_summary(shap_explanation)
            self._finalize_plot(fig, sub_dir = "explanation/SHAP/", saved_file_name = "shap_summary")

        elif shap_explanation.values.ndim == 3:
            if y_mapping_dict is not None:
                # Inverse the mapping dict (value → key)
                _inverse_mapping = {v: k for k, v in y_mapping_dict.items()}
            else:
                # If y_mapping_dict is not provided, use the index of the class
                _num_classes = shap_explanation.values.shape[2]
                _inverse_mapping = {i: str(i) for i in range(_num_classes)}

            _fig_ax_dict = _plot_shap_summary(shap_explanation)
            for class_idx, (fig, _) in _fig_ax_dict.items():
                self._finalize_plot(
                    fig,
                    sub_dir = "explanation/SHAP/shap_summary/",
                    saved_file_name = f"{str(_inverse_mapping[class_idx])}"
                )
    

    def plot_shap_dependence(
            self,
            shap_explanation: shap.Explanation, 
            y_mapping_dict: dict | None = None
        ):
        """Plot SHAP dependence
        
        Args:
            shap_explanation: SHAP explanation object
            y_mapping_dict: Dictionary mapping class names to their indices
        """

        assert isinstance(shap_explanation, shap.Explanation), \
            "shap_explanation must be a shap.Explanation object"
        assert isinstance(y_mapping_dict, dict) or y_mapping_dict is None, \
            "y_mapping_dict must be a dictionary or None"

        shap_dp_plots = _plot_shap_dependence(shap_explanation)
        if shap_explanation.values.ndim == 2:
            for fig, _, feature_name in shap_dp_plots:
                self._finalize_plot(
                    fig,
                    sub_dir = "explanation/SHAP/shap_dependence/",
                    saved_file_name = str(feature_name)
                )
                
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
                    self._finalize_plot(
                        fig, 
                        sub_dir = f"explanation/SHAP/shap_dependence/{_inverse_mapping[class_idx]}/", 
                        saved_file_name = str(feature_name)
                    )


    def plot_partial_dependence(self, optimal_model_object, x_data):
        """Plot partial dependence.
        
        Args:
            x_data: Data for partial dependence plot.
                If the data engineering was implemented, the data should be the data after the data engineering.
            optimal_model_object: Trained model object
        """
        # Implementation to be added later
        # The _plot_partial_dependence() will return a list, every element in the list is a tuple of (fig, ax).
        # fig_ax_list = _plot_partial_dependence(optimal_model_object, x_data)
        # for fig, ax in fig_ax_list:
        #     self._finalize_plot(fig, filename = "partial_dependence")
    ###########################################################################################
    





