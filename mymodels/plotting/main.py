import matplotlib.pyplot as plt
from pathlib import Path
import logging



from ._plot_diagnosed_data import _plot_category, _plot_data_distribution, _plot_correlation
from ._plot_optimizer import _plot_optimize_history
from ._plot_evaluated_classifier import _plot_roc_curve, _plot_pr_curve, _plot_confusion_matrix
from ._plot_evaluated_regressor import _plot_regression_scatter
from ._plot_explainer import _plot_summary, _plot_dependence, _plot_partial_dependence


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
        show: bool = False,
        plot_format: str = "jpg",
        plot_dpi: int = 500,
        results_dir: str | Path | None = None
    ):
        """Initialize the Plotting object.
        
        Args:
            show (bool, optional): Whether to display plots. Defaults to False.
            plot_format (str, optional): Format for saved plots. Defaults to "jpg".
            plot_dpi (int, optional): DPI for saved plots. Defaults to 500.
            results_dir (str or Path, optional): Directory to save plots. Defaults to None.
                If None, plots will not be saved.
        """
        # Check input parameters
        assert isinstance(show, bool), "show must be a boolean"
        assert plot_format in ["jpg", "png", "jpeg", "tiff", "pdf", "svg", "eps"], \
            "plot_format must be one of the following: jpg, png, jpeg, tiff, pdf, svg, eps"
        assert isinstance(plot_dpi, int), "plot_dpi must be an integer"

        assert isinstance(results_dir, Path) \
            or isinstance(results_dir, str), \
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
    # Utility functions
    ###########################################################################################
    def _save_figure(self, fig, filename):
        """Save the provided figure to the results directory.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str): Name for the saved file (without extension).
            
        Returns:
            str or None: Path to the saved file if results_dir is set, otherwise None.
        """
        if self.results_dir is None:
            logging.warning("Cannot save figure: results_dir is not set.")
            return None
        
        # Ensure filename has the correct extension
        if not filename.endswith(f".{self.plot_format}"):
            filepath = self.results_dir / f"{filename}.{self.plot_format}"
        else:
            filepath = self.results_dir / filename
        
        # Save the figure
        fig.savefig(filepath, dpi=self.plot_dpi, bbox_inches='tight')
        
        return str(filepath)
    

    def _finalize_plot(self, fig, filename=None):
        """Finalize a plot by optionally saving and showing it.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to finalize.
            filename (str, optional): Name for the saved file (without extension).
                If None, the figure will not be saved.
                
        Returns:
            str or None: Path to the saved file if saved, otherwise None.
        """
        # Save figure if filename is provided
        saved_path = None
        if filename is not None:
            saved_path = self._save_figure(fig, filename)
        
        # Show figure if requested
        if self.show:
            plt.show()
        else:
            plt.close(fig)
            
        return saved_path
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
        fig, ax = _plot_category(data, name=name, show=self.show)
        self._finalize_plot(fig, filename = "category")

        return None
    

    def plot_data_distribution(self, data, name=None):
        """Plot the data distribution.
        
        Args:
            data: Data to plot
            name: Name of the data
        """
        fig, ax = _plot_data_distribution(data, name=name, show=self.show)
        self._finalize_plot(fig, filename = "data_distribution")

        return None
    

    def plot_correlation(self, data, corr_threshold=0.8, name=None):
        """Plot the correlation matrix.
        
        Args:
            data: Data to plot
            name: Name of the data
        """
        result = _plot_correlation(data, name=name, corr_threshold=corr_threshold, show=self.show)
        pearson_fig, pearson_ax = result["pearson"]
        spearman_fig, spearman_ax = result["spearman"]

        self._finalize_plot(pearson_fig, filename = "correlation_pearson")
        self._finalize_plot(spearman_fig, filename = "correlation_spearman")

        return None
    ###########################################################################################



    ###########################################################################################
    # Plotting functions for optimizer
    ###########################################################################################
    def plot_optimize_history(self, optuna_study_object):
        """Plot the optimization history.
        
        Args:
            optuna_study_object: The completed Optuna study containing trial results.
        """
        fig, ax = _plot_optimize_history(optuna_study_object)
        self._finalize_plot(fig, filename = "optimization_history")

        return None
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
        fig, ax = _plot_roc_curve(y_test, x_test, optimal_model_object)
        self._finalize_plot(fig, filename = "roc_curve")

        return None


    def plot_pr_curve(self, y_test, x_test, optimal_model_object):
        """Plot PR curve for classification model evaluation.
        
        Args:
            y_test: Actual test target values
            x_test: Test feature data
            optimal_model_object: Trained model object
        """
        fig, ax = _plot_pr_curve(y_test, x_test, optimal_model_object)
        self._finalize_plot(fig, filename = "pr_curve")

        return None
    

    def plot_confusion_matrix(self, y_test, y_test_pred):
        """Plot confusion matrix for classification model evaluation.
        
        Args:
            y_test: Actual test target values
            y_test_pred: Predicted test target values
        """
        fig, ax = _plot_confusion_matrix(y_test, y_test_pred)
        self._finalize_plot(fig, filename = "confusion_matrix")

        return None
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
        fig, ax = _plot_regression_scatter(y_test, y_test_pred)
        self._finalize_plot(fig, filename = "regression_scatter")

        return None
    ###########################################################################################



    ###########################################################################################
    # Plotting for explainer
    ###########################################################################################
    def plot_summary(self, shap_values, title=None):
        """Plot SHAP summary.

        Args:
            shap_values: SHAP values
            title: Title of the plot
        """
        fig, ax = _plot_summary(shap_values, title=title)
        self._finalize_plot(fig, filename = "summary")

        return None
    

    def plot_dependence(self, shap_values, title=None):
        """Plot SHAP dependence.
        
        Args:
            shap_values: SHAP values
            title: Title of the plot
        """
        fig, ax = _plot_dependence(shap_values, title=title)
        self._finalize_plot(fig, filename = "dependence")

        return None
    

    def plot_partial_dependence(self, shap_values, title=None):
        """Plot partial dependence.
        
        Args:
            shap_values: SHAP values
            title: Title of the plot
        """
        fig, ax = _plot_partial_dependence(shap_values, title=title)
        self._finalize_plot(fig, filename = "partial_dependence")

        return None
    ###########################################################################################
    





