import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from imblearn.metrics import specificity_score
import logging


from ._data_loader import MyDataLoader
from ._estimator import MyEstimator
from ..plotting import Plotter
from ..output import Output


def _get_accuracy_for_regression_task(y, y_pred, eval_metric = None):
    """Calculate regression accuracy metrics.
    
    Computes R2, RMSE, and MAE metrics for regression predictions.
    
    Args:
        y: Actual target values.
        y_pred: Predicted target values.
        eval_metric: Optional dictionary of additional evaluation metrics to apply.
            Each key should be a metric name and value a callable that accepts y and y_pred.
            
    Returns:
        dict: Dictionary of regression metrics.
    """
    
    # The major metrics for regression
    accuracy_dict = dict({
        "R2": float(r2_score(y, y_pred)),
        "RMSE": float(root_mean_squared_error(y, y_pred)),
        "MAE": float(mean_absolute_error(y, y_pred)),
    })

    # The user-defined evaluation metrics
    if eval_metric is not None:
        for _key, _metric in eval_metric.items():
            try:
                result = _metric(y, y_pred)
                # Ensure result is numeric and can be serialized to JSON
                accuracy_dict[_key] = float(result)
            except Exception as e:
                logging.warning(f"Error computing metric '{_key}': {str(e)}")
                accuracy_dict[_key] = None

    return accuracy_dict



def _get_accuracy_for_classification_task(y, y_pred, eval_metric = None):
    """Calculate classification accuracy metrics.
    
    Computes accuracy, precision, recall, F1, and other metrics for classification predictions.
    
    Args:
        y: Actual target values.
        y_pred: Predicted target values.
        eval_metric: Optional dictionary of additional evaluation metrics to apply.
            Each key should be a metric name and value a callable that accepts y and y_pred.
            
    Returns:
        dict: Dictionary of classification metrics.
    """
    # Determine if it's binary or multi-class based on unique classes in y
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2

    # The major metrics for classification
    accuracy_dict = dict({
        "Overall Accuracy": float(accuracy_score(y, y_pred)),
        "Precision": float(precision_score(y, y_pred, average = "weighted")),
        "Recall": float(recall_score(y, y_pred, average = "weighted")),
        "F1": float(f1_score(y, y_pred, average = "weighted")),
        "Kappa": float(cohen_kappa_score(y, y_pred)),
    })

    # If it's binary, add matthews_corrcoef
    if is_binary:
        accuracy_dict["Matthews Correlation Coefficient"] = float(matthews_corrcoef(y, y_pred))
        accuracy_dict["Specificity"] = float(specificity_score(y, y_pred, average = "binary"))

    # The user-defined evaluation metrics
    if eval_metric is not None:
        for _key, _metric in eval_metric.items():
            try:
                result = _metric(y, y_pred)
                # Ensure result is numeric and can be serialized to JSON
                accuracy_dict[_key] = float(result)
            except Exception as e:
                logging.warning(f"Error computing metric '{_key}': {str(e)}")
                accuracy_dict[_key] = None

    return accuracy_dict



class MyEvaluator:
    def __init__(
            self,
            optimized_estimator: MyEstimator,
            optimized_dataset: MyDataLoader,
            optimized_data_engineer_pipeline: Pipeline | None = None,
            plotter: Plotter | None = None,
            output: Output | None = None
        ):
        """A class for evaluating machine learning models.

        This class handles the evaluation of machine learning models, computing various 
        accuracy metrics (R², RMSE, MAE, F1, Kappa, etc.), visualizing actual vs predicted values, 
        and saving/printing results.
        
        Args:
            optimized_dataset: Dataset containing the train and test data.
            optimized_estimator: Trained estimator to evaluate.
            optimized_data_engineer_pipeline: Optional data engineering pipeline to transform data.
            plotter: The plotter to use.
            output: The output object.

        Attributes:
            optimized_dataset (MyDataLoader): Dataset containing train and test splits.
            optimized_estimator (MyEstimator): Trained estimator to evaluate.
            optimized_data_engineer_pipeline (Pipeline | None): Data engineering pipeline.
            optimal_model_object: The actual model object from optimized_estimator.
            show_train (bool): Whether to show training set evaluation. Set in evaluate().
            eval_metric (dict | None): Custom evaluation metrics. Set in evaluate().
            _x_train: Training features.
            _x_test: Test features.
            _y_train: Training target values.
            _y_test: Test target values.
            _y_train_pred: Predicted training target values.
            _y_test_pred: Predicted test target values.
            accuracy_dict (dict): Stores evaluation results after calling evaluate().
        """

        # Validate input
        assert isinstance(optimized_dataset, MyDataLoader), \
            "optimized_dataset must be a mymodels.MyDataLoader object"
        assert isinstance(optimized_estimator, MyEstimator), \
            "optimized_estimator must be a mymodels.MyEstimator object"
        assert optimized_estimator.optimal_model_object is not None, \
            "The estimator has not been fitted yet"
        # Check data_engineer_pipeline validity
        if optimized_data_engineer_pipeline is not None:
            assert isinstance(optimized_data_engineer_pipeline, Pipeline), \
                "optimized_data_engineer_pipeline must be a `sklearn.pipeline.Pipeline` object"
        else:
            logging.warning("No data engineering will be implemented, the raw data will be used.")

        # Initialize attributes
        self.optimized_dataset = optimized_dataset
        self.optimized_estimator = optimized_estimator
        self.optimized_data_engineer_pipeline = optimized_data_engineer_pipeline
        self.optimal_model_object = self.optimized_estimator.optimal_model_object

        self.plotter = plotter
        self.output = output
        
        # Initialize attributes to be set in evaluate()
        self.show_train = False
        self.eval_metric = None
        self.accuracy_dict = dict()
        
        # Private attributes for data
        self._x_train = None
        self._x_test = None
        self._y_train = None
        self._y_test = None
        self._y_test_pred = None
        self._y_train_pred = None



    def evaluate(
            self,
            show_train: bool,
            dummy: bool,
            eval_metric: dict | None = None,
        ):

        """Evaluate the model on test and training data.
        
        Performs model evaluation by computing accuracy metrics for both the
        optimized model and optionally a dummy estimator for comparison.
        
        Args:
            show_train (bool): Whether to show the training set evaluation results.
            dummy (bool): Whether to use a dummy estimator for comparison.
            eval_metric (dict | None): Custom evaluation metrics to use.
                It must be None or a dictionary where each key is a metric name and
                each value is a callable function that takes y_true and y_pred as arguments.
                
        Returns:
            dict: Dictionary containing evaluation results.
            
        Example:
            ```python
            # Define custom metrics if needed
            custom_metrics = {'custom_metric': lambda y, y_pred: some_calculation(y, y_pred)}
            
            # Evaluate model with training results and dummy comparison
            evaluator = MyEvaluator(estimator, dataset, pipeline)
            results = evaluator.evaluate(show_train=True, dummy=True, eval_metric=custom_metrics)
            ```
        """
        
        # Assert show_train and dummy are boolean
        assert isinstance(show_train, bool), "show_train must be a boolean"
        assert isinstance(dummy, bool), "dummy must be a boolean"
        
        # Assert eval_metric is a dictionary and every item inside is callable
        if eval_metric is not None:
            assert isinstance(eval_metric, dict), "eval_metric must be a dictionary"
            for key, metric in eval_metric.items():
                assert callable(metric), f"Metric '{key}' in eval_metric must be callable"

        # Initialize the variables
        self.show_train = show_train
        self.eval_metric = eval_metric
        self.accuracy_dict = {}

        # Get the test and train data
        self._x_train = self.optimized_dataset.x_train.copy(deep = True)
        self._x_test = self.optimized_dataset.x_test.copy(deep = True)
        self._y_train = self.optimized_dataset.y_train.copy(deep = True)
        self._y_test = self.optimized_dataset.y_test.copy(deep = True)

        # Transform X data
        if self.optimized_data_engineer_pipeline is not None:
            self._x_test = self.optimized_data_engineer_pipeline.transform(self._x_test)
            self._x_train = self.optimized_data_engineer_pipeline.transform(self._x_train)
        
        # Get the optimal model object
        self.optimal_model_object = self.optimized_estimator.optimal_model_object

        # Predict
        self._y_test_pred = pd.Series(self.optimal_model_object.predict(self._x_test))
        self._y_train_pred = pd.Series(self.optimal_model_object.predict(self._x_train))

        # Trans the y data to the original label, for classification tasks
        _y_mapping_dict = self.optimized_dataset.y_mapping_dict
        if _y_mapping_dict is not None:
            # Inverse the mapping dict (value → key)
            _inverse_mapping = {v: k for k, v in _y_mapping_dict.items()}
            self._y_test = self._y_test.map(lambda x: _inverse_mapping.get(x, x))
            self._y_test_pred = self._y_test_pred.map(lambda x: _inverse_mapping.get(x, x))
            self._y_train = self._y_train.map(lambda x: _inverse_mapping.get(x, x))
            self._y_train_pred = self._y_train_pred.map(lambda x: _inverse_mapping.get(x, x))

        # Evaluate the model
        self._evaluate_model()

        # Evaluate the dummy estimator
        if dummy:
            self._evaluate_dummy()
        

        # Plot and output
        self._plot(self.plotter)
        self._output(self.output)

        return self.accuracy_dict
    


    def _evaluate_model(self):
        """Evaluate the optimized model on test and optionally training data.
        
        Predicts on test and training data (if show_train is True) and calculates accuracy metrics.
        Results are stored in self.accuracy_dict['model'] and also printed to console.
        """

        # Initialize model accuracy dictionary
        self.accuracy_dict["model"] = dict()

        # Evaluate on the test data
        if is_regressor(self.optimal_model_object):
            self.accuracy_dict["model"]["test"] = _get_accuracy_for_regression_task(
                self._y_test, self._y_test_pred, self.eval_metric
            )
        elif is_classifier(self.optimal_model_object):
            self.accuracy_dict["model"]["test"] = _get_accuracy_for_classification_task(
                self._y_test, self._y_test_pred, self.eval_metric
            )
        
        # Evaluate on the training data
        if self.show_train:
            if is_regressor(self.optimal_model_object):
                self.accuracy_dict["model"]["train"] = _get_accuracy_for_regression_task(
                    self._y_train, self._y_train_pred, self.eval_metric
                )
            elif is_classifier(self.optimal_model_object):
                self.accuracy_dict["model"]["train"] = _get_accuracy_for_classification_task(
                    self._y_train, self._y_train_pred, self.eval_metric
                )

        return None



    def _evaluate_dummy(self):
        """Evaluate using a dummy estimator for baseline comparison.
        
        Creates a dummy estimator appropriate for the task type (regression or classification)
        and evaluates its performance on the same data as the optimized model.
        Results are stored in self.accuracy_dict['dummy'] and also printed to console.

        When using dummy estimator, a warning about zero division will be printed, just ignore it.
        
        Returns:
            None
        """

        logging.warning("\nWhen using dummy estimator, a warning about zero division will be printed, JUST IGNORE IT.\n")

        # Create a dummy estimator
        _dummy_estimator = DummyRegressor() if is_regressor(self.optimal_model_object) else DummyClassifier()
        _dummy_estimator.fit(self._x_train, self._y_train)

        # Predict
        _dummy_y_test = _dummy_estimator.predict(self._x_test)
        _dummy_y_train = _dummy_estimator.predict(self._x_train)

        # Initialize dummy accuracy dictionary
        self.accuracy_dict["dummy"] = dict()

        # Evaluate on the test data
        if is_regressor(self.optimal_model_object):
            self.accuracy_dict["dummy"]["test"] = _get_accuracy_for_regression_task(
                self._y_test, _dummy_y_test, self.eval_metric
            )
        elif is_classifier(self.optimal_model_object):
            self.accuracy_dict["dummy"]["test"] = _get_accuracy_for_classification_task(
                self._y_test, _dummy_y_test, self.eval_metric
            )

        # Evaluate on the training data
        if self.show_train:
            if is_regressor(self.optimal_model_object):
                self.accuracy_dict["dummy"]["train"] = _get_accuracy_for_regression_task(
                    self._y_train, _dummy_y_train, self.eval_metric
                )
            elif is_classifier(self.optimal_model_object):
                self.accuracy_dict["dummy"]["train"] = _get_accuracy_for_classification_task(
                    self._y_train, _dummy_y_train, self.eval_metric
                )

        return None
    

    def _plot(self, _plotter: Plotter):
        """Plot the evaluation results.
        
        Args:
            _plotter: The plotter to use.
        """

        if is_classifier(self.optimal_model_object):
            _plotter.plot_roc_curve(self._y_test, self._x_test, self.optimal_model_object)
            _plotter.plot_pr_curve(self._y_test, self._x_test, self.optimal_model_object)
            _plotter.plot_confusion_matrix(self._y_test, self._y_test_pred)
        
        elif is_regressor(self.optimal_model_object):
            _plotter.plot_regression_scatter(self._y_test, self._y_test_pred)

        return None
    

    def _output(self, _output: Output):
        """Output the evaluation results.
        
        Args:
            _output: The output object.
        """
        _output.output_evaluation(
            accuracy_dict = self.accuracy_dict
        )
        _output.output_raw_data(
            y_test = self._y_test,
            y_test_pred = self._y_test_pred,
            y_train = self._y_train, 
            y_train_pred = self._y_train_pred
        )


        return None
