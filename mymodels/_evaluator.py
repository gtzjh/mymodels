import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor, clone
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from imblearn.metrics import specificity_score
import json



class MyEvaluator:
    """A class for evaluating machine learning regression models.
    
    This class handles the evaluation of machine learning models, computing various 
    accuracy metrics (R², RMSE, MAE, F1, Kappa, etc.), visualizing actual vs predicted values, 
    and saving/printing results.
    """
    def __init__(self):
        """Initialize the Evaluator with model name.""" 

        # Global variables statement
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        self.optimal_model_object = None
        self.show_train = None
        self.dummy = None
        self.eval_metric = None

        # After prediction, we will get:
        self.y_test_pred = None
        self.y_train_pred = None
        
        # Save the accuracy metrics
        self.accuracy_dict = dict()
        self.dummy_accuracy_dict = dict()


    def evaluate(self,
            x_test,
            x_train,
            y_test,
            y_train,
            optimal_model_object,
            data_engineer_pipeline,
            show_train: bool,
            dummy: bool,
            eval_metric: dict | None,
        ):

        """Evaluate the model on test and training data.
        
        Performs model evaluation by:
        1. Computing accuracy metrics (R2, RMSE, MAE for regression; accuracy, precision, recall, F1 for classification)
        2. Processing output options (saving results to files, printing to console, plotting visualizations)
        
        Args:
            x_test: Test feature data, needed for classification models' predict_proba method
            x_train: Training feature data
            y_test: Actual test target values
            y_train: Actual training target values
            data_engineer_pipeline: Data engineering pipeline
        """
        # Check the input data
        # Assert the input data is in dataframe or series format
        assert isinstance(x_test, (pd.DataFrame, pd.Series)), "x_test must be a pandas DataFrame or Series"
        assert isinstance(x_train, (pd.DataFrame, pd.Series)), "x_train must be a pandas DataFrame or Series"
        assert isinstance(y_test, (pd.DataFrame, pd.Series)), "y_test must be a pandas DataFrame or Series"
        assert isinstance(y_train, (pd.DataFrame, pd.Series)), "y_train must be a pandas DataFrame or Series"

        # Assert the optimal_model_object is callable
        assert callable(optimal_model_object), \
            "optimal_model_object must be a callable object"

        # Assert the data_engineer_pipeline belongs to the pipeline object in sklearn
        if data_engineer_pipeline is not None:
            assert isinstance(data_engineer_pipeline, Pipeline), \
                "data_engineer_pipeline must be a Pipeline object"
        
        # Assert show_train and dummy are boolean
        assert isinstance(show_train, bool), "show_train must be a boolean"
        assert isinstance(dummy, bool), "dummy must be a boolean"
        
        # Assert eval_metric is a dictionary and every item in it is callable
        if eval_metric is not None:
            assert isinstance(eval_metric, dict), "eval_metric must be a dictionary"
            for key, metric in eval_metric.items():
                assert callable(metric), f"Metric '{key}' in eval_metric must be callable"

        # Initialize the variables
        self.x_test = x_test.copy(deep=True)
        self.x_train = x_train.copy(deep=True)
        self.y_test = y_test.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.optimal_model_object = clone(optimal_model_object)
        self.show_train = show_train
        self.dummy = dummy
        self.eval_metric = eval_metric


        # Transform X data
        if data_engineer_pipeline:
            self.x_test = data_engineer_pipeline.transform(self.x_test)
            self.x_train = data_engineer_pipeline.transform(self.x_train)

        # Evaluate the model
        self._evaluate_model()

        # Evaluate the dummy estimator
        if self.dummy:
            self._evaluate_dummy()

        return None
    


    def _evaluate_model(self):
        """Evaluate the model on test data."""
        
        # Predict
        self.y_test_pred = self.optimal_model_object.predict(self.x_test)
        self.y_train_pred = self.optimal_model_object.predict(self.x_train)

        # Evaluate on the test data
        if is_regressor(self.optimal_model_object):
            self.accuracy_dict["test"] = self._get_accuracy_4_regression_task(
                self.y_test, self.y_test_pred
            )
        elif is_classifier(self.optimal_model_object):
            self.accuracy_dict["test"] = self._get_accuracy_4_classification_task(
                self.y_test, self.y_test_pred
            )
        
        # Evaluate on the training data
        if self.show_train:
            if is_regressor(self.optimal_model_object):
                self.accuracy_dict["train"] = self._get_accuracy_4_regression_task(
                    self.y_train, self.y_train_pred
                )
            elif is_classifier(self.optimal_model_object):
                self.accuracy_dict["train"] = self._get_accuracy_4_classification_task(
                    self.y_train, self.y_train_pred
                )
        
        print(f"Accuracy: \n", \
            json.dumps(self.accuracy_dict, indent=4))

        return None



    def _evaluate_dummy(self):
        """Evaluate the model using a dummy estimator.
        
        This method creates a dummy estimator and evaluates its performance against the optimal model.
        It compares the accuracy metrics of the optimal model with those of the dummy estimator.

        For regression tasks, the dummy estimator is the mean of the training data.
        For classification tasks, the dummy estimator is the majority class of the training data.
        """
        # Create a dummy estimator
        _dummy_estimator = DummyRegressor() if is_regressor(self.optimal_model_object) else DummyClassifier()
        _dummy_estimator.fit(self.x_train, self.y_train)

        # Predict
        _dummy_y_test = _dummy_estimator.predict(self.x_test)
        _dummy_y_train = _dummy_estimator.predict(self.x_train)

        # Evaluate on the test data
        if is_regressor(self.optimal_model_object):
            self.dummy_accuracy_dict["test"] = self._get_accuracy_4_regression_task(
                self.y_test, _dummy_y_test
            )
        elif is_classifier(self.optimal_model_object):
            self.dummy_accuracy_dict["test"] = self._get_accuracy_4_classification_task(
                self.y_test, _dummy_y_test
            )

        # Evaluate on the training data
        if self.show_train:
            if is_regressor(self.optimal_model_object):
                self.dummy_accuracy_dict["train"] = self._get_accuracy_4_regression_task(
                    self.y_train, _dummy_y_train
                )
            elif is_classifier(self.optimal_model_object):
                self.dummy_accuracy_dict["train"] = self._get_accuracy_4_classification_task(
                    self.y_train, _dummy_y_train
                )
        
        print(f"Dummy Accuracy: \n", \
            json.dumps(self.dummy_accuracy_dict, indent=4))

        return None
            


    def _get_accuracy_4_regression_task(self, y, y_pred):
        """Calculate regression accuracy metrics for both test and training data.
        
        Computes R2, RMSE, and MAE metrics for both test and training predictions
        and stores them in the accuracy_dict attribute.
        
        Args:
            y_test: Actual test target values.
            y_test_pred: Predicted test target values.
            y_train: Actual training target values.
            y_train_pred: Predicted training target values.
        """
        
        # The major metrics for regression
        accuracy_dict = dict({
            "R2": float(r2_score(y, y_pred)),
            "RMSE": float(root_mean_squared_error(y, y_pred)),
            "MAE": float(mean_absolute_error(y, y_pred)),
        })

        # The user-defined evaluation metrics
        if self.eval_metric is not None:
            for _key, _metric in self.eval_metric.items():
                accuracy_dict[_key] = _metric(y, y_pred)

        return accuracy_dict
    


    def _get_accuracy_4_classification_task(self, y, y_pred):
        """Calculate classification accuracy metrics for both test and training data.
        
        Computes accuracy, precision, recall, and F1 metrics for both test and training predictions
        and stores them in the accuracy_dict attribute.
        
        Args:
            y_test: Actual test target values.
            y_test_pred: Predicted test target values.
            y_train: Actual training target values.
            y_train_pred: Predicted training target values.
        """
        # Determine if it's binary or multi-class based on unique classes in y_test
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
        if self.eval_metric is not None:
            for _key, _metric in self.eval_metric.items():
                accuracy_dict[_key] = float(_metric(y, y_pred))

        return accuracy_dict