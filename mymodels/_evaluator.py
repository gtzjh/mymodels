import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import is_classifier, is_regressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from imblearn.metrics import specificity_score
import yaml, pathlib, json



class MyEvaluator:
    """A class for evaluating machine learning regression models.
    
    This class handles the evaluation of machine learning models, computing various 
    accuracy metrics (R², RMSE, MAE, F1, Kappa, etc.), visualizing actual vs predicted values, 
    and saving/printing results.
    """
    def __init__(self):
        """Initialize the Evaluator with model name."""
        # Global variables statement
        # Will change in runtime
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        self.optimal_model_object = None
        self.dummy = None
        # After predictions
        self.y_test_pred = None
        self.y_train_pred = None
        # Save the accuracy metrics
        self.accuracy_dict = None
        self.dummy_accuracy_dict = None
        # Output options
        self.results_dir = None
        self.show = None
        self.plot_format = None
        self.plot_dpi = None
        self.print_results = None
        self.save_results = None
        self.save_raw_data = None


    
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
            # Output options
            results_dir: str | pathlib.Path,
            show: bool,
            plot_format: str,
            plot_dpi: int,
            print_results: bool,
            save_results: bool,
            save_raw_data: bool,
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
        self.x_test = x_test.copy(deep=True)
        self.x_train = x_train.copy(deep=True)
        self.y_test = y_test.copy(deep=True)
        self.y_train = y_train.copy(deep=True)

        self.optimal_model_object = optimal_model_object
        self.show_train = show_train
        self.dummy = dummy
        self.eval_metric = eval_metric

        # Output options
        self.results_dir = pathlib.Path(results_dir)
        self.show = show
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.print_results = print_results
        self.save_results = save_results
        self.save_raw_data = save_raw_data

        # Transform X data
        if data_engineer_pipeline:
            self.x_test = data_engineer_pipeline.transform(self.x_test)
            self.x_train = data_engineer_pipeline.transform(self.x_train)
        
        # Infer
        self.y_test_pred = self.optimal_model_object.predict(self.x_test)
        self.y_train_pred = self.optimal_model_object.predict(self.x_train)

        # Calculate accuracy metrics
        if is_regressor(self.optimal_model_object):
            self.accuracy_dict = self._get_accuracy_4_regression_task(
                self.y_test, self.y_test_pred, self.y_train, self.y_train_pred
            )
        elif is_classifier(self.optimal_model_object):
            self.accuracy_dict = self._get_accuracy_4_classification_task(
                self.y_test, self.y_test_pred, self.y_train, self.y_train_pred
            )

        # Dummy evaluation
        if self.dummy:
            self._dummy_evaluate()

        # Save all results
        self._output()

        return None
    


    def _dummy_evaluate(self):
        """Evaluate the model using a dummy estimator.
        
        This method creates a dummy estimator and evaluates its performance against the optimal model.
        It compares the accuracy metrics of the optimal model with those of the dummy estimator.

        For regression tasks, the dummy estimator is the mean of the training data.
        For classification tasks, the dummy estimator is the majority class of the training data.
        """
        _dummy_estimator = DummyRegressor() if is_regressor(self.optimal_model_object) else DummyClassifier()
        _dummy_estimator.fit(self.x_train, self.y_train)
        _dummy_y_test = _dummy_estimator.predict(self.x_test)
        _dummy_y_train = _dummy_estimator.predict(self.x_train)


        if is_regressor(self.optimal_model_object):
            self.dummy_accuracy_dict = self._get_accuracy_4_regression_task(
                self.y_test, _dummy_y_test, self.y_train, _dummy_y_train
            )
        elif is_classifier(self.optimal_model_object):
            self.dummy_accuracy_dict = self._get_accuracy_4_classification_task(
                self.y_test, _dummy_y_test, self.y_train, _dummy_y_train
            )

        return None
            


    def _get_accuracy_4_regression_task(self, y_test, y_test_pred, y_train, y_train_pred):
        """Calculate regression accuracy metrics for both test and training data.
        
        Computes R2, RMSE, and MAE metrics for both test and training predictions
        and stores them in the accuracy_dict attribute.
        
        Args:
            y_test: Actual test target values.
            y_test_pred: Predicted test target values.
            y_train: Actual training target values.
            y_train_pred: Predicted training target values.
        """
        test_accuracy_dict = dict({
            "R2": float(r2_score(y_test, y_test_pred)),
            "RMSE": float(root_mean_squared_error(y_test, y_test_pred)),
            "MAE": float(mean_absolute_error(y_test, y_test_pred)),
        })

        train_accuracy_dict = dict({
            "R2": float(r2_score(y_train, y_train_pred)),
            "RMSE": float(root_mean_squared_error(y_train, y_train_pred)),
            "MAE": float(mean_absolute_error(y_train, y_train_pred))
        })

        if self.eval_metric is not None:
            for _key, _metric in self.eval_metric.items():
                test_accuracy_dict[_key] = _metric(y_test, y_test_pred)
                train_accuracy_dict[_key] = _metric(y_train, y_train_pred)

        if self.show_train:
            return {
                "test": test_accuracy_dict,
                "train": train_accuracy_dict
            }
        else:
            return {
                "test": test_accuracy_dict,
            }
    


    def _get_accuracy_4_classification_task(self, y_test, y_test_pred, y_train, y_train_pred):
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
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        is_binary = n_classes == 2

        test_accuracy_dict = dict({
            "Overall Accuracy": float(accuracy_score(y_test, y_test_pred)),
            "Precision": float(precision_score(y_test, y_test_pred, average = "weighted")),
            "Recall": float(recall_score(y_test, y_test_pred, average = "weighted")),
            "F1": float(f1_score(y_test, y_test_pred, average = "weighted")),
            "Kappa": float(cohen_kappa_score(y_test, y_test_pred)),
        })
        
        train_accuracy_dict = dict({            
            "Overall Accuracy": float(accuracy_score(y_train, y_train_pred)),
            "Precision": float(precision_score(y_train, y_train_pred, average = "weighted")),
            "Recall": float(recall_score(y_train, y_train_pred, average = "weighted")),
            "F1": float(f1_score(y_train, y_train_pred, average = "weighted")),
            "Kappa": float(cohen_kappa_score(y_train, y_train_pred)),
        })

        # if it's binary, add matthews_corrcoef
        if is_binary:
            test_accuracy_dict["Matthews Correlation Coefficient"] = float(matthews_corrcoef(y_test, y_test_pred))
            test_accuracy_dict["Specificity"] = float(specificity_score(y_test, y_test_pred, average = "binary"))
            train_accuracy_dict["Matthews Correlation Coefficient"] = float(matthews_corrcoef(y_train, y_train_pred))
            train_accuracy_dict["Specificity"] = float(specificity_score(y_train, y_train_pred, average = "binary"))
        else:
            test_accuracy_dict["Specificity"] = float(specificity_score(y_test, y_test_pred, average = "weighted"))
            train_accuracy_dict["Specificity"] = float(specificity_score(y_train, y_train_pred, average = "weighted"))

        # The user-defined evaluation metrics
        if self.eval_metric is not None:
            for _key, _metric in self.eval_metric.items():
                test_accuracy_dict[_key] = float(_metric(y_test, y_test_pred))
                train_accuracy_dict[_key] = float(_metric(y_train, y_train_pred))

        # Output
        if self.show_train:
            return {
                "test": test_accuracy_dict,
                "train": train_accuracy_dict
            }
        else:
            return {
                "test": test_accuracy_dict,
            }



    def _output(self):
        """Process output options based on configuration.
        
        Handles saving results to files, printing to console, generating plots,
        and saving raw prediction data based on the configuration settings.
        """
        # Save results to files
        if self.save_results:
            with open(self.results_dir.joinpath("accuracy.yml"), 'w', encoding = "utf-8") as file:
                yaml.dump(self.accuracy_dict, file)
        
        # Print results to the console
        if self.print_results:
            print(f"Accuracy: \n", \
                  json.dumps(self.accuracy_dict, indent=4))
        
        if self.dummy:
            print(f"Dummy Accuracy: \n", \
                  json.dumps(self.dummy_accuracy_dict, indent=4))


        """
        # Plot
        # Regression case
        if is_regressor(self.optimal_model_object):
            self._regression_scatter_plot(
                self.y_test,
                self.y_test_pred
            )
        # Classification case
        elif is_classifier(self.optimal_model_object):
            self._classification_plots(
                self.y_test,
                self.x_test,
                self.optimal_model_object
            )
        """


        # Output train and test results
        if self.save_raw_data:
            y_test_pred_1d = self.y_test_pred
            y_train_pred_1d = self.y_train_pred
            
            # Flatten predictions if they are 2D with second dimension of 1
            if len(y_test_pred_1d.shape) > 1 and y_test_pred_1d.shape[1] == 1:
                y_test_pred_1d = y_test_pred_1d.flatten()
            if len(y_train_pred_1d.shape) > 1 and y_train_pred_1d.shape[1] == 1:
                y_train_pred_1d = y_train_pred_1d.flatten()
            
            test_results = pd.DataFrame(data={"y_test": self.y_test,
                                              "y_test_pred": y_test_pred_1d})
            train_results = pd.DataFrame(data={"y_train": self.y_train,
                                               "y_train_pred": y_train_pred_1d})
            test_results.to_csv(self.results_dir.joinpath("test_results.csv"), index = True)
            train_results.to_csv(self.results_dir.joinpath("train_results.csv"), index = True)    

        return None







