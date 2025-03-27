import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
import yaml, pathlib
from joblib import Parallel, delayed
import optuna
from functools import partial
import json

from ._encoder import fit_transform_multi_features, transform_multi_features
from .models import MyRegressors, MyClassifiers



class MyOptimizer:
    def __init__(
            self, 
            cv: int, 
            random_state: int, 
            trials: int, 
            results_dir: pathlib.Path, 
            n_jobs: int = -1, 
            _plot_optimization: bool = True
        ):
        """A class for training and optimizing various machine learning models.
        
        This class handles hyperparameter optimization for both regression and classification
        models using Optuna.
        
        Args:
            cv: Number of folds for cross-validation.
            random_state: Random seed for reproducibility.
            trials: Number of trials to execute in Optuna optimization.
            results_dir: Directory path to store the optimization results.
            n_jobs: Number of jobs to run in parallel for cross-validation. Default is -1
                (use all available processors).
            _plot_optimization: Whether to plot the optimization history.
        """
        self.cv = cv
        self.random_state = random_state
        self.trials = trials
        self.results_dir = results_dir
        self.n_jobs = n_jobs
        self._plot_optimization = _plot_optimization

        # Global variables statement
        # In fit()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.model_name = None
        self.cat_features = None
        self.encode_method = None
        self.final_x_train = None
        self.final_x_test = None
        self._model_obj = None  # An instance of the model
        self._task_type = None
        self.show = None
        self.plot_format = None
        self.plot_dpi = None

        # After fit()
        self.optimal_params = None
        self.optimal_model = None
        self.y_train_pred = None
        self.y_test_pred = None


    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_name: str,
        cat_features: list[str] | tuple[str] | None = None,
        encode_method: str | list[str] | tuple[str] | None = None,
        show: bool = False,
        plot_format: str = "jpg",
        plot_dpi: int = 500
    ):
        """Train and optimize a machine learning model.
        
        This method handles the entire process of model training and optimization:
        1. Determines if it's a regression or classification task
        2. Selects the appropriate model and parameter space
        3. Optimizes hyperparameters using Optuna
        4. Processes categorical features if needed
        5. Fits the final model and makes predictions
        
        Args:
            x_train: Training features data.
            y_train: Training target data.
            x_test: Test features data.
            model_name: Model selection. For regression, should be one of 
                ["catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adar", 
                "svr", "knr", "mlpr"]. For classification, should be one of
                ["catc", "rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", 
                "svc", "knc", "mlpc"].
            cat_features: List of categorical feature names, if any.
            encode_method: Method(s) for encoding categorical variables.
        
        Returns:
            None. Results are stored in instance attributes.
        """
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.x_test = x_test.copy()
        self.model_name = model_name
        self.cat_features = cat_features
        self.encode_method = encode_method
        self.show = show
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi

        self.final_x_train = self.x_train  # The training set which is for final prediction after encoding
        self.final_x_test = self.x_test    # The test set which is for final prediction after encoding

        # Check the task type, regression or classification
        # Select model and load parameter space and static parameters
        self._task_type = self._check_task_type(self.model_name)
        self._model_obj, _param_space, _static_params = self._select_model(self._task_type)

        # Execute the optimization
        optuna_study = self._optimizer(_param_space, _static_params)
        
        # Save optimal parameters and model
        self.optimal_params = {**_static_params, **optuna_study.best_trial.params}
        self.optimal_model = self._model_obj(**self.optimal_params)

        #######################################################################
        # If there are categorical features and the model is not CatBoost
        # then encode the training and test set
        if (self.cat_features is not None) and (self.model_name not in ["catr", "catc"]):
            # Transform train set
            _transformed_X_train, _encoder_dict, _mapping_dict = fit_transform_multi_features(
                self.x_train.loc[:, self.cat_features],
                self.encode_method,
                self.y_train,
            )
            self.final_x_train = self.final_x_train.drop(columns = self.cat_features)
            self.final_x_train = pd.concat([self.final_x_train, _transformed_X_train], axis = 1)

            # Transform test set
            _transformed_X_test = transform_multi_features(
                self.x_test.loc[:, self.cat_features],
                _encoder_dict
            )
            self.final_x_test = self.final_x_test.drop(columns = self.cat_features)
            self.final_x_test = pd.concat([self.final_x_test, _transformed_X_test], axis = 1)

            # Save mapping dictionary
            with open(self.results_dir.joinpath("mapping.json"), 'w', encoding='utf-8') as f:
                json.dump(_mapping_dict, f, ensure_ascii=False, indent=4)
        #######################################################################

        # Fit on the whole training and validation set
        self.optimal_model.fit(self.final_x_train, self.y_train)

        # Infer
        self.y_train_pred = self.optimal_model.predict(self.final_x_train)
        self.y_test_pred = self.optimal_model.predict(self.final_x_test)

        # Plot the optimization history
        if self._plot_optimization:
            self._plot_optimize_history(optuna_study)

        # Save optimal parameters
        self.save_optimal_params()

        return None


    def _check_task_type(self, _model_name):
        """Determine if the task is regression or classification based on model name.
        
        Args:
            _model_name: The name of the model to check.
            
        Returns:
            str: Either "regression" or "classification".
            
        Raises:
            ValueError: If the model name is not recognized.
        """
        if _model_name in ["catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adar", "svr", "knr", "mlpr"]:
            _task_type = "regression"
        elif _model_name in ["catc", "rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", "svc", "knc", "mlpc"]:
            _task_type = "classification"
        else:
            raise ValueError(f"Invalid model name: {_model_name}")

        
        
        return _task_type
        

    def _select_model(self, _task_type):
        """Select the appropriate model and get its parameter space.
        
        Args:
            _task_type: Either "regression" or "classification".
            
        Returns:
            tuple: Contains (model_object, parameter_space, static_parameters).
        """
        if _task_type == "regression":
            _model_obj, param_space, static_params = MyRegressors(
                model_name = self.model_name,
                random_state = self.random_state,
                cat_features = self.cat_features
            ).get()
        else:
            _model_obj, param_space, static_params = MyClassifiers(
                model_name = self.model_name,
                random_state = self.random_state,
                cat_features = self.cat_features
            ).get()
        return _model_obj, param_space, static_params


    def _optimizer(self, _param_space: dict, _static_params: dict) -> optuna.Study:
        """Internal method for model optimization using Optuna.
        
        Args:
            _param_space: The parameter space to optimize.
            _static_params: Static parameters for the model.
            
        Returns:
            optuna.Study: The completed optimization study.
        """
        # Set log level to WARNING to avoid excessive output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        _study = optuna.create_study(
            direction = "maximize",
            sampler = TPESampler(seed=self.random_state),
        )

        # Execute the optimization
        _study.optimize(
            partial(self._objective, _param_space=_param_space, _static_params=_static_params),
            n_trials=self.trials,
            n_jobs=1,  # It is not recommended to use n_jobs > 1 in Optuna
            show_progress_bar=True
        )
        
        return _study


    def _objective(self, trial, _param_space, _static_params) -> float:
        """Objective function for the Optuna study.
        
        It performs the following steps:
        1. Creates model parameters by combining static and trial parameters
        2. Performs k-fold cross-validation in parallel
        3. Returns a score (mean CV score - 0.5*std) to optimize
        
        Args:
            trial: Optuna trial object.
            _param_space: The parameter space to sample from.
            _static_params: Static parameters for the model.
            
        Returns:
            float: The evaluation metric (R2 for regression, accuracy for classification)
                  adjusted by standard deviation.
        """
        # Get parameters for model training
        param = {**{k: v(trial) for k, v in _param_space.items()},
                 **_static_params}
        
        def _single_fold(train_idx, val_idx, param) -> float:
            X_fold_train = self.x_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.x_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            #######################################################################
            # If categorical features exist and model is not CatBoost, encode the input features
            if self.cat_features is not None:
                if self.model_name != "catr" and self.model_name != "catc":
                    _transformed_fold_train, _encoder_dict, _ = fit_transform_multi_features(
                        X_fold_train.loc[:, self.cat_features],
                        self.encode_method,
                        y_fold_train,
                    )
                    X_fold_train = X_fold_train.drop(columns = self.cat_features)
                    X_fold_train = pd.concat([X_fold_train, _transformed_fold_train], axis = 1)
                    
                    # Encode validation set
                    transformed_fold_val = transform_multi_features(
                        X_fold_val.loc[:, self.cat_features],
                        _encoder_dict
                    )                    
                    X_fold_val = X_fold_val.drop(columns = self.cat_features)
                    X_fold_val = pd.concat([X_fold_val, transformed_fold_val], axis = 1)
            #######################################################################

            # Create and train the model
            validator = self._model_obj(**param)
            validator.fit(X_fold_train, y_fold_train)

            # All models inherit from sklearn.base.RegressorMixin or sklearn.base.ClassifierMixin
            # and therefore have a score method
            # Regression returns R2, classification returns accuracy
            if self._task_type == "regression":
                return validator.score(X_fold_val, y_fold_val)
            else:
                return validator.score(X_fold_val, y_fold_val)

        # Parallel processing for validation. Initialize KFold cross validator
        kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        cv_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_single_fold)(train_idx, val_idx, param)
            for train_idx, val_idx in kf.split(self.x_train)
        )

        # Adjust CV results by subtracting 0.5*std for more stable results
        return np.mean(cv_scores) - 0.5 * np.std(cv_scores)
    

    def _plot_optimize_history(self, optuna_study_object: optuna.Study):
        """Plot the optimization history using matplotlib instead of plotly.

        This version doesn't require additional packages to save images.
        """
        # Get the optimization history data
        trials = optuna_study_object.trials
        values = [t.value for t in trials if t.value is not None]
        best_values = [max(values[:i+1]) for i in range(len(values))]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(values) + 1), values, 'o-', color='blue', alpha=0.5, label='Trial value')
        plt.plot(range(1, len(best_values) + 1), best_values, 'o-', color='red', label='Best value')
        
        # Add labels and title
        plt.xlabel('Trial number')
        plt.ylabel('Objective value')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(self.results_dir.joinpath("optimization_history." + self.plot_format), dpi = self.plot_dpi)

        if self.show:
            plt.show()
            
        plt.close()
        
        return None


    def save_optimal_params(self):
        """Save the optimal parameters to a YAML file.
        
        The parameters are saved to the results directory specified during initialization.
        
        Returns:
            None
        """
        with open(self.results_dir.joinpath("params.yml"), 'w', encoding="utf-8") as file:
            yaml.dump(self.optimal_params, file)
        return None

