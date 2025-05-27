"""Module for optimizing machine learning models using Optuna.

This module provides functionality for hyperparameter optimization of machine learning models
using Optuna. It supports both classification and regression tasks.
"""
import logging
from functools import partial

import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import accuracy_score, r2_score
from joblib import Parallel, delayed


class MyOptimizer:
    def __init__(
            self,
            dataset,
            estimator,
            data_engineer_pipeline: Pipeline | None = None,
            plotter = None,
            output = None
        ):
        """A class for training and optimizing various machine learning models.
        
        This class handles hyperparameter optimization for both regression and classification
        models using Optuna.
        
        Args:
            dataset: MyDataLoader,
            estimator: MyEstimator,
            data_engineer_pipeline: Pipeline | None = None,
            plotter: Plotter | None = None,
            output: Output | None = None
        ):
        
        Returns:
            Optimized dataset, optimized estimator, and optimized data engineer pipeline.
        """

        # Check data_engineer_pipeline validity
        if data_engineer_pipeline is not None:
            assert isinstance(data_engineer_pipeline, Pipeline), \
                "data_engineer_pipeline must be a `sklearn.pipeline.Pipeline` object"
        else:
            logging.warning("No data engineering will be implemented, the raw data will be used.")

        # Initialize
        self.dataset = dataset
        self.estimator = estimator
        self.data_engineer_pipeline = data_engineer_pipeline
        self.plotter = plotter
        self.output = output

        # Global variables statement
        self.stratify = None
        self.strategy = None
        self.cv = None
        self.trials = None
        self.n_jobs = None
        self.direction = None
        self.eval_function = None
        self.random_state = None
        self.optuna_study = None

    def fit(
        self,
        stratify: bool,
        strategy: str = "tpe",
        cv: int = 5,
        trials: int = 100,
        n_jobs: int = -1,
        direction: str = "maximize",
        eval_function = None,
        random_state: int = 0
    ):
        """This method handles the entire process of model training and optimization:
            - Optimizes hyperparameters using Optuna
            - Fits the final model and makes predictions
        
        Args:
            stratify: Whether to stratify the data.
            strategy: 
                - "tpe": Tree-structured Parzen Estimator algorithm implemented (Default)
                - "random": Random search
            cv: Number of folds for cross-validation.
            trials: Number of trials to execute in Optuna optimization.
            n_jobs: Number of jobs to run in parallel for cross-validation. Default is -1
                (use all available processors).
            direction:
                - "maximize": Maximize the objective function
                - "minimize": Minimize the objective function
            eval_function:
                - A user-defined function that takes in y_true and y_pred and returns a float
                - If None, the default evaluation function will be used
            random_state: The random state to use. Default is 0.
        """
        # Validate input
        assert isinstance(stratify, bool), \
            "stratify must be a boolean"
        assert strategy in ["tpe", "random"], \
            "strategy must be one of the following: tpe, random"
        assert isinstance(cv, int), \
            "cv must be an integer"
        assert isinstance(trials, int), \
            "trials must be an integer"
        assert isinstance(n_jobs, int), \
            "n_jobs must be an integer"
        assert direction in ["maximize", "minimize"], \
            "direction must be one of the following: maximize, minimize"
        if eval_function is not None and not callable(eval_function):
            raise TypeError("eval_function must be callable")
        assert isinstance(random_state, int), \
            "random_state must be an integer"

        # Initialize
        self.stratify = stratify
        self.strategy = strategy
        self.cv = cv
        self.trials = trials
        self.n_jobs = n_jobs
        self.direction = direction
        self.eval_function = eval_function
        self.random_state = random_state

        ###########################################################################################
        # Optimize
        ###########################################################################################
        # Get the parameter space and static parameters
        _param_space = self.estimator.param_space
        _static_params = self.estimator.static_params

        # Optimizing
        _optuna_study = self._study(_param_space, _static_params)

        # Save optimal parameters and model
        _optimal_params = {**_optuna_study.best_trial.params, **_static_params}
        ###########################################################################################

        ###########################################################################################
        # Fit the final optimal model
        ###########################################################################################
        # Prepare the data
        _x_train = self.dataset.x_train.copy(deep=True)
        _y_train = self.dataset.y_train.copy(deep=True)

        # Data engineering
        if self.data_engineer_pipeline is not None:
            _transformed_x_train = self.data_engineer_pipeline.fit_transform(_x_train)
        else:
            _transformed_x_train = _x_train

        # Use deep clone to make sure the seperate operation
        _optimal_model = clone(self.estimator.empty_model_object(**_optimal_params))

        # Fit on the whole dataset
        _optimal_model.fit(_transformed_x_train, _y_train)
        ###########################################################################################

        # Save the optimal parameters and model instance
        self.estimator.optimal_params = _optimal_params
        self.estimator.optimal_model_object = _optimal_model

        # Return the optimized dataset, estimator, and data engineer pipeline
        _optimized_dataset = self.dataset
        _optimized_estimator = self.estimator
        _optimized_data_engineer_pipeline = None
        if self.data_engineer_pipeline is not None:
            _optimized_data_engineer_pipeline = self.data_engineer_pipeline

        # Plot and output
        self._plot(self.plotter)
        self._output(self.output)

        return (
            _optimized_dataset,
            _optimized_estimator,
            _optimized_data_engineer_pipeline
        )

    def _study(
            self,
            param_space: dict,
            static_params: dict,
        ) -> optuna.Study:
        """Internal method for model optimization using Optuna.
        
        Args:
            param_space: The parameter space to optimize.
            static_params: Static parameters for the model.
            
        Returns:
            optuna.Study: The completed optimization study.
        """
        # Set log level to WARNING to avoid excessive output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Choose the sampler
        _strategy_dict = {
            "tpe": TPESampler,
            "random": RandomSampler,
        }
        self.optuna_study = optuna.create_study(
            direction=self.direction,
            sampler=_strategy_dict[self.strategy](seed=self.random_state),
        )

        # Execute the optimization
        self.optuna_study.optimize(
            partial(
                self._objective,
                param_space=param_space,
                static_params=static_params,
            ),
            n_trials=self.trials,
            n_jobs=1,  # It is not recommended to use n_jobs > 1 in Optuna
            show_progress_bar=True
        )

        return self.optuna_study

    def _objective(
            self,
            trial,
            param_space,
            static_params
        ) -> float:
        """Objective function for the Optuna study.

            1. Creates model parameters by combining static and trial parameters
            2. Performs k-fold cross-validation in parallel
            3. Returns the mean CV score to optimize
        
        Args:
            trial: Optuna trial object.
            param_space: The parameter space to sample from.
            static_params: Static parameters for the model.

        Returns:
            float: The evaluation metric (R2 for regression, accuracy for classification)
                  adjusted by standard deviation.
        """
        # Get parameters for model training
        # Make the param immutable
        _param = {
            **{k: v(trial) for k, v in param_space.items()},
            **static_params
        }

        # Parallel processing for validation. Initialize KFold cross validator
        # StratifiedKFold is recommended for classification tasks especially when the class is imbalanced
        if self.stratify:
            assert is_classifier(self.estimator.empty_model_object(**{})), \
                "StratifiedKFold can only be used for classification tasks."
            _kf = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
            _cv_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._single_fold)(train_idx, val_idx, _param, self.eval_function)
                for train_idx, val_idx in _kf.split(X=self.dataset.x_train, y=self.dataset.y_train)
            )
        else:
            _kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
            _cv_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._single_fold)(train_idx, val_idx, _param, self.eval_function)
                for train_idx, val_idx in _kf.split(self.dataset.x_train)
            )

        return np.mean(_cv_scores)

    def _single_fold(self, train_idx, val_idx, param, eval_function) -> float:
        """Single fold execution.
        
        Args:
            train_idx: The training index.
            val_idx: The validation index.
            param: The parameters for the model.
            eval_function: The user-defined evaluation function to use.
        
        Returns:
            float: The evaluation metric (R2 for regression, accuracy for classification)
        """
        # Create a validator
        validator = clone(self.estimator.empty_model_object(**param))

        # Get the training and validation data
        x_fold_train = self.dataset.x_train.iloc[train_idx].copy(deep=True)
        y_fold_train = self.dataset.y_train.iloc[train_idx].copy(deep=True)
        x_fold_val = self.dataset.x_train.iloc[val_idx].copy(deep=True)
        y_fold_val = self.dataset.y_train.iloc[val_idx].copy(deep=True)

        if self.data_engineer_pipeline:
            # Use the deep clone to make sure the seperate operation
            k_fold_data_engineer_pipeline = clone(self.data_engineer_pipeline)
            transformed_x_fold_train = k_fold_data_engineer_pipeline.fit_transform(x_fold_train)
            transformed_x_fold_val = k_fold_data_engineer_pipeline.transform(x_fold_val)

            # Fit in a single fold
            validator.fit(transformed_x_fold_train, y_fold_train)
            predicted_values = validator.predict(transformed_x_fold_val)
        else:
            # Fit in a single fold
            validator.fit(x_fold_train, y_fold_train)
            predicted_values = validator.predict(x_fold_val)

        if eval_function is not None:
            score = eval_function(y_fold_val, predicted_values)
            # Verify that the evaluation function returns a float
            if not isinstance(score, (int, float)):
                raise TypeError(f"Evaluation function must return a float, got {type(score)}")
            return float(score)

        if is_classifier(validator):
            # Use the overall accuracy score for classification task
            return accuracy_score(y_fold_val, predicted_values)
        if is_regressor(validator):
            # Use R2 score for regression task
            return r2_score(y_fold_val, predicted_values)

    def _plot(self, plotter):
        """Plot the optimization results.
        
        Args:
            plotter: The plotter to use.
        """
        if plotter:
            plotter.plot_optimize_history(self.optuna_study)

    def _output(self, output):
        """Output the optimization results.
        
        Args:
            output: The output object.
        """
        if output:
            output.output_optimal_params(self.estimator.optimal_params)
            output.output_optimal_model(self.estimator.optimal_model_object,
                                        self.estimator.save_type)
