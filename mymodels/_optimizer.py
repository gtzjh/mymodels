import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, RandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import accuracy_score, r2_score
from joblib import Parallel, delayed
from functools import partial
import logging



from mymodels import MyDataLoader, MyEstimator



class MyOptimizer:
    def __init__(
            self,
            dataset: MyDataLoader,
            estimator: MyEstimator,
            data_engineer_pipeline: Pipeline | None = None
        ):
        """A class for training and optimizing various machine learning models.
        
        This class handles hyperparameter optimization for both regression and classification
        models using Optuna.
        
        Args:
            dataset: MyDataLoader,
            estimator: MyEstimator,
            data_engineer_pipeline: Pipeline | None = None
        
        Returns:
            Optimized dataset, optimized estimator, and optimized data engineer pipeline.
        """

        # Validate input
        assert isinstance(dataset, MyDataLoader), \
            "dataset must be a mymodels.MyDataLoader object"
        assert isinstance(estimator, MyEstimator), \
            "estimator must be a mymodels.MyEstimator object"
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

        # Global variables statement
        # Input fit()
        self.strategy = None
        self.cv = None
        self.trials = None
        self.n_jobs = None
        self.direction = None
        self.eval_function = None
        
        # Inside fit()
        self._model_obj = None

    

    def fit(
        self,
        strategy = "tpe",
        cv: int = 5,
        trials: int = 100,
        n_jobs: int = -1,
        direction = "maximize",
        eval_function = None,
    ):
        """This method handles the entire process of model training and optimization:
            - Optimizes hyperparameters using Optuna
            - Fits the final model and makes predictions
        
        Args:
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
        """
    
        # Validate input
        assert strategy in ["tpe", "random"], \
            "strategy must be one of the following: tpe, random"
        assert direction in ["maximize", "minimize"], \
            "direction must be one of the following: maximize, minimize"
        if eval_function is not None and not callable(eval_function):
            raise TypeError("eval_function must be callable")

        self.strategy = strategy
        self.cv = cv
        self.trials = trials
        self.n_jobs = n_jobs
        self.direction = direction
        self.eval_function = eval_function
        

        # Select model and load parameter space and static parameters
        self._model_obj, _param_space, _static_params = 

        # Optimizing
        _optuna_study = self._study(_param_space, _static_params)
        
        # Save optimal parameters and model
        _optimal_params = {**_static_params, **_optuna_study.best_trial.params}
        # Use deep clone to make sure the seperate operation
        _optimal_model = clone(self._model_obj(**_optimal_params))


        _x_train = self.dataset.x_train
        _y_train = self.dataset.y_train

        # Data engineering
        if self.data_engineer_pipeline is not None:
            _transformed_x_train = self.data_engineer_pipeline.fit_transform(_x_train)
        
        # Fit on the whole training and validation set
        _optimal_model.fit(_transformed_x_train, _y_train)

        return _optimized_dataset, _optimized_estimator, _optimized_data_engineer_pipeline



    def _study(self, _param_space: dict, _static_params: dict) -> optuna.Study:
        """Internal method for model optimization using Optuna.
        
        Args:
            _param_space: The parameter space to optimize.
            _static_params: Static parameters for the model.
            
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
        _study = optuna.create_study(
            direction = self.direction,
            sampler = _strategy_dict[self.strategy](seed = self.random_state),
        )

        # Execute the optimization
        _study.optimize(
            partial(
                self._objective,
                _param_space = _param_space,
                _static_params = _static_params,
                _cv = self.cv,
                _n_jobs = self.n_jobs,
                _stratify = self.stratify,
                _eval_function = self.eval_function,
                _random_state = self.random_state
            ),
            n_trials = self.trials,
            n_jobs = 1,  # It is not recommended to use n_jobs > 1 in Optuna
            show_progress_bar = True
        )
        
        return _study



    def _objective(
            self,
            _trial,
            _param_space, 
            _static_params,
            _cv,
            _n_jobs,
            _stratify,
            _eval_function,
            _random_state
        ) -> float:
        """Objective function for the Optuna study.
        
        It performs the following steps:
        1. Creates model parameters by combining static and trial parameters
        2. Performs k-fold cross-validation in parallel
        3. Returns the mean CV score to optimize
        
        Args:
            _trial: Optuna trial object.
            _param_space: The parameter space to sample from.
            _static_params: Static parameters for the model.
            _cv: Number of folds for cross-validation.
            _n_jobs: Number of jobs to run in parallel for cross-validation.
            _stratify: Whether to stratify the data.
            _eval_function: The evaluation function to use.
            _random_state: The random state to use.

        Returns:
            float: The evaluation metric (R2 for regression, accuracy for classification)
                  adjusted by standard deviation.
        """

        # Get parameters for model training
        # Make the param immutable
        param = {
            **{k: v(_trial) for k, v in _param_space.items()},
            **_static_params
        }
        
        # Single fold execution
        def _single_fold(_train_idx, _val_idx, _param) -> float:
            """Single fold execution.
                - All models inherit from `sklearn.base.RegressorMixin` or `sklearn.base.ClassifierMixin`
                - and therefore have a `score` method.
                - Return R2 for regression task
                - Return the overall accuracy for classification task
            
            Args:
                _train_idx: The training index.
                _val_idx: The validation index.
                _param: The parameters for the model.
            """

            # Create a validator
            _validator = clone(self.estimator.empty_model_object(**_param))

            # Get the training and validation data
            _X_fold_train = self.dataset.x_train.iloc[_train_idx].copy(deep=True)
            _y_fold_train = self.dataset.y_train.iloc[_train_idx].copy(deep=True)
            _X_fold_val = self.dataset.x_train.iloc[_val_idx].copy(deep=True)
            _y_fold_val = self.dataset.y_train.iloc[_val_idx].copy(deep=True)


            if self.data_engineer_pipeline:
                # Use the deep clone to make sure the seperate operation
                _k_fold_data_engineer_pipeline = clone(self.data_engineer_pipeline)
                _transformed_X_fold_train = _k_fold_data_engineer_pipeline.fit_transform(_X_fold_train)
                _transformed_X_fold_val = _k_fold_data_engineer_pipeline.transform(_X_fold_val)

                # Fit in a single fold
                _validator.fit(_transformed_X_fold_train, _y_fold_train)
                _predicted_values = _validator.predict(_transformed_X_fold_val)

            else:
                # Fit in a single fold
                _validator.fit(_X_fold_train, _y_fold_train)
                _predicted_values = _validator.predict(_X_fold_val)
            

            if _eval_function is not None:
                return _eval_function(_y_fold_val, _predicted_values)
            else:
                if is_classifier(_validator):
                    # Use the overall accuracy score for classification task
                    return accuracy_score(_y_fold_val, _predicted_values)
                elif is_regressor(_validator):
                    # Use R2 score for regression task
                    return r2_score(_y_fold_val, _predicted_values)

            
        # Parallel processing for validation. Initialize KFold cross validator
        # StratifiedKFold is recommended for classification tasks especially when the class is imbalanced
        if _stratify:
            kf = StratifiedKFold(n_splits = _cv, random_state = _random_state, shuffle = True)
            cv_scores = Parallel(n_jobs = _n_jobs)(
                delayed(_single_fold)(train_idx, val_idx, param)
                for train_idx, val_idx in kf.split(X = self.dataset.x_train, y = self.dataset.y_train)
            )
        else:
            kf = KFold(n_splits = _cv, random_state = _random_state, shuffle = True)
            cv_scores = Parallel(n_jobs = _n_jobs)(
                delayed(_single_fold)(train_idx, val_idx, param)
                for train_idx, val_idx in kf.split(self.dataset.x_train)
            )
        
        return np.mean(cv_scores)
    