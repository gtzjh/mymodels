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



from ._estimator import MyEstimator



class MyOptimizer:
    def __init__(
            self, 
            random_state: int, 
            stratify: bool = False
        ):
        """A class for training and optimizing various machine learning models.
        
        This class handles hyperparameter optimization for both regression and classification
        models using Optuna.
        
        Function Call Flow:

        MyOptimizer.__init__
        |
        +-- fit()
        |   |
        |   +-- _select_model() # Select model and parameter space
        |   |
        |   +-- _optimizer() # Run Optuna optimization
        |      |
        |      +-- _objective() # Optimization objective function with CV
        |         |
        |         +-- _single_fold() # Single fold execution
        |
        +-- output() # save results
        
        Args:
            random_state: Random seed for reproducibility.
            stratify: Whether to use stratified cross-validation.
        """
        
        self.random_state = random_state
        self.stratify = stratify

        # Global variables statement
        # Input fit()
        self.x_train = None
        self.y_train = None
        self.model_name = None
        self.data_engineer_pipeline = None
        self.strategy = None
        self.cv = None
        self.trials = None
        self.n_jobs = None
        self.direction = None
        self.eval_function = None
        
        # Inside fit()
        self._model_obj = None
        
        # Inside output()
        self.optuna_study = None
        self.optimal_params = None
        self.optimal_model = None

    

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        data_engineer_pipeline: Pipeline | None = None,
        strategy = "tpe",
        cv: int = 5,
        trials: int = 100,
        n_jobs: int = -1,
        cat_features: list[str] | tuple[str] | None = None,
        direction = "maximize",
        eval_function = None,
    ):
        """This method handles the entire process of model training and optimization:
            1. Determines if it's a regression or classification task
            2. Selects the appropriate model and parameter space
            3. Optimizes hyperparameters using Optuna
            4. Fits the final model and makes predictions
        
        Args:
            x_train: Training features data.
            y_train: Training target data.
            model_name: Model selection. 
                For regression, should be one of 
                    ["lr", "catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adar", "svr", "knr", "mlpr"]. 
                For classification, should be one of
                    ["lc", "catc", "rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", "svc", "knc", "mlpc"].
            data_engineer_pipeline: A pipeline for data engineering,
                                    the pipeline should be a `sklearn.pipeline.Pipeline` object.
            strategy: 
                - "tpe": Tree-structured Parzen Estimator algorithm implemented (Default)
                - "random": Random search
            cv: Number of folds for cross-validation.
            trials: Number of trials to execute in Optuna optimization.
            n_jobs: Number of jobs to run in parallel for cross-validation. Default is -1
                (use all available processors).
            cat_features: List of categorical feature names, FOR CatBoost ONLY.
            direction:
                - "maximize": Maximize the objective function
                - "minimize": Minimize the objective function
            eval_function:
                - A user-defined function that takes in y_true and y_pred and returns a float
        """

        self.x_train = x_train.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.model_name = model_name
        self.data_engineer_pipeline = data_engineer_pipeline
        self.strategy = strategy
        self.cv = cv
        self.trials = trials
        self.n_jobs = n_jobs
        self.direction = direction
        self.eval_function = eval_function
        
        # Check if eval_function is callable
        if self.eval_function is not None and not callable(self.eval_function):
            raise TypeError("eval_function must be callable")

        # Select model and load parameter space and static parameters
        self._model_obj, _param_space, _static_params = self._select_model(cat_features)

        # Optimizing
        self.optuna_study = self._optimizer(_param_space, _static_params)
        
        # Save optimal parameters and model
        self.optimal_params = {**_static_params, **self.optuna_study.best_trial.params}
        # Use deep clone to make sure the seperate operation
        self.optimal_model = clone(self._model_obj(**self.optimal_params))

        # Data engineering
        if self.data_engineer_pipeline is not None:
            self.x_train = self.data_engineer_pipeline.fit_transform(self.x_train)
        
        # Fit on the whole training and validation set
        self.optimal_model.fit(self.x_train, self.y_train)

        return None


    def _select_model(self, cat_features = None):
        """Select the appropriate model and get its parameter space.
        
        Args:
            cat_features: List of categorical feature names, FOR CatBoost ONLY.
            
        Returns:
            tuple: Contains (model_object, parameter_space, static_parameters).
        """

        ###########################################################################################
        # If CatBoost is selected, info the user that:
        # the Encoder of categorical features is not needed in the data_engineer_pipeline
        if self.model_name in ["catr", "catc"]:
            # If the default data_engineer_pipeline includes the Encoder of categorical features,
            # info the user that the Encoder is not needed in the data_engineer_pipeline
            if self.data_engineer_pipeline is not None:
                # Check if any step name in the pipeline starts with "encode_"
                if any(step_name.startswith("encoder") for step_name, _ in self.data_engineer_pipeline.steps):
                    logging.warning("""The Encoder of categorical features is not needed for CatBoost.""")
        if self.model_name in ["svr", "svc", "knr", "knc", "mlpr", "mlpc", "lr", "lc"]:
            if self.data_engineer_pipeline is not None:
                if not any(step_name.startswith("scaler") for step_name, _ in self.data_engineer_pipeline.steps):
                    logging.warning(f"""
The Scaler is recommended for: LinearRegression, LogisticRegression, SVR, SVC, KNR, KNC, MLPRegressor, MLPClassifier
""")
        if self.model_name in ["dtr", "dtc", "rfc", "rfr", "lgbc", "lgbr", "gbdtc", "gbdtr", "xgbc", "xgbr", "catr", "catc"]:
            if self.data_engineer_pipeline is not None:
                if any(step_name.startswith("scaler") for step_name, _ in self.data_engineer_pipeline.steps):
                    logging.warning("""The Scaler is NOT recommended for tree-based models.""")
        ###########################################################################################

        _model = MyModels(
            model_name = self.model_name,
            random_state = self.random_state,
            cat_features = cat_features
        )

        return _model.model_object, _model.param_space, _model.static_params



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
        
        # Choose the sampler
        _strategy_dict = {
            "tpe": TPESampler,
            "random": RandomSampler,
        }
        _study = optuna.create_study(
            direction = self.direction,
            sampler = _strategy_dict[self.strategy](seed=self.random_state),
        )

        # Execute the optimization
        _study.optimize(
            partial(self._objective, _param_space=_param_space, _static_params=_static_params),
            n_trials = self.trials,
            n_jobs = 1,  # It is not recommended to use n_jobs > 1 in Optuna
            show_progress_bar = True
        )
        
        return _study



    def _objective(self, trial, _param_space, _static_params) -> float:
        """Objective function for the Optuna study.
        
        It performs the following steps:
        1. Creates model parameters by combining static and trial parameters
        2. Performs k-fold cross-validation in parallel
        3. Returns the mean CV score to optimize
        
        Args:
            trial: Optuna trial object.
            _param_space: The parameter space to sample from.
            _static_params: Static parameters for the model.
            
        Returns:
            float: The evaluation metric (R2 for regression, accuracy for classification)
                  adjusted by standard deviation.
        """

        # Get parameters for model training
        # Make the param immutable
        param = {
            **{k: v(trial) for k, v in _param_space.items()},
            **_static_params
        }
        
        # Single fold execution
        def _single_fold(train_idx, val_idx, param) -> float:
            """Single fold execution.
                - All models inherit from `sklearn.base.RegressorMixin` or `sklearn.base.ClassifierMixin`
                - and therefore have a `score` method.
                - Return R2 for regression task
                - Return the overall accuracy for classification task
            
            Args:
                train_idx: The training index.
                val_idx: The validation index.
                param: The parameters for the model.
            """

            # Create a validator
            _validator = clone(self._model_obj(**param))

            # Get the training and validation data
            X_fold_train = self.x_train.iloc[train_idx].copy(deep=True)
            y_fold_train = self.y_train.iloc[train_idx].copy(deep=True)
            X_fold_val = self.x_train.iloc[val_idx].copy(deep=True)
            y_fold_val = self.y_train.iloc[val_idx].copy(deep=True)


            if self.data_engineer_pipeline:
                # Use the deep clone to make sure the seperate operation
                _k_fold_data_engineer_pipeline = clone(self.data_engineer_pipeline)
                _transformed_X_fold_train = _k_fold_data_engineer_pipeline.fit_transform(X_fold_train)
                _transformed_X_fold_val = _k_fold_data_engineer_pipeline.transform(X_fold_val)

                # Fit in a single fold
                _validator.fit(_transformed_X_fold_train, y_fold_train)
                _predicted_values = _validator.predict(_transformed_X_fold_val)

            else:
                # Fit in a single fold
                _validator.fit(X_fold_train, y_fold_train)
                _predicted_values = _validator.predict(X_fold_val)
            

            if self.eval_function is not None:
                eval_value = self.eval_function(y_fold_val, _predicted_values)
            else:
                if is_classifier(_validator):
                    # Use the overall accuracy score for classification task
                    eval_value = accuracy_score(y_fold_val, _predicted_values)
                elif is_regressor(_validator):
                    # Use R2 score for regression task
                    eval_value = r2_score(y_fold_val, _predicted_values)
            
            return eval_value

            
        # Parallel processing for validation. Initialize KFold cross validator
        # StratifiedKFold is recommended for classification tasks especially when the class is imbalanced
        if self.stratify:
            kf = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
            cv_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(_single_fold)(train_idx, val_idx, param)
                for train_idx, val_idx in kf.split(X = self.x_train, y = self.y_train)
            )
        else:
            kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
            cv_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(_single_fold)(train_idx, val_idx, param)
                for train_idx, val_idx in kf.split(self.x_train)
            )
        
        return np.mean(cv_scores)
    