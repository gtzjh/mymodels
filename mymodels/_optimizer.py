import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
import yaml, pathlib
from joblib import Parallel, delayed
import optuna
from functools import partial
import json

from ._encoder import Encoder
from ._models import MyRegressors, MyClassifiers


def trans_category(X, y, _cat_features: list[str] | None, _encoder_method: str | None) \
    -> tuple[pd.DataFrame, object]:
    """This function is used to transform the categorical features of the training set."""
    encoder = Encoder(
        method=_encoder_method,
        target_col=str(y.name)
    )
    transformed_X = encoder.fit_transform(X=X, cat_cols=_cat_features, y=y)
    return transformed_X, encoder



class MyOptimizer:
    def __init__(self, cv: int, random_state: int, trials: int, results_dir: str, n_jobs: int = -1):
        """A class for training and optimizing various regression models.
        Initialize the Regr class.
        Parameters:
            cv (int): Number of folds for cross-validation
            random_state (int): Random seed for reproducibility
            trials (int): Number of trials to execute in optuna optimization
            results_dir (str or pathlib.Path): Directory path to store the optimization results
            n_jobs (int): Number of jobs to run in parallel k-fold cross validation
        """
        self.cv = cv
        self.random_state = random_state
        self.trials = trials
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(parents = True, exist_ok = True)
        self.n_jobs = n_jobs


    def fit(self, x_train, y_train, model_name: str, \
            cat_features: None | list[str] = None, \
            encoder_method: str | None = None) -> None:
        """
        Train and optimize a regression model.
        Parameters:
            x_train (pd.DataFrame): Training features data
            y_train (pd.Series): Training target data
            model_name (str): 
                Model selection,
                must be one of ["catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adac", "svrc", "knrc", "mlpr"]
            cat_features (list[str] or None): 
                List of categorical feature names, if any
            encoder_method (str or None): 
                Method for encoding categorical variables
        """
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.cat_features = cat_features
        self.encoder_method = encoder_method
        self.model_name = model_name  # Store the model type for use in _single_fold
        self.model_obj = None
        self.optimal_params = None
        self.optimal_model = None
        self.encoder = None
        self.final_x_train = self.x_train  # The training set which is for final prediction after encoding

        # Select model and load parameter space and static parameters
        self.model_obj, param_space, static_params = MyRegressors(
            model_name = self.model_name,
            random_state = self.random_state,
            cat_features = self.cat_features
        ).get()

        # Execute the optimization
        optuna_study = self._optimizer(param_space, static_params)
        
        # Save optimal parameters
        self.optimal_params = {**static_params, **optuna_study.best_trial.params}
        self.optimal_model = self.model_obj(**self.optimal_params)

        #######################################################################
        # 如果存在分类特征且模型不是CatBoost，则进行分类特征编码
        if self.cat_features is not None:
            if self.model_name != "catr" and self.model_name != "catc":
                self.final_x_train, self.encoder = trans_category(
                    self.x_train, self.y_train, self.cat_features, self.encoder_method
                )
                # 保存编码类型
                mapping = self.encoder.get_mapping(self.x_train, self.cat_features)
                mapping = self.encoder.convert_numpy_types(mapping)
                with open(self.results_dir.joinpath("mapping.json"), 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=4)
        #######################################################################

        # Train model with optimal parameters on the whole training + validation dataset
        self.optimal_model.fit(self.final_x_train, self.y_train)

        # Save optimal parameters
        with open(self.results_dir.joinpath("params.yml"), 'w', encoding="utf-8") as file:
            yaml.dump(self.optimal_params, file)

        # Return both the model and encoder
        return None
    

    def _optimizer(self, _param_space: dict, _static_params: dict) -> optuna.Study:
        """
        Internal method for model optimization using optuna.
        Parameters:
            _param_space: The parameter space to optimize
            _static_params: Static parameters for the model
        """
        # 设置日志级别为WARNING，避免输出过多日志
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create an Optuna study
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
        """
        Creating an objective function for the Optuna study.
        It performs the following steps:
            1. Get parameters for model training
            2. Parallel processing for validation. Initialize KFold cross validator
            3. Return mean R2 score across all folds
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
            # 如果类别特征存在且模型不是CatBoost，则进行对输入自变量分类特征编码
            if self.cat_features is not None:
                if self.model_name != "catr" and self.model_name != "catc":
                    X_fold_train, _encoder = trans_category(
                        X_fold_train, y_fold_train, self.cat_features, self.encoder_method
                    )
                    X_fold_val = _encoder.transform(X=X_fold_val)
            #######################################################################
            
            # Create and train the model
            validator = self.model_obj(**param)
            validator.fit(X_fold_train, y_fold_train)

            # Return R2 score for this fold
            return validator.score(X_fold_val, y_fold_val)

        # Parallel processing for validation. Initialize KFold cross validator
        kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        cv_r2_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_single_fold)(train_idx, val_idx, param)
            for train_idx, val_idx in kf.split(self.x_train)
        )

        # Return mean R2 score across all folds
        return np.mean(cv_r2_scores)

