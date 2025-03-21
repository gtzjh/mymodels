import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
import yaml, pathlib
from joblib import Parallel, delayed
import optuna
from functools import partial
import json

from ._encoder import fit_transform_multi_features, transform_multi_features
from ._models import MyRegressors, MyClassifiers



class MyOptimizer:
    def __init__(self, cv: int, random_state: int, trials: int, results_dir: pathlib.Path, n_jobs: int = -1):
        """A class for training and optimizing various regression models.
        Initialize the Optimizer class.
        Parameters:
            cv (int): Number of folds for cross-validation
            random_state (int): Random seed for reproducibility
            trials (int): Number of trials to execute in optuna optimization
            results_dir (pathlib.Path): Directory path to store the optimization results
            n_jobs (int): Number of jobs to run in parallel k-fold cross validation
        """
        self.cv = cv
        self.random_state = random_state
        self.trials = trials
        self.results_dir = results_dir
        self.n_jobs = n_jobs


    def fit(self,
            x_train,
            y_train,
            model_name: str,
            cat_features:   None | list[str] | tuple[str] = None,
            encode_method: None | str | list[str] | tuple[str] = None
        ) -> None:
        """Train and optimize a regression model.
        
        Parameters:
            x_train (pd.DataFrame): Training features data
            y_train (pd.Series): Training target data
            model_name (str): 
                Model selection,
                must be one of ["catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adac", "svrc", "knrc", "mlpr"]
            cat_features (list[str] or tuple[str] or None): 
                List of categorical feature names, if any
            encode_method (str or list[str] or tuple[str] or None): 
                Method for encoding categorical variables
        """
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.cat_features = cat_features
        self.encode_method = encode_method
        self.model_name = model_name  # Store the model type for use in _single_fold
        
        self._model_obj = None  # An instance of the model
        self._task_type = None
        
        self.optimal_params = None
        self.optimal_model = None
        self.encoder_dict = None
        self.final_x_train = self.x_train  # The training set which is for final prediction after encoding

        # 判断是回归还是分类任务
        if self.model_name in ["catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adac", "svrc", "knrc", "mlpr"]:
            self._task_type = "regression"
        elif self.model_name in ["catc", "rfc", "dtrc", "lgbrc", "gbdtrc", "xgbrc", "adacc", "svrc", "knrc", "mlprc"]:
            self._task_type = "classification"
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        # Select model and load parameter space and static parameters
        if self._task_type == "regression":
            self._model_obj, param_space, static_params = MyRegressors(
                model_name = self.model_name,
                random_state = self.random_state,
                cat_features = self.cat_features
            ).get()
        else:
            self._model_obj, param_space, static_params = MyClassifiers(
                model_name = self.model_name,
                random_state = self.random_state,
                cat_features = self.cat_features
            ).get()

        # Execute the optimization
        optuna_study = self._optimizer(param_space, static_params)
        
        # Save optimal parameters
        self.optimal_params = {**static_params, **optuna_study.best_trial.params}
        self.optimal_model = self._model_obj(**self.optimal_params)

        # 如果存在分类特征且模型不是CatBoost，则进行分类特征编码
        if self.cat_features is not None:
            if self.model_name != "catr" and self.model_name != "catc":
                transformed_X_df, self.encoder_dict, mapping_dict = fit_transform_multi_features(
                    self.x_train.loc[:, self.cat_features],
                    self.encode_method,
                    self.y_train,
                )
                self.final_x_train = self.final_x_train.drop(columns = self.cat_features)
                self.final_x_train = pd.concat([self.final_x_train, transformed_X_df], axis = 1)
                
                # 保存编码类型
                with open(self.results_dir.joinpath("mapping.json"), 'w', encoding='utf-8') as f:
                    json.dump(mapping_dict, f, ensure_ascii=False, indent=4)

        # Train model with optimal parameters on the whole training + validation dataset
        self.optimal_model.fit(self.final_x_train, self.y_train)

        # Save optimal parameters
        self.save_optimal_params()

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
                    transformed_fold_train, encoder_dict, _ = fit_transform_multi_features(
                        X_fold_train.loc[:, self.cat_features],
                        self.encode_method,
                        y_fold_train, 
                    )
                    X_fold_train = X_fold_train.drop(columns = self.cat_features)
                    X_fold_train = pd.concat([X_fold_train, transformed_fold_train], axis = 1)
                    
                    # 对验证集进行编码
                    transformed_fold_val = transform_multi_features(
                        X_fold_val.loc[:, self.cat_features],
                        encoder_dict
                    )
                    X_fold_val = X_fold_val.drop(columns = self.cat_features)
                    X_fold_val = pd.concat([X_fold_val, transformed_fold_val], axis = 1)
            #######################################################################
            
            # Create and train the model
            validator = self._model_obj(**param)
            validator.fit(X_fold_train, y_fold_train)


            # 所有模型都继承自sklearn.base.RegressorMixin或sklearn.base.ClassifierMixin
            # 因此都有score方法
            # 回归任务返回R2, 分类任务返回准确率
            # 未来如果有修改，需要注意
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

        # DeepSeek推荐在参数优化过程对交叉验证的结果减去0.5倍的标准差，可以使得结果更加稳定
        # 返回平均得分减去标准差的一半
        return np.mean(cv_scores) - 0.5 * np.std(cv_scores)
    

    def save_optimal_params(self):
        """Save the optimal parameters to a YAML file"""
        with open(self.results_dir.joinpath("params.yml"), 'w', encoding="utf-8") as file:
            yaml.dump(self.optimal_params, file)
        return None

