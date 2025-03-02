import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
import yaml, pathlib
from accuracy import RegrAccuracy
from encoder import Encoder
from joblib import Parallel, delayed
from regressors import Regrs
import optuna
from functools import partial
import json



def trans_category(X, y, _cat_features: list[str] | None, _encoder_method: str | None) \
    -> tuple[pd.DataFrame, object]:
    """This function is used to transform the categorical features of the training set."""
    encoder = Encoder(
        method=_encoder_method,
        target_col=str(y.name)
    )
    transformed_X = encoder.fit_transform(X=X, cat_cols=_cat_features, y=y)
    return transformed_X, encoder



class Pipeline:
    """
    -> __init__()
    -> fit()
        -> Regrs()  # Regrs() is a class that contains all the models
            -> _optimizer()
                -> _objective()
                    -> _single_fold()
    -> evaluate()
    """

    def __init__(self, cv: int, random_state: int, trials: int, results_dir: str):
        """A class for training and optimizing various regression models.
        Initialize the Regr class.
        Parameters:
            cv (int): Number of folds for cross-validation
            random_state (int): Random seed for reproducibility
            trials (int): Number of trials to execute in optuna optimization
            results_dir (str or pathlib.Path): Directory path to store the optimization results
        """
        self.cv = cv
        self.random_state = random_state
        self.trials = trials
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(parents = True, exist_ok = True)


    def fit(self, x_train, y_train, model_name: str, \
            cat_features = None | list[str], \
            encoder_method: str | None = None) -> None:
        """
        Train and optimize a regression model.
        Parameters:
            x_train (pd.DataFrame): Training features data
            y_train (pd.Series): Training target data
            model_name (str): 
                Model selection,
                must be one of ["cat", "rf", "dt", "lgb", "gbdt", "xgb", "ada", "svr", "knr", "mlp"]
            cat_features (list[str] or None): 
                List of categorical feature names, if any
            encoder_method (str or None): 
                Method for encoding categorical variables
        """
        self.x_train = x_train
        self.y_train = y_train
        self.cat_features = cat_features
        self.encoder_method = encoder_method
        self.model_name = model_name    # Store the model type for use in _single_fold
        self.model_obj = None
        self.optimal_params = None
        self.optimal_model = None
        self.encoder = None
        self.final_x_train = self.x_train  # The training set which is for final prediction after encoding

        # Select model and load parameters
        self.model_obj, param_space, static_params = Regrs(self.model_name, self.random_state, self.cat_features).get()

        # Execute the optimization
        optuna_study = self._optimizer(param_space, static_params)
        
        # Save optimal parameters
        self.optimal_params = {**static_params, **optuna_study.best_trial.params}
        self.optimal_model = self.model_obj(**self.optimal_params)

        #######################################################################
        # 如果存在分类特征且模型不是CatBoost，则进行分类特征编码
        if self.cat_features is not None and self.model_name != "cat":
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
        # Create an Optuna study
        _study = optuna.create_study(
            direction = "maximize",
            sampler = TPESampler(seed=self.random_state),
        )
        
        # Execute the optimization
        _study.optimize(
            partial(self._objective, _param_space=_param_space, _static_params=_static_params),
            n_trials=self.trials,
            n_jobs=1  # It is not recommended to use n_jobs > 1 in Optuna
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
            if self.cat_features is not None and self.model_name != "cat":
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
        cv_r2_scores = Parallel(n_jobs=-1)(
            delayed(_single_fold)(train_idx, val_idx, param)
            for train_idx, val_idx in kf.split(self.x_train)
        )

        # Return mean R2 score across all folds
        return np.mean(cv_r2_scores)


    def evaluate(self, opt_model, x_test, y_test) -> None:
        """
        Evaluate the optimized model and save the results.
        Parameters:
            opt_model: The optimized model
            x_test: The testing features data
            y_test: The testing target data
        """
        # 根据不同情况处理测试数据
        if self.model_name == "cat":  
            # CatBoost 直接处理分类特征，不需要编码
            _final_x_test = x_test
        elif self.encoder is not None:
            # 其他模型且有编码器时，应用编码
            _final_x_test = self.encoder.transform(X=x_test)
        else:
            # 其他情况，直接使用测试数据
            _final_x_test = x_test
        
        # 评估并保存结果
        _y_test_pred = opt_model.predict(_final_x_test)    # 测试集上的准确度
        _y_train_pred = opt_model.predict(self.final_x_train)  # 训练集上的准确度
        RegrAccuracy(y_test, _y_test_pred, self.y_train, _y_train_pred, self.results_dir)
        
        return None
