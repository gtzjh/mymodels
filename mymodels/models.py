from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from types import MappingProxyType



class MyClassifiers:
    def __init__(self, model_name: str, random_state: int, cat_features: list[str] | None = None):
        """Initialize the Classifiers class.
        
        Parameters:
            model_name (str): The name of the model to use.
            random_state (int): The random state to use for the model.
            cat_features (list[str] | None): The categorical features to use for the CatBoost ONLY.
        """

        _model_map = {
            "lc": (self._LC, LogisticRegression),
            "svc": (self._SVC, SVC),
            "knc": (self._KNC, KNeighborsClassifier),
            "mlpc": (self._MLPC, MLPClassifier),
            "dtc": (self._DTC, DecisionTreeClassifier),
            "rfc": (self._RFC, RandomForestClassifier),
            "gbdtc": (self._GBDTC, GradientBoostingClassifier),
            "adac": (self._ADAC, AdaBoostClassifier),
            "xgbc": (self._XGBC, XGBClassifier),
            "lgbc": (self._LGBC, LGBMClassifier),
            "catc": (self._CATC, CatBoostClassifier)
        }
        self.MODEL_MAP = MappingProxyType(_model_map)  # 创建不可变视图
        self.model_name = model_name
        self.random_state = random_state
        self.cat_features = cat_features
        
        assert model_name in self.MODEL_MAP, \
            f"Invalid model name: {model_name}, \
              it must be one of {list(self.MODEL_MAP.keys())}"
    
    def get(self):
        """
        Get the model object, parameter space, and static parameters.
        Returns:
            tuple: (model_object, param_space, static_params)
        """
        _model_object = self.MODEL_MAP[self.model_name][1]
        _get_method = self.MODEL_MAP[self.model_name][0]
        _param_space, _static_params = _get_method()
        return _model_object, MappingProxyType(_param_space), MappingProxyType(_static_params)
    

    def _LC(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"""
        param_space = {
            "solver": lambda t: t.suggest_categorical("solver", ["lbfgs", "saga"]),
            "penalty": lambda t: t.suggest_categorical("penalty", ["l2", None]),
            "C": lambda t: t.suggest_float("C", 0.01, 10.0, log=True),
            "max_iter": lambda t: t.suggest_int("max_iter", 100, 1000, step=100),
        }
        static_params = {
            # "multi_class": "multinomial",
            "class_weight": "balanced",
            "tol": 1e-4,
            "fit_intercept": True,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": 0,
            "warm_start": False,
        }
        return param_space, static_params
    

    def _SVC(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"""
        param_space = {
            "kernel": lambda t: t.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
            "C": lambda t: t.suggest_float("C", 0.1, 200, log=True),
            "degree": lambda t: t.suggest_int("degree", 2, 5),  # For poly kernel
        }
        static_params = {
            "class_weight": "balanced",
            "probability": True,  # Required for some methods like predict_proba
            "random_state": self.random_state,
            "verbose": 0,
        }
        return param_space, static_params


    def _KNC(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"""
        param_space = {
            "n_neighbors": lambda t: t.suggest_int("n_neighbors", 1, 50, step=1),
            "weights": lambda t: t.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": lambda t: t.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": lambda t: t.suggest_int("leaf_size", 10, 100, step=10),
            "p": lambda t: t.suggest_int("p", 1, 2),  # 1 for manhattan_distance, 2 for euclidean
        }
        static_params = {
            "n_jobs": -1,
        }
        return param_space, static_params


    def _MLPC(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"""
        param_space = {
            "activation": lambda t: t.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            "alpha": lambda t: t.suggest_float("alpha", 0.0001, 0.1, log=True),
            "learning_rate": lambda t: t.suggest_categorical("learning_rate", ["constant", "adaptive"]),
            "learning_rate_init": lambda t: t.suggest_float("learning_rate_init", 0.0001, 0.1, log=True),
            "max_iter": lambda t: t.suggest_int("max_iter", 100, 3000, step=100),
        }
        static_params = {
            "hidden_layer_sizes": (300, 300, 300),
            "solver": "adam",
            "batch_size": "auto",
            "early_stopping": True,
            "n_iter_no_change": 10,
            "random_state": self.random_state,
            "verbose": 0
        }
        return param_space, static_params


    def _DTC(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"""
        param_space = {
            "criterion": lambda t: t.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": lambda t: t.suggest_int("max_depth", 2, 20),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
            "max_features": lambda t: t.suggest_float("max_features", 0.2, 1.0),
            "class_weight": lambda t: t.suggest_categorical("class_weight", ["balanced", None]),
        }
        static_params = {
            "random_state": self.random_state
        }
        return param_space, static_params


    def _RFC(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""
        param_space = {
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
            "criterion": lambda t: t.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
            "max_features": lambda t: t.suggest_float("max_features", 0.1, 1.0, step=0.1),
            "bootstrap": lambda t: t.suggest_categorical("bootstrap", [True, False]),
            "class_weight": lambda t: t.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        }
        static_params = {
            "random_state": self.random_state,
            "verbose": 0,
            "n_jobs": -1,
        }
        return param_space, static_params


    def _GBDTC(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html"""
        param_space = {
            "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=50),
            "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "criterion": lambda t: t.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
            "max_depth": lambda t: t.suggest_int("max_depth", 3, 8),
            "max_features": lambda t: t.suggest_float("max_features", 0.2, 1.0, step=0.1),
        }
        static_params = {
            "random_state": self.random_state,
            "verbose": 0,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            "tol": 1e-4,
        }
        return param_space, static_params


    def _ADAC(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html"""
        param_space = {
            "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 3000, step=50),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 1.0, log=True),
        }
        static_params = {
            "algorithm": "SAMME",
            "random_state": self.random_state,
        }
        return param_space, static_params


    def _XGBC(self):
        """https://xgboost.readthedocs.io/en/latest/python/python_api.html"""
        param_space = {
            "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 3000, step=50),
            "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
            "colsample_bylevel": lambda t: t.suggest_float("colsample_bylevel", 0.5, 1.0, step=0.1),
            "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
            "gamma": lambda t: t.suggest_float("gamma", 0, 1, step=0.1),
            "reg_alpha": lambda t: t.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": lambda t: t.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "tree_method": lambda t: t.suggest_categorical("tree_method", ["auto", "exact", "approx", "hist"]),
        }
        static_params = {
            # "objective": "binary:logistic",  # Change based on problem type
            # "eval_metric": "logloss",  # Change based on problem type
            "booster": "gbtree",
            "enable_categorical": False,
            "seed": self.random_state,
            "verbosity": 0,
            "n_jobs": -1,
        }
        return param_space, static_params


    def _LGBC(self):
        """https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html"""
        param_space = {
            "num_leaves": lambda t: t.suggest_int("num_leaves", 20, 150),
            "max_depth": lambda t: t.suggest_int("max_depth", -1, 15),  # -1 means no limit
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 3000, step=50),
            "min_child_samples": lambda t: t.suggest_int("min_child_samples", 5, 100),
            "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "subsample_freq": lambda t: t.suggest_int("subsample_freq", 0, 10),  # 0 means no frequency
            "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": lambda t: t.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": lambda t: t.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "min_split_gain": lambda t: t.suggest_float("min_split_gain", 0, 0.5),
        }
        static_params = {
            # "objective": "binary",  # Change based on problem type
            # "metric": "binary_logloss",  # Change based on problem type
            "boosting_type": "gbdt",
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        return param_space, static_params


    def _CATC(self):
        """https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier"""
        param_space = {
            "iterations": lambda t: t.suggest_int("iterations", 100, 3000, step=100),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": lambda t: t.suggest_int("depth", 4, 10),  # Renamed from max_depth to depth
            "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": lambda t: t.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": lambda t: t.suggest_float("random_strength", 0.0, 1.0),
            "border_count": lambda t: t.suggest_int("border_count", 32, 255),
            "grow_policy": lambda t: t.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            "min_data_in_leaf": lambda t: t.suggest_int("min_data_in_leaf", 1, 50),
        }
        static_params = {
            "cat_features": self.cat_features,
            "random_seed": self.random_state,
            "verbose": 0,
            "allow_writing_files": False,
            "train_dir": None,
            "thread_count": -1,
        }
        return param_space, static_params



class MyRegressors:
    def __init__(self, model_name: str, random_state: int, cat_features: list[str] | None = None):
        """Initialize the Regrs class.
        
        Parameters:
            model_name (str): The name of the model to use.
            random_state (int): The random state to use for the model.
            cat_features (list[str] | None): The categorical features to use for the CatBoost ONLY.
        """

        _model_map = {
            "lr": (self._LR, LinearRegression),
            "svr": (self._SVR, SVR),
            "knr": (self._KNR, KNeighborsRegressor),
            "mlpr": (self._MLPR, MLPRegressor),
            "dtr": (self._DTR, DecisionTreeRegressor),
            "rfr": (self._RFR, RandomForestRegressor),
            "gbdtr": (self._GBDTR, GradientBoostingRegressor),
            "adar": (self._ADAR, AdaBoostRegressor),
            "xgbr": (self._XGBR, XGBRegressor),
            "lgbr": (self._LGBR, LGBMRegressor),
            "catr": (self._CATR, CatBoostRegressor)
        }
        self.MODEL_MAP = MappingProxyType(_model_map)  # 创建不可变视图
        self.model_name = model_name
        self.random_state = random_state
        self.cat_features = cat_features

        assert model_name in self.MODEL_MAP, \
            f"Invalid model name: {model_name}, \
              it must be one of {list(self.MODEL_MAP.keys())}"
    

    def get(self):
        """Get the model object, parameter space, and static parameters.
        
        Returns:
            tuple: (model_object, param_space, static_params)
        """
        _model_object = self.MODEL_MAP[self.model_name][1]
        _get_method = self.MODEL_MAP[self.model_name][0]
        _param_space, _static_params = _get_method()
        return _model_object, MappingProxyType(_param_space), MappingProxyType(_static_params)
    

    def _LR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"""
        param_space = {}
        static_params = {}
        return param_space, static_params
    

    def _SVR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"""
        param_space = {
            "kernel": lambda t: t.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
            "C": lambda t: t.suggest_float("C", 1, 200, step=1),
            "epsilon": lambda t: t.suggest_float("epsilon", 0.1, 10, step=0.1),
        }
        static_params = {
            "verbose": 0,
        }
        return param_space, static_params


    def _KNR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html"""
        param_space = {
            "n_neighbors": lambda t: t.suggest_int("n_neighbors", 1, 10, step=1),
            "weights": lambda t: t.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": lambda t: t.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": lambda t: t.suggest_int("leaf_size", 1, 100, step=1),
            "p": lambda t: t.suggest_int("p", 1, 5, step=1),
        }
        static_params = {
            "n_jobs": -1,
        }
        return param_space, static_params


    def _MLPR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html"""
        param_space = {
            "alpha": lambda t: t.suggest_float("alpha", 0.0001, 0.1, log=True),
            "learning_rate_init": lambda t: t.suggest_float("learning_rate_init", 0.0001, 0.1, log=True),
            "max_iter": lambda t: t.suggest_int("max_iter", 100, 3000, step = 100),
        }
        static_params = {
            "hidden_layer_sizes": (300, 300, 300),
            "activation": "relu",
            "solver": "adam",
            "batch_size": "auto",
            "random_state": self.random_state,
            "verbose": 0
        }
        return param_space, static_params


    def _DTR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html"""
        param_space = {
            "max_depth": lambda t: t.suggest_int("max_depth", 2, 20),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
            "max_features": lambda t: t.suggest_float("max_features", 0.2, 1),
        }
        static_params = {
            "random_state": self.random_state
        }
        return param_space, static_params


    def _RFR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"""
        param_space = {
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
            "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
            "max_features": lambda t: t.suggest_float("max_features", 0.1, 1.0, step=0.1),
        }
        static_params = {
            "random_state": self.random_state,
            "verbose": 0,
            "n_jobs": -1,
        }
        return param_space, static_params


    def _GBDTR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html"""
        param_space = {
            "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-8, 1.0, log=True),
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
            "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
            "subsample": lambda t: t.suggest_float("subsample", 0.2, 1.0, step=0.1),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
            "max_features": lambda t: t.suggest_float("max_features", 0.2, 1.0, step=0.1),
        }
        static_params = {
            "random_state": self.random_state,
            "verbose": 0,
        }
        return param_space, static_params


    def _ADAR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html"""
        param_space = {
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "loss": lambda t: t.suggest_categorical("loss", ["linear", "square", "exponential"]),
        }
        static_params = {
            "random_state": self.random_state,
        }
        return param_space, static_params


    def _XGBR(self):
        """https://xgboost.readthedocs.io/en/latest/python/python_api.html"""
        param_space = {
            "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-8, 1.0, log=True),
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
            "subsample": lambda t: t.suggest_float("subsample", 0.2, 1.0, step=0.1),
            "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
            "gamma": lambda t: t.suggest_float("gamma", 0, 5, step=0.1),
            "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": lambda t: t.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": lambda t: t.suggest_float("reg_lambda", 0.5, 5),
            "tree_method": lambda t: t.suggest_categorical("tree_method", ["hist", "approx"]),
        }
        static_params = {
            "enable_categorical": False,
            "seed": self.random_state,
            "verbosity": 0,
            "n_jobs": -1,
        }
        return param_space, static_params


    def _LGBR(self):
        """https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html"""
        param_space = {
            "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-8, 1.0, log=True),
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
            "num_leaves": lambda t: t.suggest_int("num_leaves", 2, 256),
            "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.2, 1.0),
            "subsample": lambda t: t.suggest_float("subsample", 0.2, 1.0),
            "subsample_freq": lambda t: t.suggest_int("subsample_freq", 1, 7),
            "reg_alpha": lambda t: t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        static_params = {
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        return param_space, static_params


    def _CATR(self):
        """https://catboost.ai/en/docs/concepts/python-reference_catboostregressor"""
        param_space = {
            "iterations": lambda t: t.suggest_int("iterations", 100, 3000, step=100),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-5, 1.0, log=True),
            "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
            "bagging_temperature": lambda t: t.suggest_float("bagging_temperature", 0.0, 1.0),
            "subsample": lambda t: t.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": lambda t: t.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": lambda t: t.suggest_int("min_data_in_leaf", 1, 100),
            "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "leaf_estimation_iterations": lambda t: t.suggest_int("leaf_estimation_iterations", 1, 10)
        }
        static_params = {
            "cat_features": self.cat_features,
            "random_seed": self.random_state,
            "verbose": 0,
            "allow_writing_files": False,
            "train_dir": None,
            "thread_count": -1,
        }
        return param_space, static_params
