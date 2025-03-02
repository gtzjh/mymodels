from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


"""每个模型的static_params都最好不要为空"""


class Regrs:
    def __init__(self, model_name: str, random_state: int, cat_features: list[str] | None = None):
        """
        Initialize the Regrs class.
        Parameters:
            model_name (str): The name of the model to use.
            random_state (int): The random state to use for the model.
            cat_features (list[str] | None): The categorical features to use for the model.
        """
        self.MODEL_MAP = {
            "svr": (self._SVR, SVR),
            "knr": (self._KNR, KNeighborsRegressor),
            "mlp": (self._MLP, MLPRegressor),
            "dt": (self._DT, DecisionTreeRegressor),
            "rf": (self._RF, RandomForestRegressor),
            "gbdt": (self._GBDT, GradientBoostingRegressor),
            "ada": (self._ADA, AdaBoostRegressor),
            "xgb": (self._XGB, XGBRegressor),
            "lgb": (self._LGB, LGBMRegressor),
            "cat": (self._CAT, CatBoostRegressor)
        }
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
        return _model_object, _param_space, _static_params


    def _SVR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"""
        param_space = {
            "kernel": lambda t: t.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
            "C": lambda t: t.suggest_float("C", 0.01, 100, log=True),
            "epsilon": lambda t: t.suggest_float("epsilon", 0.01, 1.0, step=0.01),
        }
        static_params = {
            "verbose": 0,
        }
        return param_space, static_params


    def _KNR(self):
        """https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html"""
        param_space = {
            "n_neighbors": lambda t: t.suggest_int("n_neighbors", 1, 100, step=1),
            "weights": lambda t: t.suggest_categorical("weights", ["uniform", "distance"]),
            "leaf_size": lambda t: t.suggest_int("leaf_size", 1, 100, step=1)
        }
        static_params = {
            "n_jobs": -1,
        }
        return param_space, static_params


    def _MLP(self):
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


    def _DT(self):
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


    def _RF(self):
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
        }
        return param_space, static_params


    def _GBDT(self):
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


    def _ADA(self):
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


    def _XGB(self):
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
            "enable_categorical": True if self.cat_features is not None else False,
            "seed": self.random_state,
            "verbosity": 0,
        }
        return param_space, static_params


    def _LGB(self):
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


    def _CAT(self):
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
        }
        return param_space, static_params
    


if __name__ == "__main__":
    # Simple demonstration of Regrs class usage
    print("Regrs class demonstration:")
    print("-" * 30)
    
    # Example 1: Basic usage
    model_name = "rf"  # Random Forest
    regrs = Regrs(model_name, random_state=42)
    model_obj, param_space, static_params = regrs.get()
    
    print(f"Input: model_name='{model_name}', random_state=42")
    print(f"Output: model class={model_obj.__name__}, "
          f"param_space keys={list(param_space.keys())[:3]}..., "
          f"static_params={static_params}")
    
    # Example 2: With categorical features
    print("\nWith categorical features:")
    cat_features = ["category1", "category2"]
    cat_regrs = Regrs("cat", random_state=42, cat_features=cat_features)
    _, _, cat_params = cat_regrs.get()
    
    print(f"Input: model_name='cat', cat_features={cat_features}")
    print(f"Output: cat_features in params={cat_params['cat_features']}")

