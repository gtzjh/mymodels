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
import yaml



def _convert_param_space(param_space_config):
    """Convert YAML config to Optuna parameter space definition"""
    param_space = {}
    
    for param_name, param_config in param_space_config.items():
        if param_config.get('type') == 'categorical':
            # Categorical parameter
            values = param_config['values']
            param_space[param_name] = lambda t, name=param_name, values=values: t.suggest_categorical(name, values)
        elif param_config.get('type') == 'integer':
            # Integer parameter
            values = param_config['values']
            min_val = values['min']
            max_val = values['max']
            step = values.get('step', 1)
            param_space[param_name] = lambda t, name=param_name, min_v=min_val, max_v=max_val, s=step: t.suggest_int(name, min_v, max_v, step=s)
        elif param_config.get('type') == 'float':
            # Float parameter
            values = param_config['values']
            min_val = values['min']
            max_val = values['max']
            log = values.get('log', False)
            param_space[param_name] = lambda t, name=param_name, min_v=min_val, max_v=max_val, l=log: t.suggest_float(name, min_v, max_v, log=l)
    
    return param_space



class MyModels:
    def __init__(self, model_name: str, random_state: int, cat_features: list[str] | None = None):
        """Initialize the Models class.
        
        Parameters:
            model_name (str): The name of the model to use.
            random_state (int): The random state to use for the model.
            cat_features (list[str] | None): The categorical features to use for the CatBoost ONLY.
        """
        
        # Initialize variables
        self.model_name = model_name
        self.random_state = random_state
        self.cat_features = cat_features
        # Global variables statement
        self._config = None
        self._MODEL_MAP = None

        
        # Load model configs
        _config_path = 'model_configs.yml'
        with open(_config_path, 'r') as file:
            self._config = yaml.safe_load(file)
        
        # Initialize model map
        self._init_model_map()
        
        assert self.model_name in self._MODEL_MAP, \
            f"Invalid model name: {model_name}, \
              it must be one of {list(self._MODEL_MAP.keys())}"
        
        # Initialize model object and parameters
        self.model_object = self._MODEL_MAP[self.model_name][0]
        get_method = self._MODEL_MAP[self.model_name][1]
        param_space, static_params = get_method()
        self.param_space = MappingProxyType(param_space)
        self.static_params = MappingProxyType(static_params)
    

    def _init_model_map(self):
        """Initialize model map"""
        _model_map = {
            # Classifier
            "lc": (LogisticRegression, self._get_model_config),
            "svc": (SVC, self._get_model_config),
            "knc": (KNeighborsClassifier, self._get_model_config),
            "mlpc": (MLPClassifier, self._get_model_config),
            "dtc": (DecisionTreeClassifier, self._get_model_config),
            "rfc": (RandomForestClassifier, self._get_model_config),
            "gbdtc": (GradientBoostingClassifier, self._get_model_config),
            "adac": (AdaBoostClassifier, self._get_model_config),
            "xgbc": (XGBClassifier, self._get_model_config),
            "lgbc": (LGBMClassifier, self._get_model_config),
            "catc": (CatBoostClassifier, self._get_model_config),
            # Regressor
            "lr": (LinearRegression, self._get_model_config),
            "svr": (SVR, self._get_model_config),
            "knr": (KNeighborsRegressor, self._get_model_config),
            "mlpr": (MLPRegressor, self._get_model_config),
            "dtr": (DecisionTreeRegressor, self._get_model_config),
            "rfr": (RandomForestRegressor, self._get_model_config),
            "gbdtr": (GradientBoostingRegressor, self._get_model_config),
            "adar": (AdaBoostRegressor, self._get_model_config),
            "xgbr": (XGBRegressor, self._get_model_config),
            "lgbr": (LGBMRegressor, self._get_model_config),
            "catr": (CatBoostRegressor, self._get_model_config)
        }
        self._MODEL_MAP = _model_map
    

    def _get_model_config(self):
        """Get model config from config file"""
        
        # Get parameters from config
        model_config = self._config[self.model_name]
        param_space = _convert_param_space(model_config['param_space'])
        static_params = dict(model_config['static_params'])
        
        # For all models that support random_state
        if self.model_name not in ['lr', 'svr', 'knr', 'knc']:
            static_params['random_state'] = self.random_state
            
        # For CatBoost models, add cat_features
        if self.model_name in ['catc', 'catr'] and self.cat_features is not None:
            static_params['cat_features'] = self.cat_features
            
        return param_space, static_params

