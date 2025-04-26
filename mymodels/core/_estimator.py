from types import MappingProxyType
import yaml, importlib, logging


def _convert_param_space(param_space_config):
    """Convert YAML config to Optuna parameter space definition
    
    Args:
        param_space_config (dict): The parameter space configuration.
        
    Returns:
        dict: A parameter space dictionary that can be used with Optuna for hyperparameter optimization.

    Example:
    To assess the results of the parameter space configuration, the following example can be used:
    >>> param_space_config = {
    ...     'learning_rate': {
    ...         'type': 'float',
    ...         'values': {
    ...             'min': 0.01,
    ...             'max': 0.1,
    ...             'log': True
    ...         },
    ...         'description': 'Controls the step size at each iteration while moving toward a minimum of the loss function.'
    ...     },
    ...     'n_estimators': {
    ...         'type': 'integer',
    ...         'values': {
    ...             'min': 50,
    ...             'max': 200,
    ...             'step': 10
    ...         },
    ...         'description': 'The number of trees in the ensemble.'
    ...     },
    ...     'max_depth': {
    ...         'type': 'categorical',
    ...         'values': [3, 5, 7, 9],
    ...         'description': 'The maximum depth of the tree, controlling overfitting.'
    ...     }
    ... }
    >>> param_space = _convert_param_space(param_space_config)
    >>> print(param_space)  # Assess the generated parameter space
    """

    # Check the input
    if not isinstance(param_space_config, dict):
        raise TypeError("param_space_config must be a dictionary")
    for param_name, param_config in param_space_config.items():
        if not isinstance(param_name, str):
            raise TypeError("Parameter name must be a string")
        if not isinstance(param_config, dict):
            raise TypeError(f"Configuration for '{param_name}' must be a dictionary")
        if 'type' not in param_config:
            raise KeyError(f"Configuration for '{param_name}' must include a 'type'")
        if param_config['type'] not in ['categorical', 'integer', 'float']:
            raise ValueError(f"Unsupported parameter type")
        if param_config['type'] == 'categorical':
            if 'values' not in param_config:
                raise KeyError(f"Configuration for '{param_name}' must include 'values'")
            if not isinstance(param_config['values'], (list, tuple)):
                raise TypeError(f"Values for '{param_name}' must be a list or tuple")
        elif param_config['type'] == 'integer':
            if 'values' not in param_config:
                raise KeyError(f"Configuration for '{param_name}' must include 'values'")
            if not isinstance(param_config['values'], dict):
                raise TypeError(f"Values for '{param_name}' must be a dictionary")
            if 'min' not in param_config['values'] or 'max' not in param_config['values']:
                raise KeyError(f"Configuration for '{param_name}' must include 'min' and 'max' in values")
        elif param_config['type'] == 'float':
            if 'values' not in param_config:
                raise KeyError(f"Configuration for '{param_name}' must include 'values'")
            if not isinstance(param_config['values'], dict):
                raise TypeError(f"Values for '{param_name}' must be a dictionary")
            if 'min' not in param_config['values'] or 'max' not in param_config['values']:
                raise KeyError(f"Configuration for '{param_name}' must include 'min' and 'max' in values")

    # Convert the parameter space configuration to a parameter space dictionary
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



class MyEstimator:
    def __init__(
            self,
            cat_features: list[str] | tuple[str] | None = None, 
            model_configs_path: str = 'model_configs.yml'
        ):
        """Initialize the Models class.
        
        Args:
            cat_features (list[str] | tuple[str] | None): The categorical features to use for the CatBoost ONLY.
            model_configs_path (str): The path to the model configs file. (Default: 'model_configs.yml' in the root directory)
        
        Attributes:
            empty_model_object: An empty model object.
            param_space: The parameter space for Optuna tuning.
            static_params: The static parameters.
            shap_explainer_type: The type of SHAP explainer to use.
            optimal_model_object: The optimal model object. (After optimization)
            optimal_params: The optimal parameters. (After optimization)

        Examples:
            >>> from mymodels import MyEstimator
            >>> estimator = MyEstimator()
            >>> estimator.load(model_name='lr')
            >>> estimator.empty_model_object
            >>> estimator.param_space
            >>> estimator.static_params
            >>> estimator.shap_explainer_type
        """

        # Validate the input
        assert isinstance(cat_features, (list, tuple, type(None))), "cat_features must be a list, tuple, or None"

        # Initialize variables
        self.cat_features = cat_features

        # Load model configs
        self._config = None
        try:
            with open(model_configs_path, 'r') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model configs file not found: {model_configs_path}")

        # Model attributes
        self.model_name = None
        self.empty_model_object = None
        self.param_space = None
        self.static_params = None
        self.shap_explainer_type = None
        self.save_type = None
        self.optimal_model_object = None
        self.optimal_params = None
        
        # List all available model names, and validate the input model name
        self._VALID_MODEL_NAMES = list(self._config.keys())



    def load(self, model_name: str = None):
        """Load model

        Args:
            model_name (str): The name of the model to load.
        
        Returns:
            self: The instance of the `MyModels` class
        """

        # Validate the model name
        if model_name not in self._VALID_MODEL_NAMES:
            raise ValueError(f"Model '{model_name}' not found")
        self.model_name = model_name
        
        # Get parameters from config
        _model_config = self._config[model_name]

        # Import the emptys model class
        _import_info = _model_config['IMPORTS']
        _module_name = _import_info['module']
        _module = importlib.import_module(_module_name)
        _class_name = _import_info['class']
        self.empty_model_object = getattr(_module, _class_name)

        # The tuning parameters space, unchangeable dictionary
        self.param_space = MappingProxyType(
            _convert_param_space(
                _model_config['PARAM_SPACE']
            )
        )

        # The static parameters, unchangeable dictionary
        self.static_params = dict(_model_config['STATIC_PARAMS'])

        # The SHAP explainer type
        self.shap_explainer_type = _model_config['SHAP_EXPLAINER_TYPE']

        # The save type
        self.save_type = _model_config['SAVE_TYPE']


        # For CatBoost models, the `cat_features` parameter can be accepted.
        if self.cat_features is not None:
            if self.empty_model_object.__name__ == 'CatBoostClassifier' \
                or self.empty_model_object.__name__ == 'CatBoostRegressor':
                    self.static_params['cat_features'] = self.cat_features
            else:
                logging.warning(f"Model {model_name} does not accept cat_features parameter. "
                                 "The provided cat_features value will be ignored.")
        
        return self
