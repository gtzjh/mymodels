from types import MappingProxyType
import yaml, importlib, logging


def _convert_param_space(param_space_config):
    """Convert YAML config to Optuna parameter space definition
    
    Args:
        param_space_config (dict): The parameter space configuration.
        
    Returns:
        dict: A parameter space dictionary that can be used with Optuna for hyperparameter optimization.

    Example:
    To assess the results of the parameter space configuration, you can use the following example:
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
            raise ValueError(f"Invalid type for parameter '{param_name}'")
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
            random_state: int = 0, 
            cat_features: list[str] | tuple[str] | None = None, 
            model_configs_path: str = 'model_configs.yml'
        ):
        """Initialize the Models class.
        
        Args:
            random_state (int): The random state to use for the model. (Default: 0)
            cat_features (list[str] | tuple[str] | None): The categorical features to use for the CatBoost ONLY.
            model_configs_path (str): The path to the model configs file. (Default: 'model_configs.yml' in the root directory)
        
        Attributes:
            empty_model_object: An empty model object.
            param_space: The parameter space.
            static_params: The static parameters.
            shap_explainer_type: The type of SHAP explainer to use.
            optimal_model_object: The optimal model object. (After optimization)
            optimal_params: The optimal parameters. (After optimization)

        Examples:
            >>> from mymodels import MyModels
            >>> model = MyModels(random_state=42)
            >>> model.load(model_name='lr')
            >>> model.empty_model_object
            >>> model.param_space
            >>> model.static_params
            >>> model.shap_explainer_type
            >>> model.optimal_model_object
            >>> model.optimal_params
        """

        # Initialize variables
        self.random_state = random_state
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
        
        Returns:
            self: The instance of the `MyModels` class
        """

        # Validate the model name
        assert model_name in self._VALID_MODEL_NAMES, \
            f"Invalid model name: {model_name}, it must be one of {self._VALID_MODEL_NAMES}"
        self.model_name = model_name
        
        # Get parameters from config
        model_config = self._config[model_name]

        # Import the emptys model class
        import_info = model_config['IMPORTS']
        module_name = import_info['module']
        module = importlib.import_module(module_name)
        class_name = import_info['class']
        self.empty_model_object = getattr(module, class_name)

        # The tuning parameters space, unchangeable dictionary
        self.param_space = MappingProxyType(
            _convert_param_space(
                model_config['PARAM_SPACE']
            )
        )

        # The static parameters, unchangeable dictionary
        self.static_params = dict(model_config['STATIC_PARAMS'])

        # The SHAP explainer type
        self.shap_explainer_type = model_config['SHAP_EXPLAINER_TYPE']

        # The save type
        self.save_type = model_config['SAVE_TYPE']


        # 对于catboost模型，可以接受cat_features参数
        # 对于其他模型，cat_features参数会被忽略。若此时用户还是提供了cat_features参数，会提供一个warning，告诉用户这里的cat_features参数将不会生效
        # 对于可以接受random_state参数的模型，会自动将random_state参数设置为self.random_state
        # 如果选择的模型不能接受random_state参数，但是用户还是提供了random_state参数，会提供一个warning，告诉用户这里的random_state参数将不会生效
        # 反之，如果选择的模型能够接受random_state参数，但是用户没有提供random_state参数，则会提供以后警告，
        # 告诉用户这里没有提供random_state参数，可能会影响模型的可复现性。
        
        # Create an instance of the model to check the available parameters inside
        _model_object = self.empty_model_object(
            **{
                "cat_features": self.cat_features,
                "random_state": self.random_state
            }
        )
        
        # _acceptable_params = _model_object.get_params()
        # print(_acceptable_params)
 
        from sklearn.datasets import load_iris
        iris = load_iris()
        self.X_train, self.y_train = iris.data, iris.target
        _model_object.fit(self.X_train, self.y_train)


        """
        # Check for random_state parameter compatibility
        if 'random_state' in _acceptable_params:
            if hasattr(self, 'random_state'):
                self.static_params['random_state'] = self.random_state
            else:
                logging.warning(f"Model {model_name} accepts random_state parameter, but none was provided. "
                                 "This may affect model reproducibility.")
        elif hasattr(self, 'random_state') and self.random_state is not None:
            logging.warning(f"Model {model_name} does not accept random_state parameter. "
                             "The provided random_state value will be ignored.")

        # Check for cat_features parameter compatibility
        if 'cat_features' in _acceptable_params:
            if hasattr(self, 'cat_features') and self.cat_features is not None:
                self.static_params['cat_features'] = self.cat_features
        elif hasattr(self, 'cat_features') and self.cat_features is not None:
            logging.warning(f"Model {model_name} does not accept cat_features parameter. "
                             "The provided cat_features value will be ignored.")
        """
        
        return self
    


if __name__ == "__main__":
    model = MyEstimator(random_state=0, cat_features=['cat_feature', "xxx"]).load(model_name='svc')

    for attr in dir(model):
        if not attr.startswith('_'):
            # print(f"{attr}: {getattr(model, attr)}")
            pass
