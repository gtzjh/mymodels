from types import MappingProxyType
import yaml, importlib


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



class MyModels:
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
        self.optimal_model_object = None
        self.optimal_params = None
        
        # List all available model names, and validate the input model name
        self._VALID_MODEL_NAMES = list(self._config.keys())



    def load(self, model_name: str = None):
        """Load model
        
        Returns:
            self: The instance of the `MyModels` class
        """

        assert model_name in self._VALID_MODEL_NAMES, \
            f"Invalid model name: {model_name}, \
              it must be one of {self._VALID_MODEL_NAMES}"
        
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
        self.static_params = MappingProxyType(dict(model_config['STATIC_PARAMS']))

        # The SHAP explainer type
        self.shap_explainer_type = model_config['SHAP_EXPLAINER_TYPE']

        # The save type
        self.save_type = model_config['SAVE_TYPE']

        """
        # For all models that support random_state
        if self.model_name not in ['lr', 'svr', 'knr', 'knc']:
            static_params['random_state'] = self.random_state
            
        # For CatBoost models, add cat_features
        if self.model_name in ['catc', 'catr'] and self.cat_features is not None:
            static_params['cat_features'] = self.cat_features
        """

        return self
    


if __name__ == "__main__":
    model = MyModels(random_state=42).load(model_name='rfr')
    
    print(model.empty_model_object)
    print(model.param_space)
    print(model.static_params)
    print(model.shap_explainer_type)
    print(model.save_type)

