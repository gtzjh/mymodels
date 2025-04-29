import pandas as pd
from sklearn.model_selection import train_test_split


def _label_y(y_data, verbose = False):
    """Encode non-numeric target variable to integers.
    
    Args:
        y_data (pd.Series): The target variable series to encode.
        verbose (bool, optional): Whether to print encoding mapping information. Defaults to False.
        
    Returns:
        tuple: A tuple containing (encoded_data, mapping_dict) where mapping_dict maps original categories to encoded integers.
        
    Raises:
        AssertionError: If y_data is already numeric.
        
    Examples:
        >>> import pandas as pd
        >>> series = pd.Series(['cat', 'dog', 'bird', 'cat'])
        >>> encoded_data, mapping = _label_y(series)
        >>> sorted(mapping.items())
        [('bird', 0), ('cat', 1), ('dog', 2)]
    """
    from sklearn.preprocessing import LabelEncoder

    # If y_data is already float, return original data and empty dictionary
    if pd.api.types.is_float_dtype(y_data.dtype):
        return y_data, None
    
    
    # Handle categorical/string y_data
    label_encoder = LabelEncoder()
    encoded_y_data = pd.Series(label_encoder.fit_transform(y_data), index=y_data.index)
    
    # Create mapping dictionary from original categories to encoded values
    _y_mapping_dict_keys = {
        original: encoded for original, encoded in \
            zip(label_encoder.classes_, range(len(label_encoder.classes_)))
    }
    _keys = _y_mapping_dict_keys.keys()
    y_mapping_dict = dict((_k, _y_mapping_dict_keys[_k]) for _k in _keys)

    if verbose:
        print("Label Encoding Mapping:\n")
        for _original, _encoded in y_mapping_dict.items():
            print(f"  {_original} -> {_encoded}")

    return encoded_y_data, y_mapping_dict



class MyDataLoader:
    """Load and split data for model training and testing.
    
    Args:
        input_data (pd.DataFrame): The input dataset as a pandas DataFrame.
        y (str or int): The column name or index of the target/dependent variable.
        x_list (list or tuple): List of column names or indices of the feature/independent variables.
        test_ratio (float, optional): Proportion of data to use for testing. Defaults to 0.3.
        random_state (int, optional): Seed for random number generation in train_test_split. Defaults to 0.
        stratify (bool, optional): Whether to maintain class distribution in train/test splits, 
            recommended for imbalanced datasets. Defaults to False.
    
    Attributes:
        x_train (pd.DataFrame): Training features DataFrame.
        x_test (pd.DataFrame): Testing features DataFrame.
        y_train (pd.Series): Training target Series.
        y_test (pd.Series): Testing target Series.
        y_mapping_dict (dict): Dictionary mapping original categories to encoded integers (if applicable).
    
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from mymodels._data_loader import MyDataLoader
        >>> 
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     "ID": [1, 2, 3, 4, 5],
        ...     "x1": [1.1, 2.2, 3.3, 4.4, 5.5],
        ...     "x2": [10, 20, 30, 40, 50],
        ...     "x3": ["a", "b", "c", "a", "b"],
        ...     "y": ["cat1", "cat2", "cat1", "cat2", "cat1"]
        ... }).set_index("ID")
        >>> 
        >>> # Create an instance of MyDataLoader
        >>> loader = MyDataLoader(
        ...     input_data=data,
        ...     y="y",
        ...     x_list=["x1", "x2", "x3"],
        ...     test_ratio=0.4,
        ...     random_state=42,
        ...     stratify=True
        ... )
        >>> 
        >>> # Load and split the data
        >>> dataset = loader.load()
        >>> print(f"Training set shape: {dataset.x_train.shape}")
        Training set shape: (3, 3)
        >>> print(f"Testing set shape: {dataset.x_test.shape}")
        Testing set shape: (2, 3)
    """
    
    def __init__(
            self, 
            input_data,
            y: str | int,
            x_list: list[str | int] | tuple[str | int], 
            test_ratio: float = 0.3,
            random_state: int = 0,
            stratify: bool = False
        ):
        # Check input types first
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("input_data must be a pandas DataFrame")
        
        self.input_data = input_data.copy(deep = True)
        self.y = y
        self.x_list = x_list
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.stratify = stratify

        # Initialize attributes
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_mapping_dict = None

        # Validate input parameters
        self._check_input()



    def _check_input(self):
        """Validate all input parameters.
        
        Ensures that all provided parameters meet the required specifications.
        
        Raises:
            TypeError: If input_data is not a DataFrame, random_state is not an integer, or stratify is not a boolean.
            ValueError: If test_ratio is not between 0 and 1, or x_list is empty.
            KeyError: If y column or x columns don't exist in input_data.
        
        Returns:
            None
        """
        if not isinstance(self.y, (str, int)):
            raise TypeError("y must be either a string or index in the input dataframe")
        
        if isinstance(self.y, str):
            if self.y not in self.input_data.columns:
                raise KeyError(f"Column '{self.y}' not found in input_data")
        elif isinstance(self.y, int):
            if self.y >= len(self.input_data.columns):
                raise ValueError("y index must be less than the number of columns in input_data")
    
        if not isinstance(self.x_list, (list, tuple)):
            raise TypeError("x_list must be a list or tuple")
        
        if len(self.x_list) == 0:
            raise ValueError("x_list cannot be empty")
            
        if not all(isinstance(i, (str, int)) for i in self.x_list):
            raise TypeError("x_list must contain only strings or integers")
        
        if all(isinstance(i, str) for i in self.x_list):
            missing_columns = [col for col in self.x_list if col not in self.input_data.columns]
            if missing_columns:
                raise KeyError(f"Columns {missing_columns} not found in input_data")

        if all(isinstance(i, int) for i in self.x_list):
            if not all(i >= 0 for i in self.x_list):
                raise ValueError("All indices in x_list must be greater than or equal to 0")
            if not all(i < len(self.input_data.columns) for i in self.x_list):
                raise ValueError("All indices in x_list must be less than the number of columns in input_data")

        if not isinstance(self.test_ratio, float) or self.test_ratio <= 0 or self.test_ratio > 1:
            raise ValueError("test_ratio must be between (0, 1]")
        
        if not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer")
            
        if not isinstance(self.stratify, bool):
            raise TypeError("stratify must be a boolean")
        
        return None



    def load(self):
        """Process data and split into training and testing sets.
        
        Returns:
            MyDataLoader: The current instance with populated training and testing data attributes.
            
        Raises:
            ValueError: If missing values are found in target or feature variables.
            
        Examples:
            >>> import pandas as pd
            >>> from mymodels._data_loader import MyDataLoader
            >>> data = pd.DataFrame({
            ...     "ID": [1, 2, 3, 4],
            ...     "feature": [10, 20, 30, 40],
            ...     "target": [0, 1, 0, 1]
            ... }).set_index("ID")
            >>> loader = MyDataLoader(
            ...     input_data=data,
            ...     y="target",
            ...     x_list=["feature"],
            ...     random_state=42
            ... )
            >>> result = loader.load()
            >>> result.x_train.shape[0] + result.x_test.shape[0] == data.shape[0]
            True
        """
        # Extract the target variable
        if isinstance(self.y, str):
            _y_data = self.input_data.loc[:, self.y]
        elif isinstance(self.y, int):
            _y_data = self.input_data.iloc[:, self.y]

        # Check for missing values in target
        if _y_data.isna().any():
            raise ValueError("Missing values are not allowed in the target variable")
            
        # Extract the feature variables
        if all([isinstance(i, str) for i in list(self.x_list)]):
            _x_data = self.input_data.loc[:, list(self.x_list)]
        elif all([isinstance(i, int) for i in list(self.x_list)]):
            _x_data = self.input_data.iloc[:, list(self.x_list)]

        # Encode non-numeric target variables and get mapping dictionary
        encoded_y_data, self.y_mapping_dict = _label_y(_y_data, verbose=False)
        
        # Split data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            _x_data,
            encoded_y_data,
            test_size = self.test_ratio,
            random_state = self.random_state,
            shuffle = True,
            stratify = encoded_y_data if self.stratify else None
        )

        return self
