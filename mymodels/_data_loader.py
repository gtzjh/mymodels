import numpy as np
import pandas as pd
import pathlib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def data_loader(
        file_path,
        y,
        x_list,
        index_col,
        test_ratio,
        random_state
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and preprocess data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        y (str or int): The column name or index of the dependent variable.
        x_list (list or tuple): A list of column names or indices of the independent variables.
        index_col (str or int or list or tuple or None): The column name or index of the index column.
        test_ratio (float): The ratio of the test set.
        random_state (int): The random state for the train_test_split.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - x_train: Training features DataFrame
            - x_test: Testing features DataFrame
            - y_train: Training target Series
            - y_test: Testing target Series
    """
    # Check if index_col is provided
    if index_col is None:
        logging.warning("index_col is unpresented. It's STRONGLY RECOMMENDED to set the index column if you want to output the raw data and the shap values.")

    assert isinstance(file_path, str) or isinstance(file_path, pathlib.Path), \
        "file_path must be a string or pathlib.Path"
    _df = pd.read_csv(file_path, encoding = "utf-8", na_values = np.nan, index_col = index_col)
    # print(_df.head(30))

    assert (test_ratio > 0 and test_ratio <= 1) and isinstance(test_ratio, float), \
        "test_ratio must be between (0, 1]"


    # Select column using column name (if y is a string) or integer index (if y is an integer)
    if isinstance(y, str):
        y_data = _df.loc[:, y]
    elif isinstance(y, int):
        y_data = _df.iloc[:, y]
    else:
        raise ValueError("`y` must be either a string or " \
                         "index within the whole dataset")

    # Verify x_list contains valid column identifiers and select data
    # All elements must be either strings (column names) or integers (column indices)
    x_list = list(x_list)
    if all([isinstance(i, str) for i in x_list]):
        x_data = _df.loc[:, x_list]
    elif all([isinstance(i, int) for i in x_list]):
        x_data = _df.iloc[:, x_list]
    else:
        raise ValueError("`x_list` must be either a list or tuple of strings " \
                         "or indices within the whole dataset")

    # print(x_data.head(30))
    # print(y_data.head(30))


    # Transform non-numeric target to label encoding
    if pd.api.types.is_numeric_dtype(y_data.dtype) != True:
        label_encoder = LabelEncoder()
        y_data = pd.Series(label_encoder.fit_transform(y_data), index=y_data.index)
        
        # Output the mapping from original categories to encoded integers
        mapping = {original: encoded for original, encoded in 
                  zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
        print("Label Encoding Mapping:")
        for original, encoded in mapping.items():
            print(f"  {original} -> {encoded}")

    # print(x_data.head(30))

    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test
    return train_test_split(
        x_data, y_data, 
        test_size = test_ratio,
        random_state = random_state,
        shuffle = True
    )