import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def data_loader(
        input_data,
        y,
        x_list,
        test_ratio,
        random_state,
        stratify
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    """Load and preprocess data from a CSV file.
    
    Args:
        input_data (pd.DataFrame): The input data.
        y (str or int): The column name or index of the dependent variable.
        x_list (list or tuple): A list of column names or indices of the independent variables.
        index_col (str or int or list or tuple or None): The column name or index of the index column.
        test_ratio (float): The ratio of the test set.
        random_state (int): The random state for the train_test_split.
        stratify (bool): Whether to stratify the data.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - x_train: Training features DataFrame
            - x_test: Testing features DataFrame
            - y_train: Training target Series
            - y_test: Testing target Series
    """
    
    assert isinstance(input_data, pd.DataFrame), \
        "input_data must be a pandas.DataFrame"

    assert (test_ratio > 0 and test_ratio <= 1) and isinstance(test_ratio, float), \
        "test_ratio must be between (0, 1]"


    # Select column using column name (if y is a string) or integer index (if y is an integer)
    if isinstance(y, str):
        y_data = input_data.loc[:, y]
    elif isinstance(y, int):
        y_data = input_data.iloc[:, y]
    else:
        raise ValueError("`y` must be either a string or index in the input dataframe")

    # Verify x_list contains valid column identifiers and select data
    # All elements must be either strings (column names) or integers (column indices)
    x_list = list(x_list)
    if all([isinstance(i, str) for i in x_list]):
        x_data = input_data.loc[:, x_list]
    elif all([isinstance(i, int) for i in x_list]):
        x_data = input_data.iloc[:, x_list]
    else:
        raise ValueError("`x_list` must be either a list or tuple of strings or indices in the input dataframe")


    # Transform non-numeric target to label encoding
    # Save the mapping dict as wll.
    y_mapping_dict = None
    if pd.api.types.is_numeric_dtype(y_data.dtype) != True:
        label_encoder = LabelEncoder()
        y_data = pd.Series(label_encoder.fit_transform(y_data), index = y_data.index)
        
        # Output the mapping from original categories to encoded integers
        y_mapping_dict = {original: encoded for original, encoded in 
                  zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
        print("Label Encoding Mapping:")
        for original, encoded in y_mapping_dict.items():
            print(f"  {original} -> {encoded}")

    # print(x_data.head(30))

    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test, y_mapping_dict (optional)
    return (
                train_test_split(
                    x_data, y_data, 
                    test_size = test_ratio,
                    random_state = random_state,
                    shuffle = True,  # Default value in `train_test_split()` is True
                    stratify = y_data if stratify else None
                ),
                y_mapping_dict
            )
