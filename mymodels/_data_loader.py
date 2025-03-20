import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from utils._encoder import transform_multi_features, fit_transform_multi_features


def data_loader(file_path, y, x_list, cat_features, test_ratio, random_state) \
    -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and preprocess data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        y (str or int): The column name or index of the dependent variable.
        x_list (list): A list of column names or indices of the independent variables.
        cat_features (a list of strings or None): 
            A list of column names (str) for representing the categorical features.
        test_ratio (float): The ratio of the test set.
        random_state (int): The random state for the train_test_split.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - x_train: Training features DataFrame
            - x_test: Testing features DataFrame
            - y_train: Training target Series
            - y_test: Testing target Series
    """

    _df = pd.read_csv(file_path, encoding = "utf-8", na_values = np.nan)


    """Convert categorical features to category dtype"""
    if cat_features is not None:
        assert isinstance(cat_features, list) \
            and all([isinstance(i, str) for i in cat_features]) \
            and all([i in _df.columns for i in cat_features]), \
            "The `cat_features` must be a list of strings " \
            "and all elements must be in the column names of the inputted dataset."
        
        try:
            for cat_col in cat_features:
                _df[cat_col] = _df[cat_col].astype('category')
        except Exception as e:
            raise ValueError(f"Error converting categorical features to category dtype: \n{e}")


    """Select column using column name (if y is a string) or integer index (if y is an integer)"""
    if isinstance(y, str):
        y_data = _df.loc[:, y]
    elif isinstance(y, int):
        y_data = _df.iloc[:, y]
    else:
        raise ValueError("`y` must be either a string or " \
                         "index within the whole dataset")


    """
    Verify that x_list is a list and all elements are either strings or integers.
    If x_list contains strings, select columns using column names.
    If x_list contains integers, select columns using integer indices.
    Then verify that all non-categorical columns are numeric.
    """
    if all([isinstance(i, str) for i in x_list]):
        x_data = _df.loc[:, x_list]
    elif all([isinstance(i, int) for i in x_list]):
        x_data = _df.iloc[:, x_list]
    else:
        raise ValueError("`x_list` must be either a list of strings " \
                         "or indices within the whole dataset")
    # Verify that all columns except for the specified categorical features are numeric
    if cat_features is not None:
        x_data_numeric = x_data.loc[:, ~x_data.columns.isin(cat_features)]
    else:
        x_data_numeric = x_data
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x_data_numeric.dtypes):
        raise ValueError("There are non-numeric columns in `x_data`, " \
                         "Please check the `x_list` seriously. " \
                         "You may need to specify the `cat_features` " \
                         "if there are any categorical features.")
    

    """Clean data
    Drop rows with missing values
    """
    _data = pd.concat([x_data, y_data], axis = 1)
    _data = _data.dropna()
    _data = _data.reset_index(drop = True)
    x_data = _data.iloc[:, :-1]
    y_data = _data.iloc[:, -1]


    # x_train, x_test, y_train, y_test
    return train_test_split(
        x_data, y_data, 
        test_size = test_ratio,
        random_state = random_state,
        shuffle = True
    )


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_loader(
        file_path = "data/titanic/train.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
        cat_features = ["Sex", "Embarked"],
        test_ratio = 0.3,
        random_state = 0
    )
    # print(y_train)
    # print(x_train)

    transformed_x_train, encoder_dict, mapping_dict = fit_transform_multi_features(
        x_train.loc[:, ["Sex", "Embarked"]],
        encoder_methods = ["onehot", "binary"],
        y = y_train
    )
    x_train = x_train.drop(columns = ["Sex", "Embarked"])
    x_train = pd.concat([x_train, transformed_x_train], axis = 1)
    print(x_train.info())
    print(encoder_dict)
    print(mapping_dict)