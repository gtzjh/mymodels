from pathlib import Path

import numpy as np
import pandas as pd


def _output_raw_data(
        results_dir: str | Path,
        y_test: pd.Series | pd.DataFrame,
        y_test_pred: pd.Series | pd.DataFrame | np.ndarray,
        y_train: pd.Series | pd.DataFrame,
        y_train_pred: pd.Series | pd.DataFrame | np.ndarray
    ):

    """Save the raw data to a CSV file.
    """
    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    assert isinstance(y_test, (pd.Series, pd.DataFrame)), \
        "y_test must be a pandas Series or DataFrame"
    assert isinstance(y_test_pred, (pd.Series, pd.DataFrame, np.ndarray)), \
        "y_test_pred must be a pandas Series or DataFrame or numpy array"
    assert isinstance(y_train, (pd.Series, pd.DataFrame)), \
        "y_train must be a pandas Series or DataFrame"
    assert isinstance(y_train_pred, (pd.Series, pd.DataFrame, np.ndarray)), \
        "y_train_pred must be a pandas Series or DataFrame or numpy array"


    y_test_pred = y_test_pred
    y_train_pred = y_train_pred
    
    # Flatten predictions if they are 2D with second dimension of 1
    if len(y_test_pred.shape) > 1 and y_test_pred.shape[1] == 1:
        y_test_pred = y_test_pred.flatten()
    if len(y_train_pred.shape) > 1 and y_train_pred.shape[1] == 1:
        y_train_pred = y_train_pred.flatten()
    
    test_results = pd.DataFrame(data={"y_test": y_test,
                                      "y_test_pred": y_test_pred},
                                index = y_test.index)
    train_results = pd.DataFrame(data={"y_train": y_train,
                                       "y_train_pred": y_train_pred},
                                 index = y_train.index)
    test_results.to_csv(results_dir.joinpath("test_results.csv"), index = True)
    train_results.to_csv(results_dir.joinpath("train_results.csv"), index = True)

    return None

