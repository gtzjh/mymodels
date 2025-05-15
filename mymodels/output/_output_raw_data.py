from pathlib import Path
import pandas as pd


def _output_raw_data(
        results_dir: str | Path,
        y_test: pd.Series,
        y_test_pred: pd.Series,
        y_train: pd.Series,
        y_train_pred: pd.Series
    ):

    """Save the raw data to a CSV file.
    """
    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    assert isinstance(y_test, pd.Series), \
        "y_test must be a pandas Series"
    assert isinstance(y_test_pred, pd.Series), \
        "y_test_pred must be a pandas Series"
    assert isinstance(y_train, pd.Series), \
        "y_train must be a pandas Series"
    assert isinstance(y_train_pred, pd.Series), \
        "y_train_pred must be a pandas Series"

    # Assemble the results
    test_results = pd.DataFrame(data={"y_test": y_test.to_numpy(),
                                      "y_test_pred": y_test_pred.to_numpy()},
                                index = y_test.index)
    train_results = pd.DataFrame(data={"y_train": y_train.to_numpy(),
                                       "y_train_pred": y_train_pred.to_numpy()},
                                 index = y_train.index)

    test_results.to_csv(results_dir.joinpath("test_results.csv"), encoding="utf-8", index = True)
    train_results.to_csv(results_dir.joinpath("train_results.csv"), encoding="utf-8", index = True)
