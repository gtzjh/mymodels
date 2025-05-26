import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import is_regressor


def pdp_explainer(
    model,
    explain_data: pd.DataFrame,
    results_dir: str | pathlib.Path,
    dpi: int,
    format: str,
    y_mapping_dict: dict | None = None
):
    """
    PDP (Partial Dependence Plot)
    """
    assert isinstance(explain_data, pd.DataFrame), \
        "explain_data must be a pandas DataFrame"
    assert isinstance(results_dir, str | pathlib.Path), \
        "results_dir must be a string or pathlib.Path"
    assert isinstance(dpi, int), \
        "dpi must be an integer"
    assert isinstance(format, str), \
        "format must be a string"

    feature_names = explain_data.columns.tolist()

    # Create a directory for PDP plots
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_dir = results_dir.joinpath("explanation/PDP/")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create PDP plots
    # For regression and binary classification
    if is_regressor(model) or \
        (hasattr(model, "classes_") and len(model.classes_) == 2):
        for i, n in enumerate(feature_names):
            PartialDependenceDisplay.from_estimator(
                estimator=model,
                X=explain_data,
                features=[i],
                n_jobs=-1
            )
            plt.savefig(
                results_dir.joinpath(f"{n}.{format}"),
                dpi=dpi,
                bbox_inches="tight"
            )

    # For multi-class classification
    elif len(model.classes_) > 2:
        for c in model.classes_:
            for i, n in enumerate(feature_names):
                PartialDependenceDisplay.from_estimator(
                    estimator=model,
                    X=explain_data,
                    features=[i],
                    target=c,
                    n_jobs=-1
                )
                plt.savefig(
                    results_dir.joinpath(f"class_{c}_{n}.{format}"),
                    dpi=dpi,
                    bbox_inches="tight"
                )
