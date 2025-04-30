import numpy as np
import pandas as pd
import pytest



import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from mymodels.plotting import Plotter



def test_plot_regression():
    """
    Test whether the regression scatter plot of the regression model is drawn correctly

    1. Generate two columns of data, one for actual values and one for predicted values, all numeric types, should correctly draw a scatter plot
    2. Generate two columns of data with different lengths, should catch the error
    3. Generate two columns of data, one numeric type and one string type, should catch the error
    4. Generate two columns of data, both numeric types but both containing null values, should catch the error
    5. Generate two columns of data, both are 3-dimensional ndarray, should catch the error
    """
    # Create Plotter object
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir="./results/test_plotting_plot_evaluated_regressor"
    )
    
    # Test case 1: Normal data
    y_true = pd.Series(np.random.randn(100))
    y_pred = pd.Series(np.random.randn(100))
    
    # Should plot normally, no exceptions should be raised
    plotter.plot_regression_scatter(y_true, y_pred)
    
    # Test case 2: Different length data
    y_true_diff_len = pd.Series(np.random.randn(100))
    y_pred_diff_len = pd.Series(np.random.randn(50))
    
    # Should raise an exception
    with pytest.raises(AssertionError) as excinfo:
        plotter.plot_regression_scatter(y_true_diff_len, y_pred_diff_len)
    assert "The length of _y and _y_pred must be the same" in str(excinfo.value)
    
    # Test case 3: Type mismatch
    y_true_numeric = pd.Series(np.random.randn(100))
    y_pred_string = pd.Series(['a'] * 100)
    
    # Should raise an exception
    with pytest.raises(AssertionError) as excinfo:
        plotter.plot_regression_scatter(y_true_numeric, y_pred_string)
    assert "All values in _y_pred must be numeric" in str(excinfo.value)
    
    # Test case 4: Contains null values
    y_true_with_nan = pd.Series([1, 2, 3, np.nan, 5])
    y_pred_with_nan = pd.Series([1, 2, np.nan, 4, 5])
    
    # Should raise an exception (since NaN values will affect min/max calculation or polyfit)
    with pytest.raises(Exception):
        plotter.plot_regression_scatter(y_true_with_nan, y_pred_with_nan)
    
    # Test case 5: 3D arrays
    y_true_3d = np.random.randn(10, 10, 10)
    y_pred_3d = np.random.randn(10, 10, 10)
    
    # Should raise an exception
    with pytest.raises(AssertionError) as excinfo:
        plotter.plot_regression_scatter(y_true_3d, y_pred_3d)
    assert "Input _y must be a 1D" in str(excinfo.value)
    
    return None



if __name__ == "__main__":
    plotter = Plotter(
        show = False,
        plot_format = "png",
        plot_dpi = 300,
        results_dir = "./results/test_plotting_plot_evaluated_regressor"
    )
    
    test_plot_regression()
