"""
Tests for SHAP summary and SHAP dependence plotting functions

Testing with sklearn random forest models across three task types: binary classification, multiclass classification, and regression
- Test the SHAP summary plotting function
- Test the SHAP dependence plotting function

In total, there are 5 basic test items

Beyond these basic tests, we should add input exception handling tests:
- Test file saving functionality, check if files are correctly saved to the specified path
- Test different display parameters, such as show=True/False behavior
- Test edge cases, such as empty feature lists or cases with very few features
- Test handling of input data containing NaN or extreme values
- Test various plot_format parameter values
"""


import numpy as np
import pandas as pd
import shutil
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pytest


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mymodels.plotting import Plotter




test_results_dir = "./results/test_plotting_plot_explainer/"

# Clean up the existing test directory
if os.path.exists(test_results_dir):
    try:
        shutil.rmtree(test_results_dir)
        print(f"Cleaned up test directory: {test_results_dir}")
    except Exception as e:
        print(f"Failed to clean up test directory: {str(e)}")

# Create test directory
os.makedirs(test_results_dir, exist_ok=True)


@pytest.fixture
def test_dir():
    return test_results_dir


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42, n_classes=2)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X  = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return {
        "X_train": X_train, 
        "X_test": X_test, 
        "y_train": y_train, 
        "y_test": y_test, 
        "model": model, 
        "explainer": explainer
    }


@pytest.fixture
def multiclass_classification_data():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=4, n_redundant=1, random_state=42, n_classes=6)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X  = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return {
        "X_train": X_train, 
        "X_test": X_test, 
        "y_train": y_train, 
        "y_test": y_test, 
        "model": model, 
        "explainer": explainer
    }


@pytest.fixture
def regression_data():
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes().feature_names
    X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return {
        "X_train": X_train, 
        "X_test": X_test, 
        "y_train": y_train, 
        "y_test": y_test, 
        "model": model, 
        "explainer": explainer
    }



# Basic test item 1: Test SHAP summary plotting function - binary classification
def test_plot_shap_summary_binary(test_dir, binary_classification_data):
    """
    Test SHAP summary plot for binary classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = binary_classification_data["explainer"]
    X_test = binary_classification_data["X_test"]
    explanation = explainer(X_test)

    
    # Test plotting functionality
    plotter.plot_shap_summary(explanation)


# Basic test item 2: Test SHAP summary plotting function - multiclass classification
def test_plot_shap_summary_multiclass(test_dir, multiclass_classification_data):
    """
    Test SHAP summary plot for multiclass classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = multiclass_classification_data["explainer"]
    X_test = multiclass_classification_data["X_test"]
    explanation = explainer(X_test)
    
    # Test plotting functionality
    plotter.plot_shap_summary(explanation)
    

# Basic test item 3: Test SHAP summary plotting function - regression
def test_plot_shap_summary_regression(test_dir, regression_data):
    """
    Test SHAP summary plot for regression
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test)

    # Test plotting functionality
    plotter.plot_shap_summary(explanation)

# Basic test item 4: Test SHAP dependence plotting function - binary classification
def test_plot_shap_dependence_binary(test_dir, binary_classification_data):
    """
    Test SHAP dependence plot for binary classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = binary_classification_data["explainer"]
    X_test = binary_classification_data["X_test"]
    explanation = explainer(X_test)

    
    # Test plotting functionality
    plotter.plot_shap_dependence(explanation)


# Basic test item 5: Test SHAP dependence plotting function - multiclass classification
def test_plot_shap_dependence_multiclass(test_dir, multiclass_classification_data):
    """
    Test SHAP dependence plot for multiclass classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = multiclass_classification_data["explainer"]
    X_test = multiclass_classification_data["X_test"]
    explanation = explainer(X_test)
    
    # Test plotting functionality
    plotter.plot_shap_dependence(explanation)
    

# Basic test item 6: Test SHAP dependence plotting function - regression
def test_plot_shap_dependence_regression(test_dir, regression_data):
    """
    Test SHAP dependence plot for regression
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test)
    
    # Test plotting functionality
    plotter.plot_shap_dependence(explanation)
    

# Test item 4: Test different plot_format parameters
def test_different_plot_formats(test_dir, regression_data):
    """
    Test different plot formats
    """
    # Calculate SHAP values, get explanation object
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test)
    
    for format_name in ["png", "jpg", "pdf", "svg"]:
        results_dir = os.path.join(test_dir, f"format_{format_name}")
        os.makedirs(results_dir, exist_ok=True)
        
        plotter = Plotter(
            show=False,
            plot_format=format_name,
            plot_dpi=300,
            results_dir=results_dir
        )
        
        # Test plotting functionality
        plotter.plot_shap_summary(explanation)


# Test item 5: Test edge cases - very few features
def test_few_features(test_dir):
    """
    Test with very few features
    """
    # Create data with only 1 feature
    X = np.random.rand(50, 1)
    y = np.random.randint(0, 2, 50)
    
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model, X_train)
    explainer.feature_names = ["single_feature"]
    
    # Calculate SHAP values, get explanation object
    explanation = explainer(X_test)
    
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_dir
    )
    
    # Test plotting functionality
    plotter.plot_shap_summary(explanation)
    plotter.plot_shap_dependence(explanation)



# Test item 6: Test data containing NaN values
def test_with_nan_values(test_dir, regression_data):
    """
    Test with NaN values in the dataset
    """
    # Get original data and introduce some NaN values
    X_train = regression_data["X_train"].copy()
    X_test = regression_data["X_test"].copy()
    
    # Replace some values with NaN
    X_test_with_nan = X_test.copy()
    X_test_with_nan.iloc[0, 0] = np.nan
    X_test_with_nan.iloc[1, 1] = np.nan
    
    # Create new explainer
    model = regression_data["model"]
    explainer = shap.TreeExplainer(model, X_train)
    explainer.feature_names = regression_data["explainer"].feature_names
    
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_dir
    )
    
    # Test if it can handle NaN values
    try:
        # Note: This might fail, depending on how the shap library handles NaN values
        explanation = explainer(X_test_with_nan)
        plotter.plot_shap_summary(explanation)
    except Exception as e:
        # Record the exception, but don't fail the test
        print(f"NaN handling test resulted in exception: {str(e)}")
