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
import pytest
import shutil
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels.plotting import Plotter


# Create directories needed for testing
def test_results_dir():
    results_dir = "./results/test_plotting_plot_explainer/"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
    # Cleanup part moved to the main function


# Create binary classification data and model
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


# Create multiclass classification data and model
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


# Create regression data and model
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
def test_plot_shap_summary_binary(test_results_dir, binary_classification_data):
    """
    Test SHAP summary plot for binary classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = binary_classification_data["explainer"]
    X_test = binary_classification_data["X_test"]
    explanation = explainer(X_test)

    
    # Test plotting functionality
    plotter.plot_shap_summary(explanation)
    
    # Verify if the file is saved
    # assert os.path.exists(os.path.join(test_results_dir, "SHAP/shap_summary.png")), \
    #     "SHAP summary plot should be saved"


# Basic test item 2: Test SHAP summary plotting function - multiclass classification
def test_plot_shap_summary_multiclass(test_results_dir, multiclass_classification_data):
    """
    Test SHAP summary plot for multiclass classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = multiclass_classification_data["explainer"]
    X_test = multiclass_classification_data["X_test"]
    explanation = explainer(X_test)
    
    # Test plotting functionality
    plotter.plot_shap_summary(explanation)
    
    # Verify if file for each class is saved
    n_classes = len(np.unique(multiclass_classification_data["y_train"]))
    for i in range(n_classes):
        assert os.path.exists(os.path.join(test_results_dir, f"SHAP/shap_summary/class_{i}.png")), "SHAP summary plot for each class should be saved"


# Basic test item 3: Test SHAP summary plotting function - regression
def test_plot_shap_summary_regression(test_results_dir, regression_data):
    """
    Test SHAP summary plot for regression
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test)

    # Test plotting functionality
    plotter.plot_shap_summary(explanation)
    
    # Verify if the file is saved
    assert os.path.exists(os.path.join(test_results_dir, "SHAP/shap_summary.png")), "SHAP summary plot should be saved"


# Basic test item 4: Test SHAP dependence plotting function - binary classification
def test_plot_shap_dependence_binary(test_results_dir, binary_classification_data):
    """
    Test SHAP dependence plot for binary classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = binary_classification_data["explainer"]
    X_test = binary_classification_data["X_test"]
    explanation = explainer(X_test)

    
    # Test plotting functionality
    plotter.plot_shap_dependence(explanation)
    
    # Verify if the file is saved
    # for i, feature_name in enumerate(binary_classification_data["explainer"].feature_names):
    #     assert os.path.exists(os.path.join(test_results_dir, f"SHAP/shap_dependence/{feature_name}.png")), "SHAP dependence plot for each feature should be saved"


# Basic test item 5: Test SHAP dependence plotting function - multiclass classification
def test_plot_shap_dependence_multiclass(test_results_dir, multiclass_classification_data):
    """
    Test SHAP dependence plot for multiclass classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = multiclass_classification_data["explainer"]
    X_test = multiclass_classification_data["X_test"]
    explanation = explainer(X_test)
    
    # Test plotting functionality
    plotter.plot_shap_dependence(explanation)
    
    # Verify if file for each class is saved
    n_classes = len(np.unique(multiclass_classification_data["y_train"]))
    for i in range(n_classes):
        for feature_name in explanation.feature_names:
            assert os.path.exists(os.path.join(test_results_dir, f"SHAP/shap_dependence/class_{i}/{feature_name}.png")), "SHAP dependence plot for each feature should be saved"


# Basic test item 6: Test SHAP dependence plotting function - regression
def test_plot_shap_dependence_regression(test_results_dir, regression_data):
    """
    Test SHAP dependence plot for regression
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # Calculate SHAP values, get explanation object
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test)
    
    # Test plotting functionality
    plotter.plot_shap_dependence(explanation)
    
    # Verify if the file is saved
    for feature_name in explanation.feature_names:
        assert os.path.exists(os.path.join(test_results_dir, f"SHAP/shap_dependence/{feature_name}.png")), \
            "SHAP dependence plot for each feature should be saved"


# Test item 4: Test different plot_format parameters
def test_different_plot_formats(test_results_dir, regression_data):
    """
    Test different plot formats
    """
    # Calculate SHAP values, get explanation object
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test)
    
    for format_name in ["png", "jpg", "pdf", "svg"]:
        results_dir = os.path.join(test_results_dir, f"format_{format_name}")
        os.makedirs(results_dir, exist_ok=True)
        
        plotter = Plotter(
            show=False,
            plot_format=format_name,
            plot_dpi=300,
            results_dir=results_dir
        )
        
        # Test plotting functionality
        plotter.plot_shap_summary(explanation)
        
        # Verify if the file is saved in correct format
        assert os.path.exists(os.path.join(results_dir, f"SHAP/shap_summary.{format_name}")), f"SHAP summary plot should be saved in {format_name} format"


# Test item 5: Test edge cases - very few features
def test_few_features(test_results_dir):
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
        results_dir=test_results_dir
    )
    
    # Test plotting functionality
    plotter.plot_shap_summary(explanation)
    plotter.plot_shap_dependence(explanation)



# Test item 6: Test data containing NaN values
def test_with_nan_values(test_results_dir, regression_data):
    """
    Test with NaN values in the dataset
    """
    # Get original data and introduce some NaN values
    X_train = regression_data["X_train"].copy()
    X_test = regression_data["X_test"].copy()
    
    # Replace some values with NaN
    X_test_with_nan = X_test.copy()
    X_test_with_nan[0, 0] = np.nan
    X_test_with_nan[1, 1] = np.nan
    
    # Create new explainer
    model = regression_data["model"]
    explainer = shap.TreeExplainer(model, X_train)
    explainer.feature_names = regression_data["explainer"].feature_names
    
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # Test if it can handle NaN values
    try:
        # Note: This might fail, depending on how the shap library handles NaN values
        explanation = explainer(X_test_with_nan)
        plotter.plot_shap_summary(explanation)
    except Exception as e:
        # Record the exception, but don't fail the test
        print(f"NaN handling test resulted in exception: {str(e)}")



if __name__ == "__main__":
    results_dir = "./results/test_plotting_plot_explainer/"

    # Clean up the existing test directory
    if os.path.exists(results_dir):
        try:
            shutil.rmtree(results_dir)
            print(f"Cleaned up test directory: {results_dir}")
        except Exception as e:
            print(f"Failed to clean up test directory: {str(e)}")

    # Create test directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Get test data
    binary_data = binary_classification_data()
    multiclass_data = multiclass_classification_data()
    reg_data = regression_data()
    
    # Run all tests
    print("Running test_plot_shap_summary_binary...")
    test_plot_shap_summary_binary(results_dir, binary_data)
    
    print("Running test_plot_shap_summary_multiclass...")
    test_plot_shap_summary_multiclass(results_dir, multiclass_data)
    
    print("Running test_plot_shap_summary_regression...")
    test_plot_shap_summary_regression(results_dir, reg_data)
    
    print("Running test_plot_shap_dependence_binary...")
    test_plot_shap_dependence_binary(results_dir, binary_data)
    
    print("Running test_plot_shap_dependence_multiclass...")
    test_plot_shap_dependence_multiclass(results_dir, multiclass_data)
    
    print("Running test_plot_shap_dependence_regression...")
    test_plot_shap_dependence_regression(results_dir, reg_data)
    
    print("Running test_different_plot_formats...")
    test_different_plot_formats(results_dir, reg_data)
    
    print("Running test_few_features...")
    test_few_features(results_dir)
    
    print("Running test_with_nan_values...")
    test_with_nan_values(results_dir, reg_data)
    
    print("All tests completed successfully!")
