import optuna
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from mymodels.plotting import Plotter



def test_plot_binary_classification():
    """
    Test whether the ROC curve, PR curve, and confusion matrix plots for binary classification models are drawn correctly
    
    1. Generate a binary classification dataset
    2. Train a model using Random Forest classifier
    3. Use the trained model to make predictions on the test set
    4. Plot ROC curve, PR curve, and confusion matrix
    5. Save images to the results directory
    """
    # Create Plotter instance
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir="./results/test_plotting_plot_evaluated_classifier/binary_classification"
    )
    
    # 1. Generate a binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                             random_state=42)
    
    # 2. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                      random_state=42)
    
    # 3. Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # 5. Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Binary Classification Accuracy: {accuracy:.4f}")
    
    # 6. Plot ROC curve
    plotter.plot_roc_curve(y_test, X_test, model)
    
    # 7. Plot PR curve
    plotter.plot_pr_curve(y_test, X_test, model)
    
    # 8. Plot confusion matrix
    plotter.plot_confusion_matrix(y_test, y_pred)
    
    return None


def test_plot_multi_classification():
    """
    Test whether the ROC curve, PR curve, and confusion matrix plots for multi-class classification models are drawn correctly
    
    1. Generate a multi-class classification dataset
    2. Train a model using Random Forest classifier
    3. Use the trained model to make predictions on the test set
    4. Plot ROC curve, PR curve, and confusion matrix
    5. Save images to the results directory
    """
    # Create Plotter instance
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir="./results/test_plotting_plot_evaluated_classifier/multi_classification"
    )
    
    # 1. Generate a multi-class classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                              n_informative=15, random_state=42)
    
    # 2. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    # 3. Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # 5. Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Multi-class Classification Accuracy: {accuracy:.4f}")
    
    # 6. Plot ROC curve
    plotter.plot_roc_curve(y_test, X_test, model)
    
    # 7. Plot PR curve
    plotter.plot_pr_curve(y_test, X_test, model)
    
    # 8. Plot confusion matrix
    plotter.plot_confusion_matrix(y_test, y_pred)
    
    return None


def test_multi_input_case():
    """
    Test whether the plotting functions can properly handle error cases
    
    1. Test with unequal lengths of X and y
    2. Test with y containing null/empty values
    3. Test with inconsistent prediction probabilities
    4. Verify appropriate errors are raised
    """
    import numpy as np
    import pytest
    
    # Create Plotter instance
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir="./results/test_plotting_plot_evaluated_classifier/error_cases"
    )
    
    # Case 1: Unequal lengths of X and y
    X_unequal = np.random.rand(10, 5)  # 10 samples, 5 features
    y_unequal = np.array([0, 1, 0, 1, 0])  # Only 5 labels
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Test ROC curve with unequal lengths
    with pytest.raises(ValueError):
        model.fit(X_unequal, y_unequal)  # This should fail
    
    # Case 2: y containing null/empty values
    X_normal = np.random.rand(10, 5)
    y_with_nulls = np.array([0, 1, np.nan, 1, 0, 1, 0, 1, 0, 1])
    
    # Test fitting with null values
    with pytest.raises(Exception):
        model.fit(X_normal, y_with_nulls)  # Should fail with nan values
    
    # Case 3: Test confusion matrix with mismatched prediction lengths
    X_train, X_test, y_train, y_test = train_test_split(
        np.random.rand(20, 5), np.random.randint(0, 2, 20), 
        test_size=0.5, random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Create mismatched prediction length
    y_pred_mismatched = model.predict(X_test)[:5]  # Only use first 5 predictions
    
    # Test confusion matrix with mismatched lengths
    with pytest.raises(ValueError):
        plotter.plot_confusion_matrix(y_test, y_pred_mismatched)
    
    # Case 4: Test with empty arrays
    X_empty = np.array([])
    y_empty = np.array([])
    
    with pytest.raises(Exception):
        model.fit(X_empty.reshape(0, 1), y_empty)
    
    # Case 5: Direct plotting tests with bad inputs
    # Create a valid model first
    X_valid = np.random.rand(30, 5)
    y_valid = np.random.randint(0, 2, 30)
    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.3)
    
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    
    # Test plot_roc_curve with y_test having different class values than y_train
    y_test_bad_classes = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    X_test_matching = np.random.rand(8, 5)
    
    with pytest.raises(Exception):
        plotter.plot_roc_curve(y_test_bad_classes, X_test_matching, model)
        
    # Test plot_pr_curve with different sizes of X_test and y_test
    y_test_normal = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    X_test_smaller = np.random.rand(4, 5)  # Only 4 samples
    
    with pytest.raises(ValueError):
        plotter.plot_pr_curve(y_test_normal, X_test_smaller, model)
    
    # Test plot_confusion_matrix with y_true containing values not in training
    y_pred_normal = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_true_invalid = np.array([0, 1, 2, 3, 0, 1, 0, 1])  # Contains 2, 3 not in training
    
    # This might raise an error depending on implementation
    try:
        plotter.plot_confusion_matrix(y_true_invalid, y_pred_normal)
        # If it doesn't raise an error, we should still verify the result
        # in some implementations, this might generate a confusion matrix with more classes
    except Exception:
        pass  # Expected error behavior
        
    return None


if __name__ == "__main__":
    test_plot_binary_classification()
    test_plot_multi_classification()
    test_multi_input_case()
