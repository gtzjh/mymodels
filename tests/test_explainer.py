# Test the PDP and SHAP explainers
import os, sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mymodels.explainer import pdp_explainer, shap_explainer


def test_regression_explainers():
    """Test PDP and SHAP explainers with a regression task using RandomForest"""
    
    # Create a small regression dataset
    X, y = make_regression(
        n_samples=100,  # small dataset
        n_features=5,
        n_informative=3,
        random_state=42
    )
    
    # Convert to dataframe with feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=42
    )
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)

    results_dir = "results/test_regression_explainers"
    
    # Test PDP explainer
    
    pdp_explainer(
        model=model,
        explain_data=X_test,
        results_dir=results_dir,
        dpi=600,
        format="tif"
    )

    # Test SHAP explainer
    shap_explainer(
        model=model,
        background_data=X_train,
        shap_data=X_test,
        explainer_type="tree",
        results_dir=results_dir,
        dpi=100,
        format="jpg"
    )


def test_binary_explainers():
    """Test PDP and SHAP explainers with a binary classification task using RandomForest"""
    # Create a small binary classification dataset
    X, y = make_classification(
        n_samples=100,  # small dataset
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    
    # Convert to dataframe with feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=42
    )
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)

    results_dir = "results/test_binary_explainers"
    
    # Test PDP explainer
    pdp_explainer(
        model=model,
        explain_data=X_test,
        results_dir=results_dir,
        dpi=600,
        format="tif"
    )

    # Test SHAP explainer
    shap_explainer(
        model=model,
        background_data=X_train,
        shap_data=X_test,
        explainer_type="tree",
        results_dir=results_dir,
        dpi=100,
        format="jpg"
    )



def test_multi_class_explainers():
    """Test PDP and SHAP explainers with a multi-class classification task using RandomForest"""
    # Create a small multi-class classification dataset
    X, y = make_classification(
        n_samples=100,  # small dataset
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=5,
        random_state=42
    )
    
    # Convert to dataframe with feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=42
    )
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)

    results_dir = "results/test_multi_class_explainers"
    
    # Test PDP explainer
    pdp_explainer(
        model=model,
        explain_data=X_test,
        results_dir=results_dir,
        dpi=300,
        format="jpg"
    )

    # Test SHAP explainer
    shap_explainer(
        model=model,
        background_data=X_train,
        shap_data=X_test,
        explainer_type="tree",
        results_dir=results_dir,
        dpi=300,
        format="jpg"
    )
    

if __name__ == "__main__":
    # test_regression_explainers()
    # test_binary_explainers()
    test_multi_class_explainers()
