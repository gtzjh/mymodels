# mymodels: Automated Interpretable Machine Learning Workflow

## Supported Models

### Regression Tasks
- [Support Vector Regression (SVR)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [K-Nearest Neighbors Regression (KNR)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [Multi-Layer Perceptron (MLP)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
- [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Gradient Boosted Decision Trees (GBDT)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)
- [CatBoost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

### Classification Tasks
- [Support Vector Classification (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [K-Nearest Neighbors Classification (KNC)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Multi-Layer Perceptron (MLP)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Gradient Boosted Decision Trees (GBDT)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
- [CatBoost](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier)

**Note:** SHAP currently doesn't support multi-class classification tasks when using GBDT models.

## Prerequisites

1. **Python Proficiency**  
   Recommended resource: [Liao Xuefeng's Python Tutorial](https://liaoxuefeng.com/books/python/introduction/index.html). Key chapters:
   - Basic syntax (Chapters 7-11)
   - Built-in modules (Chapter 17)
   - Practical project implementation

2. **Machine Learning Fundamentals**  
   [CS229 by Andrew Ng](https://www.bilibili.com/video/BV1JE411w7Ub) provides essential theoretical foundations.

3. **Technical Skills**:
   - Environment management with conda/pip
   - Terminal/Command Line proficiency
   - Version control with Git ([Learning Resources](https://github.com/gtzjh/learngit))

## Environment Setup (Windows)

**Requirements**:
- Python 3.10
- 1.75 GB available disk space

```bash
conda env create -f requirement.yml -n mymodels -y
conda activate mymodels
```

## Basic Usage

Example implementation (`run_titanic.py`):

```python
# 1. Initialize pipeline
mymodels = MyPipeline(
    results_dir="results/catc",  # Output directory
    random_state=0,              # Reproducibility seed
    show=True,                   # Display plots inline
    plot_format="pdf",           # Export format
    plot_dpi=300                 # Image resolution
)

# 2. Load and preprocess data
mymodels.load(
    file_path="data/titanic.csv",
    target="Survived",
    features=["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
    test_ratio=0.4,             # Train-test split ratio
    inspect=False               # Set to True for data exploration
)

# 3. Model optimization
mymodels.optimize(
    model_name="catc",
    categorical_features=["Pclass", "Sex", "Embarked"],
    encoding_strategy=None,     # Auto-detection enabled
    cv_folds=5,                 # Cross-validation
    optimization_trials=6,      # Hyperparameter search iterations
    parallel_jobs=-1,           # Use all available cores
    visualize_optim=True        # Generate optimization plots
)

# 4. Model evaluation
mymodels.evaluate()

# 5. Explainability analysis
mymodels.explain(
    train_subsample=0.5,        # Fraction of training data to use
    test_subsample=0.5          # Fraction of test data to use
)
```

## Example Implementations

- `run_housing.py`: Regression task  
  Dataset source: [Kaggle Housing Data](https://www.kaggle.com/datasets/jamalshah811/housingdata)

- `run_obesity.py`: Multi-class classification  
  Dataset source: [Obesity Risk Dataset](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)

- `run_titanic.py`: Binary classification  
  Dataset source: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)