# mymodels: An automated and efficient workflow for Interpretable Machine Learning.

## Supported Models

### For Regression Task

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

### For Classification Task

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


*SHAP DOES NOT CURRENTLY SUPPORT MULTI-CLASSIFICATION TASKS WHEN USING GBDT.*


## 0. Prerequisites

1. Python programming proficiency. [Liao Xuefeng's Python Tutorial](https://liaoxuefeng.com/books/python/introduction/index.html) provides excellent beginner guidance. Study up to Chapter 17 (Built-in Modules), with special focus on Chapters 7-11. Complete the exercises after each section. Most importantly, **validate your learning through a practical project** - for example, writing a web scraper or implementing small utilities [(here's my example web scraper)](https://github.com/gtzjh/WundergroundSpider). Avoid using ChatGPT during initial learning, but you can use it later for code optimization suggestions.

2. Machine learning fundamentals. [CS229 by Andrew Ng](https://www.bilibili.com/video/BV1JE411w7Ub) is an excellent resource.

3. Additional skills:

    **Understanding how to create and manage environments using conda and pip**, and how to use them in editors (VSCode, Cursor etc.)

    **Proficiency with Terminal/Command Line**

    Recommended to learn [Git](https://github.com/gtzjh/learngit) - try creating your own GitHub project and learn to manage code with version control.

## 1. Environment Setup (Windows)

**Python 3.10 required**
*Approximately 1.75 GiB disk space required*

Using Conda:
```bash
conda env create -f requirement.yml -n mymodels -y
conda activate mymodels
```

## 2. Usage

Take `run_titanic.py` as an example.

1. Build an instance of the pipeline

```python
mymodels = MyPipeline(
    results_dir = "results/catc",
    random_state = 0,
    show = True,
    plot_format = "pdf",
    plot_dpi = 300
)
```

2. Load the data

```python
mymodels.load(
    file_path = "data/titanic.csv",
    y = "Survived",
    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
    test_ratio = 0.4,
    inspect = False
)
```

3. Optimize model

```python
mymodels.optimize(
    model_name = "catc",
    cat_features = ["Pclass", "Sex", "Embarked"],
    encode_method = None,
    cv = 5,
    trials = 6,
    n_jobs = -1,
    plot_optimization = True
)
```

4. Evaluate model
```python
mymodels.evaluate()
```

```python
mymodels.explain(
    sample_train_k = 0.5,
    sample_test_k = 0.5,
)
```

## 3. Other Usage

`run_housing.py`: For regression task.

> https://www.kaggle.com/datasets/jamalshah811/housingdata

`run_obesity.py`: For multi-class classification task.

> https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset

`run_titanic.py`: For binary classification task.

> https://www.kaggle.com/c/titanic/data