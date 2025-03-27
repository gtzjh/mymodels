**Machine learning pipeline with automated hyperparameter tuning using Optuna and model interpretation with SHAP (SHapley Additive exPlanations).**

## Supported Models

### Regression Models

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

### Classification Models

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

Configure parameters in `main.py`:

1. Build an instance of the pipeline:

```python
mymodels = MyPipeline(
        results_dir = "results/catc",  # 结果保存路径
        random_state = 0,  # 随机种子
        show = True,  # 是否显示图表
        plot_format = "pdf",  # 图表格式
        plot_dpi = 300  # 图表分辨率
    )
```

2. Load the data:
```python
mymodels.load(
        file_path = "data/titanic.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        test_ratio = 0.4,
        inspect = False
    )
```

3. Run the pipeline step by step:

```python
# Optimize model with hyperparameter tuning
mymodels.optimize(
        model_name = "catc",
        cat_features = ["Pclass", "Sex", "Embarked"],
        encode_method = None,
        cv = 5,
        trials = 6,
        n_jobs = -1,
        plot_optimization = True
    )
# Evaluate the model
mymodels.evaluate()
# Explain model with SHAP
mymodels.explain(
        sample_train_k = 0.5,
        sample_test_k = 0.5,
    )
```


Description of parameters:

| Parameter     | Description                                                                                                  |
|---------------|--------------------------------------------------------------------------------------------------------------|
| model_name    | Model selection: "svr", "knr", "mlpr", "dtr", "rfr", "gbdtr", "adar", "xgbr", "lgbr", "catr" for regression;<br>"svc", "knc", "mlpc", "dtc", "rfc", "gbdtc", "adac", "xgbc", "lgbc", "catc" for classification |
| results_dir   | Directory to save results (string or pathlib.Path object)                                                   |
| cat_features  | List/tuple of categorical feature names (required if using non-CatBoost models)                             |
| encode_method | Encoding method(s) for categorical features: "onehot", "binary", "ordinal", "label", "target", "frequency".<br>Must match length of cat_features when using list/tuple. |
| random_state  | Controls randomness in: data splitting, model training, cross-validation, and SHAP sampling                |
| file_path     | Path to the data file (string or pathlib.Path object)                                                      |
| y             | Target variable name or index                                                                               |
| x_list        | List of feature names or indices to use                                                                     |
| test_ratio    | Proportion of data to use for testing (default: 0.3)                                                        |
| inspect       | Whether to print data information during loading (default: True)                                            |
| cv            | Number of cross-validation folds for optimization (default: 5)                                              |
| trials        | Number of optimization trials (default: 50)                                                                 |
| n_jobs        | Number of parallel jobs for optimization (default: 5)                                                       |
| shap_ratio    | Proportion of test data to use for SHAP analysis (default: 0.3)                                             |
| interpret     | Whether to run explanation with SHAP (default: True)                                                        |

Execute the command:
```bash
python main.py
```