[English version](#English-Documentation)

[中文版本](#中文使用说明)

# English Documentation

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
conda env create -f requirement.yml -n mymodels
conda activate mymodels
```

## 2. Usage

Configure parameters in `main.py`:

1. Build an instance of the pipeline:

```python
the_pipeline = MyPipeline(
    model_name = "rfc",
    results_dir = "results/rfc_single_test",
    cat_features = ['Pclass', 'Sex', 'Embarked'],
    encode_method = "onehot",
    random_state = 0,
)
```

2. Load the data:
```python
the_pipeline.load(
    file_path = "data/titanic/train.csv",
    y = "Survived",
    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
    test_ratio = 0.4,
    inspect = True
)
```

3. Run the pipeline step by step:

```python
# Optimize model with hyperparameter tuning
the_pipeline.optimize(cv = 5, trials = 30, n_jobs = 5)
# Evaluate the model
the_pipeline.evaluate()
# Explain model with SHAP
the_pipeline.explain(shap_ratio = 0.3, _plot = True)
```

4. Or run the pipeline in one step (includes load, optimize, evaluate, explain):

```python
the_pipeline.run(
    file_path = "data/titanic/train.csv",
    y = "Survived",
    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
    test_ratio = 0.4,
    cv = 5,
    trials = 30,
    n_jobs = 5,
    shap_ratio = 0.3,
    inspect = True,
    interpret = True
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


# 中文使用说明

存储我常用的机器学习模型，并使用Optuna进行贝叶斯调参，使用SHAP进行模型解释。

## 本项目包含的模型

### 回归模型

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

### 分类模型

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



## 前置知识

1. 熟悉Python编程。[廖雪峰老师的教程](https://liaoxuefeng.com/books/python/introduction/index.html)提供了非常好的入门指引，建议学到 17.常用内建模块即可，而对7，8，9，10，11则要重点掌握。每一节学完以后尝试去完成课后习题。最后，**一定要以一个实践项目来检验自己的学习成果**，比如设计一段爬虫，或是实现一些小功能等 [(这是我写的一个小爬虫)](https://github.com/gtzjh/WundergroundSpider)。建议不要在这一阶段使用ChatGPT，但可以再写出来以后让其给出代码优化建议让自己进步。

2. 机器学习的基础。吴恩达老师的[CS229](https://www.bilibili.com/video/BV1JE411w7Ub)课程是非常棒的资料。

3. 其他

    **明白如何使用conda和pip创建和管理环境**，并明白如何在编辑器（vscode、Cursor等）中使用它

    **明白如何使用终端（Terminal）**

    建议学会使用[Git](https://github.com/gtzjh/learngit)，尝试自己在GitHub上建一个项目并学会用它来管理代码。


## 环境准备（Windows平台，其余平台同理）

**使用 Python 3.10**

*环境安装大约使用1.75 GiB存储空间*

conda

```bash
conda env create -f requirement.yml -n mymodels
conda activate mymodels
```

## 使用

根据自己需要修改 `main.py` 中的以下内容：

1. 构建模型实例：

```python
the_pipeline = MyPipeline(
    model_name = "rfc",
    results_dir = "results/rfc_single_test",
    cat_features = ['Pclass', 'Sex', 'Embarked'],
    encode_method = "onehot",
    random_state = 0,
)
```

2. 加载数据：
```python
the_pipeline.load(
    file_path = "data/titanic/train.csv",
    y = "Survived",
    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
    test_ratio = 0.4,
    inspect = True
)
```

3. 逐步运行：

```python
# 优化模型
the_pipeline.optimize(cv = 5, trials = 30, n_jobs = 5)
# 评估模型
the_pipeline.evaluate()
# 解释模型
the_pipeline.explain(shap_ratio = 0.3, _plot = True)
```

4. 或一步到位（包含load, optimize, evaluate, explain）：

```python
the_pipeline.run(
    file_path = "data/titanic/train.csv",
    y = "Survived",
    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
    test_ratio = 0.4,
    cv = 5,
    trials = 30,
    n_jobs = 5,
    shap_ratio = 0.3,
    inspect = True,
    interpret = True
)
```

参数说明：

| 参数          | 说明                                                                                                      |
|---------------|----------------------------------------------------------------------------------------------------------|
| model_name    | 模型选择：回归任务使用 "svr", "knr", "mlpr", "dtr", "rfr", "gbdtr", "adar", "xgbr", "lgbr", "catr";<br>分类任务使用 "svc", "knc", "mlpc", "dtc", "rfc", "gbdtc", "adac", "xgbc", "lgbc", "catc" |
| results_dir   | 结果保存目录（支持字符串或pathlib.Path对象）                                                            |
| cat_features  | 分类特征列表/元组（使用非CatBoost模型时必须指定）                                                       |
| encode_method | 分类特征编码方法："onehot", "binary", "ordinal", "label", "target", "frequency"<br>使用列表/元组时必须与cat_features长度一致 |
| random_state  | 控制以下随机性：数据分割、模型训练、交叉验证、SHAP抽样                                                  |
| file_path     | 数据文件路径（字符串或pathlib.Path对象）                                                                 |
| y             | 目标变量名称或索引                                                                                        |
| x_list        | 要使用的特征名称或索引列表                                                                               |
| test_ratio    | 用于测试的数据比例（默认：0.3）                                                                          |
| inspect       | 是否在加载过程中打印数据信息（默认：True）                                                                |
| cv            | 优化时的交叉验证折数（默认：5）                                                                           |
| trials        | 优化尝试次数（默认：50）                                                                                  |
| n_jobs        | 优化的并行作业数（默认：5）                                                                               |
| shap_ratio    | 用于SHAP分析的测试数据比例（默认：0.3）                                                                   |
| interpret     | 是否使用SHAP进行解释（默认：True）                                                                        |

运行 `main.py` 。(命令行中或用Debug模式)

```bash
python main.py
```