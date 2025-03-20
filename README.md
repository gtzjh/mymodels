[English version](#English-Documentation)

[中文版本](#中文使用说明)

# English Documentation

**Machine learning pipeline with automated hyperparameter tuning using Optuna and model interpretation with SHAP (SHapley Additive exPlanations).**

**Remember: All models are wrong, but some are useful. - George Box**

*Support for classification tasks and categorical features is currently being developed.*

Supported Models with Hyperparameter Optimization:

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

Build an instance of the pipeline:

```python
the_model = MLPipeline(
    file_path = "data.csv",
    y = "y", 
    x_list = list(range(1, 16)),
    model = "cat",
    results_dir = "results/cat",
    cat_features = ["x16", "x17"],
    trials = 50,
    test_ratio = 0.3,
    shap_ratio = 0.3,
    cross_valid = 5,
    random_state = 0
)
```

Run the pipeline directly:

```python
the_model.run()
```

Or run the pipeline step by step:

```python
the_model.load_data()  # Load data
the_model.optimize()   # Optimize model
the_model.explain()    # Explain model
```

Description of parameters:

| Parameter | Description |
|-----------|-------------|
| file_path | Path to the data file |
| y | Column name of the dependent variable (y) |
| x_list | List of column indices for independent variables (x), can also be a list of column name strings |
| model | Model selection: "dt", "rf", "gbdt", "ada", "xgb", "lgb", "cat", "svr", "knr", "mlp" |
| results_dir | Directory to save results, can use model name as directory name or pass a pathlib object |
| cat_features | List of categorical features, e.g. ["x16", "x17"], None if none exist |
| trials | Number of trials for Optuna hyperparameter optimization |
| test_ratio | Proportion of test set in total dataset (30% of total data) |
| shap_ratio | Proportion of data used for SHAP value calculation (30% of total data) |
| cross_valid | Number of folds for cross-validation during Optuna optimization |
| random_state | Global random seed to control randomness in model training, cross-validation and testing |

Execute the command:
```bash
python main.py
```


# 中文使用说明

存储我常用的机器学习模型，并使用Optuna进行贝叶斯调参，使用SHAP进行模型解释。用最少的时间完成机器学习任务。

**请记住：所有模型都是错的，但有一些是有用的。**

*对分类任务和类别自变量的支持正在完善中。*

本项目包含的模型：

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


## 0. 前置知识

1. 熟悉Python编程。[廖雪峰老师的教程](https://liaoxuefeng.com/books/python/introduction/index.html)提供了非常好的入门指引，建议学到 17.常用内建模块即可，而对7，8，9，10，11则要重点掌握。每一节学完以后尝试去完成课后习题。最后，**一定要以一个实践项目来检验自己的学习成果**，比如设计一段爬虫，或是实现一些小功能等 [(这是我写的一个小爬虫)](https://github.com/gtzjh/WundergroundSpider)。建议不要在这一阶段使用ChatGPT，但可以再写出来以后让其给出代码优化建议让自己进步。

2. 机器学习的基础。吴恩达老师的[CS229](https://www.bilibili.com/video/BV1JE411w7Ub)课程是非常棒的资料。

3. 其他

    **明白如何使用conda和pip创建和管理环境**，并明白如何在编辑器（vscode、Cursor等）中使用它

    **明白如何使用终端（Terminal）**

    建议学会使用[Git](https://github.com/gtzjh/learngit)，尝试自己在GitHub上建一个项目并学会用它来管理代码。


## 1. 环境准备（Windows平台，其余平台同理）

**使用 Python 3.10**

*环境安装大约使用1.75 GiB存储空间*

conda

```bash
conda env create -f requirement.yml -n mymodels
conda activate mymodels
```

## 2. 使用

根据自己需要修改 `main.py` 中的以下内容：

构建模型实例：

```python
the_model = MLPipeline(
    file_path = "data.csv",
    y = "y", 
    x_list = list(range(1, 16)),
    model = "cat",
    results_dir = "results/cat",
    cat_features = ["x16", "x17"],
    trials = 50,
    test_ratio = 0.3,
    shap_ratio = 0.3,
    cross_valid = 5,
    random_state = 0
)
```

运行模型：
```python
the_model.run()
```

或逐步运行：

```python
the_model.load_data()  # 加载数据
the_model.optimize()   # 优化模型
the_model.explain()    # 解释模型
```

参数说明：

| 参数 | 说明 |
|------|------|
| file_path | 数据文件路径 |
| y | 选择因变量（y）的列名 |
| x_list | 选择自变量（x）的列索引列表，也可以传入变量名字符串列表 |
| model | 模型选择："dt"、"rf"、"gbdt"、"ada"、"xgb"、"lgb"、"cat"、"svr"、"knr"、"mlp" |
| results_dir | 结果保存目录，可以使用模型名称作为目录名，也可以传入pathlib对象 |
| cat_features | 分类特征列表，需显式指定分类特征的名称，没有则为None |
| trials | Optuna超参数优化的迭代次数（默认迭代50次） |
| test_ratio | 测试集占总数据集的比例（默认占总数据集的30%） |
| shap_ratio | 用于SHAP值计算的数据比例（默认占总数据集的30%） |
| cross_valid | Optuna超参数优化时每一次迭代的交叉验证折数（默认5折交叉验证） |
| random_state | 全局随机种子，用于控制模型训练、交叉验证和测试的随机性 |

运行 `main.py` 。(命令行中或用Debug模式)

```bash
python main.py
```