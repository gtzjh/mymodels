<div style="text-align: center;">

<h1 align="center">🚀 mymodels 🚀 : 节省您的时间！自动化可解释机器学习工作流</h1>

</div>

**关注公众号：👉GT地学志👈 获取项目更新。**

<img src="qrcode.jpg" alt="mymodels" width="130">


## 🎯 谁应该使用mymodels？

科研人员或（硕士/博士）学生希望在他们的研究项目中使用可解释性机器学习，但又不希望将过多的时间耗费在无意义的工具安装、环境配置、以及从0-1的编程学习当中（例如，临床医学生通常需要将大部分时间用于临床工作，科研的时间所剩无几）。


## 🤔 为什么选择 mymodels？

可解释机器学习在地理学、遥感和城市规划等各个领域已经获得了显著的关注。机器学习模型因其通过复杂拟合算法捕捉数据中复杂关系的强大能力而备受推崇。作为这些模型的补充，基于博弈论的可解释性框架——如SHapley Additive exPlanations (SHAP)——提供了揭示这些"黑盒"模型的重要工具。这些可解释方法通过对特征重要性排序、识别非线性响应阈值以及分析因素之间的交互关系，提供了关键见解。

尽管有这些优势，实现可解释机器学习工作流仍然是一个复杂且耗时的过程，特别是对于该领域的新手。在有效执行这些工作流程的综合性、用户友好的工具方面存在明显的差距。mymodels项目通过自动化可解释机器学习过程解决了这一差距，在保持分析严谨性的同时显著减少了实施时间。

## 👨‍🎓 你要先学一些东西

1. 💡 **Python基础**

    推荐资源：

    - [W3SCHOOL的Python教程](https://www.w3schools.com/python/default.asp)
    
    - [廖雪峰的Python教程](https://liaoxuefeng.com/books/python/introduction/index.html)

    必要内容：
    - 基础知识
    - 面向对象开发(OOP)
    - 一些常用的Python内置模块
    
    > **请记住**：在完成上述学习后制作一个个人小项目以加强所学内容（例如，一个小型网络爬虫）。[这是我的一个练习项目](https://github.com/gtzjh/WundergroundSpider)

2. 💡 **机器学习基础**

    [吴恩达机器学习教程](https://www.bilibili.com/video/BV1Bq421A74G) 吴恩达老师的课程提供了必要的理论基础。


3. 💡 **技术技能**

    - 使用conda/pip进行环境管理
    - 熟悉终端/命令行
    - 使用Git进行版本控制 ([我的Git学习笔记](https://github.com/gtzjh/learngit))

## 🛠️ 环境设置

支持的平台：

- Windows (X86) [在Windows 10/11上测试通过]
- Linux (X86) [在WSL2.0 (Ubuntu)上测试通过]
- macOS (ARM) [在Apple Silicon (M1)上测试通过]

### 创建环境

**要求**：
- Python 3.10.X
- 1.75 GB可用磁盘空间

在终端中运行以下命令：

```bash
conda env create -f requirement.yml -n mymodels -y
```

> 您也可以根据`requirement.yml`文件使用`pip`手动创建环境：

---

### 激活环境

在终端中运行以下命令：

```bash
conda activate mymodels
```

---

## 🚀 如何使用

我们以`run_titanic.py`为例：


### 导入包

#### 示例

```python
from mymodels.pipeline import MyPipeline
```

---

### 构建工作流对象

#### 参数

- **results_dir**：存储结果的目录路径。接受字符串或pathlib.Path对象。如果目录不存在，将创建该目录。

- **random_state**：整个流水线的随机种子（数据分割、模型调优等）。（默认为0）

- **show**：是否在屏幕上显示图像。（默认为`False`）

- **plot_format**：图像输出格式。（默认为jpg）

- **plot_dpi**：控制输出图像的分辨率。（默认为500）

#### 示例

```python
mymodel = MyPipeline(
    results_dir = "results/titanic",
    random_state = 0,
    show = False,
    plot_format = "jpg",
    plot_dpi = 500
)
```

---

### 加载数据

#### 参数

- **file_path**：您要输入的数据所在位置。**.csv格式是必需的**。

- **y**：您要预测的目标。表示列名的`str`对象或表示列索引的`int`对象都是可以接受的。

- **x_list**：独立变量的`list`对象（或`tuple`对象）。`list`（或`tuple`）中的每个元素必须是表示列名的`str`对象或表示列索引的`int`对象。

- **index_col**：表示索引列的`int`对象或`str`对象。（默认为`None`）

    > 如果您想输出原始数据和shap值，强烈建议设置索引列。同样，提供表示多个索引列的`list`对象（或`tuple`对象）也是可以接受的。

- **test_ratio**：测试数据的比例。（默认为0.3）

- **inspect**：是否在终端中显示您选择的y列或独立变量。（默认为`True`）

#### 示例

```python
mymodel.load(
    file_path = "data/titanic.csv",
    y = "Survived",
    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
    index_col = ["PassengerId", "Name"],
    test_ratio = 0.3,
    inspect = False
)
```

---

### 执行优化

#### 参数

- **model_name**：您要使用的模型。在本例中，`xgbc`代表XGBoost分类器，其他模型名称如`catr`表示CatBoost回归器。表示不同模型和任务的完整模型名称列表可在末尾找到。

- **cat_features**：为模型指定的分类特征的`list`（或`tuple`）。表示列名的`str`的`list`（或`tuple`）或表示列索引的`int`都是可以接受的。（默认为`None`）

- **encode_method**：表示编码方法的`str`对象，或编码方法的`list`（或`tuple`）都是可以接受的。（默认为`None`）

  如果仅呈现单个编码方法（如下所示，仅呈现`onehot`编码方法），它将应用于所有分类特征（如`encode_method`中列出的）。

  如果呈现编码方法的`list`（或`tuple`），例如`["onehot", "binary", "target"]`，它们将分别应用于三个分类特征。因此，`encode_method`的长度必须与`cat_features`的长度相同。

  > 支持的编码方法完整列表可在末尾找到。

  

- **cv**：调优过程中的交叉验证。（默认为5）

- **trials**：贝叶斯调优过程中的试验次数（基于[Optuna](https://optuna.org/)）。（默认为50）

- **n_jobs**：交叉验证过程中将使用的核心数量。建议使用与`cv`相同的值。（默认为5）

- **optimize_history**：是否保存优化历史。（默认为`True`）

- **save_optimal_params**：是否保存最佳参数。（默认为`True`）

- **save_optimal_model**：是否保存最优模型。（默认为`True`）

#### 输出

结果目录中将输出几个文件：

- `params.yml`将记录最佳参数。

- `mapping.json`将记录分类特征与编码特征之间的映射关系。

- `optimal-model.joblib`将保存来自sklearn的最优模型。

- `optimal-model.cbm`将保存来自CatBoost的最优模型。

- `optimal-model.txt`将保存来自LightGBM的最优模型。

- `optimal-model.json`将保存来自XGBoost的最优模型。

- `optimal-model.pkl`将保存所有类型的最优模型以实现兼容性。

#### 示例

```python
mymodel.optimize(
    model_name = "xgbc",
    cat_features = ["Pclass", "Sex", "Embarked"],
    encode_method = "onehot",
    cv = 5,
    trials = 10,
    n_jobs = 5,
    optimize_history = True,
    save_optimal_params = True,
    save_optimal_model = True
)
```

---

### 评估模型的准确性

#### 参数

- **save_raw_data**：是否保存原始预测数据。默认为`True`。

#### 输出

准确性结果将输出到您上面定义的目录：

- 名为`accuracy`的`.yml`文件将记录模型准确性的结果。

- 名为`roc_curve_plot`的图形记录分类准确性。

- 或者名为`accuracy_plot`的图形（这是一个散点图）用于回归任务。

#### 示例
```python
mymodel.evaluate(save_raw_data = True)
```

---

### 使用SHAP（SHapley Additive exPlanations）解释模型

#### 参数

- **select_background_data**：用于**背景值计算**的数据。（默认为`"train"`）

    默认为`"train"`，表示将使用训练集中的所有数据。`"test"`表示将使用测试集中的所有数据。`"all"`表示将使用训练集和测试集中的所有数据。

- **select_shap_data**：用于**计算SHAP值**的数据。默认为`"test"`，表示将使用测试集中的所有数据。`"all"`表示将使用训练集和测试集中的所有数据。（默认为`"test"`）

- **sample_background_data_k**：对训练集中的样本进行抽样，用于**背景值计算**。（默认为`None`）
    
    默认`None`，表示将使用训练集中的所有数据。整数值表示实际数据数量，而浮点数（例如0.5）表示训练集中的比例。

- **sample_shap_data_k**：与`sample_background_data_k`含义类似。测试集将用于**SHAP值计算**。（默认为`None`）

- **output_raw_data**：是否保存原始数据。默认为`False`。

#### 输出

图形（摘要图、依赖图）将输出到您上面定义的目录。

#### 示例

```python
mymodel.explain(
    select_background_data = "train",
    select_shap_data = "test",
    sample_background_data_k = 50,
    sample_shap_data_k = 50,
    output_raw_data = True
)
```

---

### 运行代码

在VSCode中按`F5`以调试模式运行。

或者在终端中运行以下命令：

```bash
python run_titanic.py
```

---

### 完整代码

```python
from mymodels.pipeline import MyPipeline


def main():
    mymodel = MyPipeline(
        results_dir = "results/titanic",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 500
    )
    mymodel.load(
        file_path = "data/titanic.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        index_col = ["PassengerId", "Name"],
        test_ratio = 0.3,
        inspect = False
    )
    mymodel.optimize(
        model_name = "rfc",
        cat_features = ["Pclass", "Sex", "Embarked"],
        encode_method = "onehot",
        cv = 5,
        trials = 10,
        n_jobs = 5,
        optimize_history = True,
        save_optimal_params = True,
        save_optimal_model = True
    )
    mymodel.evaluate(save_raw_data = True)
    mymodel.explain(
        select_background_data = "train",
        select_shap_data = "test",
        sample_background_data_k = 50,
        sample_shap_data_k = 50,
        output_raw_data = True
    )

    return None


if __name__ == "__main__":
    main()
```


## 🎯 尝试这些示例

- `run_housing.py`：回归任务  
  数据集来源：[Kaggle住房数据](https://www.kaggle.com/datasets/jamalshah811/housingdata)

- `run_obesity.py`：多类分类  
  数据集来源：[肥胖风险数据集](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)

- `run_titanic.py`：二元分类  
  数据集来源：[泰坦尼克：从灾难中学习机器学习](https://www.kaggle.com/c/titanic/data)


## 📚 补充信息

### 🛠️ 所需包

需要以下包：
  - catboost=1.2.7
  - ipython=8.30.0
  - lightgbm=4.5.0
  - matplotlib-base=3.9.3
  - numba=0.60.0
  - numpy=1.26.4
  - optuna=4.1.0
  - pandas=2.2.3
  - pip=24.3.1
  - plotly=5.24.1
  - py-xgboost=2.1.4
  - python-graphviz=0.20.3
  - python=3.10.16
  - scikit-learn=1.5.2
  - scipy=1.14.1
  - setuptools=75.6.0
  - shap=0.46.0
  - tqdm=4.67.1
  - wheel=0.45.1
  - category_encoders=2.6.3


### 🛠️ 支持的模型

*点击第二列中的链接查看官方文档。*

#### 回归任务
| `model_name` | 模型|
|------------|-------|
| svr        | [支持向量回归](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) |
| knr        | [K近邻回归](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) |
| mlpr       | [多层感知器回归器](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) |
| dtr        | [决策树回归器](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) |
| rfr        | [随机森林回归器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) |
| gbdtr        | [梯度提升决策树(GBDT)回归器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) |
| adar        | [AdaBoost回归器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) |
| xgbr       | [XGBoost回归器](https://xgboost.readthedocs.io/en/latest/python/python_api.html) |
| lgbr      | [LightGBM回归器](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) |
| catr       | [CatBoost回归器](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor) |

#### 分类任务

| `model_name` | 模型|
|------------|-------|
| svc        | [支持向量分类](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| knc        | [K近邻分类](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
| mlpc       | [多层感知器分类器](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) |
| dtc        | [决策树分类器](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) |
| rfc        | [随机森林分类器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| gbdtc        | [梯度提升决策树(GBDT)分类器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) |
| adac        | [AdaBoost分类器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) |
| xgbc       | [XGBoost分类器](https://xgboost.readthedocs.io/en/latest/python/python_api.html) |
| lgbc      | [LightGBM分类器](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) |
| catc       | [CatBoost分类器](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier) |


### 🛠️ 支持的编码方法

| `encode_method` | 描述 |
|------------|------------------|
| onehot   | 独热编码   |
| binary   | 二进制编码    |
| target   | 目标编码    |
| ordinal  | 序数编码   |
| label    | 标签编码     |
| frequency| 频率编码 |


### ⚠️ 您应该知道的事情

- 当使用`catc`模型进行分类任务，或使用`catr`模型进行回归任务时，`encode_method`必须为`None`。

- 当使用GBDT模型时，SHAP目前不支持多类分类任务。