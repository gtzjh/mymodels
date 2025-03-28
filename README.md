<div style="text-align: center;">

<h1>🚀 mymodels: Save Your Time ! Automated Interpretable Machine Learning Workflow</h1>

</div>

🔍 中文介绍请参见[此处](docs/20250328mymodels.md)。

In recent years, interpretable machine learning has become increasingly prominent in fields like geography, remote sensing, and urban planning. Machine learning models excel at capturing complex data relationships due to their powerful fitting capabilities. Meanwhile, interpretability frameworks based on game theory—such as SHapley Additive exPlanations (SHAP)—help demystify these "black-box" models. Interpretable machine learning provides valuable insights into ranking feature importance, revealing nonlinear response thresholds, and analyzing interaction relationships between factors.

However, the process of building an interpretable machine learning model is complex and time-consuming, expecially for the beginners. 
And there is a lack of a comprehensive and easy-to-use tool for excuting the interpretable machine learning workflow.
This project aims to automate this process and SAVE YOUR TIME !


## Prerequisites for Beginners

1. **✨ Python Proficiency**

    Recommended resource:

    - [Python tutorial on W3SCHOOL](https://www.w3schools.com/python/default.asp)
    
    - [Liao Xuefeng's Python Tutorial](https://liaoxuefeng.com/books/python/introduction/index.html)

    Essential contents:
    - Basic
    - Object-oriented Development (OOP)
    - Some commonly used Python built-in modules
    
    **DO REMEMBER**: Make a practical demo project after you finish the above learning to enhance what you have learned (i.e., a tiny web crawler). [Here is one of my practice projects](https://github.com/gtzjh/WundergroundSpider)

2. **✨ Machine Learning Fundamentals**

    [Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) provides essential theoretical foundations.

3. **🛠️ Technical Skills**

    - Environment management with conda/pip
    - Terminal/Command Line proficiency
    - Version control with Git ([Git学习资源](https://github.com/gtzjh/learngit))

## Environment Setup (Windows, same for Linux and macOS)

**Requirements**:
- Python 3.10
- 1.75 GB available disk space

The following packages will be installed:
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

Create the environment

```bash
conda env create -f requirement.yml -n mymodels -y
```

Activate environment

```base
conda activate mymodels
```

## Usage

We take `run_titanic.py` as an example:

### Code Explanations

#### Import the package.

```python
from mymodels.pipeline import MyPipeline
```

#### Construct an object for workflow. 

The created instance here named mymodel.

- `results_dir`: Directory path where your results will be stored. Accepts either a string or pathlib.Path object. The directory will be created if it doesn't exist.

- `random_state`: Random seed for the entire pipeline (data splitting, model tuning, etc.). (Default is 0)

- `show`: Whether to display the figure on the screen. (Default is `False`)

- `plot_format`: Output format for figures. (Default is jpg)

- `plot_dpi`: Controlling the resolution of output figures. (Default is 500)

```python
mymodel = MyPipeline(
    results_dir = "results/titanic",
    random_state = 0,
    show = False,
    plot_format = "jpg",
    plot_dpi = 500
)
```

#### Load data

- `file_path`: In which the data you want to input. **.csv format is mandatory**. 

- `y`: The target you want to predict. A `str` object represented column name or a `int` object represented the column index are both acceptable.

- `x_list`: A `list` object (or a `tuple` object) of the independent variables. Each element in `list` (or `tuple`) must be a `str` object represented column name or a `int` object represented the column index.

- `test_ratio`: The proportion of test data. (Default is 0.3)

- `inspect`: Whether to display the y column or the independent variables you chose in the terminal. (Default is `True`)

```python
mymodel.load(
    file_path = "data/titanic.csv",
    y = "Survived",
    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
    test_ratio = 0.3,
    inspect = False
)
```

#### Execute the optimization

- `model_name`: the model you want to use. In this example, `xgbc` represented XGBoost classifier, other model name like `catr` means CatBoost regressor. A full list of model names representing different models and tasks can be found at the end.

- `cat_features`: A `list` (or a `tuple`) of categorical features to specify for model. A `list` (or a `tuple`) of `str` representing the column names or `int` representing the index of column are both acceptable. (Default is `None`)

- `encode_method`: A `str` object representing the encode method, or a `list` (or a `tuple`) of encode methods are both acceptable.

  If the `cat_features` is presented, and the `model_name` is not `catr` or `catc`, then the `encode_method` must be presented

  If a single encode method is presented (like below, only `onehot` encode method is presented), it will be implemented to all categorical features (as listed in `encode_method`); 

  If a `list` (or a `tuple`) of encode methods is presented, i.e. `["onehot", "binary", "target"]`, they will be implemented to the three categorical features respectively.

  A full list of supported encode methods can be found at the end.

  (Default is `None`)

- `cv`: Cross-validation in the tuning process. (Default is 5)

- `trials`: How many trials in the Bayesian tuning process (Based on [Optuna](https://optuna.org/)). (Default is 50)

- `n_jobs`: How many cores will be used in the cross-validation process. It's recommended to use the same value as `cv`. (Default is 5)

- `plot_optimization`: Whether to display the tuning process. Default is `True`. A figure named `optimization_history` will be output in the results directory. (Default is `True`)


```python
mymodel.optimize(
    model_name = "xgbc",
    cat_features = ["Pclass", "Sex", "Embarked"],
    encode_method = "onehot",
    cv = 5,
    trials = 10,
    n_jobs = 5,
    plot_optimization = True
)
```

Evaluate the model's accuracy.

The accuracy results will be output to the directory you defined above:

- A `.yml` file named `accuracy` will document the results of model's accuracy.

- A figure named `roc_curve_plot` document the classification accuracy.

- Or a figure named `accuracy_plot` (it is a scatter plot) for regression task.


```python
mymodel.evaluate()
```

Explain the model using SHAP (SHapley Additive exPlanations):

`sample_train_k`: Sampling the samples in the training set for **background value calculation**. Default `None`, meaning that all data in the training set will be used. An integer value means an actual number of data, while a float (i.e., 0.5) means the proportion in the training set for it. (Default is `None`)

`sample_test_k`: Similar meaning to the `sample_train_k`. The test set will be implemented for **SHAP value calculation**. (Default is `None`)

The figures (Summary plot, Dependence plots) will be output to the directory you defined above.


```python
mymodel.explain(
    sample_train_k = 50,
    sample_test_k = 50,
)
```

Run the code

```bash
python run_titanic.py
```

### Full Code

```python
from mymodels.pipeline import MyPipeline

def main():
    mymodels = MyPipeline(
        results_dir = "results/titanic",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 500
    )
    mymodels.load(
        file_path = "data/titanic.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        test_ratio = 0.3,
        inspect = False
    )
    mymodels.optimize(
        model_name = "xgbc",
        cat_features = ["Pclass", "Sex", "Embarked"],
        encode_method = "onehot",
        cv = 5,
        trials = 10,
        n_jobs = 5,
        plot_optimization = True
    )
    mymodels.evaluate()
    mymodels.explain(
        sample_train_k = 50,
        sample_test_k = 50,
    )

    return None

if __name__ == "__main__":
    main()
```


## Other Examples

- `run_housing.py`: Regression task  
  Dataset source: [Kaggle Housing Data](https://www.kaggle.com/datasets/jamalshah811/housingdata)

- `run_obesity.py`: Multi-class classification  
  Dataset source: [Obesity Risk Dataset](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)

- `run_titanic.py`: Binary classification  
  Dataset source: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)


## Supplementary Information

### Supported Models

Click the link to see the official documentation.

#### For Regression Tasks
| `model_name` | Models|
|------------|-------|
| svr        | [Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) |
| knr        | [K-Nearest Neighbors Regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) |
| mlpr       | [Multi-Layer Perceptron Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) |
| dtr        | [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) |
| rfr        | [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) |
| gbr        | [Gradient Boosted Decision Trees (GBDT) Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) |
| abr        | [AdaBoost Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) |
| xgbr       | [XGBoost Regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html) |
| lgbmr      | [LightGBM Regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) |
| catr       | [CatBoost Regressor](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor) |

#### For Classification Tasks

| `model_name` | Models|
|------------|-------|
| svc        | [Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| knc        | [K-Nearest Neighbors Classification](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
| mlpc       | [Multi-Layer Perceptron Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) |
| dtc        | [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) |
| rfc        | [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| gbc        | [Gradient Boosted Decision Trees (GBDT) Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) |
| abc        | [AdaBoost Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) |
| xgbc       | [XGBoost Classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html) |
| lgbmc      | [LightGBM Classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) |
| catc       | [CatBoost Classifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier) |

**Note:** SHAP currently doesn't support multi-class classification tasks when using GBDT models.


### Supported Encoding Methods

| `encode_method`     | Description                          |
|------------|--------------------------------------|
| onehot   | One-hot encoding                     |
| binary   | Binary encoding                      |
| target   | Target/mean encoding                 |
| ordinal  | Ordinal encoding                     |
| count    | Count encoding                       |
| frequency| Frequency encoding                   |


