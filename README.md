<div style="text-align: center;">

<h1 align="center">🚀 mymodels 🚀 : Save Your Time ! Efficient Interpretable Machine Learning Workflow</h1>

</div>


**关注公众号：👉GT地学志👈 获取项目更新。**

<img src="docs/qrcode.jpg" alt="mymodels" width="130">

Feel free to contact me: [zhongjh86@outlook.com](mailto:zhongjh86@outlook.com)


## **STATEMENTS**

- The **open-source** project is under **very active** development, and is not yet ready for production use. The software is provided without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

- Users must independently verify the suitability and safety of the software for their specific use case. Any application of the software in safety-critical systems is expressly prohibited.

- Third-party dependencies are used as-is. The project does not guarantee the security, reliability, or compatibility of any third-party libraries.

- This software is subject to export control laws and regulations. Users are responsible for compliance with all applicable export and import regulations.

- In this project, the `random_state` is set to `0` for demonstration purposes only. Users should try different `random_state` in their actual applications to ensure the robustness of their results.

- This project **is not suitable** for time-series tasks.

- The explanation of this project is based on the [SHapley Additive exPlanations (SHAP)](https://shap.readthedocs.io/en/latest/index.html) framework. Users are solely responsible for validating the appropriateness of explanation methods for their specific use cases. Other explanation methods are comming soon.




## 🎯 Who is `mymodels` for?

Scientific researchers or (master/PhD) students seeking to implement interpretable machine learning in their work, but requiring minimal time investment in coding infrastructure. (e.g., the medical students who have to spend most of their time on clinical work).

The ideal solution for those who want to focus on domain-specific insights rather than practical implementation details.


## 🤔 Why `mymodels`?

Interpretable machine learning has gained significant prominence across various fields including geography, remote sensing, and urban planning. Machine learning models are valued for their robust capability to capture complex relationships within data through sophisticated fitting algorithms. Complementing these models, interpretability frameworks based on game theory—such as SHapley Additive exPlanations (SHAP)—provide essential tools for revealing such "black-box" models. These interpretable approaches deliver critical insights by ranking feature importance, identifying nonlinear response thresholds, and analyzing interaction relationships between factors.

Despite these advantages, implementing interpretable machine learning workflows remains a complex and time-intensive process, particularly for those new to the field. There exists a notable gap in comprehensive, user-friendly tooling for executing these workflows efficiently. The mymodels project addresses this gap by automating the interpretable machine learning process, significantly reducing implementation time while maintaining analytical rigor.


## 👨‍🎓 Prerequisites for Beginners

1. 💡 **Python Proficiency**

    Recommended resource:

    - [Python tutorial on W3SCHOOL](https://www.w3schools.com/python/default.asp)
    
    - [Liao Xuefeng's Python Tutorial](https://liaoxuefeng.com/books/python/introduction/index.html)

    Essential contents:
    - Basic
    - Object-oriented Development (OOP)
    - Some commonly used Python built-in modules
    
    > **DO REMEMBER**: Make a practical demo project after you finish the above learning to enhance what you have learned (i.e., a tiny web crawler). [Here is one of my practice projects](https://github.com/gtzjh/WundergroundSpider)

2. 💡 **Machine Learning Fundamentals**

    [Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) provides essential theoretical foundations.

3. 💡 **Technical Skills**

    - Environment management with conda/pip
    - Terminal/Command Line proficiency
    - Version control with Git ([My note of learning Git](https://github.com/gtzjh/learngit))

## 🛠️ Environment Setup

Supported platforms:

- Windows (X86) [Tested on Windows 10/11]
- Linux (X86) [Tested on WSL2.0 (Ubuntu)]
- macOS (ARM) [Tested on Apple Silicon (M1)]

### Create environment

**Requirements**:
- Python 3.10.X
- 1.75 GB available disk space

Run the following command in terminal:

```bash
conda env create -f requirement.yml -n mymodels -y
```

> You can also create the environment manually using `pip` according to the `requirement.yml` file:

---

### Activate environment

Run the following command in terminal:

```bash
conda activate mymodels
```

---

## 🚀 How to Use

We take `run_titanic.py` as an example:


### Import the package

#### Example

```python
from mymodels.pipeline import MyPipeline
```

---

### Construct an object for workflow

#### Parameters

- **results_dir**: Directory path where your results will be stored. Accepts either a string or pathlib.Path object. The directory will be created if it doesn't exist.

- **random_state**: Random seed for the entire pipeline (data splitting, model tuning, etc.). (Default is 0)

- **show**: Whether to display the figure on the screen. (Default is `False`)

- **plot_format**: Output format for figures. (Default is jpg)

- **plot_dpi**: Controlling the resolution of output figures. (Default is 500)

#### Example

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

### Load data

#### Parameters

- **file_path**: In which the data you want to input. **.csv format is mandatory**. 

- **y**: The target you want to predict. A `str` object represented column name or a `int` object represented the column index are both acceptable.

- **x_list**: A `list` object (or a `tuple` object) of the independent variables. Each element in `list` (or `tuple`) must be a `str` object represented column name or a `int` object represented the column index.

- **index_col**: An `int` object or `str` object representing the index column. (Default is `None`)

    > It's STRONGLY RECOMMENDED to set the index column if you want to output the raw data and the shap values. Also, it's acceptable to provide a `list` object (or a `tuple` object) for representing multiple index columns. 

- **test_ratio**: The proportion of test data. (Default is 0.3)

- **inspect**: Whether to display the y column or the independent variables you chose in the terminal. (Default is `True`)

#### Example

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

### Execute the optimization

#### Parameters

- **model_name**: the model you want to use. In this example, `xgbc` represented XGBoost classifier, other model name like `catr` means CatBoost regressor. A full list of model names representing different models and tasks can be found at the end.

- **cat_features**: A `list` (or a `tuple`) of categorical features to specify for model. A `list` (or a `tuple`) of `str` representing the column names or `int` representing the index of column are both acceptable. (Default is `None`)

- **encode_method**: A `str` object representing the encode method, or a `list` (or a `tuple`) of encode methods are both acceptable. (Default is `None`)

  If a single encode method is presented (like below, only `onehot` encode method is presented), it will be implemented to all categorical features (as listed in `encode_method`).

  If a `list` (or a `tuple`) of encode methods is presented, i.e. `["onehot", "binary", "target"]`, they will be implemented to the three categorical features respectively. Hence, the length of `encode_method` must be the same as the length of `cat_features`.

  > A full list of supported encode methods can be found at the end.


- **cv**: Cross-validation in the tuning process. (Default is 5)

- **trials**: How many trials in the Bayesian tuning process (Based on [Optuna](https://optuna.org/)). (Default is 50)

- **n_jobs**: How many cores will be used in the cross-validation process. It's recommended to use the same value as `cv`. (Default is 5)

- **optimize_history**: Whether to save the optimization history. (Default is `True`)

- **save_optimal_params**: Whether to save the best parameters. (Default is `True`)

- **save_optimal_model**: Whether to save the optimal model. (Default is `True`)

> Attention: When using the `catc` model for classification tasks, or `catr` model for regression tasks, the `encode_method` must be `None`. Users are responsible for ensuring proper configuration of model parameters.

#### Output

Several files will be output in the results directory:

- `params.yml` will document the best parameters.

- `mapping.json` will document the mapping relationship between the categorical features and the encoded features.

- `optimal-model.joblib` will save the optimal model from sklearn.

- `optimal-model.cbm` will save the optimal model from CatBoost.

- `optimal-model.txt` will save the optimal model from LightGBM.

- `optimal-model.json` will save the optimal model from XGBoost.

- `optimal-model.pkl` will save all types of optimal model for compatibility.

#### Example

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

### Evaluate the model's accuracy

#### Parameters

- **save_raw_data**: Whether to save the raw prediction data. Default is `True`.

#### Output

The accuracy results will be output to the directory you defined above:

- A `.yml` file named `accuracy` will document the results of model's accuracy.

- A figure named `roc_curve_plot` document the classification accuracy.

- Or a figure named `accuracy_plot` (it is a scatter plot) for regression task.

#### Example
```python
mymodel.evaluate(save_raw_data = True)
```

---

### Explain the model using SHAP (SHapley Additive exPlanations)

#### Parameters

- **select_background_data**: The data used for **background value calculation**. (Default is `"train"`)

    Default is `"train"`, meaning that all data in the training set will be used. `"test"` means that all data in the test set will be used. `"all"` means that all data in the training and test set will be used. 

- **select_shap_data**: The data used for **calculating SHAP values**. Default is `"test"`, meaning that all data in the test set will be used. `"all"` means that all data in the training and test set will be used. (Default is `"test"`)

- **sample_background_data_k**: Sampling the samples in the training set for **background value calculation**. (Default is `None`)
    
    Default `None`, meaning that all data in the training set will be used. An integer value means an actual number of data, while a float (i.e., 0.5) means the proportion in the training set for it. 

- **sample_shap_data_k**: Similar meaning to the `sample_background_data_k`. The test set will be implemented for **SHAP value calculation**. (Default is `None`)

- **output_raw_data**: Whether to save the raw data. Default is `False`.

> SHAP currently doesn't support multi-class classification tasks when using **GBDT** models. This limitation may affect the interpretability results and users should verify compatibility with their use case.

#### Output

The figures (Summary plot, Dependence plots) will be output to the directory you defined above.

#### Example

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

### Run the code

Tap `F5` to run in Debug mode in VSCode.

Or run the following command in terminal:

```bash
python run_titanic.py
```

---

### Full Code

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


## 🎯 Try These Examples

- `run_housing.py`: Regression task  
  Dataset source: [Kaggle Housing Data](https://www.kaggle.com/datasets/jamalshah811/housingdata)

- `run_obesity.py`: Multi-class classification  
  Dataset source: [Obesity Risk Dataset](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)

- `run_titanic.py`: Binary classification  
  Dataset source: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)


## 📚 Supplementary Information

### 🛠️ Required Packages

The following packages are required:
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


### 🛠️ Supported Models

*Click the following links in the second column to see the official documentation.*

#### For Regression Tasks
| `model_name` | Models|
|------------|-------|
| svr        | [Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) |
| knr        | [K-Nearest Neighbors Regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) |
| mlpr       | [Multi-Layer Perceptron Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) |
| dtr        | [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) |
| rfr        | [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) |
| gbdtr        | [Gradient Boosted Decision Trees (GBDT) Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) |
| adar        | [AdaBoost Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) |
| xgbr       | [XGBoost Regressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html) |
| lgbr      | [LightGBM Regressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) |
| catr       | [CatBoost Regressor](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor) |

#### For Classification Tasks

| `model_name` | Models|
|------------|-------|
| svc        | [Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| knc        | [K-Nearest Neighbors Classification](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
| mlpc       | [Multi-Layer Perceptron Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) |
| dtc        | [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) |
| rfc        | [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| gbdtc        | [Gradient Boosted Decision Trees (GBDT) Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) |
| adac        | [AdaBoost Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) |
| xgbc       | [XGBoost Classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html) |
| lgbc      | [LightGBM Classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) |
| catc       | [CatBoost Classifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier) |



### 🛠️ Supported Encoding Methods

| `encode_method` | Description |
|------------|------------------|
| onehot   | One-hot encoding   |
| binary   | Binary encoding    |
| target   | Target encoding    |
| ordinal  | Ordinal encoding   |
| label    | Label encoding     |
| frequency| Frequency encoding |