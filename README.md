<div style="text-align: center;">

<h1 align="center">🚀 mymodels 🚀 : Save Your Time ! Efficient Interpretable Machine Learning Workflow</h1>

</div>

**USERS MUST THOROUGHLY REVIEW THIS DOCUMENTATION BEFORE IMPLEMENTING THE PROJECT.**

Feel free to contact me: [gtzjh86@outlook.com](mailto:gtzjh86@outlook.com)

<div style="border: 2px solid #cc0000; padding: 10px; background-color: rgba(255, 204, 204, 0.6); border-radius: 5px;">
    <p style="color: #333333;">⚠️ <strong>Note:</strong> Support for <code>LabelEncoder</code>, <code>TargetEncoder</code>, and <code>FrequencyEncoder</code> is currently unavailable. However, these features are under active development, with an updated version expected to be released within the next month. In the meantime, users can implement their own custom data engineering pipelines to incorporate these encoding methods.</p>
</div>



## 🤔 Why `mymodels`?

Interpretable machine learning has gained significant prominence across various fields including geography, remote sensing, and urban planning. Machine learning models are valued for their robust capability to capture complex relationships within data through sophisticated fitting algorithms. Complementing these models, interpretability frameworks based on game theory—such as SHapley Additive exPlanations (SHAP)—provide essential tools for revealing such "black-box" models. These interpretable approaches deliver critical insights by ranking feature importance, identifying nonlinear response thresholds, and analyzing interaction relationships between factors. 

Despite these advantages, implementing interpretable machine learning workflows remains a complex and time-intensive process, particularly for those new to the field. There exists a notable gap in comprehensive, user-friendly tooling for executing these workflows efficiently.

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

### Activate environment

Run the following command in terminal:

```bash
conda activate mymodels
```

### Try These Notebooks

- Binary classification: [run_titanic.ipynb](run_titanic.ipynb)

  Dataset source: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

- Regression task: [run_housing.ipynb](run_housing.ipynb)

  Dataset source: [Kaggle Housing Data](https://www.kaggle.com/datasets/jamalshah811/housingdata)

- Multi-class classification: [run_obesity.ipynb](run_obesity.ipynb)

  Dataset source: [Obesity Risk Dataset](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)


## 📚 Supplementary Information

### 🛠️ Supported Models

*Click the following links in the second column to see the official documentation.*

#### For Regression Tasks
| `model_name` | Models|
|------------|-------|
| lr         | [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) |
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
| lc         | [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) |
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

| `encode_method` | Description   |
|------------|--------------------|
| onehot     | One-hot encoding   |
| binary     | Binary encoding    |
| ordinal    | Ordinal encoding   |
| label (coming soon)      | Label encoding     |
| frequency (coming soon)  | Frequency encoding |
| target (coming soon)     | Target encoding    |


## ⚠️ **STATEMENTS**

Project `mymodels`, **IS NOT, and WILL NEVER BE**, a framework including all the models and methods about interpretable machine learning. 

It's targeting on building a **tiny, user-friendly, and efficient toolkit**, for the scientific researchers or (master/PhD) students who are seeking to implement interpretable machine learning in their work efficiently.

The developer will try to meet the common needs of the target users to the best extent, but several statements should be clarified:

- This **open-source** project is in **active** development, and is not yet ready for production use. The software is provided without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

- Users must independently verify the suitability and safety of the software for their specific use case. Any application of the software in safety-critical systems is expressly prohibited.

- Third-party dependencies are used as-is. The project does not guarantee the security, reliability, or compatibility of any third-party libraries.

- This software is subject to export control laws and regulations. Users are responsible for compliance with all applicable export and import regulations.


## ⚠️ The Users Should Know

- The project **is not suitable** for time-series tasks.

- The hyperparameters shown in `models.py` are only for demonstration purposes. Users should try different hyperparameters in their actual applications to ensure the robustness of their results.

- The `random_state` is set to `0` for demonstration purposes only. Users should try different `random_state` in their actual applications to ensure the robustness of their results.

- The explanation in this project is currently based on [SHapley Additive exPlanations (SHAP)](https://shap.readthedocs.io/en/latest/index.html), Other explanation methods are coming soon. 

- Note that explanations may not always be meaningful for real-world tasks, especially after data engineering. Users are solely responsible for validating the appropriateness of explanation methods for their specific use cases. **It's strongly recommended to compare the explanation results before and after data engineering**.

- The Partial Dependence Plot (PDP) is not supported for classification tasks currently.
