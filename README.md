<div style="text-align: center;">

<h1 align="center">🚀 mymodels : Build an efficient interpretable machine learning workflow</h1>

</div>

Feel free to contact me: [gtzjh86@outlook.com](mailto:gtzjh86@outlook.com)

**4/2/2025: Support for <code>LabelEncoder</code>, <code>TargetEncoder</code>, and <code>FrequencyEncoder</code> is under developing.**


## 🤔 Why `mymodels`?

Interpretable machine learning has gained significant prominence across various fields. Machine learning models are valued for their robust capability to capture complex relationships within data through sophisticated fitting algorithms. Complementing these models, interpretability frameworks provide essential tools for revealing such "black-box" models. These interpretable approaches deliver critical insights by ranking feature importance, identifying nonlinear response thresholds, and analyzing interaction relationships between factors. 

Project `mymodels`, is targeting on building a **tiny, user-friendly, and efficient** workflow, for the scientific researchers and students who are seeking to implement interpretable machine learning in their their research works.

## 👨‍🎓 Prerequisites for Beginners

1. **Python Proficiency**

    - [Python tutorial on W3SCHOOL](https://www.w3schools.com/python/default.asp)
    
    - [Liao Xuefeng's Python Tutorial](https://liaoxuefeng.com/books/python/introduction/index.html)
    
    > **DO REMEMBER**: Make a practical demo project after you finish the above learning to enhance what you have learned (i.e., a tiny web crawler). [Here is one of my practice projects](https://github.com/gtzjh/WundergroundSpider)

2. **Machine Learning Fundamentals**

    - [Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) provides essential theoretical foundations.

3. **Technical Skills**

    - Environment management with conda/pip
    - Terminal/Command Line proficiency
    - Version control with Git ([My note about Git](https://github.com/gtzjh/learngit))
  
> The above recommended tutorials are selected based solely on personal experience.

## 🛠️ Environment Setup

**Supported platforms**:

- Windows (X86) - Tested on Windows 10/11
- Linux (X86) - Tested on WSL2.0 (Ubuntu)
- macOS (ARM) - Tested on Apple Silicon (M1)

**Requirements**:
- Python 3.10.X
- 1.75 GB available disk space

**Create environment**

```bash
conda env create -f requirement.yml -n mymodels -y
```

**Activate**

```bash
conda activate mymodels
```

## :point_right: Try

**Try the Titanic demon first**

- Binary classification: [run_titanic.ipynb](run_titanic.ipynb)

  > Dataset source: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

**And try other demons**

- Multi-class classification: [run_obesity.ipynb](run_obesity.ipynb)

  > Dataset source: [Obesity Risk Dataset](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)

- Regression task: [run_housing.ipynb](run_housing.ipynb)

  > Dataset source: [Kaggle Housing Data](https://www.kaggle.com/datasets/jamalshah811/housingdata)

## ⚠️ The Users Should Know

- The project **is not suitable** for time-series tasks.

- The hyperparameters shown in `models.py` are only for demonstration purposes. Users should try different hyperparameters in their actual applications to ensure the robustness of their results.

- The `random_state` is set to `0` for demonstration purposes only. Users should try different `random_state` in their actual applications to ensure the robustness of their results.

- The explanation in this project is currently based on [SHapley Additive exPlanations (SHAP)](https://shap.readthedocs.io/en/latest/index.html), Other explanation methods are coming soon. 

- Note that explanations may not always be meaningful for real-world tasks, especially after data engineering. Users are solely responsible for validating the appropriateness of explanation methods for their specific use cases. **It's strongly recommended to compare the explanation results before and after data engineering**.

- The Partial Dependence Plot (PDP) is not supported for classification tasks currently.


## 📚 Supplementary Information

### Supported Models

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


### Supported Encoding Methods

| `encode_method` | Description   |
|------------|--------------------|
| onehot     | One-hot encoding   |
| binary     | Binary encoding    |
| ordinal    | Ordinal encoding   |
| label (coming soon)      | Label encoding     |
| frequency (coming soon)  | Frequency encoding |
| target (coming soon)     | Target encoding    |

**This project is intended solely for scientific reference. It may contain calculation errors or logical inaccuracies. Users are responsible for verifying the accuracy of the results independently, and the author shall not be held liable for any consequences arising from the use of this code.**
