# Welcome to mymodels!

Interpretable machine learning has gained significant prominence across various fields. Machine learning models are valued for their robust capability to capture complex relationships within data through sophisticated fitting algorithms. Complementing these models, interpretability frameworks provide essential tools for revealing such "black-box" models. These interpretable approaches deliver critical insights by ranking feature importance, identifying nonlinear response thresholds, and analyzing interaction relationships between factors. 

Project `mymodels`, is targeting on building a **tiny, user-friendly, and efficient** workflow, for the scientific researchers and students who are seeking to implement interpretable machine learning in their their research works.


# Supported encode methods

- Onehot
- Ordinal
- Binary

# Supported models

<table>
  <thead>
    <tr>
      <th colspan="2">Regression</th>
      <th colspan="2">Classification</th>
    </tr>
    <tr>
      <th>model_name</th>
      <th>Models</th>
      <th>model_name</th>
      <th>Models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>lr</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Linear Regression</a></td>
      <td>lc</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Logistic Regression</a></td>
    </tr>
    <tr>
      <td>svr</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html">Support Vector Regression</a></td>
      <td>svc</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Support Vector Classification</a></td>
    </tr>
    <tr>
      <td>knr</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">K-Nearest Neighbors Regression</a></td>
      <td>knc</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">K-Nearest Neighbors Classification</a></td>
    </tr>
    <tr>
      <td>mlpr</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html">Multi-Layer Perceptron Regressor</a></td>
      <td>mlpc</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">Multi-Layer Perceptron Classifier</a></td>
    </tr>
    <tr>
      <td>dtr</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html">Decision Tree Regressor</a></td>
      <td>dtc</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Decision Tree Classifier</a></td>
    </tr>
    <tr>
      <td>rfr</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">Random Forest Regressor</a></td>
      <td>rfc</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Random Forest Classifier</a></td>
    </tr>
    <tr>
      <td>gbdtr</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">Gradient Boosted Decision Trees (GBDT) Regressor</a></td>
      <td>gbdtc</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">Gradient Boosted Decision Trees (GBDT) Classifier</a></td>
    </tr>
    <tr>
      <td>adar</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html">AdaBoost Regressor</a></td>
      <td>adac</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">AdaBoost Classifier</a></td>
    </tr>
    <tr>
      <td>xgbr</td>
      <td><a href="https://xgboost.readthedocs.io/en/latest/python/python_api.html">XGBoost Regressor</a></td>
      <td>xgbc</td>
      <td><a href="https://xgboost.readthedocs.io/en/latest/python/python_api.html">XGBoost Classifier</a></td>
    </tr>
    <tr>
      <td>lgbr</td>
      <td><a href="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html">LightGBM Regressor</a></td>
      <td>lgbc</td>
      <td><a href="https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html">LightGBM Classifier</a></td>
    </tr>
    <tr>
      <td>catr</td>
      <td><a href="https://catboost.ai/en/docs/concepts/python-reference_catboostregressor">CatBoost Regressor</a></td>
      <td>catc</td>
      <td><a href="https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier">CatBoost Classifier</a></td>
    </tr>
  </tbody>
</table>


# Recommanded usage

## For running on the multi-cores platform

To ensure system stability and optimal performance, thread management must adhere to one of the following protocols:

1.  **Eliminate Nesting:** Refactor the process to remove nested parallelism. Parallel execution should be restricted to the single, most appropriate layer.
2.  **Constrain Thread Product:** If nested parallelism is required, the product of the thread counts at all levels must be less than or equal to ($\le$) the total number of available logical cores.

For example:

* **System Specification:** 32 available logical cores.
* **Process Architecture:**
    * Level 1: Main process (executing 5-fold cross-validation).
    * Level 2: Model (internal parallelism).
* **Recommended Configuration:**
    * Set Level 1 parallelism = 5
    * Set Level 2 (model) maximum parallelism = 6
* **Result:** The total concurrent threads ($5 \times 6 = 30$) remains within the system capacity of 32 cores.

* **Performance Tuning:** Initial testing indicates that performance is optimized when the main process parallelism (Level 1) is set equal to the number of cross-validation folds.
* **Resource Monitoring:** Concurrently monitor memory utilization. Increasing parallelism introduces significant memory overhead, which must be managed.
* **Validation:** It is standard procedure to profile multiple nested parallelism combinations to identify the optimal configuration for the specific workload.


# The users should know

- This project is intended solely for scientific reference. It may contain calculation errors or logical inaccuracies. Users are responsible for verifying the accuracy of the results independently, and the author shall not be held liable for any consequences arising from the use of this code.

- Due to the developer's limited personal capabilities and time constraints, the project may inevitably have shortcomings. We sincerely welcome fellow professionals to provide critiques and suggestions for improvement.

- Note that explanations may not always be meaningful for real-world tasks, especially after data engineering. Users are solely responsible for validating the appropriateness of explanation methods for their specific use cases.

- The project is not suitable for time-series tasks.

- The hyperparameters shown in `models_configs.yml` are only for demonstration purposes. Users should try different hyperparameters in their actual applications to ensure the robustness of their results.

- The `random_state` is set to `0` for demonstration purposes only. Users should try different `random_state` in their actual applications to ensure the robustness of their results.

- The explanation in this project is currently based on [SHAP](https://shap.readthedocs.io/en/latest/index.html) and PDP (Partial Dependence Plot). Other explanation methods (i.e., ALE, FI) are under supporting.
