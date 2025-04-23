import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


"""
我正在构建一个机器学习的工作流，现在正在构建其中的一个模块，这个模块的功能是获取指定的模型。
这个模块的调用方式如下：
>>> from mymodels import MyEstimator
>>> estimator = MyEstimator()
>>> estimator.load(model_name='lr')

具体地，MyEstimator类的接口如下：
cat_features (list[str] | tuple[str] | None): The categorical features to use for the CatBoost ONLY.
model_configs_path (str): The path to the model configs file. (Default: 'model_configs.yml' in the root directory)

其中，model_cofigs_path是一个模型的配置文件，它包含每一个模型的参数配置，它的每个参数大概是这样的：

```yaml
# HOW TO CUSTOMIZE YOUR OWN MODEL
# ----------------------------
# This configuration file allows you to define machine learning models with their hyperparameters.
# To add a new model, follow this structure:
#
# model_key:  # Short name for your model (e.g., 'my_classifier')
#   IMPORTS:
#     module: package.submodule  # Python module where the model class is located
#     class: ClassName  # Name of the model class to import
#   PARAM_SPACE:  # Parameters to tune during optimization
#     parameter_name:
#       type: categorical|float|integer  # Parameter type
#       values: [val1, val2] or {min: min_val, max: max_val, step: step_val, log: bool}
#   STATIC_PARAMS:  # Fixed parameters that won't be tuned
#     parameter_name: value
#   SHAP_EXPLAINER_TYPE: linear|kernel|tree|permutation  # Type of SHAP explainer to use
#   SAVE_TYPE: joblib|xgboost|lightgbm|catboost  # Method used to save the model
#
# PARAMETER TYPES:
# - categorical: discrete set of choices (list of values)
# - float: continuous numerical value (min/max range with optional step)
#   - log: true makes the sampling logarithmic
# - integer: whole numbers (min/max range with optional step)
#
# SHAP EXPLAINER TYPES:
# - linear: for linear models (faster for LinearRegression, LogisticRegression)
# - kernel: model-agnostic explainer (can be used with any model, but slower)
# - tree: optimized for tree-based models (RandomForest, XGBoost, etc.)
# - permutation: feature permutation-based approach for black-box models
#
# SAVE TYPES:
# - joblib: default for scikit-learn models
# - xgboost: for XGBoost models (saves as .json)
# - lightgbm: for LightGBM models (saves as .txt)
# - catboost: for CatBoost models (saves as .cbm)
```

当然，用户也可以按照以上的方式，定义自己的模型和对应的参数



通过调用`load()`的方法，它返回一个MyEstimator的对象：

其中，load() 方法的接口是这样的：
Args:
    model_name (str): The name of the model to load.

Returns:
    self: The instance of the `MyModels` class

返回的内容是这样的

Attributes:
    empty_model_object: An empty model object.
    param_space: The parameter space for Optuna tuning.
    static_params: The static parameters.
    shap_explainer_type: The type of SHAP explainer to use.
    optimal_model_object: The optimal model object. (After optimization)
    optimal_params: The optimal parameters. (After optimization)
    

需要注意，model_name这个参数，要与上面给出的yaml设置的key一致，比如像上面的随机森林模型，如果我想要调用它，那我就要输入 `rfr`
也就是，在load方法这里，还会需要检查用户输入的model_name，是否在指定的yaml文件的key中被找到。


接下来，我需要为这个模块，编写单元测试，首先是基准测试，需要确保模块能被正常调用，同时得到的模型能够被用来调优
所以你需要构建基准单元测试，涵盖所有模型，并且包含回归、二分类、多分类任务，并且创建指定的数据集，用来进行模拟测试
然后你要进行异常测试，你需要列出这里面可能存在的输入，确保模块都能捕捉到非法的输入，要涵盖机器学习任务中任何可能的情况，以及关键的边界条件。
我要用pytest
每个测试用例都应该用google python的风格编写注释，并且注明这个测试的内容，以及对期待的输出进行校验

对于PARAM_SPACE获取的参数空间，最重要的是把这个参数空间放到optuna study中，看看能否被正常调用，能否正常执行参数调优的过程
因此你要删掉原本的基准测试的代码，然后导入optuna包，然后尝试对每个模型构建一个study，然后把参数空间放进去，看看能否正常运行

此外，你需要涵盖的模型类型包括一下，我以一个html表格的形式给出


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
"""


import pytest
import optuna
import yaml
from pathlib import Path
from sklearn.datasets import make_regression, make_classification
from mymodels import MyEstimator


# --------------------------
# 基准测试
# --------------------------






# --------------------------
# 异常测试
# --------------------------
def test_invalid_model_name():
    """测试无效模型名称异常处理
    
    Expect:
        - 输入不存在的模型名称时抛出ValueError
    """
    with pytest.raises(ValueError, match="Model 'invalid_model' not found"):
        MyEstimator().load(model_name='invalid_model')

def test_malformed_config_file(tmp_path):
    """测试格式错误的配置文件
    
    Args:
        tmp_path: pytest临时目录fixture
    
    Expect:
        - 当配置文件缺少必要字段时抛出KeyError
    """
    # 创建损坏的配置文件
    bad_config = {'lr': {'IMPORTS': {'module': 'sklearn.linear_model'}}}  # 缺少class定义
    config_path = tmp_path / "bad_config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(bad_config, f)
    
    with pytest.raises(KeyError):
        MyEstimator(model_configs_path=str(config_path)).load(model_name='lr')

def test_invalid_param_space(tmp_path):
    """测试无效参数空间定义
    
    Args:
        tmp_path: pytest临时目录fixture
    
    Expect:
        - 当参数空间定义包含非法类型时抛出ValueError
    """
    # 创建包含非法参数类型的配置文件
    bad_config = {
        'lr': {
            'IMPORTS': {
                'module': 'sklearn.linear_model',
                'class': 'LinearRegression'
            },
            'PARAM_SPACE': {
                'normalize': {'type': 'invalid_type', 'values': [True, False]}
            }
        }
    }
    config_path = tmp_path / "bad_param_config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(bad_config, f)
    
    with pytest.raises(ValueError, match="Unsupported parameter type"):
        MyEstimator(model_configs_path=str(config_path)).load(model_name='lr')

def test_missing_config_file():
    """测试缺失配置文件异常处理
    
    Expect:
        - 当配置文件不存在时抛出FileNotFoundError
    """
    with pytest.raises(FileNotFoundError):
        MyEstimator(model_configs_path='non_existent.yml').load(model_name='lr')

# --------------------------
# 模型特殊属性测试
# --------------------------
def test_catboost_categorical_features():
    """测试CatBoost分类特征处理
    
    Expect:
        - 当加载CatBoost模型时，正确传递categorical_features参数
    """
    estimator = MyEstimator(cat_features=['feature1', 'feature2']).load(model_name='catr')
    assert estimator.static_params["cat_features"] == ['feature1', 'feature2']

def test_xgboost_save_format():
    """测试XGBoost保存格式配置
    
    Expect:
        - XGBoost模型的SAVE_TYPE应为xgboost
    """
    estimator = MyEstimator().load(model_name='xgbr')
    assert estimator.save_type == 'json', "XGBoost模型保存格式配置错误"