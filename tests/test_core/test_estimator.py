import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



import pandas as pd
import pytest
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import yaml
from sklearn.datasets import make_regression, make_classification
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from functools import partial
from optuna.samplers import TPESampler



from mymodels.core import MyEstimator


# --------------------------
# 模型功能基准测试
# --------------------------

# Create a regression dataset
X_reg, y_reg = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=0)
X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
y_reg = pd.Series(y_reg, name='target')

X_binary, y_binary = make_classification(n_samples=100, n_features=10, n_classes=2, n_informative=5, random_state=0)
X_binary = pd.DataFrame(X_binary, columns=[f'feature_{i}' for i in range(X_binary.shape[1])])
y_binary = pd.Series(y_binary, name='target')

X_multi, y_multi = make_classification(n_samples=100, n_features=10, n_classes=6, n_informative=5, random_state=0)
X_multi = pd.DataFrame(X_multi, columns=[f'feature_{i}' for i in range(X_multi.shape[1])])
y_multi = pd.Series(y_multi, name='target')


# Construct the optimization
class TestEstimator:
    def __init__(self, model_name: str): 
        self.model_name = model_name

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

        estimator = MyEstimator()
        estimator.load(model_name=self.model_name)
        self._model_obj = estimator.empty_model_object
        param_space = estimator.param_space
        static_params = estimator.static_params

        # Create a study
        _study = optuna.create_study(
            direction = "maximize",
            sampler = TPESampler(seed=0),
        )

        # Optimization
        _study.optimize(
            partial(self._objective, _param_space=param_space, _static_params=static_params),
            n_trials = 5,
            n_jobs = 1,
            show_progress_bar = True
        )

        return None

    def _objective(self, trial, _param_space, _static_params) -> float:
        """Objective function for the Optuna study."""

        # Get parameters for model training
        # Make the param immutable
        param = {
            **{k: v(trial) for k, v in _param_space.items()},
            **_static_params
        }

        # Data split
        _x_train, _x_test, _y_train, _y_test = train_test_split(
            self.x_train, self.y_train
        )

        # Create a validator
        _validator = clone(self._model_obj(**param))
        _validator.fit(_x_train, _y_train)
        
        return _validator.score(_x_test, _y_test)


def test_regression_models_optimization():
    """测试回归模型优化功能
    
    测试所有回归模型是否能成功完成trial，使用回归数据集
    """
    regression_models = ['lr', 'svr', 'knr', 'mlpr', 'dtr', 'rfr', 'gbdtr', 'adar', 'xgbr', 'lgbr', 'catr']
    
    for model_name in regression_models:
        print(f"Testing regression model: {model_name}")
        try:
            test_estimator = TestEstimator(model_name=model_name)
            test_estimator.fit(X_reg, y_reg)
            print(f"Model {model_name} completed successfully")
        except Exception as e:
            pytest.fail(f"Model {model_name} failed with error: {str(e)}")

def test_binary_classification_models_optimization():
    """测试二分类模型优化功能
    
    测试所有分类模型是否能成功完成trial，使用二分类数据集
    """
    classification_models = ['lc', 'svc', 'knc', 'mlpc', 'dtc', 'rfc', 'gbdtc', 'adac', 'xgbc', 'lgbc', 'catc']
    
    for model_name in classification_models:
        print(f"Testing binary classification model: {model_name}")
        try:
            test_estimator = TestEstimator(model_name=model_name)
            test_estimator.fit(X_binary, y_binary)
            print(f"Model {model_name} completed successfully")
        except Exception as e:
            pytest.fail(f"Model {model_name} failed with error: {str(e)}")

def test_multiclass_classification_models_optimization():
    """测试多分类模型优化功能
    
    测试所有分类模型是否能成功完成trial，使用多分类数据集
    """
    classification_models = ['lc', 'svc', 'knc', 'mlpc', 'dtc', 'rfc', 'gbdtc', 'adac', 'xgbc', 'lgbc', 'catc']
    
    for model_name in classification_models:
        print(f"Testing multiclass classification model: {model_name}")
        try:
            test_estimator = TestEstimator(model_name=model_name)
            test_estimator.fit(X_multi, y_multi)
            print(f"Model {model_name} completed successfully")
        except Exception as e:
            pytest.fail(f"Model {model_name} failed with error: {str(e)}")


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
