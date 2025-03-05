import pytest
from unittest import TestCase
import numpy as np
from _regressors import MyRegressors

# 测试用数据
X_sample = np.random.rand(100, 5)
y_sample = np.random.rand(100)
CAT_FEATURES = ['feat1', 'feat3']

class TestRegressors(TestCase):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.valid_models = ["svr", "knr", "mlp", "dt", "rf", "gbdt", "ada", "xgb", "lgb", "cat"]
        self.invalid_model = "invalid_model"
        self.non_cat_models = ["svr", "knr", "mlp", "dt", "rf", "gbdt", "ada"]
        self.cat_models = ["xgb", "lgb", "cat"]

    # 基础功能测试
    @pytest.mark.parametrize("model_name", [
        "svr", "knr", "mlp", "dt", "rf", 
        "gbdt", "ada", "xgb", "lgb", "cat"
    ])
    def test_model_initialization(self, model_name):
        """测试所有有效模型能否正确初始化"""
        reg = MyRegressors(model_name, random_state=42)
        model_obj, param_space, static_params = reg.get()
        
        # 验证返回类型
        assert callable(model_obj)
        assert isinstance(param_space, dict)
        assert isinstance(static_params, dict)
        
        # 验证参数空间结构
        for param in param_space.values():
            assert callable(param), "参数空间值必须是可调用函数"
            
        # 验证随机种子设置
        if "random_state" in static_params:
            assert static_params["random_state"] == 42
        elif "seed" in static_params:  # XGBoost的特殊情况
            assert static_params["seed"] == 42

    # 异常情况测试
    def test_invalid_model_name(self):
        """测试无效模型名称引发异常"""
        with pytest.raises(AssertionError) as e:
            MyRegressors(self.invalid_model, random_state=42)
        assert "Invalid model name" in str(e.value)

    def test_non_integer_random_state(self):
        """测试非整数随机种子类型检查"""
        with pytest.raises(AssertionError):
            MyRegressors("rf", random_state="42")

    # 类别特征处理测试
    @pytest.mark.parametrize("model_name", ["xgb", "lgb", "cat"])
    def test_cat_features_handling(self, model_name):
        """测试支持类别特征的模型"""
        reg = MyRegressors(model_name, random_state=42, cat_features=CAT_FEATURES)
        _, _, static_params = reg.get()
        
        if model_name == "cat":
            assert static_params["cat_features"] == CAT_FEATURES
        elif model_name == "xgb":
            assert static_params["enable_categorical"] is True

    # 参数空间验证
    def test_param_space_structure(self):
        """验证参数空间结构完整性"""
        for model_name in self.valid_models:
            reg = MyRegressors(model_name, random_state=42)
            param_space = reg.get()[1]
            
            # 检查参数项是否都是可调用函数
            for param_func in param_space.values():
                assert callable(param_func), f"{model_name}参数项必须是可调用函数"
                
            # 检查关键参数是否存在
            if model_name in ["rf", "gbdt"]:
                assert "n_estimators" in param_space

    # 静态参数验证
    def test_static_params_consistency(self):
        """验证静态参数一致性"""
        test_cases = [
            ("mlp", {"hidden_layer_sizes": (300, 300, 300)}),
            ("lgb", {"verbose": -1}),
            ("ada", {"random_state": 42})
        ]
        
        for model_name, expected in test_cases:
            reg = MyRegressors(model_name, random_state=42)
            static_params = reg.get()[2]
            for key, value in expected.items():
                assert static_params[key] == value

    # 边界条件测试
    def test_empty_cat_features(self):
        """Test the case when there are no categorical features"""
        reg = MyRegressors("cat", random_state=42, cat_features=[])
        _, _, static_params = reg.get()
        assert static_params["cat_features"] == []

    # 模型特定功能测试
    def test_xgb_cat_feature_handling(self):
        """Test XGBoost's categorical feature handling"""
        # When there are categorical features
        reg_with_cat = MyRegressors("xgb", random_state=42, cat_features=CAT_FEATURES)
        assert reg_with_cat.get()[2]["enable_categorical"] is True
        
        # When there are no categorical features
        reg_without_cat = MyRegressors("xgb", random_state=42)
        assert reg_without_cat.get()[2]["enable_categorical"] is False

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_regressors.py"])
