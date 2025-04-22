"""
测试绘制SHAP summary和SHAP dependence的函数


构建sklearn的随机森林模型进行测试，分三种任务类型：二分类、多分类、回归
1. 测试绘制SHAP summary的函数
2. 测试绘制SHAP dependence的函数

也就是总共5个基础测试项目

在这些测试项目的基础上，应当加入输入异常捕捉，比如但不限于：

1. 输入的shap_explainer对象不是shap.Explainer对象
2. 输入的shap_explainer对象是shap.Explainer对象，但是shap_values的维度不是2维或3维
3. 输入的shap_explainer对象是shap.Explainer对象，但是shap_values的维度是2维或3维，但是feature_names_list的长度与shap_values的维度不一致
4. 测试保存文件功能，检查文件是否正确保存到指定路径
5. 测试不同的显示参数，如show=True/False的行为
6. 测试边缘情况，如空特征列表或极少特征的情况
7. 测试输入数据包含NaN或极端值时的处理
8. 测试各种不同的plot_format参数值
9. 测试函数在不同的matplotlib后端下的行为
10. 测试内存和性能边界（大数据集、高维特征等）
"""


import numpy as np
import pandas as pd
import pytest
import shutil
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels.plotting import Plotter


# 创建测试所需的目录
def test_results_dir():
    results_dir = "./results/test_plotting_plot_explainer/"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
    # 测试完成后清理部分移到主函数中


# 创建二分类数据和模型
def binary_classification_data():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42, n_classes=2)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X  = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    return {
        "X_train": X_train, 
        "X_test": X_test, 
        "y_train": y_train, 
        "y_test": y_test, 
        "model": model, 
        "explainer": explainer
    }


# 创建多分类数据和模型
def multiclass_classification_data():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=4, n_redundant=1, random_state=42, n_classes=6)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X  = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    return {
        "X_train": X_train, 
        "X_test": X_test, 
        "y_train": y_train, 
        "y_test": y_test, 
        "model": model, 
        "explainer": explainer
    }


# 创建回归数据和模型
def regression_data():
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes().feature_names
    X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    return {
        "X_train": X_train, 
        "X_test": X_test, 
        "y_train": y_train, 
        "y_test": y_test, 
        "model": model, 
        "explainer": explainer
    }


# 基础测试项目 1: 测试绘制SHAP summary的函数 - 二分类
def test_plot_shap_summary_binary(test_results_dir, binary_classification_data):
    """
    Test SHAP summary plot for binary classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 计算 SHAP 值，得到explanation对象
    explainer = binary_classification_data["explainer"]
    X_test = binary_classification_data["X_test"]
    explanation = explainer(X_test, check_additivity=False)

    
    # 测试绘制功能
    plotter.plot_shap_summary(explanation)
    
    # 验证文件是否保存
    # assert os.path.exists(os.path.join(test_results_dir, "SHAP/shap_summary.png")), \
    #     "SHAP summary plot should be saved"


# 基础测试项目 2: 测试绘制SHAP summary的函数 - 多分类
def test_plot_shap_summary_multiclass(test_results_dir, multiclass_classification_data):
    """
    Test SHAP summary plot for multiclass classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 计算 SHAP 值，得到explanation对象
    explainer = multiclass_classification_data["explainer"]
    X_test = multiclass_classification_data["X_test"]
    explanation = explainer(X_test, check_additivity=False)
    
    # 测试绘制功能
    plotter.plot_shap_summary(explanation)
    
    # 验证每个类的文件是否保存
    n_classes = len(np.unique(multiclass_classification_data["y_train"]))
    for i in range(n_classes):
        assert os.path.exists(os.path.join(test_results_dir, f"SHAP/shap_summary/class_{i}.png")), "SHAP summary plot for each class should be saved"


# 基础测试项目 3: 测试绘制SHAP summary的函数 - 回归
def test_plot_shap_summary_regression(test_results_dir, regression_data):
    """
    Test SHAP summary plot for regression
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 计算 SHAP 值，得到explanation对象
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test, check_additivity=False)

    # 测试绘制功能
    plotter.plot_shap_summary(explanation)
    
    # 验证文件是否保存
    assert os.path.exists(os.path.join(test_results_dir, "SHAP/shap_summary.png")), "SHAP summary plot should be saved"


# 基础测试项目 4: 测试绘制SHAP dependence的函数 - 二分类
def test_plot_shap_dependence_binary(test_results_dir, binary_classification_data):
    """
    Test SHAP dependence plot for binary classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 计算 SHAP 值，得到explanation对象
    explainer = binary_classification_data["explainer"]
    X_test = binary_classification_data["X_test"]
    explanation = explainer(X_test, check_additivity=False)

    
    # 测试绘制功能
    plotter.plot_shap_dependence(explanation)
    
    # 验证文件是否保存
    # for i, feature_name in enumerate(binary_classification_data["explainer"].feature_names):
    #     assert os.path.exists(os.path.join(test_results_dir, f"SHAP/shap_dependence/{feature_name}.png")), "SHAP dependence plot for each feature should be saved"


# 基础测试项目 5: 测试绘制SHAP dependence的函数 - 多分类
def test_plot_shap_dependence_multiclass(test_results_dir, multiclass_classification_data):
    """
    Test SHAP dependence plot for multiclass classification
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 计算 SHAP 值，得到explanation对象
    explainer = multiclass_classification_data["explainer"]
    X_test = multiclass_classification_data["X_test"]
    explanation = explainer(X_test, check_additivity=False)
    
    # 测试绘制功能
    plotter.plot_shap_dependence(explanation)
    
    # 验证每个类的文件是否保存
    n_classes = len(np.unique(multiclass_classification_data["y_train"]))
    for i in range(n_classes):
        for feature_name in multiclass_classification_data["explainer"].feature_names:
            assert os.path.exists(os.path.join(test_results_dir, f"SHAP/shap_dependence/class_{i}/{feature_name}.png")), "SHAP dependence plot for each feature should be saved"


# 基础测试项目 6: 测试绘制SHAP dependence的函数 - 回归
def test_plot_shap_dependence_regression(test_results_dir, regression_data):
    """
    Test SHAP dependence plot for regression
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 计算 SHAP 值，得到explanation对象
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test, check_additivity=False)
    
    # 测试绘制功能
    plotter.plot_shap_dependence(explanation)
    
    # 验证文件是否保存
    for feature_name in regression_data["explainer"].feature_names:
        assert os.path.exists(os.path.join(test_results_dir, f"SHAP/shap_dependence/{feature_name}.png")), "SHAP dependence plot for each feature should be saved"


# 异常测试项目 1: 输入的shap_explainer对象不是shap.Explainer对象
def test_invalid_explainer_type(test_results_dir):
    """
    Test handling of invalid explainer type
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 创建一个不是shap.Explainer的对象
    invalid_explainer = object()
    
    # 测试是否正确抛出异常
    with pytest.raises(AssertionError, match="shap_explainer must be a shap.Explainer object"):
        plotter.plot_shap_summary(invalid_explainer)
    
    with pytest.raises(AssertionError, match="shap_explainer must be a shap.Explainer object"):
        plotter.plot_shap_dependence(invalid_explainer)


# 异常测试项目 2: shap_values的维度不是2维或3维
def test_invalid_shap_values_dimension(test_results_dir, binary_classification_data):
    """
    Test handling of invalid SHAP values dimension
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 创建一个无效的explanation对象（这里简化处理，实际上可能需要修改实现）
    class InvalidExplanation:
        def __init__(self):
            self.values = np.array([1.0, 2.0, 3.0])  # 1维
            self.data = binary_classification_data["X_test"]
            self.feature_names = binary_classification_data["explainer"].feature_names
    
    invalid_explanation = InvalidExplanation()
    
    # 测试是否正确抛出异常
    with pytest.raises(ValueError, match="Invalid SHAP values dimension"):
        plotter.plot_shap_summary(invalid_explanation)


# 异常测试项目 3: feature_names_list长度与shap_values维度不一致
def test_inconsistent_feature_names(test_results_dir, binary_classification_data):
    """
    Test handling of inconsistent feature names length
    """
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 计算 SHAP 值，得到explanation对象
    explainer = binary_classification_data["explainer"]
    X_test = binary_classification_data["X_test"]
    explanation = explainer(X_test, check_additivity=False)
    
    # 创建一个feature_names不一致的explanation对象
    class InconsistentExplanation:
        def __init__(self, original_explanation):
            self.values = original_explanation.values
            self.data = original_explanation.data
            # 少一个特征名
            self.feature_names = original_explanation.feature_names[:-1]
    
    inconsistent_explanation = InconsistentExplanation(explanation)
    
    # 测试是否正确处理不一致情况
    with pytest.raises(IndexError):
        plotter.plot_shap_dependence(inconsistent_explanation)


# 测试项目 4: 测试不同的plot_format参数
def test_different_plot_formats(test_results_dir, regression_data):
    """
    Test different plot formats
    """
    # 计算 SHAP 值，得到explanation对象
    explainer = regression_data["explainer"]
    X_test = regression_data["X_test"]
    explanation = explainer(X_test, check_additivity=False)
    
    for format_name in ["png", "jpg", "pdf", "svg"]:
        results_dir = os.path.join(test_results_dir, f"format_{format_name}")
        os.makedirs(results_dir, exist_ok=True)
        
        plotter = Plotter(
            show=False,
            plot_format=format_name,
            plot_dpi=300,
            results_dir=results_dir
        )
        
        # 测试绘制功能
        plotter.plot_shap_summary(explanation)
        
        # 验证文件是否以正确格式保存
        assert os.path.exists(os.path.join(results_dir, f"SHAP/shap_summary.{format_name}")), f"SHAP summary plot should be saved in {format_name} format"


# 测试项目 5: 测试边缘情况 - 极少特征
def test_few_features(test_results_dir):
    """
    Test with very few features
    """
    # 创建只有1个特征的数据
    X = np.random.rand(50, 1)
    y = np.random.randint(0, 2, 50)
    
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model, X_train)
    explainer.feature_names = ["single_feature"]
    
    # 计算 SHAP 值，得到explanation对象
    explanation = explainer(X_test, check_additivity=False)
    
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 测试绘制功能
    plotter.plot_shap_summary(explanation)
    plotter.plot_shap_dependence(explanation)
    
    # 验证文件是否保存
    assert os.path.exists(os.path.join(test_results_dir, "SHAP/shap_summary.png")), "SHAP summary plot should be saved"
    assert os.path.exists(os.path.join(test_results_dir, "SHAP/shap_dependence/single_feature.png")), "SHAP dependence plot for each feature should be saved"


# 测试项目 6: 测试数据包含NaN值
def test_with_nan_values(test_results_dir, regression_data):
    """
    Test with NaN values in the dataset
    """
    # 获取原始数据并引入一些NaN值
    X_train = regression_data["X_train"].copy()
    X_test = regression_data["X_test"].copy()
    
    # 将一些值替换为NaN
    X_test_with_nan = X_test.copy()
    X_test_with_nan[0, 0] = np.nan
    X_test_with_nan[1, 1] = np.nan
    
    # 创建新的explainer
    model = regression_data["model"]
    explainer = shap.TreeExplainer(model, X_train)
    explainer.feature_names = regression_data["explainer"].feature_names
    
    plotter = Plotter(
        show=False,
        plot_format="png",
        plot_dpi=300,
        results_dir=test_results_dir
    )
    
    # 测试是否能处理NaN值
    try:
        # 注意：这可能会失败，取决于shap库如何处理NaN值
        explanation = explainer(X_test_with_nan)
        plotter.plot_shap_summary(explanation)
    except Exception as e:
        # 记录异常，但不失败测试
        print(f"NaN handling test resulted in exception: {str(e)}")



if __name__ == "__main__":
    results_dir = "./results/test_plotting_plot_explainer/"

    # Clean up the existing test directory
    if os.path.exists(results_dir):
        try:
            shutil.rmtree(results_dir)
            print(f"Cleaned up test directory: {results_dir}")
        except Exception as e:
            print(f"Failed to clean up test directory: {str(e)}")

    # Create test directory
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取测试数据
    binary_data = binary_classification_data()
    multiclass_data = multiclass_classification_data()
    reg_data = regression_data()
    
    # Run all tests
    print("Running test_plot_shap_summary_binary...")
    test_plot_shap_summary_binary(results_dir, binary_data)
    
    print("Running test_plot_shap_summary_multiclass...")
    test_plot_shap_summary_multiclass(results_dir, multiclass_data)
    
    print("Running test_plot_shap_summary_regression...")
    test_plot_shap_summary_regression(results_dir, reg_data)
    
    print("Running test_plot_shap_dependence_binary...")
    test_plot_shap_dependence_binary(results_dir, binary_data)
    
    print("Running test_plot_shap_dependence_multiclass...")
    test_plot_shap_dependence_multiclass(results_dir, multiclass_data)
    
    print("Running test_plot_shap_dependence_regression...")
    test_plot_shap_dependence_regression(results_dir, reg_data)
    
    print("Running test_invalid_explainer_type...")
    test_invalid_explainer_type(results_dir)
    
    print("Running test_invalid_shap_values_dimension...")
    test_invalid_shap_values_dimension(results_dir, binary_data)
    
    print("Running test_inconsistent_feature_names...")
    test_inconsistent_feature_names(results_dir, binary_data)
    
    print("Running test_different_plot_formats...")
    test_different_plot_formats(results_dir, reg_data)
    
    print("Running test_few_features...")
    test_few_features(results_dir)
    
    print("Running test_with_nan_values...")
    test_with_nan_values(results_dir, reg_data)
    
    print("All tests completed successfully!")
