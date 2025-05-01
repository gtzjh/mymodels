import numpy as np
import pandas as pd
import pytest
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))



from mymodels.core import MyDataLoader



# 样本数据用于基准测试
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'y_str': ['a', 'b', 'a', 'b', 'a'],
        'y_bool': [True, False, True, False, True],
        'y_num': [0, 1, 0, 1, 0],
        'x1': [1, 2, 3, 4, 5],
        'x2': [6, 7, 8, 9, 10],
        'x3': [11, 12, 13, 14, 15]
    })
    return data

# 基准测试用例
def test_y_string_conversion(sample_data):
    loader = MyDataLoader(
        input_data=sample_data,
        y='y_str',
        x_list=['x1', 'x2', 'x3'],
        test_ratio=0.4,
        random_state=42,
        stratify=True
    )
    dataset = loader.load()
    # 检查y是否被编码为数字
    assert dataset.y_mapping_dict == {'a': 0, 'b': 1}
    assert set(dataset.y_train) <= {0, 1}
    assert set(dataset.y_test) <= {0, 1}


def test_boolean_y_no_conversion(sample_data):
    # 当y列已经是布尔类型时不应进行编码
    loader = MyDataLoader(
        input_data=sample_data,
        y='y_bool',
        x_list=['x1', 'x2', 'x3'],
        test_ratio=0.4,
        random_state=42
    )
    dataset = loader.load()

    
    # 验证无编码字典
    assert not dataset.y_mapping_dict
    # 验证y值保持布尔型
    assert dataset.y_train.dtype == bool
    assert dataset.y_test.dtype == bool
    # 验证值范围
    assert set(dataset.y_train) <= {True, False}
    assert set(dataset.y_test) <= {True, False}



def test_string_boolean_y_conversion():
    """当y列是字符串型布尔值时应进行编码"""
    data = pd.DataFrame({
        'y': ['True', 'False', 'True', 'False', 'True'],
        'x1': np.random.randn(5)
    })
    loader = MyDataLoader(
        input_data=data,
        y='y',
        x_list=['x1'],
        test_ratio=0.4,
        random_state=42
    )
    dataset = loader.load()
    
    # 验证编码字典
    assert dataset.y_mapping_dict == {'False': 0, 'True': 1}
    # 验证数值转换
    assert set(dataset.y_train) <= {0, 1}
    assert set(dataset.y_test) <= {0, 1}

def test_y_boolean_conversion(sample_data):
    # 创建字符串型布尔列
    sample_data = sample_data.copy()
    sample_data['y_str_bool'] = sample_data['y_bool'].astype(str)
    
    loader = MyDataLoader(
        input_data=sample_data,
        y='y_str_bool',
        x_list=['x1', 'x2', 'x3'],
        test_ratio=0.4,
        random_state=42
    )
    dataset = loader.load()
    
    # 验证字符串布尔被转换
    assert dataset.y_mapping_dict == {'False': 0, 'True': 1}
    assert set(dataset.y_train) <= {0, 1}


def test_y_numeric_no_conversion(sample_data):
    loader = MyDataLoader(
        input_data=sample_data,
        y='y_num',
        x_list=['x1', 'x2', 'x3'],
        test_ratio=0.4,
        random_state=42
    )
    dataset = loader.load()

    # y已经是数字，无需转换
    assert set(dataset.y_train) <= {0, 1}
    assert dataset.y_mapping_dict == None
    

def test_x_list_selection(sample_data):
    loader = MyDataLoader(
        input_data=sample_data,
        y='y_str',
        x_list=['x1', 'x3'],
        test_ratio=0.4,
        random_state=42
    )
    dataset = loader.load()
    # 检查x_list是否正确选择
    assert list(dataset.x_train.columns) == ['x1', 'x3']
    assert list(dataset.x_test.columns) == ['x1', 'x3']

def test_train_test_split_ratio():
    data = pd.DataFrame({
        'y': np.random.choice([0, 1], 100),
        'x1': np.random.randn(100),
        'x2': np.random.randn(100)
    })
    loader = MyDataLoader(
        input_data=data,
        y='y',
        x_list=['x1', 'x2'],
        test_ratio=0.3,
        random_state=42
    )
    dataset = loader.load()
    # 检查分割比例
    assert len(dataset.x_test) == 30
    assert len(dataset.x_train) == 70

def test_random_state_reproducibility():
    data = pd.DataFrame({
        'y': np.random.choice([0, 1], 100),
        'x1': np.random.randn(100),
        'x2': np.random.randn(100)
    })
    loader1 = MyDataLoader(data, y='y', x_list=['x1', 'x2'], test_ratio=0.3, random_state=42)
    dataset1 = loader1.load()
    loader2 = MyDataLoader(data, y='y', x_list=['x1', 'x2'], test_ratio=0.3, random_state=42)
    dataset2 = loader2.load()
    # 检查随机种子是否确保结果一致
    pd.testing.assert_frame_equal(dataset1.x_train, dataset2.x_train)
    pd.testing.assert_frame_equal(dataset1.x_test, dataset2.x_test)

def test_stratify_true():
    y = [0] * 80 + [1] * 20
    data = pd.DataFrame({
        'y': y,
        'x1': np.random.randn(100),
        'x2': np.random.randn(100)
    })
    loader = MyDataLoader(data, y='y', x_list=['x1', 'x2'], test_ratio=0.3, stratify=True, random_state=42)
    dataset = loader.load()
    # 检查分层抽样的比例
    test_counts = dataset.y_test.value_counts()
    assert test_counts[0] == 24  # 80% of 30
    assert test_counts[1] == 6   # 20% of 30

# 异常测试用例
def test_invalid_input_data_type():
    with pytest.raises(TypeError):
        MyDataLoader(input_data="invalid", y='y', x_list=['x1'])

def test_y_column_not_exist(sample_data):
    with pytest.raises(KeyError):
        MyDataLoader(sample_data, y='invalid_col', x_list=['x1'])

def test_x_list_column_not_exist(sample_data):
    with pytest.raises(KeyError):
        MyDataLoader(sample_data, y='y_str', x_list=['x1', 'invalid_col'])

def test_invalid_test_ratio(sample_data):
    with pytest.raises(ValueError):
        MyDataLoader(sample_data, y='y_str', x_list=['x1'], test_ratio=1.5)
    with pytest.raises(ValueError):
        MyDataLoader(sample_data, y='y_str', x_list=['x1'], test_ratio=-0.5)

def test_invalid_random_state_type(sample_data):
    with pytest.raises(TypeError):
        MyDataLoader(sample_data, y='y_str', x_list=['x1'], random_state='invalid')

def test_invalid_stratify_type(sample_data):
    with pytest.raises(TypeError):
        MyDataLoader(sample_data, y='y_str', x_list=['x1'], stratify='invalid')

def test_empty_x_list(sample_data):
    with pytest.raises(ValueError):
        MyDataLoader(sample_data, y='y_str', x_list=[])

def test_missing_values_in_y():
    data = pd.DataFrame({
        'y': [0, 1, np.nan, 0, 1],
        'x1': [1, 2, 3, 4, 5]
    })
    with pytest.raises(ValueError):
        MyDataLoader(data, y='y', x_list=['x1']).load()

