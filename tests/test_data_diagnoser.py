import numpy as np
import pandas as pd
import logging
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels.main import MyPipeline


logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


def test_data_diagnoser():
    mymodel = MyPipeline(
        results_dir = "results/housing",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 500
    )

    mymodel.load(
        file_path = "data/housing.csv",
        y = "MEDV",
        x_list = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", \
                "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
        index_col = ["ID"],
        test_ratio = 0.3,
        inspect = False
    )

    mymodel.diagnose(sample_k = 0.5)

    return None


"""
def test_index_mismatch_warning():
    # 创建具有不同索引的数据
    x_data = pd.DataFrame({'feature': [1, 2, 3]}, index=[1, 2, 3])
    y_data = pd.Series([4, 5, 6], index=[4, 5, 6])

    diagnoser = MyDataDiagnoser(x_data, y_data)
"""



if __name__ == "__main__":
    # test_index_mismatch_warning()
    test_data_diagnoser()
