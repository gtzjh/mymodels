import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels.main import MyModel



def test_data_diagnoser():
    mymodel = MyModel(
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
