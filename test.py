import numpy as np
import pandas as pd


from mymodels import data_engineer
from mymodels import MyModel


# Construct the pipeline
mymodel = MyModel(random_state = 0)


data_engineer_pipeline = data_engineer(
    outlier_cols = None,
    missing_values_cols = ["CRIM", "ZN", "INDUS", "CHAS", "AGE", "LSTAT"],
    impute_method = ["median", "median", "median", "median", "median", "median"],
    cat_features = None,
    encode_method = None,
    # scale_cols = ["CRIM", "ZN"],
    # scale_method = ["standard", "minmax"],
    n_jobs = -1,
    verbose = False
)


data = pd.read_csv("data/housing.zip", encoding = "utf-8", 
                   na_values = np.nan, index_col = ["ID"])


mymodel.load(
    model_name = "rfr",
    input_data = data,
    y = "MEDV",
    x_list = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", \
              "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
    test_ratio = 0.3,
    stratify = False,
    data_engineer_pipeline = data_engineer_pipeline,
    model_configs_path = "model_configs.yml"
)


# Configure the plotting and output
mymodel.format(
    results_dir = "results/",
    show = False,
    plot_format = "jpg",
    plot_dpi = 500,
    save_optimal_model = True,
    save_raw_data = True,
    save_shap_values = True
)

# Data diagnosis
mymodel.diagnose(sample_k = 100)

# Optimize
mymodel.optimize(
    strategy = "tpe",
    cv = 5,
    trials = 10,
    n_jobs = 5,
    direction = "maximize",
    eval_function = None
)

# Evaluate
mymodel.evaluate(
    show_train = True,
    dummy = True,
    eval_metric = None
)

# Explain
mymodel.explain(
    select_background_data = "train",
    select_shap_data = "test",
    sample_background_data_k = 50,
    sample_shap_data_k = 50
)

