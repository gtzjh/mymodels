import numpy as np
import pandas as pd


from mymodels.data_engineer import data_engineer
from mymodels import MyModel


# Construct the pipeline
mymodel = MyModel(random_state = 0)


"""
data = pd.read_csv("data/titanic.csv", encoding="utf-8",
                   na_values=np.nan, index_col = ["PassengerId"])

# Construct the data engineer pipeline
data_engineer_pipeline = data_engineer(
    outlier_cols = None,
    missing_values_cols = ["Age", "Embarked"],
    impute_method = ["mean", "most_frequent"],
    cat_features = ["Sex", "Embarked"],
    encode_method = ["onehot", "onehot"],
    # scale_cols = ["Fare"],
    # scale_method = ["standard"],
    n_jobs = 5,
    verbose = False
)

# Load dateset, model, and data engineer pipeline
mymodel.load(
    model_name = "rfc",
    input_data = data,
    y = "Survived",
    x_list = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    test_ratio = 0.3,
    stratify = False,
    data_engineer_pipeline = data_engineer_pipeline,
    cat_features = ["Sex", "Embarked"],
    model_configs_path = "model_configs.yml"
)
"""

data = pd.read_csv("data/obesity.csv", encoding="utf-8",
                   na_values=np.nan, index_col=["id"])

# Return an instance of `sklearn.pipeline.Pipeline` object
data_engineer_pipeline = data_engineer(
    outlier_cols = None,
    missing_values_cols = None,
    impute_method = None,
    cat_features = ["Gender", "CAEC", "CALC", "MTRANS"],
    encode_method = ["onehot", "ordinal", "ordinal", "ordinal"],
    scale_cols = ["Age", "Height", "Weight"],
    scale_method = ["standard", "standard", "standard"],
    n_jobs = 5,
    verbose = False
)

mymodel.load(
    model_name = "rfc",
    input_data = data,
    y = "0be1dad",
    x_list = ["Gender","Age","Height","Weight",\
              "family_history_with_overweight",\
              "FAVC","FCVC","NCP","CAEC","SMOKE",\
              "CH2O","SCC","FAF","TUE","CALC","MTRANS"],
    test_ratio = 0.3,
    stratify = False,
    data_engineer_pipeline = data_engineer_pipeline,
    # cat_features = ["Sex", "Embarked"],
    model_configs_path = "model_configs.yml"
)

# Configure the plotting and output
mymodel.format(
    results_dir = "results/",
    show = False,
    plot_format = "jpg",
    plot_dpi = 500,
    # save_optimal_params = True,
    # save_optimal_model = True,
    # output_evaluation = True,
    # save_raw_data = True,
    # output_shap_values = True
)

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

mymodel.explain(
    select_background_data = "train",
    select_shap_data = "test",
    sample_background_data_k = 50,
    sample_shap_data_k = 50
)

