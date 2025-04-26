import numpy as np
import pandas as pd


from mymodels.data_engineer import data_engineer
from mymodels import MyPipeline


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


# Construct the pipeline
mymodel = MyPipeline(random_state = 0)

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

# Optimize
mymodel.optimize(
    strategy = "tpe",
    cv = 5,
    trials = 10,
    n_jobs = 5,
    direction = "maximize",
    eval_function = None
)

"""
# Evaluate
mymodel.evaluate(
    show_train = True,
    dummy = True,
    eval_metric = None
)

mymodel.explain(
    select_background_data = "train",
    select_shap_data = "test",
    sample_background_data_k = None,
    sample_shap_data_k = None
)
"""


