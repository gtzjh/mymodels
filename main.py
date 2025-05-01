import numpy as np
import pandas as pd


from mymodels import data_engineer
from mymodels import MyModel


# Construct the pipeline
mymodel = MyModel(random_state = 0)


# Data engineering
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


# Load data
data = pd.read_csv("data/titanic.zip", encoding="utf-8",
                   na_values=np.nan, index_col=["PassengerId"])

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


# Configure the plotting and output
mymodel.format(
    results_dir = "results/titanic",
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


# Predict
data_pred = pd.read_csv("data/titanic_test.csv", encoding = "utf-8",
                        na_values = np.nan, index_col = ["PassengerId"])

data_pred = data_pred.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

y_pred = mymodel.predict(data = data_pred)
y_pred.to_csv("results/titanic/prediction.csv", encoding = "utf-8", index = True)
