import numpy as np
import pandas as pd


from mymodels import data_engineer
from mymodels import MyModel


def test_binary_classification(model_name: str):
    try:
        _results_dir = f"results/titanic_{model_name}"

        # Construct the pipeline
        mymodel = MyModel(random_state = 0)

        # Data engineering
        data_engineer_pipeline = data_engineer(
            outlier_cols = None,
            missing_values_cols = ["Age", "Embarked"],
            impute_method = ["mean", "most_frequent"],
            cat_features = ["Sex", "Embarked"],
            encode_method = ["ordinal", "binary"],
            # scale_cols = ["Fare"],
            # scale_method = ["standard"],
            n_jobs = 5,
            verbose = False
        )

        # Load data
        data = pd.read_csv("data/titanic.zip", encoding="utf-8",
                        na_values=np.nan, index_col=["PassengerId"]).sample(300)

        mymodel.load(
            model_name = model_name,
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
            results_dir = _results_dir,
            show = False,
            plot_format = "jpg",
            plot_dpi = 100,
            save_optimal_model = True,
            save_raw_data = True,
            save_shap_values = True
        )

        # Data diagnosis
        mymodel.diagnose(sample_k = 100)

        # Optimize
        mymodel.optimize(
            strategy = "tpe",
            cv = 3,
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

        y_pred.name = "Survived"
        y_pred.to_csv(_results_dir + "/prediction.csv", encoding = "utf-8", index = True)
    
    except Exception as e:
        print(e)


def test_multi_classification(model_name: str):
    try:
        _results_dir = f"results/obesity_{model_name}"

        # Construct the pipeline
        mymodel = MyModel(random_state = 0)

        # Data engineering
        data_engineer_pipeline = data_engineer(
            outlier_cols = None,
            missing_values_cols = None,
            impute_method = None,
            cat_features = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"],
            encode_method = ["ordinal", "ordinal", "ordinal", "ordinal", "ordinal", "ordinal", "ordinal", "ordinal"],
            # scale_cols = ["Age", "Height", "Weight"],
            # scale_method = ["standard", "standard", "standard"],
            n_jobs = 5,
            verbose = False
        )

        # Load data
        data = pd.read_csv("data/obesity.zip", encoding="utf-8",
                        na_values=np.nan, index_col=["id"]).sample(300)

        mymodel.load(
            model_name = model_name,
            input_data = data,
            y = "NObeyesdad",
            x_list = ["Gender","Age","Height","Weight",\
                    "family_history_with_overweight",\
                    "FAVC","FCVC","NCP","CAEC","SMOKE",\
                    "CH2O","SCC","FAF","TUE","CALC","MTRANS"],
            test_ratio = 0.3,
            stratify = False,
            data_engineer_pipeline = data_engineer_pipeline,
            model_configs_path = "model_configs.yml"
        )

        # Configure the plotting and output
        mymodel.format(
            results_dir = _results_dir,
            show = False,
            plot_format = "jpg",
            plot_dpi = 100,
            save_optimal_model = True,
            save_raw_data = True,
            save_shap_values = True
        )

        # Data diagnosis
        mymodel.diagnose(sample_k = 100)

        # Optimize
        mymodel.optimize(
            strategy = "tpe",
            cv = 3,
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
        data_pred = pd.read_csv("data/obesity_test.csv", encoding = "utf-8",
                                na_values = np.nan, index_col = ["id"])

        data_pred = data_pred.loc[:, ["Gender","Age","Height","Weight",\
                                    "family_history_with_overweight",\
                                    "FAVC","FCVC","NCP","CAEC","SMOKE",\
                                    "CH2O","SCC","FAF","TUE","CALC","MTRANS"]]

        y_pred = mymodel.predict(data = data_pred)
        y_pred.name = "NObeyesdad"
        y_pred.to_csv(_results_dir + "/prediction.csv", encoding = "utf-8", index = True)
    
    except Exception as e:
        print(e)


def test_regression(model_name: str):
    try:
        _results_dir = f"results/housing_{model_name}"

        # Construct the pipeline
        mymodel = MyModel(random_state = 0)

        # Data engineering
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

        # Load data
        data = pd.read_csv("data/housing.zip", encoding = "utf-8", 
                        na_values = np.nan, index_col = ["ID"])

        mymodel.load(
            model_name = model_name,
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
            results_dir = _results_dir,
            show = False,
            plot_format = "jpg",
            plot_dpi = 100,
            save_optimal_model = True,
            save_raw_data = True,
            save_shap_values = True
        )

        # Data diagnosis
        mymodel.diagnose(sample_k = 100)

        # Optimize
        mymodel.optimize(
            strategy = "tpe",
            cv = 3,
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

    except Exception as e:
        print(e)


if __name__ == "__main__":
    test_binary_classification("lgbc")
    test_multi_classification("lgbc")
    test_regression("lgbr")

