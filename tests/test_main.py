import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



from mymodels.data_engineer import data_engineer
from mymodels import MyPipeline



def test_regression():
    for i in [
        # "lr", "catr", 
        "rfr", 
        # "dtr", 
        # "lgbr",
        # "gbdtr", "xgbr", "adar", "svr", "knr", "mlpr"
    ]:
        mymodel = MyPipeline(
            results_dir = f"results/test_{i}_regression",
            random_state = 0,
            stratify = False,
            show = False,
            plot_format = "jpg",
            plot_dpi = 300
        )

        data = pd.read_csv("data/housing.csv", encoding = "utf-8", 
                           na_values = np.nan, index_col = ["ID"])
        mymodel.load(
            input_data = data,
            y = "MEDV",
            x_list = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", \
                      "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
            test_ratio = 0.3,
            inspect = False
        )

        mymodel.diagnose(sample_k=None)

        # Return an instance of `sklearn.pipeline.Pipeline` object
        # User can define their own pipeline
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

        mymodel.optimize(
            model_name = i,
            data_engineer_pipeline = data_engineer_pipeline,
            strategy = "tpe",
            cv = 5,
            trials = 100,
            n_jobs = -1,
            # cat_features = None,
            direction = "maximize",
            eval_function = None,
            optimize_history = True,
            save_optimal_params = True,
            save_optimal_model = True
        )
        mymodel.evaluate(
            show_train = True,
            dummy = True,
            save_raw_data = True,
            eval_metric = None
        )

        mymodel.explain(
            select_background_data = "train",
            select_shap_data = "test",
            sample_background_data_k = 50,
            sample_shap_data_k = 50,
            output_raw_data = True
        )

    return None



def test_binary():
    for i in ["lc", "catc", "rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", "svc", "knc", "mlpc"]:
        mymodel = MyPipeline(
            results_dir = f"results/test_{i}_binary",
            random_state = 0,
            stratify = True,
            show = False,
            plot_format = "jpg",
            plot_dpi = 300
        )

        data = pd.read_csv("data/titanic.csv", encoding="utf-8",
                           na_values=np.nan, index_col=["PassengerId"])
        mymodel.load(
            input_data = data,
            y = "Survived",
            x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
            test_ratio = 0.3,
            inspect = False
        )

        mymodel.diagnose(sample_k=None)

        data_engineer_pipeline = data_engineer(
            outlier_cols = None,
            missing_values_cols = ["Age", "Embarked"],
            impute_method = ["mean", "most_frequent"],
            cat_features = ["Sex", "Embarked"],
            encode_method = ["onehot", "onehot"],
            # scale_cols = ["Fare"],
            # scale_method = ["standard"],
            n_jobs = -1,
            verbose = False
        )

        mymodel.optimize(
            model_name = i,
            data_engineer_pipeline = data_engineer_pipeline,
            strategy = "tpe",
            cv = 5,
            trials = 30,
            n_jobs = -1,
            # cat_features = None,  # For CatBoost ONLY
            direction = "maximize",
            eval_function = None,
            optimize_history = True,
            save_optimal_params = True,
            save_optimal_model = True
        )

        mymodel.evaluate(
            show_train = True,
            dummy = True,
            save_raw_data = True,
            eval_metric = None
        )

        mymodel.explain(
            select_background_data = "train",
            select_shap_data = "test",
            sample_background_data_k = 50,
            sample_shap_data_k = 50,
            output_raw_data = True
        )

    return None



def test_multiclass():
    for i in [
        # "lc", "catc", 
        # "rfc", 
        # "dtc", 
        "lgbc", 
        # "xgbc", "adac", "svc", "knc", "mlpc"
    ]:
        mymodel = MyPipeline(
            results_dir = f"results/test_{i}_multiclass",
            random_state = 0,
            stratify = True,
            show = False,
            plot_format = "jpg",
            plot_dpi = 300
        )

        data = pd.read_csv("data/obesity.csv", encoding="utf-8",
                           na_values=np.nan, index_col=["id"])
        mymodel.load(
            input_data = data,
            y = "0be1dad",
            x_list = ["Gender","Age","Height","Weight",\
                      "family_history_with_overweight",\
                      "FAVC","FCVC","NCP","CAEC","SMOKE",\
                      "CH2O","SCC","FAF","TUE","CALC","MTRANS"],
            test_ratio = 0.3,
            inspect = False
        )

        mymodel.diagnose(sample_k=None)

        # Return an instance of `sklearn.pipeline.Pipeline` object
        data_engineer_pipeline = data_engineer(
            outlier_cols = None,
            missing_values_cols = None,
            impute_method = None,
            cat_features = ["Gender", "CAEC", "CALC", "MTRANS"],
            encode_method = ["onehot", "ordinal", "ordinal", "ordinal"],
            # scale_cols = ["Age", "Height", "Weight"],
            # scale_method = ["standard", "standard", "standard"],
            n_jobs = -1,
            verbose = False
        )

        mymodel.optimize(
            model_name = i,
            data_engineer_pipeline = data_engineer_pipeline,
            strategy = "tpe",
            cv = 5,
            trials = 10,
            n_jobs = -1,
            # cat_features = ["Gender", "CAEC", "CALC", "MTRANS"],  # For CatBoost ONLY
            direction = "maximize",
            eval_function = None,
            optimize_history = True,
            save_optimal_params = True,
            save_optimal_model = True
        )


        from sklearn.metrics import fbeta_score, log_loss
        def my_fbeta_score(y_test, y_test_pred):
            return fbeta_score(y_test, y_test_pred, average = "weighted", beta = 0.5)
        # def my_log_loss(y_test, y_test_pred):
        #     return log_loss(y_test, y_test_pred)
        self_defined_eval_metric = {
            "Fbeta Score": my_fbeta_score,
            # "Log Loss": my_log_loss
        }

        mymodel.evaluate(
            show_train = True,
            dummy = True,
            save_raw_data = True,
            eval_metric = self_defined_eval_metric
        )

        mymodel.explain(
            select_background_data = "train",
            select_shap_data = "test",
            sample_background_data_k = 100,
            sample_shap_data_k = 100,
            output_raw_data = True
        )

    return None



if __name__ == "__main__":
    # test_regression()
    # test_binary()
    test_multiclass()
