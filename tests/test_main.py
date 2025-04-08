# For debugging
"""
import logging
logging.basicConfig(
    level = logging.WARNING,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)
"""


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels.data_engineer import data_engineer
from mymodels.pipeline import MyPipeline


def test_linear_regression():
    mymodel = MyPipeline(
        results_dir = "results/test_xgbr_regression",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 300
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
        n_jobs = 5,
        verbose = False
    )

    mymodel.optimize(
        model_name = "xgbr",
        data_engineer_pipeline = data_engineer_pipeline,
        cv = 5,
        trials = 10,
        n_jobs = 5,
        # cat_features = None,
        optimize_history = True,
        save_optimal_params = True,
        save_optimal_model = True
    )
    mymodel.evaluate(save_raw_data = True)

    mymodel.explain(
        select_background_data = "train",
        select_shap_data = "test",
        sample_background_data_k = 50,
        sample_shap_data_k = 50,
        output_raw_data = True
    )

    return None



def test_logistic_binary():
    mymodel = MyPipeline(
        results_dir = "results/test_xgb_binary",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 300
    )

    mymodel.load(
        file_path = "data/titanic.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        index_col = ["PassengerId"],
        test_ratio = 0.3,
        inspect = False
    )

    # mymodel.diagnose(sample_k=None)

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

    mymodel.diagnose(sample_k=None)

    mymodel.optimize(
        model_name = "xgbc",
        data_engineer_pipeline = data_engineer_pipeline,
        cv = 5,
        trials = 10,
        n_jobs = 5,
        # cat_features = None,  # For CatBoost ONLY
        optimize_history = True,
        save_optimal_params = True,
        save_optimal_model = True
    )

    mymodel.evaluate(save_raw_data = True)

    mymodel.explain(
        select_background_data = "train",
        select_shap_data = "test",
        sample_background_data_k = 50,
        sample_shap_data_k = 50,
        output_raw_data = True
    )

    return None



def test_logistic_multiclass():
    mymodel = MyPipeline(
        results_dir = "results/test_xgb_multiclass",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 300
    )

    mymodel.load(
        file_path = "data/obesity.csv",
        y = "0be1dad",
        x_list = ["Gender","Age","Height","Weight",\
                    "family_history_with_overweight",\
                    "FAVC","FCVC","NCP","CAEC","SMOKE",\
                    "CH2O","SCC","FAF","TUE","CALC","MTRANS"],
        index_col = "id",
        test_ratio = 0.3,
        inspect = False
    )

    # Return an instance of `sklearn.pipeline.Pipeline` object
    data_engineer_pipeline = data_engineer(
        outlier_cols = None,
        missing_values_cols = None,
        impute_method = None,
        cat_features = ["Gender", "CAEC", "CALC", "MTRANS"],
        encode_method = ["onehot", "ordinal", "ordinal", "ordinal"],
        # scale_cols = ["Age", "Height", "Weight"],
        # scale_method = ["standard", "standard", "standard"],
        n_jobs = 5,
        verbose = False
    )

    mymodel.diagnose(sample_k=None)

    mymodel.optimize(
        model_name = "xgbc",
        data_engineer_pipeline = data_engineer_pipeline,
        cv = 5,
        trials = 10,
        n_jobs = 5,
        # cat_features = ["Gender", "CAEC", "CALC", "MTRANS"],  # For CatBoost ONLY
        optimize_history = True,
        save_optimal_params = True,
        save_optimal_model = True
    )

    mymodel.evaluate(save_raw_data = True)

    mymodel.explain(
        select_background_data = "train",
        select_shap_data = "test",
        sample_background_data_k = 100,
        sample_shap_data_k = 100,
        output_raw_data = True
    )

    return None




if __name__ == "__main__":
    # test_linear_regression()
    test_logistic_binary()
    # test_logistic_multiclass()
