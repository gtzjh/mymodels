# For debugging
import logging
logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


from mymodels.data_engineer import data_engineer
from mymodels.pipeline import MyPipeline


def test_linear_regression():
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

    # Return an instance of `sklearn.pipeline.Pipeline` object
    # User can define their own pipeline
    data_engineer_pipeline = data_engineer(
        outlier_cols = None,
        missing_values_cols = ["CRIM", "ZN", "INDUS", "CHAS", "AGE", "LSTAT"],
        impute_method = ["median", "median", "median", "median", "median", "median"],
        cat_features = None,
        encode_method = None,
        scale_cols = ["CRIM", "ZN"],
        scale_method = ["standard", "minmax"],
        n_jobs = 5,
        verbose = False
    )


    mymodel.optimize(
        model_name = "lr",
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



if __name__ == "__main__":
    test_linear_regression()
