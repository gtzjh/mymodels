import numpy as np
import pandas as pd


from mymodels.pipeline import MyPipeline


def clean_data():
    data = pd.read_csv("data/housing.csv",
                       encoding = "utf-8",
                       na_values = np.nan,
                       index_col = ["ID"])
    
    return None



def main():
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
        x_list = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
        index_col = ["ID"],
        test_ratio = 0.3,
        inspect = False
    )
    mymodel.engineering(
        missing_values_cols = ["CRIM", "ZN", "INDUS", "CHAS", "AGE", "LSTAT"],
        impute_method = ["median", "median", "median", "median", "median", "median"],
        cat_features = None,
        encode_method = None,
        scale_cols = ["CRIM", "ZN"],
        scale_method = ["standard", "minmax"]
        # scale_cols = None,
        # scale_method = None
    )
    mymodel.optimize(
        model_name = "rfr",
        cv = 5,
        trials = 10,
        n_jobs = 5,
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
    # clean_data()
    main()