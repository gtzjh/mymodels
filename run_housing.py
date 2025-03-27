import numpy as np
import pandas as pd


from mymodels.pipeline import MyPipeline


def clean_data():
    data = pd.read_csv("data/housing.csv", encoding = "utf-8", na_values = np.nan)
    print(data.info())

    # Identify non-numerical features
    non_numerical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Calculate and print unique values for non-numerical features
    print("\nUnique values for non-numerical features:")
    for feature in non_numerical_features:
        unique_values = data[feature].unique()
        print(f"\n{feature} (Count: {len(unique_values)}):")
        print(unique_values)
    
    return None



def main():
    mymodels = MyPipeline(
        results_dir = "results/housing",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 500
    )
    mymodels.load(
        file_path = "data/housing.csv",
        y = "MEDV",
        x_list = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
        test_ratio = 0.3,
        inspect = False
    )
    mymodels.optimize(
        model_name = "xgbr",
        cat_features = None,
        encode_method = None,
        cv = 5,
        trials = 10,
        n_jobs = 5,
        plot_optimization = True
    )
    mymodels.evaluate()
    mymodels.explain(
        sample_train_k = 50,
        sample_test_k = 50,
    )

    return None


if __name__ == "__main__":
    clean_data()
    main()