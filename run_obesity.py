import numpy as np
import pandas as pd


from mymodels.pipeline import MyPipeline


def clean_data():
    data = pd.read_csv("data/obesity.csv", encoding = "utf-8", na_values = np.nan)
    # print(data.info())

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
    mymodel = MyPipeline(
        results_dir = "results/obesity",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 500
    )
    mymodel.load(
        file_path = "data/obesity.csv",
        y = "0be1dad",
        x_list = ["Gender","Age","Height","Weight","family_history_with_overweight",\
                  "FAVC","FCVC","NCP","CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS"],
        index_col = "id",
        test_ratio = 0.3,
        inspect = False
    )
    mymodel.optimize(
        model_name = "lgbc",
        cat_features = ["Gender", "CAEC", "CALC", "MTRANS"],
        encode_method = ["label", "ordinal", "ordinal", "frequency"],
        cv = 5,
        trials = 10,
        n_jobs = 5,
        plot_optimization = True
    )
    mymodel.evaluate(save_raw_data = True)
    mymodel.explain(
        select_background_data = "all",
        select_shap_data = "all",
        sample_background_data_k = None,
        sample_shap_data_k = None,
        output_raw_data = True
    )

    return None


if __name__ == "__main__":
    clean_data()
    main()