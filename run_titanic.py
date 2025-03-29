from mymodels.pipeline import MyPipeline


def main():
    mymodel = MyPipeline(
        results_dir = "results/titanic",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 500
    )
    mymodel.load(
        file_path = "data/titanic.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        index_col = ["PassengerId", "Name"],
        test_ratio = 0.3,
        inspect = False
    )
    mymodel.optimize(
        model_name = "xgbc",
        cat_features = ["Pclass", "Sex", "Embarked"],
        encode_method = "onehot",
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
    main()