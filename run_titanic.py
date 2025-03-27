from mymodels.pipeline import MyPipeline


def main():
    mymodels = MyPipeline(
        results_dir = "results/titanic",
        random_state = 0,
        show = False,
        plot_format = "jpg",
        plot_dpi = 500
    )
    mymodels.load(
        file_path = "data/titanic.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        test_ratio = 0.3,
        inspect = False
    )
    mymodels.optimize(
        model_name = "xgbc",
        cat_features = ["Pclass", "Sex", "Embarked"],
        encode_method = "onehot",
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
    main()