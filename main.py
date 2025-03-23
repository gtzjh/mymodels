from mymodels.pipeline import MyPipeline


def main():
    mymodels = MyPipeline(
        results_dir = "results/catc",
        random_state = 0,
    )
    mymodels.load(
        file_path = "data/titanic.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        test_ratio = 0.4,
        inspect = False
    )
    mymodels.optimize(
        model_name = "catc",
        cat_features = ["Pclass", "Sex", "Embarked"],
        encode_method = None,
        cv = 5,
        trials = 10,
        n_jobs = -1,
        plot_optimization = True
    )
    mymodels.evaluate()
    mymodels.explain(
        sample_train_k = 0.5,
        sample_test_k = 0.5,
        plot = True
    )

    return None


if __name__ == "__main__":
    main()