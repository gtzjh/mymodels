from mymodels.pipeline import MyPipeline


def main():
    the_pipeline = MyPipeline(
        results_dir = "results/lgbc",
        random_state = 0,
    )
    
    the_pipeline.load(
        file_path = "data/titanic.csv",
        y = "Survived",
        x_list = ("Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"),
        test_ratio = 0.4,
        inspect = False
    )
    the_pipeline.optimize(
        model_name = "lgbc",
        cat_features = ["Pclass", "Sex", "Embarked"],
        encode_method = "onehot",
        cv = 5,
        trials = 10,
        n_jobs = -1,
        plot_optimization = False
    )
    the_pipeline.evaluate()
    the_pipeline.explain(sample_train_k = 50, sample_test_k = 50, _plot = False)

    return None


if __name__ == "__main__":
    main()