from mymodels.pipeline import MyPipeline


def main():
    the_pipeline = MyPipeline(
        model_name = "lgbc",
        results_dir = "results/lgbc",
        cat_features = ['Pclass', 'Sex', 'Embarked'],
        encode_method = "binary",
        random_state = 0,
    )
    
    the_pipeline.load(
        file_path = "data/titanic/train.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        test_ratio = 0.4,
        inspect = True
    )
    the_pipeline.optimize(cv = 5, trials = 10, n_jobs = 5)
    the_pipeline.evaluate()
    the_pipeline.explain(sample_train_k = 50, sample_test_k = 50, _plot = True)

    return None


if __name__ == "__main__":
    main()