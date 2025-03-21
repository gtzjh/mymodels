from mymodels.pipeline import MyPipeline


def main():
    the_pipeline = MyPipeline(
        model_name = "rfc",
        results_dir = "results/rfc_single_test",
        cat_features = ['Pclass', 'Sex', 'Embarked'],
        encode_method = "onehot",
        random_state = 0,
    )
    
    the_pipeline.load(
        file_path = "data/titanic/train.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        test_ratio = 0.4,
        inspect = True
    )
    the_pipeline.optimize(cv = 5, trials = 30, n_jobs = 5)
    the_pipeline.evaluate()
    the_pipeline.explain(shap_ratio = 0.3, _plot = True)

    """一步到位
    the_pipeline.run(
        file_path = "data/titanic/train.csv",
        y = "Survived",
        x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
        test_ratio = 0.4,
        cv = 5,
        trials = 30,
        n_jobs = 5,
        shap_ratio = 0.3,
        inspect=True,
        interpret=True
    )
    """

    return None



if __name__ == "__main__":
    main()