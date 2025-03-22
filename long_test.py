from mymodels.pipeline import MyPipeline


def long_test():
    try:
        the_pipeline = MyPipeline(
            model_name = "catc",
            results_dir = f"results/catc",
            cat_features = ['Pclass', 'Sex', 'Embarked'],
            encode_method = None,
            random_state = 0,
        )
        the_pipeline.load(
            file_path = "data/titanic/train.csv",
            y = "Survived",
            x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
            test_ratio = 0.4,
            inspect = False
        )
        the_pipeline.optimize(cv = 5, trials = 10, n_jobs = -1)
        the_pipeline.evaluate()
        # the_pipeline.explain(shap_ratio = 0.1, _plot = True)
    except Exception as e:
        with open("error.txt", "a") as f:
            f.write(f"Error in catboost: \n{e}\n")


    for n in ["rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", "svc", "knc", "mlpc"]:
        for e in ["onehot", "binary", "ordinal", "label", "target", "frequency"]:
            try:
                the_pipeline = MyPipeline(
                    model_name = n,
                    results_dir = f"results/{n}/{e}",
                    cat_features = ['Pclass', 'Sex', 'Embarked'],
                    encode_method = e,
                    random_state = 0,
                )
                the_pipeline.load(
                    file_path = "data/titanic/train.csv",
                    y = "Survived",
                    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
                    test_ratio = 0.4,
                    inspect = False
                )
                the_pipeline.optimize(cv = 5, trials = 10, n_jobs = -1)
                the_pipeline.evaluate()
                # the_pipeline.explain(shap_ratio = 0.1, _plot = True)
            except Exception as error:
                with open("error.txt", "a") as f:
                    f.write(f"Error in {n} model, encoder: {e}: \n{error}\n")
                continue
    
    return None


if __name__ == "__main__":
    long_test()
