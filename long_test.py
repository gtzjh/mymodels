from mymodels.pipeline import MyPipeline


def titanic_test():
    for n in ["rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", "svc", "knc", "mlpc"]:
        # for e in ["onehot", "binary", "label", "ordinal", "target", "frequency"]:
        try:
            mymodels = MyPipeline(
                results_dir = f"results/{n}",
                random_state = 0,
                show = False,
                plot_format = "jpg",
                plot_dpi = 300
            )
            mymodels.load(
                file_path = "data/titanic.csv",
                y = "Survived",
                x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
                index_col = ["PassengerId"],
                test_ratio = 0.4,
                inspect = False
            )
            mymodels.optimize(
                model_name = n,
                cat_features = ['Pclass', 'Sex', 'Embarked'],
                encode_method = "onehot",
                cv = 5,
                trials = 10,
                n_jobs = -1,
                optimize_history = True,
                save_optimal_params = True,
                save_optimal_model = True
            )
            mymodels.evaluate(save_raw_data = True)
            mymodels.explain(
                select_background_data = "train",
                select_shap_data = "test",
                sample_background_data_k = 50,
                sample_shap_data_k = 50,
                output_raw_data = True
            )
        except Exception as error:
            with open("error-titanic-models.txt", "a") as f:
                f.write(f"Error in model [ {n} ]: \n{error}\n")
            continue
    
    try:
        mymodels = MyPipeline(
            results_dir = f"results/catc",
            random_state = 0,
            show = False,
            plot_format = "jpg",
            plot_dpi = 300
        )
        mymodels.load(
            file_path = "data/titanic.csv",
            y = "Survived",
            x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
            index_col = ["PassengerId"],
            test_ratio = 0.4,
            inspect = False
        )
        mymodels.optimize(
            model_name = "catc",
            cat_features = ['Pclass', 'Sex', 'Embarked'],
            encode_method = None,
            cv = 5,
            trials = 10,
            n_jobs = -1,
            optimize_history = True,
            save_optimal_params = True,
            save_optimal_model = True
        )
        mymodels.evaluate(save_raw_data = True)
        mymodels.explain(
            select_background_data = "train",
            select_shap_data = "test",
            sample_background_data_k = 50,
            sample_shap_data_k = 50,
            output_raw_data = True
        )
    except Exception as e:
        with open("error-titanic-catboost.txt", "a") as f:
            f.write(f"Error in catboost: \n{e}\n")
    
    return None


if __name__ == "__main__":
    titanic_test()
