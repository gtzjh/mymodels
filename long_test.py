from mymodels.pipeline import MyPipeline


def titanic_test():
    try:
        mymodels = MyPipeline(
            results_dir = f"results/catc",
            random_state = 0,
            show = True,  # 是否显示图表
            plot_format = "pdf",  # 图表格式
            plot_dpi = 300  # 图表分辨率
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
            cat_features = ['Pclass', 'Sex', 'Embarked'],
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
        )
    except Exception as e:
        with open("error-titanic-catboost.txt", "a") as f:
            f.write(f"Error in catboost: \n{e}\n")


    for n in ["rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", "svc", "knc", "mlpc"]:
        for e in ["onehot", "binary", "label", "ordinal", "target", "frequency"]:
            try:
                mymodels = MyPipeline(
                    results_dir = f"results/{n}/{e}",
                    random_state = 0,
                    show = True,  # 是否显示图表
                    plot_format = "pdf",  # 图表格式
                    plot_dpi = 300  # 图表分辨率
                )
                mymodels.load(
                    file_path = "data/titanic.csv",
                    y = "Survived",
                    x_list = ["Pclass", "Sex", "Embarked", "Age", "SibSp", "Parch", "Fare"],
                    test_ratio = 0.4,
                    inspect = False
                )
                mymodels.optimize(
                    model_name = n,
                    cat_features = ['Pclass', 'Sex', 'Embarked'],
                    encode_method = e,
                    cv = 5,
                    trials = 10,
                    n_jobs = -1,
                    plot_optimization = True
                )
                mymodels.evaluate()
                mymodels.explain(
                    sample_train_k = 0.5,
                    sample_test_k = 0.5,
                )
            except Exception as error:
                with open("error-titanic-models.txt", "a") as f:
                    f.write(f"Error in {n} model, encoder: {e}: \n{error}\n")
                continue
    
    return None


if __name__ == "__main__":
    titanic_test()
