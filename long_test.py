from mymodels.pipeline import MyPipeline


def car_prices_test():
    try:
        mymodels = MyPipeline(
            results_dir = f"results/catr",
            random_state = 0,
        )
        mymodels.load(
            file_path = "data/car_prices_cleaned.csv",
            y = "Price",
            x_list = ["Levy", "Manufacturer", "Model", "Prod. year", "Category", "Leather interior",
                      "Fuel type", "Engine volume", "Mileage", "Cylinders", "Gear box type",
                      "Drive wheels", "Doors", "Wheel", "Color", "Airbags"],
            test_ratio = 0.4,
            inspect = False
        )
        mymodels.optimize(
            model_name = "catr",
            cat_features = ['Manufacturer', 'Model', 'Category', 'Color',
                            'Fuel type', 'Gear box type', 'Drive wheels', 'Doors'],
            encode_method = None,
            cv = 5,
            trials = 30,
            n_jobs = 5,
            plot_optimization = True
        )
        mymodels.evaluate()
        mymodels.explain(
            sample_train_k = 50,
            sample_test_k = 50,
            plot = True
        )
    except Exception as e:
        with open("error-car-prices-catboost.txt", "a") as f:
            f.write(f"Error in catboost: \n{e}\n")

    """
    # car prices的数据集太大, 所以不进行其他模型的测试
    for n in ["lgbr", "rfr", "dtr", "gbdtr", "xgbr", "adar", "svr", "knr", "mlpr"]:
        for e in ["onehot", "binary", "label", "ordinal", "target", "frequency"]:
            try:
                mymodels = MyPipeline(
                    results_dir = f"results/{n}/{e}",
                    random_state = 0,
                )
                mymodels.load(
                    file_path = "data/car_prices_cleaned.csv",
                    y = "Price",
                    x_list = ["Levy", "Manufacturer", "Model", "Prod. year", "Category", "Leather interior",
                              "Fuel type", "Engine volume", "Mileage", "Cylinders", "Gear box type",
                              "Drive wheels", "Doors", "Wheel", "Color", "Airbags"],
                    test_ratio = 0.4,
                    inspect = False
                )
                mymodels.optimize(
                    model_name = n,
                    cat_features = ['Manufacturer', 'Model', 'Category', 'Color',
                                    'Fuel type', 'Gear box type', 'Drive wheels', 'Doors'],
                    encode_method = e,
                    cv = 5,
                    trials = 30,
                    n_jobs = 5,
                    plot_optimization = True
                )
                mymodels.evaluate()
                mymodels.explain(
                    sample_train_k = 50,
                    sample_test_k = 50,
                    plot = True
                )
            except Exception as error:
                with open("error-car-prices-models.txt", "a") as f:
                    f.write(f"Error in {n} model, encoder: {e}: \n{error}\n")
                continue
    """

    return None


def titanic_test():
    try:
        mymodels = MyPipeline(
            results_dir = f"results/catc",
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
            plot = True
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
                    n_jobs = 5,
                    plot_optimization = True
                )
                mymodels.evaluate()
                mymodels.explain(
                    sample_train_k = 50,
                    sample_test_k = 50,
                    plot = True
                )
            except Exception as error:
                with open("error-titanic-models.txt", "a") as f:
                    f.write(f"Error in {n} model, encoder: {e}: \n{error}\n")
                continue
    
    return None


if __name__ == "__main__":
    car_prices_test()
    titanic_test()
