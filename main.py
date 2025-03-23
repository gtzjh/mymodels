from mymodels.pipeline import MyPipeline


def main():
    mymodels = MyPipeline(
        results_dir = "results/lgbc",
        random_state = 0,
    )
    mymodels.load(
        file_path = "data/iris.csv",
        y = "Species",
        x_list = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        test_ratio = 0.4,
        inspect = False
    )
    mymodels.optimize(
        model_name = "lgbc",
        cat_features = None,
        encode_method = None,
        cv = 5,
        trials = 10,
        n_jobs = -1,
        plot_optimization = False
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