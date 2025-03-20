from mymodels.pipeline import MyPipeline


def single_test(model_name, encoder_method):
    the_pipeline = MyPipeline(
        model_name = model_name,
        results_dir = "results/" + model_name + "_single_test",
        shap_ratio = 0.5,
        encoder_method = encoder_method,
        **config
    )
    the_pipeline.run()
    return None


def loop_test():
    """Test the combination effects of multiple machine learning models and feature encoding methods
      - Includes 10 models: svr, knr, mlpr, adar, dtr, gbdtr, xgbr, lgbr, rfr, catr
      - Uses 6 encoding methods for categorical features: frequency, onehot, label, target, binary, ordinal
      - Performs hyperparameter optimization and cross-validation for each combination
      - Errors are logged to error.log
      - No encoding is performed if the model is CatBoost
      - For svr, knr, mlpr, adar models, shap_ratio=0.1 (using 10% of test set for SHAP values), shap_ratio=1 for others
    """
    for i in [
        # "svr",
        "knr", 
        # "mlpr", "adar",
        # "dtr", "gbdtr", "xgbr", "lgbr", "rfr",
        # "catr"
    ]:
        # 如果模型是CatBoost，则不进行编码
        if i == "catr":
            try:
                print(f"Running model: {i}")
                the_model = MyPipeline(
                    model_name = i,
                    results_dir = "results/" + i,
                    shap_ratio = 0.5,
                    encoder_method = None,
                    **config
                )
                the_model.run()
            except Exception as error:
                with open("error.log", "a") as log_file:
                    error_message = f"Error with model={i}: {str(error)}\n"
                    log_file.write(error_message)
                print(error_message)
                continue
        else:
            for e in [
                "onehot", "binary",
                "frequency", "label", "ordinal", 
                "target"
            ]:
                if i in ["svr", "knr", "mlpr", "adar"]:
                    # For kernel explainer
                    shap_ratio = 0.1
                else:
                    shap_ratio = 1
                try:
                    print(f"Running model: {i}, encoder: {e}")
                    the_model = MyPipeline(
                        model_name = i,
                        results_dir = "results/" + i + "_" + e,
                        shap_ratio = shap_ratio,
                        encoder_method = e,
                        **config
                    )
                    the_model.run()
                except Exception as error:
                    with open("error.log", "a") as log_file:
                        error_message = f"Error with model={i}, encoder={e}: {str(error)}\n"
                        log_file.write(error_message)
                    print(error_message)
                    continue
    return None



if __name__ == "__main__":
    config = {
        "file_path": "data/data.csv",
        "y": "y",
        "x_list": list(range(1, 18)),
        "cat_features": ['x16', 'x17'],
        "trials": 30,
        "test_ratio": 0.4,
        "random_state": 0,
        "cross_valid": 5,
        "n_jobs": 5,  # Number of jobs to run in parallel k-fold cross validation
    }


    # Test on a single model.
    single_test("rfr", ["onehot", "binary"])


    # Test on all models and encoders.
    # loop_test()
    