from mymodels.pipeline import MyPipeline


def single_test(model_name, encoder_method):
    the_pipeline = MyPipeline(
        file_path = "data.csv",
        y = "y",    
        x_list = list(range(1, 18)),
        model_name = model_name,
        results_dir = "results/" + model_name + "_" + encoder_method,
        cat_features = ['x16', 'x17'],
        encoder_method = encoder_method,
        trials = 15,
        test_ratio = 0.3,
        shap_ratio = 0.5,
        cross_valid = 5,
        random_state = 6,
        n_jobs = 5,  # Number of jobs to run in parallel k-fold cross validation
    )
    the_pipeline.run()
    return None


def loop_test():
    """
    测试多种机器学习模型和特征编码方法的组合效果:
    - 包含10种模型：svr, knr, mlpr, adar, dtr, gbdtr, xgbr, lgbr, rfr, catr
    - 使用6种编码方法处理分类特征：frequency, onehot, label, target, binary, ordinal
    - 对每种组合进行超参数优化和交叉验证
    - 错误会记录到error.
    - 如果模型是CatBoost，则不进行编码
    - 如果模型是svr, knr, mlpr, adar，则shap_ratio=0.1，表示用测试集的10%来计算shap值，其他模型shap_ratio=1
    """
    for i in [
        "svr", "knr", "mlpr", "adar",
        "dtr", "gbdtr", "xgbr", "lgbr", "rfr",
        "catr"
    ]:
        if i == "catr":  # 如果模型是CatBoost，则不进行编码
            try:
                print(f"Running model: {i}")
                the_model = MyPipeline(
                    file_path = "data.csv",
                    y = "y",
                    x_list = list(range(1, 18)),
                    model_name = i,
                    results_dir = "results/" + i,
                    cat_features = ['x16', 'x17'],
                    encoder_method = None,
                    trials = 50,
                    test_ratio = 0.3,
                    shap_ratio = 1,
                    cross_valid = 5,
                    random_state = 6,
                    n_jobs = 5,
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
                "onehot", 
                "frequency", "label", "binary", "ordinal", 
                "target"
            ]:
                if i in ["svr", "knr", "mlpr", "adar"]:
                    _shap_ratio = 0.1  # For kernel explainer
                else:
                    _shap_ratio = 0.5  # For tree explainer
                try:
                    print(f"Running model: {i}, encoder: {e}")
                    the_model = MyPipeline(
                        file_path = "data.csv",
                        y = "y",
                        x_list = list(range(1, 18)),
                        model_name = i,
                        results_dir = "results/" + i + "_" + e,
                        cat_features = ['x16', 'x17'],
                        encoder_method = e,
                        trials = 50,
                        test_ratio = 0.3,
                        shap_ratio = _shap_ratio,
                        cross_valid = 5,
                        random_state = 6,
                        n_jobs = 5,
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
    # Test on a single model.
    # single_test("rfr", "onehot")

    # Test on all models and encoders.
    loop_test()
    
