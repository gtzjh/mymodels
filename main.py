from mymodels.pipeline import MyPipeline


def main():
    the_model = MyPipeline(
        file_path = "data.csv",
        y = "y",
        x_list = list(range(1, 18)),
        model_name = "cat",
        results_dir = "results/cat",
        cat_features = ['x16', 'x17'],
        encoder_method = None,
        trials = 20,
        test_ratio = 0.3,
        shap_ratio = 0.3,
        cross_valid = 5,
        random_state = 0,
    )
    the_model.run()
    return None


if __name__ == "__main__":
    main()


    """
    The following code performs a comprehensive machine learning model evaluation test:
    1. Tests 10 different machine learning models: Support Vector Regression (svr), K-Nearest Neighbors Regression (knr), 
       Multi-layer Perceptron (mlp), AdaBoost (ada), Decision Tree (dt), Gradient Boosting Decision Tree (gbdt), 
       XGBoost (xgb), LightGBM (lgb), Random Forest (rf), and CatBoost (cat)
    2. For each model, tests 6 different categorical feature encoding methods: frequency encoding (frequency), 
       one-hot encoding (onehot), label encoding (label), target encoding (target), binary encoding (binary), 
       and ordinal encoding (ordinal)
    3. Uses the data.csv dataset, with the 'y' column as the target variable and columns 1-17 as features
    4. Treats 'x16' and 'x17' columns as categorical features for encoding
    5. Performs 50 hyperparameter optimization attempts for each model configuration
    6. Uses 30% of the data as the test set with 5-fold cross-validation
    7. All errors are logged to the error.log file to ensure the testing process continues even if 
       individual models or encoding methods fail
    """

    """
    def test_model(i, e):
        print(f"Running model: {i}, encoder: {e}")
        the_model = MyPipeline(
            file_path = "data.csv",
            y = "y",
            x_list = list(range(1, 18)),
            model_name = i,
            results_dir = "results/" + i + "_" + e,
            cat_features = ['x16', 'x17'],
            encoder_method = e,
            trials = 100,
            test_ratio = 0.3,
            shap_ratio = 0.3,
            cross_valid = 5,
            random_state = 0,
        )
        the_model.run()
        return None
    
    for i in [
        "svr", "knr", "mlp", "ada", "dt", "gbdt", "xgb", "lgb", 
        "rf",
        "cat"
    ]:
        if i == "cat":  # 如果模型是CatBoost，则不进行编码
            try:
                test_model(i, "frequency")  # 实际上这个frequency是不会生效
            except Exception as error:
                # 记录错误信息到error.log文件
                with open("error.log", "a") as log_file:
                    error_message = f"Error with model={i}: {str(error)}\n"
                    log_file.write(error_message)
                    log_file.write("-" * 80 + "\n")
                print(f"Error occurred with model={i}. Details logged to error.log")
                # 继续下一个循环
                continue
        else:
            for e in [
                "frequency", "onehot", "label", "binary", "ordinal", 
                "target"
            ]:
                try:
                    test_model(i, e)
                except Exception as error:
                    with open("error.log", "a") as log_file:
                        error_message = f"Error with model={i}, encoder={e}: {str(error)}\n"
                        log_file.write(error_message)
                        log_file.write("-" * 80 + "\n")
                    print(f"Error occurred with model={i}, encoder={e}. Details logged to error.log")
                    continue
    """
    

