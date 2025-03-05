import numpy as np
import pandas as pd
import pathlib

from data_loader import data_loader
from myoptimizer import MyOptimizer
from myshap import MySHAP



class MyPipeline:
    """Machine Learning Pipeline for Model Training and Evaluation
    A class that handles data loading, model training, and evaluation with SHAP analysis.
    Supports various regression models with hyperparameter optimization and cross-validation.
    """
    def __init__(
            self, 
            file_path: str | pathlib.Path, 
            y: str | int, 
            x_list: list[str | int], 
            model_name: str, 
            results_dir: str | pathlib.Path,
            cat_features: None | list[str] = None, 
            encoder_method: str | None = None,
            trials: int = 50,
            test_ratio: float = 0.3,
            shap_ratio: float = 0.3,
            cross_valid: int = 5,
            random_state: int | None = 0
        ):
        self.file_path = file_path
        self.y = y
        self.x_list = x_list
        self.model_name = model_name
        self.results_dir = results_dir
        self.cat_features = cat_features
        self.encoder_method = encoder_method
        self.trials = trials
        self.test_ratio = test_ratio
        self.shap_ratio = shap_ratio
        self.cross_valid = cross_valid
        self.random_state = random_state
        

    def load(self):
        """Prepare training and test data"""
        self.x_train, self.x_test, self.y_train, self.y_test = data_loader(
            file_path=self.file_path,
            y=self.y,
            x_list=self.x_list,
            cat_features=self.cat_features,
            test_ratio=self.test_ratio,
            random_state=self.random_state
        )


    def optimize(self):
        """Optimize and evaluate the model"""
        optimizer = MyOptimizer(
            cv=self.cross_valid,
            random_state=self.random_state,
            trials=self.trials,
            results_dir=self.results_dir,
        )

        optimizer.fit(
            self.x_train, self.y_train,
            self.model_name,
            self.cat_features, self.encoder_method
        )
        self.optimal_model = optimizer.optimal_model

        optimizer.evaluate(
            self.optimal_model,
            self.x_test,
            self.y_test
        )


    def explain(self):
        """Use SHAP for explanation"""
        np.random.seed(self.random_state)
        all_data = pd.concat([self.x_train, self.x_test])
        
        # 直接随机选择数据行，而不是通过索引
        shap_sample_size = int(len(all_data) * self.shap_ratio)
        shap_data = all_data.sample(n=shap_sample_size, random_state=self.random_state)
        
        myshap = MySHAP(self.optimal_model, self.model_name, shap_data, self.results_dir)
        myshap.summary_plot()
        myshap.dependence_plot()
        myshap.partial_dependence_plot()
        
        
    def run(self):
        """Execute the whole pipeline"""
        self.load()
        self.optimize()
        self.explain()
        return None


if __name__ == "__main__":

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
            trials = 20,
            test_ratio = 0.3,
            shap_ratio = 0.3,
            cross_valid = 5,
            random_state = 0,
        )
        the_model.run()
        return None
    test_model("cat", "frequency")


    """
    This code performs a comprehensive machine learning model evaluation test:
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
    