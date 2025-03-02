import numpy as np
import pandas as pd
from data_loader import data_loader
from myshap import myshap
from pipeline import Pipeline
import pathlib


"""
Machine Learning Pipeline for Model Training and Evaluation
A class that handles data loading, model training, and evaluation with SHAP analysis.
Supports various regression models with hyperparameter optimization and cross-validation.
"""
class MyModels:
    def __init__(self, file_path, y, x_list, model, results_dir,
                 cat_features: None | list[str] = None, encoder_method=None, trials=50, test_ratio=0.3,
                 shap_ratio=0.3, cross_valid=5, random_state=0):
        self.file_path = file_path
        self.y = y
        self.x_list = x_list
        self.model = model
        self.results_dir = results_dir
        self.cat_features = cat_features
        self.encoder_method = encoder_method
        self.trials = trials
        self.test_ratio = test_ratio
        self.shap_ratio = shap_ratio
        self.cross_valid = cross_valid
        self.random_state = random_state
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input parameters"""
        assert isinstance(self.file_path, (str, pathlib.Path)), \
            "`file_path` must be string or Path object"
        assert isinstance(self.y, (str, int)), \
            "`y` must be either a string or index within the whole dataset"
        assert (isinstance(self.x_list, list) and len(self.x_list) > 0) \
            and all(isinstance(x, (str, int)) for x in self.x_list), \
            "`x_list` must be non-empty list of strings or integers"
        assert isinstance(self.model, str) and self.model in ["cat", "rf", "dt", "lgb", "gbdt", "xgb", "ada", "svr", "knr", "mlp"], \
            "`model` must be a string and must be one of ['cat', 'rf', 'dt', 'lgb', 'gbdt', 'xgb', 'ada', 'svr', 'knr', 'mlp']"
        assert isinstance(self.results_dir, (str, pathlib.Path)), \
            "`results_dir` must be string or Path object"
        assert isinstance(self.cat_features, list) or self.cat_features is None, \
            "`cat_features` must be a list or None"
        
        # Add validation for encoder_method
        VALID_ENCODERS = ['onehot', 'label', 'target', 'frequency', 'binary', 'ordinal']
        if self.cat_features is not None:
            assert isinstance(self.encoder_method, str) and self.encoder_method in VALID_ENCODERS, \
                f"`encoder_method` must be one of {VALID_ENCODERS} when cat_features is not None"
        
        assert isinstance(self.trials, int) and self.trials > 0, \
            "`trials` must be a positive integer"
        assert isinstance(self.test_ratio, float) and 0.0 < self.test_ratio < 1.0, \
            "`test_ratio` must be a float between 0 and 1"
        assert isinstance(self.shap_ratio, float) and 0.0 < self.shap_ratio < 1.0, \
            "`shap_ratio` must be a float between 0 and 1"
        assert isinstance(self.cross_valid, int) and self.cross_valid > 0, \
            "`cv` must be a positive integer"
        assert isinstance(self.random_state, int), \
            "`random_state` must be an integer"
        

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
        optimizer = Pipeline(
            cv=self.cross_valid,
            random_state=self.random_state,
            trials=self.trials,
            results_dir=self.results_dir,
        )

        optimizer.fit(
            self.x_train, self.y_train,
            self.model,
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
        
        myshap(self.optimal_model, self.model, shap_data, self.results_dir)
        
        
    def run(self):
        """Execute the whole pipeline"""
        self.load()
        self.optimize()
        # self.explain()
        return None


if __name__ == "__main__":
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

    def test_model(i, e):
        print(f"Running model: {i}, encoder: {e}")
        the_model = MyModels(
            file_path = "data.csv",
            y = "y",
            x_list = list(range(1, 18)),
            model = i,
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