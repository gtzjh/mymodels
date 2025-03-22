import pathlib
import pandas as pd

from ._data_loader import data_loader
from ._optimizer import MyOptimizer
from ._evaluator import Evaluator
from ._explainer import MyExplainer


class MyPipeline:
    """Machine Learning Pipeline for Model Training and Evaluation
    A class that handles data loading, model training, and evaluation with SHAP analysis.
    Supports various regression models with hyperparameter optimization and cross-validation.
    """
    def __init__(
            self,
            model_name: str, 
            results_dir: str | pathlib.Path,
            cat_features:  None | list[str] | tuple[str] = None, 
            encode_method: None | str | list[str] | tuple[str] = None,
            random_state: int = 0,
        ):
        self.model_name = model_name
        self.results_dir = pathlib.Path(results_dir)
        self.cat_features = cat_features
        self.encode_method = encode_method
        self.random_state = random_state

        # 先声明全局变量
        self._x_train = None
        self._x_test = None
        self._y_train = None
        self._y_test = None
        self._optimal_model = None
        
        self._check_input()
        self.results_dir.mkdir(parents = True, exist_ok = True)


    def _check_input(self):
        """Check input"""
        assert self.model_name in \
            ["svr", "knr", "mlpr", "dtr", "rfr", "gbdtr", "adar", "xgbr", "lgbr", "catr",
             "svc", "knc", "mlpc", "dtc", "rfc", "gbdtc", "adac", "xgbc", "lgbc", "catc"], \
            "model_name is invalid. Check the document for more details."
        
        assert isinstance(self.results_dir, str) \
            or isinstance(self.results_dir, pathlib.Path), \
            "results_dir must be a string or pathlib.Path"
        
        if self.model_name in ["catr", "catc"]:
            assert self.encode_method is None, "encode_method must be None in which catboost is implemented"
        
        if isinstance(self.cat_features, list) or isinstance(self.cat_features, tuple):
            if self.model_name not in ["catr", "catc"]:
                valid_encoder_methods = ["onehot", "binary", "ordinal", "label", "target", "frequency"]
                if isinstance(self.encode_method, list) or isinstance(self.encode_method, tuple):
                    assert len(self.encode_method) == len(self.cat_features), \
                        "encode_method must have the same length as cat_features"
                    assert all(e in valid_encoder_methods for e in self.encode_method), \
                        "encode_method must be one of the following: \n" \
                        f"[{' '.join(valid_encoder_methods)}]"
                elif isinstance(self.encode_method, str):
                    assert self.encode_method in valid_encoder_methods, \
                        "encode_method must be one of the following: \n" \
                        f"[{' '.join(valid_encoder_methods)}]"
                else:
                    raise ValueError("encode_method must be a list or tuple or str, \
                                     in valid encode_method list as well")
        elif self.cat_features is None:
            pass
        else:
            raise ValueError("cat_features must be a list, tuple or None")

        return None


    def load(self,
             file_path: str | pathlib.Path,
             y: str | int, 
             x_list: list[str | int],
             test_ratio: float = 0.3,
             inspect: bool = True
        ):
        """Prepare training and test data"""
        self._x_train, self._x_test, self._y_train, self._y_test = data_loader(
            file_path=file_path,
            y=y,
            x_list=x_list,
            test_ratio=test_ratio,
            cat_features=self.cat_features,
            random_state=self.random_state
        )
        if inspect:
            print(f"\nTotal samples: {len(self._x_train) + len(self._x_test)}")
            print(f"\nTrain X data info:")
            print(self._x_train.info())
            print(f"\nTrain X data head:")
            print(self._x_train.head(10))
            print(f"\nTrain y data info:")
            print(self._y_train.info())
            print(f"\nTrain y data head:")
            print(self._y_train.head(10))
            print(f"\nTotally features: {self._x_train.shape[1]}")
            print("\nCategorical features:")
            cat_cols = [col for col in self._x_train.columns \
                        if pd.api.types.is_categorical_dtype(self._x_train[col])]
            print(cat_cols)
            print("\nNumerical features:")
            num_cols = [col for col in self._x_train.columns if pd.api.types.is_numeric_dtype(self._x_train[col])]
            print(num_cols, "\n")

        return None


    def optimize(self, cv: int = 5, trials: int = 50, n_jobs: int = 5):
        """Optimize, output the optimal model and encoder objects as well"""
        optimizer = MyOptimizer(
            cv=cv,
            trials=trials,
            random_state=self.random_state,
            results_dir=self.results_dir,
            n_jobs=n_jobs
        )
        optimizer.fit(
            x_train=self._x_train,
            y_train=self._y_train,
            x_test=self._x_test,
            model_name=self.model_name,
            cat_features=self.cat_features,
            encode_method=self.encode_method
        )
        self._optimal_model = optimizer.optimal_model

        self._used_X_train = optimizer.final_x_train
        self._used_X_test = optimizer.final_x_test

        self._y_train_pred = optimizer.y_train_pred
        self._y_test_pred = optimizer.y_test_pred

        return None
    

    def evaluate(self):
        """Evaluate the model
        the results will be saved in the results_dir,
        and output to the console
        """
        evaluator = Evaluator(
            model_name=self.model_name,
            results_dir=self.results_dir,
            plot=False,
            print_results=True,
            save_results=True,
            save_raw_data=True
        )
        evaluator.evaluate(
            y_test = self._y_test,
            y_test_pred = self._y_test_pred,
            y_train = self._y_train,
            y_train_pred = self._y_train_pred
        )

        return None


    def explain(
            self,
            sample_train_k: int | None = None,
            sample_test_k:  int | None = None,
            _plot: bool = True
        ):
        """Use SHAP for explanation
        Use training set to build the explainer, use test set to calculate SHAP values
        """

        assert isinstance(sample_train_k, int) or sample_train_k is None, \
            "sample_train_k must be an integer or None, 100 is recommended when using non-tree model."
        assert isinstance(sample_test_k, int) or sample_test_k is None, \
            "sample_test_k must be an integer or None, 100 is recommended when using non-tree model."
        
        # Explain the model
        explainer = MyExplainer(
            results_dir = self.results_dir,
            model_object = self._optimal_model,
            model_name = self.model_name,
            used_X_train = self._used_X_train,
            used_X_test = self._used_X_test,
            sample_train_k = sample_train_k,
            sample_test_k = sample_test_k,
            cat_features = self.cat_features
        )
        explainer.explain()
        
        if _plot:
            explainer.summary_plot()
            explainer.dependence_plot()
            # explainer.partial_dependence_plot()
        
        return None
    
    
