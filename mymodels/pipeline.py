import pathlib

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
            cat_features:   None | list[str] | tuple[str] = None, 
            encode_method: None | str | list[str] | tuple[str] = None,
            random_state: int = 0,
        ):
        self.model_name = model_name
        self.results_dir = pathlib.Path(results_dir)
        self.cat_features = cat_features
        self.encode_method = encode_method
        self.random_state = random_state
        
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
        
        if isinstance(self.cat_features, list) or isinstance(self.cat_features, tuple):
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
                raise ValueError("encode_method must be a list or tuple or str, in valid encode_method list as well")
        elif self.cat_features is None:
            pass
        else:
            raise ValueError("cat_features must be a list, tuple or None")

        return None


    def load(self,
             file_path: str | pathlib.Path,
             y: str | int, 
             x_list: list[str | int],
             test_ratio: float = 0.3
        ):
        """Prepare training and test data"""
        self.x_train, self.x_test, self.y_train, self.y_test = data_loader(
            file_path=file_path,
            y=y,
            x_list=x_list,
            test_ratio=test_ratio,
            cat_features=self.cat_features,
            random_state=self.random_state
        )
        return None


    def inspect(self):
        """Inspect the data after loading
        
        Raises:
            AttributeError: If data has not been loaded via load() method
            ValueError: If loaded data attributes are not in expected format
        """
        # Check if data has been loaded
        required_attrs = ['x_train', 'x_test', 'y_train', 'y_test']
        if not all(hasattr(self, attr) for attr in required_attrs):
            raise AttributeError(
                "Data has not been loaded. Please call load() method first."
            )

        # Validate data format and content
        if not (len(self.x_train) > 0 and len(self.x_test) > 0 \
                and len(self.y_train) > 0 and len(self.y_test) > 0):
            raise ValueError("Loaded data appears to be empty")

        if not (self.x_train.shape[0] == self.y_train.shape[0] \
                and self.x_test.shape[0] == self.y_test.shape[0]):
            raise ValueError("Mismatch between X and y data dimensions")

        """
        print(f"\nDATA INFO:")
        print(f"Total samples:             {len(self.x_train) + len(self.x_test)}")
        print(f"Train & Validate samples:  {len(self.x_train)} ({(1-self.test_ratio)*100:.1f}%)")
        print(f"Test samples:              {len(self.x_test)} ({self.test_ratio*100:.1f}%)")
        print(f"\nCROSS VALIDATION INFO:")
        print(f"Totally {self.cross_valid} folds")
        fold_train = len(self.x_train) * (self.cross_valid-1)/self.cross_valid
        fold_val = len(self.x_train) / self.cross_valid
        print(f"Per fold\n  Train:     {fold_train:.0f}\n  Validate:  {fold_val:.0f}")
        print(f"\nFEATURE INFO:")
        print(f"Total features:        {self.x_train.shape[1]}")
        if self.cat_features:
            print(f"Categorical features:  {len(self.cat_features)}")
            print(f"Numerical features:    {self.x_train.shape[1] - len(self.cat_features)}")

        print(f"\nTRAIN X DATA INFO:")
        print(self.x_train.info())
        print(f"\nTRAIN X DATA HEAD:")
        print(self.x_train.head(10))
        print(f"\nTRAIN Y DATA INFO:")
        print(self.y_train.info())
        print(f"\nTRAIN Y DATA HEAD:")
        print(self.y_train.head(10), "\n")
        """

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
            x_train=self.x_train,
            y_train=self.y_train,
            model_name=self.model_name,
            cat_features=self.cat_features,
            encode_method=self.encode_method
        )
        self.optimal_model = optimizer.optimal_model
        self.encoder_dict = optimizer.encoder_dict
    

    def evaluate(self):
        """Evaluate the model
        the results will be saved in the results_dir,
        and output to the console
        """
        evaluator = Evaluator(
            model_name=self.model_name,
            model_obj=self.optimal_model,
            results_dir=self.results_dir,
            encoder_dict=self.encoder_dict,
            cat_features=self.cat_features,
            plot=False,
            print_results=True,
            save_results=True,
            save_raw_data=True
        )
        evaluator.evaluate(
            x_test=self.x_test,
            y_test=self.y_test,
            x_train=self.x_train,
            y_train=self.y_train
        )


    def explain(self, shap_ratio: float = 0.3, _plot: bool = True):
        """Use SHAP for explanation"""
        # Sampling for reduce the time cost
        shap_data = self.x_test.sample(
            n = int(len(self.x_test) * shap_ratio),
            random_state = self.random_state
        )
        # Explain the model
        explainer = MyExplainer(
            model_object = self.optimal_model,
            model_name = self.model_name,
            shap_data = shap_data,
            results_dir = self.results_dir,
            encoder_dict = self.encoder_dict,
            cat_features = self.cat_features
        )
        explainer.explain()
        if _plot:
            explainer.summary_plot()
            explainer.dependence_plot()
            explainer.partial_dependence_plot()
        
        
    # 一步到位
    def run(self,
            file_path: str | pathlib.Path,
            y: str | int, 
            x_list: list[str | int],
            test_ratio: float = 0.3,
            cv: int = 5,
            trials: int = 100,
            n_jobs: int = 5,
            shap_ratio: float = 0.3,
            inspect: bool = True,
            interpret: bool = True
        ):
        """Execute the whole pipeline"""
        self.load(
            file_path = file_path,
            y = y,
            x_list = x_list,
            test_ratio = test_ratio
        )
        
        if inspect:
            self.inspect()

        self.optimize(cv = cv, trials = trials, n_jobs = n_jobs)  # 可能这里可以考虑，调用已经训练好的模型，这个以后再弄
        
        self.evaluate()
        
        if interpret:
            self.explain(shap_ratio = shap_ratio, _plot = True)

        return None
    
