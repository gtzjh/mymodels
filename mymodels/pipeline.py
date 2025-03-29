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
            results_dir: str | pathlib.Path,
            random_state: int = 0,
            show: bool = False,
            plot_format: str = "jpg",
            plot_dpi: int = 500
        ):
        self.results_dir = results_dir
        self.random_state = random_state
        self.show = show
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi

        # Global variables statement
        # In load()
        self._x_train = None
        self._x_test = None
        self._y_train = None
        self._y_test = None

        # In optimize()
        self.model_name = None
        self.cat_features = None
        self.encode_method = None

        # After optimization
        self._optimal_model = None
        self._used_X_train = None
        self._used_X_test = None
        self._y_train_pred = None
        self._y_test_pred = None

        self._check_pipeline_input()

    

    def _check_pipeline_input(self):
        """Check input parameters for pipeline.
        
        This function validates the following parameters:
            - results_dir: Must be a valid directory path
            - random_state: Must be an integer
            - show: Must be a boolean

        Raises:
            ValueError: If any parameter validation fails
        """
        assert isinstance(self.results_dir, pathlib.Path) or isinstance(self.results_dir, str), \
            "results_dir must be a valid directory path"
        if isinstance(self.results_dir, str):
            self.results_dir = pathlib.Path(self.results_dir)
        self.results_dir.mkdir(parents = True, exist_ok = True)
        assert isinstance(self.random_state, int), "random_state must be an integer"
        assert isinstance(self.show, bool), "show must be a boolean"
        assert isinstance(self.plot_format, str), "plot_format must be a string"
        assert self.plot_format in ["jpg", "png", "jpeg", "tiff", "pdf", "svg", "eps"], \
            "plot_format must be one of the following: jpg, png, jpeg, tiff, pdf, svg, eps"
        assert isinstance(self.plot_dpi, int), "plot_dpi must be an integer"

        return None



    def load(
        self,
        file_path: str | pathlib.Path,
        y: str | int, 
        x_list: list[str | int],
        index_col: str | int | list[str | int] | tuple[str | int] | None = None,
        test_ratio: float = 0.3,
        inspect: bool = True
    ):
        """Prepare training and test data"""
        self._x_train, self._x_test, self._y_train, self._y_test = data_loader(
            file_path=file_path,
            y=y,
            x_list=x_list,
            index_col=index_col,
            test_ratio=test_ratio,
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

        return None



    def optimize(
        self,
        model_name: str,
        cat_features: list[str] | tuple[str] | None = None,
        encode_method: str | list[str] | tuple[str] | None = None,
        cv: int = 5,
        trials: int = 50,
        n_jobs: int = 5,
        plot_optimization: bool = True
    ):
        """Optimize, output the optimal model and encoder objects as well"""

        self.model_name = model_name
        self.cat_features = list(cat_features) if cat_features is not None else None
        self.encode_method = encode_method
        self._check_optimize_input()

        optimizer = MyOptimizer(
            cv=cv,
            trials=trials,
            random_state=self.random_state,
            results_dir=self.results_dir,
            n_jobs=n_jobs,
            _plot_optimization=plot_optimization
        )

        optimizer.fit(
            x_train=self._x_train,
            y_train=self._y_train,
            x_test=self._x_test,
            model_name=self.model_name,
            cat_features=self.cat_features,
            encode_method=self.encode_method,
            show = self.show,
            plot_format = self.plot_format,
            plot_dpi = self.plot_dpi
        )
        
        # For evaluate()
        self._y_train_pred = optimizer.y_train_pred
        self._y_test_pred = optimizer.y_test_pred

        # For explain()
        self._optimal_model = optimizer.optimal_model
        self._used_X_train = optimizer.final_x_train
        self._used_X_test = optimizer.final_x_test

        return None
    


    def evaluate(
            self,
            save_raw_data: bool = True
        ):
        """Evaluate the model

        Args:
            save_raw_data (bool): Whether to save the raw prediction data. Default is True.
        """
        evaluator = Evaluator(model_name=self.model_name)
        evaluator.evaluate(
            y_test = self._y_test,
            y_test_pred = self._y_test_pred,
            y_train = self._y_train,
            y_train_pred = self._y_train_pred,
            results_dir = self.results_dir,
            show = self.show,
            plot_format = self.plot_format,
            plot_dpi = self.plot_dpi,
            print_results = True,
            save_results = True,
            save_raw_data = save_raw_data
        )

        return None



    def explain(
            self,
            select_background_data: str = "train",
            select_shap_data: str = "test",
            sample_background_data_k: int | float | None = None,
            sample_shap_data_k:  int | float | None = None,
            output_raw_data: bool = False
        ):
        """Use SHAP for explanation
        Use training set to build the explainer, use test set to calculate SHAP values is the default behavior.
        """
        # Check input parameters
        assert select_background_data in ["train", "test", "all"], \
            "select_background_data must be one of the following: train, test, all"
        assert select_shap_data in ["train", "test", "all"], \
            "select_shap_data must be one of the following: train, test, all"

        assert isinstance(sample_background_data_k, (int, float)) or sample_background_data_k is None, \
            "sample_background_data_k must be an integer or float or None, 100 is recommended for explaining non-tree model."
        assert isinstance(sample_shap_data_k, (int, float)) or sample_shap_data_k is None, \
            "sample_shap_data_k must be an integer or float or None, 100 is recommended for explaining non-tree model."
        
        # Background data for building the explainer
        if select_background_data == "train":
            _background_data = self._used_X_train
        elif select_background_data == "test":
            _background_data = self._used_X_test
        elif select_background_data == "all":
            _background_data = pd.concat([self._used_X_train, self._used_X_test]).sort_index()

        # SHAP data for calculating SHAP values
        if select_shap_data == "train":
            _shap_data = self._used_X_train
        elif select_shap_data == "test":
            _shap_data = self._used_X_test
        elif select_shap_data == "all":
            _shap_data = pd.concat([self._used_X_train, self._used_X_test]).sort_index()


        # Explain the model
        explainer = MyExplainer(
            results_dir = self.results_dir,
            model_object = self._optimal_model,
            model_name = self.model_name,
            background_data = _background_data,
            shap_data = _shap_data,
            sample_background_data_k = sample_background_data_k,
            sample_shap_data_k = sample_shap_data_k,
            cat_features = self.cat_features
        )

        # Output the explanation results
        explainer.explain(
            plot = True,
            show = self.show,
            plot_format = self.plot_format,
            plot_dpi = self.plot_dpi,
            output_raw_data = output_raw_data
        )
        
        return None
    


    def _check_optimize_input(self):
        """Check input parameters for optimization.
        
        This function validates the following parameters:
            - model_name: Must be one of the supported model types
            - cat_features: Must be a valid list/tuple of string column names or None
            - encode_method: Must be compatible with the selected model and cat_features
        
        The function enforces these validation rules:
            1. model_name must be one of the supported regression or classification models
            2. For CatBoost models (catr, catc), encode_method must be None
            3. If cat_features is provided:
               - It must be a non-empty list/tuple of strings
               - It must not contain duplicates
               - For non-CatBoost models, encode_method must be provided and valid
            4. If cat_features is None, encode_method must also be None
            5. If y is float, regression model must be choosed
        
        Raises:
            ValueError: If any parameter validation fails
            AssertionError: If model_name is invalid or encode_method is provided with CatBoost
        """

        # Check model_name validity
        assert self.model_name in \
            ["svr", "knr", "mlpr", "dtr", "rfr", "gbdtr", "adar", "xgbr", "lgbr", "catr",
             "svc", "knc", "mlpc", "dtc", "rfc", "gbdtc", "adac", "xgbc", "lgbc", "catc"], \
            "model_name is invalid"

        # Check if the target variable is a continuous variable
        if pd.api.types.is_float_dtype(self._y_train.dtype):
            assert self.model_name in ["svr", "knr", "mlpr", "dtr", "rfr", "gbdtr", "adar", "xgbr", "lgbr", "catr"], \
                "The target variable is a continuous variable, but you haven't choosed a regression model"
        
        # CatBoost models specific validation
        if self.model_name in ["catr", "catc"]:
            assert self.encode_method is None, "encode_method must be None when using CatBoost models"
        
        # Categorical features validation
        if self.cat_features is not None:
            # Type checking
            if not isinstance(self.cat_features, (list, tuple)):
                raise ValueError("cat_features must be a list, tuple or None")
            
            # Empty list check
            if len(self.cat_features) == 0:
                raise ValueError("cat_features should be None instead of an empty list or tuple")
            
            # Element type check
            if not all(isinstance(feature, str) for feature in self.cat_features):
                raise ValueError("All elements in cat_features must be strings (column names)")
            
            # Duplicate check
            if len(self.cat_features) != len(set(self.cat_features)):
                raise ValueError("cat_features contains duplicate feature names")
            
            # For non-CatBoost models, encoding method is required
            if self.model_name not in ["catr", "catc"]:
                valid_encoder_methods = ["onehot", "binary", "ordinal", "label", "target", "frequency"]
                
                # encode_method must be provided
                if self.encode_method is None:
                    raise ValueError("encode_method must be provided when using categorical features with non-CatBoost models")
                
                # List/tuple of encoding methods
                if isinstance(self.encode_method, (list, tuple)):
                    # Check length match
                    if len(self.encode_method) != len(self.cat_features):
                        raise ValueError("encode_method must have the same length as cat_features")
                    
                    # Check element types
                    if not all(isinstance(e, str) for e in self.encode_method):
                        raise ValueError("All elements in encode_method must be strings")
                    
                    # Check valid methods
                    invalid_methods = [e for e in self.encode_method if e not in valid_encoder_methods]
                    if invalid_methods:
                        raise ValueError(f"Invalid encoding methods: {invalid_methods}. "
                                        f"Valid methods are: {valid_encoder_methods}")
                
                # Single encoding method for all features
                elif isinstance(self.encode_method, str):
                    if self.encode_method not in valid_encoder_methods:
                        raise ValueError(f"Invalid encoding method: {self.encode_method}. "
                                        f"Valid methods are: {valid_encoder_methods}")
                
                # Invalid encode_method type
                else:
                    raise ValueError("encode_method must be a list, tuple, or string")
        else:
            # When cat_features is None, encode_method should also be None
            if self.encode_method is not None:
                raise ValueError("encode_method should be None when cat_features is None")

        return None
