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

        # In engineering()
        self.missing_values_cols = None
        self.impute_method = None
        self.cat_features = None
        self.encode_method = None
        self.scale_cols = None
        self.scale_method = None

        # In optimize()
        self.model_name = None

        # After optimization
        self._optimal_model = None
        self._used_X_train = None
        self._used_X_test = None
        self._y_train_pred = None
        self._y_test_pred = None

        self._check_init_input()

    

    def _check_init_input(self):
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
    

    def engineering(
        self,
        missing_values_cols: list[str] | tuple[str] | None = None,
        impute_method: list[str] | tuple[str] | None = None,
        cat_features: list[str] | tuple[str] | None = None,
        encode_method: list[str] | tuple[str] | None = None,
        scale_cols: list[str] | tuple[str] | None = None,
        scale_method: list[str] | tuple[str] | None = None,
    ):
        """Data engineering for the training and test set.
        
        Step:
        1. Imputation of missing values
        2. Removal of outliers
        3. Encoding of categorical features
        4. Data standardization or normalization
        """
        self.missing_values_cols = missing_values_cols
        self.impute_method = impute_method
        self.cat_features = cat_features
        self.encode_method = encode_method
        self.scale_cols = scale_cols
        self.scale_method = scale_method

        return None



    def optimize(
        self,
        model_name: str,
        cv: int = 5,
        trials: int = 50,
        n_jobs: int = 5,
        optimize_history: bool = True,
        save_optimal_params: bool = True,
        save_optimal_model: bool = True,
    ):
        """Optimize using Optuna"""
        
        # Check model_name validity
        assert model_name in \
            ["svr", "knr", "mlpr", "dtr", "rfr", "gbdtr", "adar", "xgbr", "lgbr", "catr",
             "svc", "knc", "mlpc", "dtc", "rfc", "gbdtc", "adac", "xgbc", "lgbc", "catc"], \
            "model_name is invalid"
        self.model_name = model_name

        
        # Initialize optimizer
        optimizer = MyOptimizer(
            random_state=self.random_state,
            results_dir=self.results_dir,
        )

        # Data engineering
        optimizer.engineering(
            missing_values_cols = self.missing_values_cols,
            impute_method = self.impute_method,
            cat_features = self.cat_features,
            encode_method = self.encode_method,
            scale_cols = self.scale_cols,
            scale_method = self.scale_method,
        )

        # Fit the optimizer
        optimizer.fit(
            x_train=self._x_train,
            y_train=self._y_train,
            x_test=self._x_test,
            model_name=self.model_name,
            cv = cv,
            trials = trials,
            n_jobs = n_jobs,
        )

        # Output the optimization history, optimal model and parameters
        optimizer.output(
            optimize_history = optimize_history,
            save_optimal_params = save_optimal_params,
            save_optimal_model = save_optimal_model,
            show = self.show,
            plot_format = self.plot_format,
            plot_dpi = self.plot_dpi
        )
        
        # Output data for evaluate()
        self._y_train_pred = optimizer.y_train_pred
        self._y_test_pred = optimizer.y_test_pred

        # Output data for explain()
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
        )

        # Output the explanation results
        """Input the numeric features for Partial Dependence Plot
        I known this is very ugly, but it's the only way to make the Partial Dependence Plot (PDP) work correctly.
        If you have any better idea, please let me know [zhongjh86@outlook.com].
        """
        _numeric_features = self._x_train.columns.tolist()
        if self.cat_features:
            _numeric_features = [x for x in _numeric_features if x not in self.cat_features]
        explainer.explain(
            numeric_features = _numeric_features,
            plot = True,
            show = self.show,
            plot_format = self.plot_format,
            plot_dpi = self.plot_dpi,
            output_raw_data = output_raw_data
        )
        
        return None
    