import pandas as pd
from sklearn.pipeline import Pipeline
import logging, pathlib



from ._data_loader import data_loader
from ._data_diagnoser import MyDataDiagnoser
from ._optimizer import MyOptimizer
from ._evaluator import MyEvaluator
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
        input_data: pd.DataFrame,
        y: str | int, 
        x_list: list[str | int],
        test_ratio: float = 0.3,
        inspect: bool = True
    ):
        """Prepare training and test data"""
        self._x_train, self._x_test, self._y_train, self._y_test = data_loader(
            input_data=input_data,
            y=y,
            x_list=x_list,
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

    

    def diagnose(self, sample_k: int | float | None = None):
        """Diagnose the data"""

        diagnose_x_data = self._x_train.copy()
        diagnose_y_data = self._y_train.copy()

        logging.debug(f"Diagnose data have shapes X_train | Y_train: {diagnose_x_data.shape} {diagnose_y_data.shape}")

        # Sample diagnosis data
        assert sample_k is None or isinstance(sample_k, int) or isinstance(sample_k, float), \
            "sample_k must be an integer or float or None"
        if sample_k is not None:
            if isinstance(sample_k, float):
                sample_k = int(sample_k * len(self._x_train))
                logging.debug(f"Sample {sample_k} samples from {len(self._x_train)}")

            diagnose_data = diagnose_x_data.merge(diagnose_y_data,
                                                  left_index=True,
                                                  right_index=True).sample(sample_k,
                                                                           random_state=self.random_state)
            diagnose_x_data = diagnose_data.iloc[:, :-1]
            diagnose_y_data = diagnose_data.iloc[:, -1]


        diagnoser = MyDataDiagnoser(
            diagnose_x_data,
            diagnose_y_data,
            results_dir = self.results_dir,
            show = self.show,
            plot_format = self.plot_format,
            plot_dpi = self.plot_dpi
        )
        diagnoser.diagnose()
        
        return None



    def optimize(
        self,
        model_name: str,
        data_engineer_pipeline: Pipeline | None = None,
        cv: int = 5,
        trials: int = 50,
        n_jobs: int = 5,
        cat_features: list[str] | tuple[str] | None = None,
        optimize_history: bool = True,
        save_optimal_params: bool = True,
        save_optimal_model: bool = True,
    ):
        """Optimization"""
        
        ###########################################################################################
        # Input validation
        assert model_name in \
            ["lr", "svr", "knr", "mlpr", "dtr", "rfr", "gbdtr", "adar", "xgbr", "lgbr", "catr",
             "lc", "svc", "knc", "mlpc", "dtc", "rfc", "gbdtc", "adac", "xgbc", "lgbc", "catc"], \
            "model_name is invalid"
        self.model_name = model_name

        # Check data_engineer_pipeline validity
        if data_engineer_pipeline is not None:
            assert isinstance(data_engineer_pipeline, Pipeline), \
                "data_engineer_pipeline must be a `sklearn.pipeline.Pipeline` object"
        else:
            logging.warning("No data engineering will be implemented, the raw data will be used.")

        # Check if `cat_features` is explicitly provided for the CatBoost model
        if cat_features is not None:
            assert self.model_name in ["catr", "catc"], \
                "`cat_features` is only supported for CatBoost"
        ###########################################################################################
        
        # Initialize optimizer
        optimizer = MyOptimizer(
            random_state=self.random_state,
            results_dir=self.results_dir,
        )

        # Fit the optimizer
        optimizer.fit(
            x_train=self._x_train,
            y_train=self._y_train,
            x_test=self._x_test,
            model_name=self.model_name,
            data_engineer_pipeline=data_engineer_pipeline,
            cv = cv,
            trials = trials,
            n_jobs = n_jobs,
            cat_features = cat_features
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
    


    def evaluate(self, save_raw_data: bool = True):
        """Evaluate the model

        Args:
            save_raw_data (bool): Whether to save the raw prediction data. Default is True.
        """
        evaluator = MyEvaluator(
            model_name=self.model_name,
            optimal_model_object=self._optimal_model
            )
        evaluator.evaluate(
            y_test = self._y_test,
            y_test_pred = self._y_test_pred,
            y_train = self._y_train,
            y_train_pred = self._y_train_pred,
            X_test = self._used_X_test,
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

        # Identify numeric features by excluding category and object dtypes
        _numeric_features = []
        for col, dtype in self._x_train.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                _numeric_features.append(col)
                
        explainer.explain(
            numeric_features = _numeric_features,
            plot = True,
            show = self.show,
            plot_format = self.plot_format,
            plot_dpi = self.plot_dpi,
            output_raw_data = output_raw_data
        )
        
        return None


    