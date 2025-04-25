import pandas as pd
from sklearn.pipeline import Pipeline
import logging, pathlib


from mymodels import MyDataLoader, MyDataDiagnoser, MyEstimator, MyOptimizer, MyEvaluator, MyExplainer



class MyPipeline:
    """Machine Learning Pipeline for Model Training and Evaluation
    A class that handles data loading, model training, and evaluation with SHAP analysis.
    Supports various regression models with hyperparameter optimization and cross-validation.
    """
    def __init__(
            self,
            random_state: int = 0,
            stratify: bool = False,
        ):

        self.random_state = random_state
        self.stratify = stratify

        # Global variables statement
        self.dataset = None
        self.estimator = None
        self.data_engineer_pipeline = None

        self.optimized_estimator = None
        self.optimized_dataset = None
        self.optimized_data_engineer_pipeline = None


    def load(
        self,
        model_name: str,
        input_data: pd.DataFrame,
        y: str | int, 
        x_list: list[str | int],
        data_engineer_pipeline: Pipeline | None = None,
        test_ratio: float = 0.3,
        cat_features: list[str] | tuple[str] | None = None,
        model_configs_path: str = 'model_configs.yml',
    ):
        """Load the estimator, (train/test) dataset, and data engineer pipeline"""

        # Load the estimator
        self.estimator = MyEstimator(
            cat_features = cat_features,
            model_configs_path = model_configs_path
        ).load(model_name = model_name)

        # Load the dataset
        self.dataset = MyDataLoader(
            input_data=input_data,
            y=y,
            x_list=x_list,
            test_ratio=test_ratio,
            random_state=self.random_state,
            stratify=self.stratify
        ).load()

        # Load the data engineer pipeline
        self.data_engineer_pipeline = data_engineer_pipeline
    
        return None


    """
    def diagnose(self, sample_k: int | float | None = None):

        diagnose_x_data = self._x_train.copy(deep=True)
        diagnose_y_data = self._y_train.copy(deep=True)

        # Sample diagnosis data
        assert sample_k is None or isinstance(sample_k, int) or isinstance(sample_k, float), \
            "sample_k must be an integer or float or None"
        if sample_k is not None:
            if isinstance(sample_k, float):
                sample_k = int(sample_k * len(self._x_train))

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
    """



    def optimize(
        self,
        strategy = "tpe",
        cv: int = 5,
        trials: int = 50,
        n_jobs: int = 5,
        direction: str = "maximize",
        eval_function: None = None,
    ):
        """Optimization"""
        
        # Initialize optimizer
        optimizer = MyOptimizer(
            random_state=self.random_state,
            stratify = self.stratify
        )

        # Fit the optimizer
        self.optimized_estimator, self.optimized_dataset, self.optimized_data_engineer_pipeline = optimizer.fit(
            dateset = self.dataset,
            estimator = self.estimator,
            data_engineer_pipeline=self.data_engineer_pipeline,
            strategy = strategy,
            cv = cv,
            trials = trials,
            n_jobs = n_jobs,
            direction = direction,
            eval_function = eval_function
        )

        return None
    


    def evaluate(
            self,
            show_train: bool = True,
            dummy: bool = True,
            eval_metric: dict | None = None
        ):
        """Evaluate the model

        Args:
            show_train (bool): Whether to show the training set evaluation results. Default is True.
            dummy (bool): Whether to use a dummy estimator for comparison. Default is True.
            save_raw_data (bool): Whether to save the raw prediction data. Default is True.
            eval_metric (None): The self-defined evaluation metric to use. 
                It must be None or a dictionary where each item is a callable function. Default is None.
        """

        evaluator = MyEvaluator()
        evaluator.evaluate(

            data_engineer_pipeline = self.data_engineer_pipeline,
            show_train = show_train,
            dummy = dummy,
            eval_metric = eval_metric,
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
        

        # Transform X data
        if self.data_engineer_pipeline:
            _used_x_train = self.data_engineer_pipeline.transform(self._x_train)
            _used_x_test = self.data_engineer_pipeline.transform(self._x_test)
        else:
            _used_x_train = self._x_train
            _used_x_test = self._x_test
        
        # Background data for building the explainer
        if select_background_data == "train":
            _background_data = _used_x_train
        elif select_background_data == "test":
            _background_data = _used_x_test
        elif select_background_data == "all":
            _background_data = pd.concat([_used_x_train, _used_x_test]).sort_index()

        # SHAP data for calculating SHAP values
        if select_shap_data == "train":
            _shap_data = _used_x_train
        elif select_shap_data == "test":
            _shap_data = _used_x_test
        elif select_shap_data == "all":
            _shap_data = pd.concat([_used_x_train, _used_x_test]).sort_index()


        # Explain the model
        explainer = MyExplainer(
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
            results_dir = self.results_dir,
            numeric_features = _numeric_features,
            output_raw_data = output_raw_data
        )
        
        return None


    