import pandas as pd
from sklearn.pipeline import Pipeline



from mymodels import MyDataLoader, MyEstimator
from mymodels import MyDataDiagnoser, MyOptimizer, MyEvaluator, MyExplainer, Plotter, Output



class MyPipeline:
    """Machine Learning Pipeline for Model Training and Evaluation
    A class that handles data loading, model training, and evaluation with SHAP analysis.
    Supports various regression models with hyperparameter optimization and cross-validation.
    """
    def __init__(
            self,
            random_state: int = 0
        ):

        self.random_state = random_state

        # Global variables statement
        self.dataset = None
        self.estimator = None
        self.data_engineer_pipeline = None
        self.stratify = None

        # After optimization
        self.optimized_estimator = None
        self.optimized_dataset = None
        self.optimized_data_engineer_pipeline = None


    def load(
        self,
        model_name: str,
        input_data: pd.DataFrame,
        y: str | int, 
        x_list: list[str | int] | tuple[str | int],
        test_ratio: float = 0.3,
        stratify: bool = False,
        data_engineer_pipeline: Pipeline | None = None,
        cat_features: list[str] | tuple[str] | None = None,
        model_configs_path: str = 'model_configs.yml'
    ):
        """Load the estimator, (train/test) dataset, and data engineer pipeline
        
        Args:
            model_name (str): The name of the model to load.
            input_data (pd.DataFrame): The input data for the model.
            y (str | int): The target (dependent) variable.
            x_list (list[str | int] | tuple[str | int]): The list of independent variables.
            test_ratio (float, optional): The ratio of the dataset to include in the test split. Defaults to 0.3.
            stratify (bool, optional): Whether to stratify the dataset. Defaults to False.
            data_engineer_pipeline (Pipeline | None, optional): The data engineering pipeline to use. Defaults to None.
            cat_features (list[str] | tuple[str] | None, optional): Categorical features to be used. Defaults to None.
            model_configs_path (str, optional): Path to the model configuration file. Defaults is 'model_configs.yml'.
        """

        self.stratify = stratify

        # Load the estimator
        self.estimator = MyEstimator(
            cat_features = cat_features,
            model_configs_path = model_configs_path
        ).load(model_name = model_name)

        # Load the dataset
        self.dataset = MyDataLoader(
            input_data = input_data,
            y = y,
            x_list = x_list,
            test_ratio = test_ratio,
            random_state = self.random_state,
            stratify = self.stratify
        ).load()

        # Load the data engineer pipeline
        self.data_engineer_pipeline = data_engineer_pipeline
    
        return None


    def diagnose(self, sample_k: int | float | None = None):
        """Diagnose the dataset

        Args:
            sample_k (int | float | None, optional): The number of samples to use for diagnosis. Defaults to None.`
        """
        
        """
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
        """

        diagnoser = MyDataDiagnoser(dataset = self.dataset)
        diagnoser.diagnose(sample_k = sample_k)
        
        return None


    def optimize(
        self,
        strategy: str = "tpe",
        cv: int = 5,
        trials: int = 100,
        n_jobs: int = 5,
        direction: str = "maximize",
        eval_function: None | callable = None,
    ):
        """Optimization
        
        Args:
            strategy (str): The optimization strategy to use. Default is "tpe".
            cv (int): The number of cross-validation folds. Default is 5.
            trials (int): The number of trials to run. Default is 100.
            n_jobs (int): The number of jobs to run in parallel. Default is 5.
            direction (str): The direction of the optimization. Default is "maximize".
            eval_function (None | callable): The evaluation function to use. Default is None.

        Returns:
            The optimized estimator, dataset, and data engineer pipeline.
        """
        
        # Initialize optimizer
        optimizer = MyOptimizer(
            dataset = self.dataset,
            estimator = self.estimator,
            data_engineer_pipeline = self.data_engineer_pipeline,
        )

        # Fit the optimizer
        self.optimized_dataset, self.optimized_estimator, self.optimized_data_engineer_pipeline = optimizer.fit(
            random_state = self.random_state,
            stratify = self.stratify,
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
            eval_metric (None): The self-defined evaluation metric to use. 
                It must be None or a dictionary where each item is a callable function. Default is None.
        """

        evaluator = MyEvaluator(
            optimized_estimator = self.optimized_estimator,
            optimized_dataset = self.optimized_dataset,
            optimized_data_engineer_pipeline = self.optimized_data_engineer_pipeline,
        )
        evaluator.evaluate(
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
        ):
        """Use training set to build the explainer, use test set to calculate SHAP values.

        Args:
            select_background_data (str): The data to use to build the explainer.
            select_shap_data (str): The data to use to calculate SHAP values.
            sample_background_data_k (int | float | None): The number of samples to use to build the explainer.
            sample_shap_data_k (int | float | None): The number of samples to use to calculate SHAP values.
        """

        explainer = MyExplainer(
            optimized_estimator = self.optimized_estimator,
            optimized_dataset = self.optimized_dataset,
            optimized_data_engineer_pipeline = self.optimized_data_engineer_pipeline,
        )
        explainer.explain(
            select_background_data = select_background_data,
            select_shap_data = select_shap_data,
            sample_background_data_k = sample_background_data_k,
            sample_shap_data_k = sample_shap_data_k
        )
        
        return None
    

    def predict(self):
        """Predict the model
        
        predictor = MyPredictor(
            optimized_estimator = self.optimized_estimator,
            optimized_dataset = self.optimized_dataset,
            optimized_data_engineer_pipeline = self.optimized_data_engineer_pipeline,
        )
        predictor.predict()
        """

        return None
    

    def plot(self):
        """Plot everything in the pipeline
        """

        """
        plotter = Plotter(
            optimized_estimator = self.optimized_estimator,
            optimized_dataset = self.optimized_dataset,
            optimized_data_engineer_pipeline = self.optimized_data_engineer_pipeline,
        )
        plotter.plot()

        """
        pass

        return None

    
    def output(self):
        """Output everything in the pipeline
        """

        pass

        return None
