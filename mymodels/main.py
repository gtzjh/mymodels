import pandas as pd
from sklearn.pipeline import Pipeline


from .core import MyDataLoader
from .core import MyEstimator
from .core import MyOptimizer
from .core import MyEvaluator
from .core import MyExplainer
from .data_diagnoser import MyDataDiagnoser
from .plotting import Plotter
from .output import Output


class MyModel:
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

        self.plotter = None
        self.output = None

        # After optimization
        self.optimized_estimator = None
        self.optimized_dataset = None
        self.optimized_data_engineer_pipeline = None

        # After evaluation
        self.evaluated_accuracy_dict = None


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
    

    def format(
            self,
            results_dir: str = "results/",
            show: bool = False,
            plot_format: str = "jpg",
            plot_dpi: int = 500,
            save_optimal_model: bool = False,
            save_raw_data: bool = False,
            save_shap_values: bool = False
        ):
        """Format the plotting and output

        Args:
            results_dir (str): The directory to save the results.
            show (bool): Whether to show the plots.
            plot_format (str): The format of the plots.
            plot_dpi (int): The DPI of the plots.
            save_optimal_model (bool): Whether to save the optimal model.
            save_raw_data (bool): Whether to save the raw data.
            save_shap_values (bool): Whether to save the SHAP values.
        """

        self.plotter = Plotter(
            show = show,
            plot_format = plot_format,
            plot_dpi = plot_dpi,
            results_dir = results_dir,
        )

        self.output = Output(
            results_dir = results_dir,
            save_optimal_model = save_optimal_model,
            save_raw_data = save_raw_data,
            save_shap_values = save_shap_values,
        )

        return None


    def diagnose(self, sample_k: int | float | None = None):
        """Diagnose the dataset

        Args:
            sample_k (int | float | None, optional): 
                The number of samples to use for diagnosis. Defaults to None.`
        """

        diagnoser = MyDataDiagnoser(
            dataset = self.dataset,
            plotter = self.plotter,
        )
        
        diagnoser.diagnose(sample_k = sample_k, random_state = self.random_state)
        
        return None


    def optimize(
        self,
        strategy: str = "tpe",
        cv: int = 5,
        trials: int = 100,
        n_jobs: int = 5,
        direction: str = "maximize",
        eval_function = None ,
    ):
        """Optimization
        
        Args:
            strategy (str): The optimization strategy to use. Default is "tpe".
            cv (int): The number of cross-validation folds. Default is 5.
            trials (int): The number of trials to run. Default is 100.
            n_jobs (int): The number of jobs to run in parallel. Default is 5.
            direction (str): The direction of the optimization. Default is "maximize".
            eval_function: The evaluation function to use. Default is None.

        Returns:
            The optimized estimator, dataset, and data engineer pipeline.
        """
        
        # Initialize optimizer
        optimizer = MyOptimizer(
            dataset = self.dataset,
            estimator = self.estimator,
            data_engineer_pipeline = self.data_engineer_pipeline,
            plotter = self.plotter,
            output = self.output
        )

        # Fit the optimizer
        self.optimized_dataset, self.optimized_estimator, self.optimized_data_engineer_pipeline = optimizer.fit(
            stratify = self.stratify,
            strategy = strategy,
            cv = cv,
            trials = trials,
            n_jobs = n_jobs,
            direction = direction,
            eval_function = eval_function,
            random_state = self.random_state,
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
            plotter = self.plotter,
            output = self.output
        )
        self.evaluated_accuracy_dict = evaluator.evaluate(
            show_train = show_train,
            dummy = dummy,
            eval_metric = eval_metric,
        )

        import json
        print(json.dumps(self.evaluated_accuracy_dict, indent=4))

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
            plotter = self.plotter,
            output = self.output
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
        pass

        return None
