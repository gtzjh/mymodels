import pathlib

from mymodels._data_loader import data_loader
from mymodels._optimizer import MyOptimizer
from mymodels._evaluator import evaluate
from mymodels._explainer import MyExplainer



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
            random_state: int = 0
        ):
        self.file_path = file_path
        self.y = y
        self.x_list = x_list
        self.model_name = model_name
        self.results_dir = pathlib.Path(results_dir)
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
        self.encoder = optimizer.encoder
    

    def evaluate(self):
        """Evaluate the model
        the results will be saved in the results_dir,
        and output to the console
        """
        evaluate(
            model_name=self.model_name,
            model_obj=self.optimal_model,
            x_test=self.x_test,
            y_test=self.y_test,
            x_train=self.x_train,
            y_train=self.y_train,
            results_dir=self.results_dir,
            encoder_obj=self.encoder
        )


    def explain(self):
        """Use SHAP for explanation"""
        # Sampling
        shap_data = self.x_test.sample(
            n = int(len(self.x_test) * self.shap_ratio),
            random_state = self.random_state
        )
        # Explain
        explainer = MyExplainer(self.optimal_model, self.model_name, shap_data, self.results_dir)
        explainer.summary_plot()
        explainer.dependence_plot()
        explainer.partial_dependence_plot()
        
        
    def run(self):
        """Execute the whole pipeline"""
        self.load()
        self.optimize()
        self.evaluate()
        self.explain()
        return None


    


    