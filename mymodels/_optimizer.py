import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed, dump
from functools import partial
import pickle, yaml, pathlib, logging
from types import MappingProxyType



from .models import MyModels



class MyOptimizer:
    def __init__(
            self, 
            random_state: int, 
            results_dir: pathlib.Path, 
        ):
        """A class for training and optimizing various machine learning models.
        
        This class handles hyperparameter optimization for both regression and classification
        models using Optuna.
        
        Function Call Flow:

        MyOptimizer.__init__
        |
        +-- fit()
        |   |
        |   +-- _check_task_type() # Determine regression or classification
        |   |
        |   +-- _select_model() # Select model and parameter space
        |   |
        |   +-- _optimizer() # Run Optuna optimization
        |      |
        |      +-- _objective() # Optimization objective function with CV
        |         |
        |         +-- _single_fold() # Single fold execution
        |
        +-- output() # Process results
            |
            +-- _plot_optimize_history() # Plot optimization history
            |
            +-- _save_optimal_params() # Save optimal parameters
            |
            +-- _save_optimal_model() # Save optimal model
        
        Args:
            random_state: Random seed for reproducibility.
            results_dir: Directory path to store the optimization results.
        """
        
        self.random_state = random_state
        self.results_dir = results_dir

        # Global variables statement
        # Input fit()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.model_name = None
        self.data_engineer_pipeline = None
        self.cv = None
        self.trials = None
        self.n_jobs = None
        # Inside fit()
        self.final_x_train = None
        self.final_x_test = None
        self._model_obj = None
        # Inside output()
        self.optuna_study = None
        self.optimal_params = None
        self.optimal_model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.show = None
        self.plot_format = None
        self.plot_dpi = None

    

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_name: str,
        data_engineer_pipeline: Pipeline | None = None,
        cv: int = 5,
        trials: int = 50,
        n_jobs: int = -1,
        cat_features: list[str] | tuple[str] | None = None,
    ):
        """This method handles the entire process of model training and optimization:
            1. Determines if it's a regression or classification task
            2. Selects the appropriate model and parameter space
            3. Optimizes hyperparameters using Optuna
            4. Fits the final model and makes predictions
        
        Args:
            x_train: Training features data.
            y_train: Training target data.
            x_test: Test features data.
            model_name: Model selection. 
                For regression, should be one of 
                    ["lr", "catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adar", "svr", "knr", "mlpr"]. 
                For classification, should be one of
                    ["lc", "catc", "rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", "svc", "knc", "mlpc"].
            data_engineer_pipeline: A pipeline for data engineering,
                                    the pipeline should be a `sklearn.pipeline.Pipeline` object.
            cv: Number of folds for cross-validation.
            trials: Number of trials to execute in Optuna optimization.
            n_jobs: Number of jobs to run in parallel for cross-validation. Default is -1
                (use all available processors).
            cat_features: List of categorical feature names, FOR CatBoost ONLY.
        """

        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.x_test = x_test.copy()
        self.model_name = model_name
        self.data_engineer_pipeline = data_engineer_pipeline
        self.cv = cv
        self.trials = trials
        self.n_jobs = n_jobs

        # The training set which is for final prediction after encoding
        self.final_x_train = self.x_train
        # The test set which is for final prediction after encoding
        self.final_x_test = self.x_test

        # Select model and load parameter space and static parameters
        self._model_obj, _param_space, _static_params = self._select_model(cat_features)

        # Optimizing
        self.optuna_study = self._optimizer(_param_space, _static_params)
        
        # Save optimal parameters and model
        self.optimal_params = {**_static_params, **self.optuna_study.best_trial.params}
        # Use deep clone to make sure the seperate operation
        self.optimal_model = clone(self._model_obj(**self.optimal_params))

        # Data engineering
        if self.data_engineer_pipeline is not None:
            # Use the deep clone to make sure the seperate operation
            final_data_engineer_pipeline = clone(self.data_engineer_pipeline)
            self.final_x_train = final_data_engineer_pipeline.fit_transform(self.final_x_train)
            self.final_x_test = final_data_engineer_pipeline.transform(self.final_x_test)
        
        # Fit on the whole training and validation set
        self.optimal_model.fit(self.final_x_train, self.y_train)

        # Infer
        self.y_train_pred = self.optimal_model.predict(self.final_x_train)
        self.y_test_pred = self.optimal_model.predict(self.final_x_test)

        return None


    def _select_model(self, _cat_features=None):
        """Select the appropriate model and get its parameter space.
        
        Args:
            _cat_features: List of categorical feature names, FOR CatBoost ONLY.
            
        Returns:
            tuple: Contains (model_object, parameter_space, static_parameters).
        """

        ###########################################################################################
        # If CatBoost is selected, info the user that:
        # the Encoder of categorical features is not needed in the data_engineer_pipeline
        if self.model_name in ["catr", "catc"]:
            # If the default data_engineer_pipeline includes the Encoder of categorical features,
            # info the user that the Encoder is not needed in the data_engineer_pipeline
            if self.data_engineer_pipeline is not None:
                # Check if any step name in the pipeline starts with "encode_"
                if any(step_name.startswith("encoder") for step_name, _ in self.data_engineer_pipeline.steps):
                    logging.warning("""The Encoder of categorical features is not needed for CatBoost.""")
        if self.model_name in ["svr", "svc", "knr", "knc", "mlpr", "mlpc", "lr", "lc"]:
            if self.data_engineer_pipeline is not None:
                if not any(step_name.startswith("scaler") for step_name, _ in self.data_engineer_pipeline.steps):
                    logging.warning(f"""
The Scaler is recommended for:
    - LinearRegression
    - LogisticRegression
    - SVR
    - SVC
    - KNR
    - KNC
    - MLPRegressor
    - MLPClassifier
""")
        if self.model_name in ["dtr", "dtc", "rfc", "rfr", "lgbc", "lgbr", "gbdtc", "gbdtr", "xgbc", "xgbr", "catr", "catc"]:
            if self.data_engineer_pipeline is not None:
                if any(step_name.startswith("scaler") for step_name, _ in self.data_engineer_pipeline.steps):
                    logging.warning("""The Scaler is NOT recommended for tree-based models.""")
        ###########################################################################################

        _model_obj, param_space, static_params = MyModels(
            model_name = self.model_name,
            random_state = self.random_state,
            cat_features = _cat_features
        ).get()

        return _model_obj, param_space, static_params



    def _optimizer(self, _param_space: dict, _static_params: dict) -> optuna.Study:
        """Internal method for model optimization using Optuna.
        
        Args:
            _param_space: The parameter space to optimize.
            _static_params: Static parameters for the model.
            
        Returns:
            optuna.Study: The completed optimization study.
        """
        # Set log level to WARNING to avoid excessive output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        _study = optuna.create_study(
            direction = "maximize",
            sampler = TPESampler(seed=self.random_state),
        )

        # Execute the optimization
        _study.optimize(
            partial(self._objective, _param_space=_param_space, _static_params=_static_params),
            n_trials=self.trials,
            n_jobs=1,  # It is not recommended to use n_jobs > 1 in Optuna
            show_progress_bar=True
        )
        
        return _study



    def _objective(self, trial, _param_space, _static_params) -> float:
        """Objective function for the Optuna study.
        
        It performs the following steps:
        1. Creates model parameters by combining static and trial parameters
        2. Performs k-fold cross-validation in parallel
        3. Returns a score (mean CV score - 0.5*std) to optimize
        
        Args:
            trial: Optuna trial object.
            _param_space: The parameter space to sample from.
            _static_params: Static parameters for the model.
            
        Returns:
            float: The evaluation metric (R2 for regression, accuracy for classification)
                  adjusted by standard deviation.
        """

        # Get parameters for model training
        # Make the param immutable
        param = MappingProxyType({
            **{k: v(trial) for k, v in _param_space.items()},
            **_static_params
        })
        
        # Single fold execution
        def _single_fold(train_idx, val_idx, param) -> float:
            """Single fold execution.
                - All models inherit from `sklearn.base.RegressorMixin` or `sklearn.base.ClassifierMixin`
                - and therefore have a `score` method.
                - Return R2 for regression task
                - Return the overall accuracy for classification task
            
            Args:
                train_idx: The training index.
                val_idx: The validation index.
                param: The parameters for the model.
            """

            # Create a validator
            _validator = clone(self._model_obj(**param))

            # Get the training and validation data
            X_fold_train = self.x_train.iloc[train_idx].copy(deep=True)
            y_fold_train = self.y_train.iloc[train_idx].copy(deep=True)
            X_fold_val = self.x_train.iloc[val_idx].copy(deep=True)
            y_fold_val = self.y_train.iloc[val_idx].copy(deep=True)


            if self.data_engineer_pipeline is not None:
                # Use the deep clone to make sure the seperate operation
                _k_fold_data_engineer_pipeline = clone(self.data_engineer_pipeline)
                _transformed_X_fold_train = _k_fold_data_engineer_pipeline.fit_transform(X_fold_train)
                _transformed_X_fold_val = _k_fold_data_engineer_pipeline.transform(X_fold_val)

                # Fit in a single fold
                _validator.fit(_transformed_X_fold_train, y_fold_train)
                return _validator.score(_transformed_X_fold_val, y_fold_val)

            else:
                # Fit in a single fold
                _validator.fit(X_fold_train, y_fold_train)
                return _validator.score(X_fold_val, y_fold_val)

            
        # Parallel processing for validation. Initialize KFold cross validator
        kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        
        
        """
        # Use the for-in loop for debugging
        cv_scores = list()
        for train_idx, val_idx in kf.split(self.x_train):
            cv_scores.append(_single_fold(train_idx, val_idx, param))
        """
        
        cv_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_single_fold)(train_idx, val_idx, param)
            for train_idx, val_idx in kf.split(self.x_train)
        )
        
        # Adjust CV results by subtracting 0.5*std for more stable results
        return np.mean(cv_scores) - 0.5 * np.std(cv_scores)



    def output(
        self,
        optimize_history: bool = True,
        save_optimal_params: bool = True,
        save_optimal_model: bool = True,
        show: bool = False,
        plot_format: str = "jpg",
        plot_dpi: int = 500
    ):
        """Output the optimization history, optimal model and parameters.
        
        This method handles the results visualization and saving:
        1. Plots the optimization history
        2. Saves the optimal parameters to a YAML file
        3. Optionally saves the optimal model to a pickle file
        
        Args:
            optimize_history: Whether to plot and save the optimization history.
            save_optimal_params: Whether to save the optimal parameters to a YAML file.
            save_optimal_model: Whether to save the optimal model to a pickle file.
            show: Whether to display the plots.
            plot_format: Format for saving plots (jpg, png, pdf, etc.).
            plot_dpi: Resolution for saved plots.
            
        Returns:
            None. Results are saved to files in the results directory.
        """
        self.show = show
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi

        # Plot the optimization history
        if optimize_history:
            self._plot_optimize_history(self.optuna_study)

        # Save optimal parameters
        if save_optimal_params:
            self._save_optimal_params()

        # Save optimal model
        if save_optimal_model:
            self._save_optimal_model()

        return None
    


    def _plot_optimize_history(self, optuna_study_object: optuna.Study):
        """Plot the optimization history using matplotlib instead of plotly.

        This version doesn't require additional packages to save images.
        Creates a plot showing trial values and best values across optimization trials.
        
        Args:
            optuna_study_object: The completed Optuna study containing trial results.
            
        Returns:
            None. The plot is saved to the results directory.
        """
        # Get the optimization history data
        trials = optuna_study_object.trials
        values = [t.value for t in trials if t.value is not None]
        best_values = [max(values[:i+1]) for i in range(len(values))]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(values) + 1), values, 'o-', color='blue', alpha=0.5, label='Trial value')
        plt.plot(range(1, len(best_values) + 1), best_values, 'o-', color='red', label='Best value')
        
        # Add labels and title
        plt.xlabel('Trial number')
        plt.ylabel('Objective value')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(self.results_dir.joinpath("optimization_history." + self.plot_format), dpi = self.plot_dpi)

        if self.show:
            plt.show()

        plt.close("all")
        
        return None



    def _save_optimal_params(self):
        """Save the optimal parameters to a YAML file.
        """
        with open(self.results_dir.joinpath("params.yml"), 'w', encoding="utf-8") as file:
            yaml.dump(self.optimal_params, file)
        return None



    def _save_optimal_model(self):
        """Save the optimal model using the recommended export method based on model type.
        
        Different models have different recommended export methods:
        - CatBoost: save_model() method to save in binary format
        - XGBoost: save_model() method to save in binary format  
        - LightGBM: booster_.save_model() method to save in text format
        - Scikit-learn models: joblib is recommended over pickle
        """
        model_path = self.results_dir.joinpath("optimal-model")
        
        # XGBoost models
        if self.model_name in ["xgbr", "xgbc"]:
            self.optimal_model.save_model(f"{model_path}.json")
            
        # LightGBM models
        elif self.model_name in ["lgbr", "lgbc"]:
            self.optimal_model.booster_.save_model(f"{model_path}.txt")
            
        # CatBoost models
        elif self.model_name in ["catr", "catc"]:
            self.optimal_model.save_model(f"{model_path}.cbm")
            
        # For scikit-learn based models, use joblib which is more efficient for numpy arrays
        else:
            dump(self.optimal_model, f"{model_path}.joblib")
            
        # Also save a pickle version for backward compatibility
        with open(self.results_dir.joinpath("optimal-model.pkl"), 'wb') as file:
            pickle.dump(self.optimal_model, file)
            
        return None
    