import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from functools import partial
import json, pickle, yaml, pathlib, logging
from types import MappingProxyType


from ._data_engineer import MyEngineer
from ._encoder import fit_transform_multi_features, transform_multi_features
from .models import MyRegressors, MyClassifiers



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
        # Input engineering()
        self.cat_features = None
        self.encode_method = None
        # Input fit()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.model_name = None
        self.cv = None
        self.trials = None
        self.n_jobs = None
        # Inside fit()
        self.final_x_train = None
        self.final_x_test = None
        self._model_obj = None
        self._task_type = None
        # Inside output()
        self.optuna_study = None
        self.optimal_params = None
        self.optimal_model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.show = None
        self.plot_format = None
        self.plot_dpi = None

    
    def engineering(self):
        """Engineer the data.
        Plant to use the MyEngineer class to engineer the data instead.
        """
        pass

    

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        model_name: str,
        cv: int = 5,
        trials: int = 50,
        n_jobs: int = -1,
    ):
        """Train and optimize a machine learning model.
        
        This method handles the entire process of model training and optimization:
        1. Determines if it's a regression or classification task
        2. Selects the appropriate model and parameter space
        3. Optimizes hyperparameters using Optuna
        4. Processes categorical features if needed
        5. Fits the final model and makes predictions
        
        Args:
            x_train: Training features data.
            y_train: Training target data.
            x_test: Test features data.
            model_name: Model selection. For regression, should be one of 
                ["catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adar", 
                "svr", "knr", "mlpr"]. For classification, should be one of
                ["catc", "rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", 
                "svc", "knc", "mlpc"].
            cat_features: List of categorical feature names, if any.
            encode_method: Method(s) for encoding categorical variables.
            cv: Number of folds for cross-validation.
            trials: Number of trials to execute in Optuna optimization.
            n_jobs: Number of jobs to run in parallel for cross-validation. Default is -1
                (use all available processors).
        
        Returns:
            None. Results are stored in instance attributes.
        """

        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.x_test = x_test.copy()
        self.model_name = model_name
        self.cv = cv
        self.trials = trials
        self.n_jobs = n_jobs

        self.final_x_train = self.x_train  # The training set which is for final prediction after encoding
        self.final_x_test = self.x_test    # The test set which is for final prediction after encoding

        # Check the task type, regression or classification
        # Select model and load parameter space and static parameters
        self._task_type = self._check_task_type(self.model_name)
        self._model_obj, _param_space, _static_params = self._select_model(self._task_type)

        # Execute the optimization
        self.optuna_study = self._optimizer(_param_space, _static_params)
        
        # Save optimal parameters and model
        self.optimal_params = {**_static_params, **self.optuna_study.best_trial.params}
        self.optimal_model = self._model_obj(**self.optimal_params)


        """        
        # Data engineering
        _final_column_transformer = self.column_transformer
        _transformed_x_train_matrix = _final_column_transformer.fit_transform(self.x_train)
        _transformed_x_test_matrix = _final_column_transformer.transform(self.x_test)

        """


        #######################################################################
        # If there are categorical features and the model is not CatBoost
        # then encode the training and test set
        if (self.cat_features is not None) and (self.model_name not in ["catr", "catc"]):
            # Transform train set
            _transformed_X_train, _encoder_dict, _mapping_dict = fit_transform_multi_features(
                self.x_train.loc[:, self.cat_features],
                self.encode_method,
                self.y_train,
            )
            self.final_x_train = self.final_x_train.drop(columns = self.cat_features)
            self.final_x_train = pd.concat([self.final_x_train, _transformed_X_train],
                                           axis = 1,
                                           join="inner",
                                           verify_integrity=True)

            # Transform test set
            _transformed_X_test = transform_multi_features(
                self.x_test.loc[:, self.cat_features],
                _encoder_dict
            )
            self.final_x_test = self.final_x_test.drop(columns = self.cat_features)
            self.final_x_test = pd.concat([self.final_x_test, _transformed_X_test],
                                          axis = 1,
                                          join="inner",
                                          verify_integrity=True)

            # Save mapping dictionary
            with open(self.results_dir.joinpath("mapping.json"), 'w', encoding='utf-8') as f:
                json.dump(_mapping_dict, f, ensure_ascii=False, indent=4)
        #######################################################################

        # Fit on the whole training and validation set
        self.optimal_model.fit(self.final_x_train, self.y_train)

        # Infer
        self.y_train_pred = self.optimal_model.predict(self.final_x_train)
        self.y_test_pred = self.optimal_model.predict(self.final_x_test)

        return None



    def _check_task_type(self, _model_name):
        """Determine if the task is regression or classification based on model name.
        
        Args:
            _model_name: The name of the model to check.
            
        Returns:
            str: Either "regression" or "classification".
            
        Raises:
            ValueError: If the model name is not recognized.
        """
        if _model_name in ["catr", "rfr", "dtr", "lgbr", "gbdtr", "xgbr", "adar", "svr", "knr", "mlpr"]:
            _task_type = "regression"
        elif _model_name in ["catc", "rfc", "dtc", "lgbc", "gbdtc", "xgbc", "adac", "svc", "knc", "mlpc"]:
            _task_type = "classification"
        else:
            raise ValueError(f"Invalid model name: {_model_name}")

        # Check if the target variable is suitable for the task type
        if _task_type == "classification" and pd.api.types.is_float_dtype(self.y_train):
            logging.warning(f"""
The target variable is a float type, 
which is not suitable for classification tasks. 
Please check the configuration CAREFULLY!
""")
        if _task_type == "regression" and self.y_train.nunique() <= 3:
            logging.warning(f"""
The target variable has only {self.y_train.nunique()} unique values, 
which might not be suitable for regression tasks. 
Consider using classification instead.
""")
        
        return _task_type
        


    def _select_model(self, _task_type):
        """Select the appropriate model and get its parameter space.
        
        Args:
            _task_type: Either "regression" or "classification".
            
        Returns:
            tuple: Contains (model_object, parameter_space, static_parameters).
        """
        if _task_type == "regression":
            _model_obj, param_space, static_params = MyRegressors(
                model_name = self.model_name,
                random_state = self.random_state,
                cat_features = self.cat_features
            ).get()
        else:
            _model_obj, param_space, static_params = MyClassifiers(
                model_name = self.model_name,
                random_state = self.random_state,
                cat_features = self.cat_features
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
            X_fold_train = self.x_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.x_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]


            """
            # Data engineering
            _train_row_index = X_fold_train.index
            _val_row_index = X_fold_val.index

            k_fold_column_transformer = self.column_transformer
            X_fold_train = k_fold_column_transformer.fit_transform(X_fold_train)
            X_fold_val = k_fold_column_transformer.transform(X_fold_val)
            _column_names = k_fold_column_transformer.get_feature_names_out()
            
            X_fold_train = pd.DataFrame(X_fold_train, index = _train_row_index, columns = _column_names)
            X_fold_val = pd.DataFrame(X_fold_val, index = _val_row_index, columns = _column_names)
            """
            
            #######################################################################
            # If categorical features exist and model is not CatBoost, encode the input features
            if self.cat_features is not None:
                if self.model_name != "catr" and self.model_name != "catc":
                    _transformed_fold_train, _encoder_dict, _ = fit_transform_multi_features(
                        X_fold_train.loc[:, self.cat_features],
                        self.encode_method,
                        y_fold_train,
                    )
                    X_fold_train = X_fold_train.drop(columns = self.cat_features)
                    X_fold_train = pd.concat([X_fold_train, _transformed_fold_train],
                                             axis = 1, 
                                             join="inner", 
                                             verify_integrity=True)
                    
                    # Encode validation set
                    transformed_fold_val = transform_multi_features(
                        X_fold_val.loc[:, self.cat_features],
                        _encoder_dict
                    )                    
                    X_fold_val = X_fold_val.drop(columns = self.cat_features)
                    X_fold_val = pd.concat([X_fold_val, transformed_fold_val],
                                           axis = 1, 
                                           join="inner", 
                                           verify_integrity=True)
            #######################################################################

            # Create and train the model
            _validator = self._model_obj(**param)
            _validator.fit(X_fold_train, y_fold_train)

            # All models inherit from sklearn.base.RegressorMixin or sklearn.base.ClassifierMixin
            # and therefore have a score method
            # Return R2 for regression task
            if self._task_type == "regression":
                return _validator.score(X_fold_val, y_fold_val)
            # Return the overall accuracy for classification task
            else:
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
            from joblib import dump
            dump(self.optimal_model, f"{model_path}.joblib")
            
        # Also save a pickle version for backward compatibility
        with open(self.results_dir.joinpath("optimal-model.pkl"), 'wb') as file:
            pickle.dump(self.optimal_model, file)
            
        return None
    