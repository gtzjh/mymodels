import optuna
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels.plotting import Plotter



def test_plot_optimize_history():
    # Generate a synthetic classification dataset
    X, y = make_classification(
        n_samples=100, 
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define the objective function for optimization
    def objective(trial):
        # Define the hyperparameters to optimize
        n_estimators = trial.suggest_int("n_estimators", 10, 100)
        max_depth = trial.suggest_int("max_depth", 2, 10)
        
        # Create and train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions and calculate the accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    # Create an Optuna study that maximizes the objective
    study = optuna.create_study(direction="maximize")
    
    # Run the optimization with 10 trials
    study.optimize(objective, n_trials=10)
    
    # Plot the optimization history
    plotter.plot_optimize_history(study)
    
    return study


if __name__ == "__main__":
    plotter = Plotter(
        show = False,
        plot_format = "png",
        plot_dpi = 300,
        results_dir = "./results/test_plotting_plot_optimizer"
    )
    
    test_plot_optimize_history()
