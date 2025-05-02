from pathlib import Path
from joblib import dump
import pickle


def _output_optimal_model(results_dir, optimal_model, save_type: str | None = None):
    """Save the optimal model using the recommended export method based on model type.
    
    Different models have different recommended export methods:
    - CatBoost: save_model() method to save in binary format
    - XGBoost: save_model() method to save in binary format  
    - LightGBM: booster_.save_model() method to save in text format
    - Other scikit-learn models: joblib is recommended over pickle
    
    Args:
        optimal_model: The trained model to save
        save_type: String identifier of the model type (e.g., "txt", "cbm")
    """

    # Check if the results_dir is a valid directory
    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir.joinpath("optimal_model")
    
    # XGBoost models
    if save_type == "json":
        optimal_model.save_model(f"{model_path}.json")
        
    # LightGBM models
    elif save_type == "txt":
        optimal_model.booster_.save_model(f"{model_path}.txt")
        
    # CatBoost models
    elif save_type == "cbm":
        optimal_model.save_model(f"{model_path}.cbm")
    
    # For scikit-learn based models, use joblib which is more efficient for numpy arrays
    else:
        dump(optimal_model, f"{model_path}.joblib")
        
    # Also save a pickle version for backward compatibility
    with open(results_dir.joinpath("optimal_model.pkl"), 'wb') as file:
        pickle.dump(optimal_model, file)
        
    return None