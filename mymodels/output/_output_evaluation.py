import yaml
from pathlib import Path


def _output_evaluation(
        results_dir: str | Path,
        accuracy_dict: dict, 
    ):
    """
    Output the evaluation results to files.

    Args:
        results_dir: The directory to save the evaluation results
        accuracy_dict: The evaluation results
    """

    # Check if the results_dir is a valid directory
    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    assert isinstance(accuracy_dict, dict), \
        "accuracy_dict must be a dictionary"
    
    # Save results to files
    with open(results_dir.joinpath("accuracy.yml"), 'w', encoding = "utf-8") as file:
        yaml.dump(accuracy_dict, file)
