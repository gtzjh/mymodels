import yaml
from pathlib import Path


def _output_optimal_params(results_dir,
                         optimal_params):
    """Save the optimal parameters to a YAML file.
    """

    assert isinstance(optimal_params, dict), \
        "optimal_params must be a dictionary"

    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir.joinpath("params.yml"), 'w', encoding="utf-8") as file:
        yaml.dump(optimal_params, file)
    return None