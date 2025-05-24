"""Optimization visualization module.

This module provides functions to visualize optimization history from Optuna studies.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna


def _plot_optimize_history(optuna_study_object: optuna.Study):
    """Plot the optimization history using matplotlib instead of plotly.

    This version doesn't require additional packages to save images.
    Creates a plot showing trial values and best values across optimization trials.
    
    Args:
        optuna_study_object: The completed Optuna study containing trial results.
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects.
    """
    assert isinstance(optuna_study_object, optuna.Study), \
        "Input optuna_study_object must be an Optuna study object."

    # Get the optimization history data
    trials = optuna_study_object.trials
    values = [t.value for t in trials if t.value is not None]
    best_values = [max(values[:i+1]) for i in range(len(values))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(values) + 1), values, 'o-', color='blue', alpha=0.5, label='Trial value')
    ax.plot(range(1, len(best_values) + 1), best_values, 'o-', color='red', label='Best value')
    
    # Add labels and title
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Objective value')
    ax.set_title('Optimization History')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Apply tight layout
    fig.tight_layout()
    
    return fig, ax
