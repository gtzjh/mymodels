import numpy as np
import pandas as pd
import sys
import os
from sklearn.datasets import load_diabetes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from mymodels import MyDataDiagnoser
from mymodels.core import MyDataLoader
from mymodels.plotting import Plotter


def prepare_test_data(sample_size=442, random_state=42):
    """
    Prepare test data with missing values, categorical features, and skewed distributions.
    
    Args:
        sample_size (int): Number of samples to use (max 442 for diabetes dataset)
        random_state (int): Random state for reproducibility
        
    Returns:
        pd.DataFrame: Prepared dataframe with target and features
    """
    # Load diabetes dataset
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    # Sample the data (diabetes dataset has 442 samples total)
    sample_size = min(sample_size, 442)  # Ensure we don't try to sample more than available
    df = df.sample(sample_size, random_state=random_state)

    # Add categorical features by discretizing some continuous features
    df['bmi_category'] = pd.cut(df['bmi'], bins=3, labels=['low', 'medium', 'high'])
    df['age_group'] = pd.cut(df['age'], bins=3, labels=['young', 'middle', 'old'])
    
    # Convert to string type to ensure they're treated as categorical
    df['bmi_category'] = df['bmi_category'].astype(str)
    df['age_group'] = df['age_group'].astype(str)

    # Create a strongly skewed feature
    df['skewed_feature'] = np.exp(df['bmi'] * 2)

    # Introduce missing values in some columns (about 8% of the data)
    for col in ['bmi', 'bp', 's1', 's2']:
        mask = np.random.rand(len(df)) < 0.08
        df.loc[mask, col] = np.nan
        
    return df


# Get test data
df = prepare_test_data()


# Create the dataset using MyDataLoader
dataset = MyDataLoader(
    input_data = df,
    y = 'target',
    x_list = [col for col in df.columns if col != 'target'],
    test_ratio = 0.3,
    random_state = 0,
    stratify = False
).load()


plotter = Plotter(
    show = False,
    plot_format = "jpg",
    plot_dpi = 500,
    results_dir = "results/test_data_diagnoser/",
)


def test_data_diagnoser():
    """Test the data diagnoser functionality whether can run without errors"""
    
    try:
        diagnoser = MyDataDiagnoser(
            dataset,
            plotter,
        )
        diagnoser.diagnose(sample_k=0.1, random_state=42)
        # If we reach here without exceptions, the test passes
        assert True
    except Exception as e:
        assert False, f"Data diagnoser failed with error: {str(e)}"

    try:
        diagnoser = MyDataDiagnoser(
            dataset,
            plotter,
        )
        diagnoser.diagnose(sample_k=100, random_state=42)
        # If we reach here without exceptions, the test passes
        assert True
    except Exception as e:
        assert False, f"Data diagnoser failed with error: {str(e)}"
    


if __name__ == "__main__":
    test_data_diagnoser()
