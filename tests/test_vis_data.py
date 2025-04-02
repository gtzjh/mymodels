import numpy as np
import pandas as pd
import logging
import pathlib


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels._vis_data import _vis_data_distribution, _vis_correlation


logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)



def test_vis_data_distribution():
    # Create different distributions for testing
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    skewed_data = np.random.exponential(2, 1000)
    bimodal_data = np.concatenate([
        np.random.normal(-3, 1, 500),
        np.random.normal(3, 1, 500)
    ])
    
    # Visualize distributions
    print("Generating visualizations for test data...")
    
    # Test with normal distribution
    _vis_data_distribution(
        data=normal_data,
        name="Normal Distribution",
        save_dir=results_dir,
        show=True
    )
    
    # Test with skewed distribution
    _vis_data_distribution(
        data=skewed_data,
        name="Skewed Distribution",
        save_dir=results_dir,
        show=True
    )
    
    # Test with bimodal distribution
    _vis_data_distribution(
        data=bimodal_data,
        name="Bimodal Distribution",
        save_dir=results_dir,
        show=True
    )
    
    # Test with DataFrame input
    print("Testing distribution visualization with DataFrame input...")
    np.random.seed(123)
    
    # Create a DataFrame with different distributions in each column
    n_samples = 1500
    df_test = pd.DataFrame({
        'normal': np.random.normal(0, 1, n_samples),
        'uniform': np.random.uniform(-3, 3, n_samples),
        'exponential': np.random.exponential(2, n_samples),
        'lognormal': np.random.lognormal(0, 0.5, n_samples),
        'bimodal': np.concatenate([
            np.random.normal(-2, 0.8, n_samples // 2),
            np.random.normal(2, 0.8, n_samples // 2)
        ])
    })
    
    # Add a column with mixed distribution
    df_test['mixed'] = df_test['normal'] * 0.6 + df_test['exponential'] * 0.4
    
    # Test visualization of each column in the DataFrame
    for column in df_test.columns:
        _vis_data_distribution(
            data=df_test[column],
            name=f"DataFrame-{column}",
            save_dir=results_dir,
            show=True
        )
    
    # Test with a subset of rows
    _vis_data_distribution(
        data=df_test.iloc[100:600]['lognormal'],
        name="DataFrame-Subset-Lognormal",
        save_dir=results_dir,
        show=True
    )
    
    # Test with a Series created from DataFrame
    series_test = df_test['bimodal'].copy()
    series_test.name = "Series from DataFrame"
    _vis_data_distribution(
        data=series_test,
        name="DataFrame-Series",
        save_dir=results_dir,
        show=True
    )

    return None
    

def test_vis_correlation():
    """Test correlation visualization function and NaN handling mechanism
    
    Test items:
    1. Basic correlation testing
       - No-NaN dataset test: Test basic correlation calculation and visualization
       - With-NaN dataset test: Test data processing with 10% random NaN values
       
    2. Large dataset testing
       - Dataset with 8 features
       - 15% random NaN value distribution
       - Verify performance and NaN handling on larger datasets
       
    3. Strong correlation dataset testing
       - Test detection of highly correlated feature pairs (threshold warnings)
       - Using varying NaN densities: 5%, 10%, and 20%
       - Verify the impact of different NaN value densities on correlation calculations
       
    4. Mixed data type testing
       - Dataset containing both numeric and non-numeric columns
       - Add 12% NaN values only to numeric columns
       - Test automatic filtering and handling of non-numeric columns
       
    5. Large-scale complex dataset testing
       - 25 features, 1000 samples
       - Complex correlation patterns and groupings
       - Using stratified NaN value densities (ranging from 5% to 25%)
       - Verify algorithm performance and accuracy on large-scale complex data
    """
    # Test correlation visualization
    test_df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(0, 1, 100),
        'C': np.random.normal(0, 1, 100)
    })
    # Make B and C correlated
    test_df['C'] = test_df['B'] * 0.8 + np.random.normal(0, 0.5, 100)
    
    # Add some random NaN values (10% of the data)
    test_df_with_nans = test_df.copy()
    random_mask = np.random.random(test_df.shape) < 0.1  # 10% chance of True
    test_df_with_nans = test_df_with_nans.mask(random_mask)
    
    print(f"Added {random_mask.sum()} NaN values to test_df")
    
    # Test correlation with the original data
    _vis_correlation(
        data=test_df,
        name="Test Correlation (No NaNs)",
        save_dir=results_dir,
        show=True
    )
    
    # Test correlation with NaNs
    _vis_correlation(
        data=test_df_with_nans,
        name="Test Correlation (With NaNs)",
        save_dir=results_dir,
        show=True
    )
    
    print("Testing correlation visualization with additional datasets...")
    
    # Test with larger dataset and more varied correlations
    np.random.seed(123)
    n_samples = 200
    large_test_df = pd.DataFrame({
        'Feature1': np.random.normal(0, 1, n_samples),
        'Feature2': np.random.normal(0, 1, n_samples),
        'Feature3': np.random.normal(0, 1, n_samples),
        'Feature4': np.random.normal(0, 1, n_samples),
        'Feature5': np.random.normal(0, 1, n_samples),
    })
    
    # Create some correlated features
    large_test_df['Feature6'] = large_test_df['Feature1'] * 0.7 + np.random.normal(0, 0.7, n_samples)
    large_test_df['Feature7'] = large_test_df['Feature2'] * -0.6 + np.random.normal(0, 0.8, n_samples)
    large_test_df['Feature8'] = large_test_df['Feature3'] * 0.5 + large_test_df['Feature4'] * 0.3 + np.random.normal(0, 0.6, n_samples)
    
    # Add some random NaN values (15% of the data)
    large_test_df_with_nans = large_test_df.copy()
    random_mask = np.random.random(large_test_df.shape) < 0.15  # 15% chance of True
    large_test_df_with_nans = large_test_df_with_nans.mask(random_mask)
    
    print(f"Added {random_mask.sum()} NaN values to large_test_df")
    
    _vis_correlation(
        data=large_test_df_with_nans,
        name="Large Dataset Correlation (With NaNs)",
        save_dir=results_dir,
        show=True
    )
    
    # Test with very strong correlations (should trigger warning)
    strong_corr_df = pd.DataFrame({
        'X1': np.random.normal(0, 1, 150),
        'X2': np.random.normal(0, 1, 150),
        'X3': np.random.normal(0, 1, 150),
    })
    
    # Create nearly perfect correlations
    strong_corr_df['X4'] = strong_corr_df['X1'] * 0.95 + np.random.normal(0, 0.2, 150)  # Strong positive
    strong_corr_df['X5'] = -strong_corr_df['X2'] * 0.9 + np.random.normal(0, 0.3, 150)  # Strong negative
    strong_corr_df['X6'] = strong_corr_df['X3'] * 0.85 + np.random.normal(0, 0.4, 150)  # Moderate-strong
    
    # Create specific NaN patterns - varying density of NaNs across features
    strong_corr_df_with_nans = strong_corr_df.copy()
    # X1 and X4 have 5% NaNs
    strong_corr_df_with_nans['X1'] = strong_corr_df_with_nans['X1'].mask(np.random.random(150) < 0.05)
    strong_corr_df_with_nans['X4'] = strong_corr_df_with_nans['X4'].mask(np.random.random(150) < 0.05)
    # X2 and X5 have 10% NaNs
    strong_corr_df_with_nans['X2'] = strong_corr_df_with_nans['X2'].mask(np.random.random(150) < 0.10)
    strong_corr_df_with_nans['X5'] = strong_corr_df_with_nans['X5'].mask(np.random.random(150) < 0.10)
    # X3 and X6 have 20% NaNs
    strong_corr_df_with_nans['X3'] = strong_corr_df_with_nans['X3'].mask(np.random.random(150) < 0.20)
    strong_corr_df_with_nans['X6'] = strong_corr_df_with_nans['X6'].mask(np.random.random(150) < 0.20)
    
    nan_count = strong_corr_df_with_nans.isna().sum().sum()
    print(f"Added {nan_count} NaN values to strong_corr_df with varying densities")
    
    _vis_correlation(
        data=strong_corr_df_with_nans,
        corr_threshold=0.7,  # Lower threshold to ensure warning is triggered
        name="Strong Correlations (With NaNs)",
        save_dir=results_dir,
        show=True
    )
    
    # Test with mixed data types (some non-numeric columns)
    mixed_df = pd.DataFrame({
        'Numeric1': np.random.normal(0, 1, 100),
        'Numeric2': np.random.normal(0, 1, 100),
        'Numeric3': np.random.normal(0, 1, 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Text': ['Sample' + str(i) for i in range(100)]
    })
    
    # Create correlation between Numeric1 and Numeric3
    mixed_df['Numeric3'] = mixed_df['Numeric1'] * 0.75 + np.random.normal(0, 0.5, 100)
    
    # Add NaNs to numeric columns only
    mixed_df_with_nans = mixed_df.copy()
    for col in ['Numeric1', 'Numeric2', 'Numeric3']:
        mixed_df_with_nans[col] = mixed_df_with_nans[col].mask(np.random.random(100) < 0.12)
    
    nan_count = mixed_df_with_nans[['Numeric1', 'Numeric2', 'Numeric3']].isna().sum().sum()
    print(f"Added {nan_count} NaN values to numeric columns in mixed_df")
    
    _vis_correlation(
        data=mixed_df_with_nans,
        name="Mixed Data Types (With NaNs)",
        save_dir=results_dir,
        show=True
    )
    
    # Test with much larger dataset and many more features
    print("Testing correlation visualization with extensive dataset...")
    np.random.seed(456)
    large_samples = 1000
    
    # Create base dataframe with 20 features
    feature_names = [f'Feature{i}' for i in range(1, 21)]
    extensive_df = pd.DataFrame({
        name: np.random.normal(0, 1, large_samples) for name in feature_names
    })
    
    # Create complex correlation patterns between features
    # Group 1: Features 1-5 correlated with each other
    for i in range(2, 6):
        extensive_df[f'Feature{i}'] = extensive_df['Feature1'] * (0.3 + i/20) + np.random.normal(0, 0.7, large_samples)
    
    # Group 2: Features 6-10 negatively correlated with each other
    for i in range(7, 11):
        extensive_df[f'Feature{i}'] = -extensive_df['Feature6'] * (0.3 + (i-5)/20) + np.random.normal(0, 0.7, large_samples)
    
    # Group 3: Features 11-15 moderately correlated with Feature1
    for i in range(11, 16):
        extensive_df[f'Feature{i}'] = extensive_df['Feature1'] * 0.4 + np.random.normal(0, 0.9, large_samples)
    
    # Group 4: Features 16-20 complex correlations
    extensive_df['Feature16'] = extensive_df['Feature6'] * 0.3 + extensive_df['Feature11'] * 0.3 + np.random.normal(0, 0.8, large_samples)
    extensive_df['Feature17'] = extensive_df['Feature7'] * 0.4 + extensive_df['Feature12'] * 0.2 + np.random.normal(0, 0.8, large_samples)
    extensive_df['Feature18'] = extensive_df['Feature8'] * 0.3 + extensive_df['Feature13'] * 0.3 + np.random.normal(0, 0.8, large_samples)
    extensive_df['Feature19'] = extensive_df['Feature9'] * 0.2 + extensive_df['Feature14'] * 0.4 + np.random.normal(0, 0.8, large_samples)
    extensive_df['Feature20'] = extensive_df['Feature10'] * 0.25 + extensive_df['Feature15'] * 0.35 + np.random.normal(0, 0.8, large_samples)
    
    # Add 5 more features with varying correlations
    extensive_df['Feature21'] = np.random.normal(0, 1, large_samples)  # Independent
    extensive_df['Feature22'] = extensive_df['Feature21'] * 0.9 + np.random.normal(0, 0.3, large_samples)  # Strong correlation with 21
    extensive_df['Feature23'] = extensive_df['Feature1'] * 0.6 + extensive_df['Feature6'] * -0.3 + np.random.normal(0, 0.7, large_samples)  # Mixed correlation
    extensive_df['Feature24'] = extensive_df['Feature10'] * -0.5 + extensive_df['Feature20'] * 0.3 + np.random.normal(0, 0.7, large_samples)  # Mixed correlation
    extensive_df['Feature25'] = extensive_df['Feature5'] * 0.45 + extensive_df['Feature15'] * 0.45 + np.random.normal(0, 0.6, large_samples)  # Balanced correlation
    
    # Add complex NaN patterns - different NaN rates for different groups of features
    extensive_df_with_nans = extensive_df.copy()
    
    # Group 1: Low NaN rate (5%)
    for i in range(1, 6):
        extensive_df_with_nans[f'Feature{i}'] = extensive_df_with_nans[f'Feature{i}'].mask(np.random.random(large_samples) < 0.05)
    
    # Group 2: Medium NaN rate (10%)
    for i in range(6, 11):
        extensive_df_with_nans[f'Feature{i}'] = extensive_df_with_nans[f'Feature{i}'].mask(np.random.random(large_samples) < 0.10)
    
    # Group 3: High NaN rate (15%)
    for i in range(11, 16):
        extensive_df_with_nans[f'Feature{i}'] = extensive_df_with_nans[f'Feature{i}'].mask(np.random.random(large_samples) < 0.15)
    
    # Group 4: Very high NaN rate (20%)
    for i in range(16, 21):
        extensive_df_with_nans[f'Feature{i}'] = extensive_df_with_nans[f'Feature{i}'].mask(np.random.random(large_samples) < 0.20)
    
    # Group 5: Varying NaN rates
    extensive_df_with_nans['Feature21'] = extensive_df_with_nans['Feature21'].mask(np.random.random(large_samples) < 0.05)
    extensive_df_with_nans['Feature22'] = extensive_df_with_nans['Feature22'].mask(np.random.random(large_samples) < 0.10)
    extensive_df_with_nans['Feature23'] = extensive_df_with_nans['Feature23'].mask(np.random.random(large_samples) < 0.15)
    extensive_df_with_nans['Feature24'] = extensive_df_with_nans['Feature24'].mask(np.random.random(large_samples) < 0.20)
    extensive_df_with_nans['Feature25'] = extensive_df_with_nans['Feature25'].mask(np.random.random(large_samples) < 0.25)
    
    nan_count = extensive_df_with_nans.isna().sum().sum()
    nan_percent = 100 * nan_count / (extensive_df_with_nans.shape[0] * extensive_df_with_nans.shape[1])
    print(f"Added {nan_count} NaN values to extensive_df ({nan_percent:.1f}% of data)")
    
    _vis_correlation(
        data=extensive_df_with_nans,
        name="Extensive Dataset (With NaNs)",
        save_dir=results_dir,
        show=True
    )
    
    print(f"Visualizations saved to {results_dir.absolute()}")

    return None


if __name__ == "__main__":
    results_dir = pathlib.Path("./results/test_vis")
    
    test_vis_data_distribution()
    test_vis_correlation()

    
