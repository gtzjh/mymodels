import numpy as np
import pandas as pd
import logging
import pytest
# from sklearn.datasets import load_boston
from numpy.testing import assert_almost_equal, assert_array_equal


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels._data_engineer import MyEngineer



logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


"""
1. Core Transformer Functionality
- Input/Output Validation
  - Verify fit/transform outputs for custom transformers (e.g., missing value handlers, encoders) 
    match expectations (correct bin boundaries, matching text vectorization dimensions).
  - Numerical calculation accuracy (e.g., standardized mean ≈ 0, standard deviation ≈ 1).

2. Pipeline Process Integrity
- Step Ordering & Data Flow
  - Validate pipeline output equivalence with manual step-by-step execution 
    (e.g., cleaning → encoding → feature generation).
  - Intermediate data format compatibility (e.g., previous step's output types 
    match next step's input requirements).

3. ColumnTransformer Column Assignment
- Column Matching & Concatenation
  - Ensure specified columns (numerical/text) are correctly routed to corresponding transformers.
  - Validate merged feature matrix dimensions (e.g., one-hot encoded columns = original categories count).

4. Parameter Passing & Overrides
- Critical Parameter Validation
  - Verify global params (e.g., verbose) and step-level params (e.g., StandardScaler.with_mean) 
    take effect as expected.
  - Test dynamic parameter updates via set_params and subsequent re-fitting/output changes.

5. Error Handling & Edge Cases
- Defensive Exception Handling
  - Validate proper exceptions (ValueError) for empty data, fully missing columns, invalid types.
  - Test clear error messaging for unconfigured required transformers/parameters.

6. End-to-End Flow Validation
- Real-World Scenario Simulation
  - Full workflow test from raw data input to final feature matrix generation.
  - Verify compatibility with sklearn models (e.g., ensure feature matrix can be directly 
    fed to RandomForest for training).
"""

# Fixture for sample test data
@pytest.fixture
def sample_data():
    """Create a sample DataFrame with some missing values for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
        'B': [np.nan, 20, 30, 40, 50, np.nan, 70, 80, 90, 100],
        'C': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
        'D': ['cat', 'dog', 'cat', 'fish', 'dog', 'cat', 'fish', 'dog', 'cat', 'dog']
    })
    return df

# ====================== 1. Core Transformer Functionality ======================

def test_median_imputation(sample_data):
    """
    Test that median imputation correctly replaces missing values with the median.
    
    Expected result: 
    - NaN values in column A should be replaced with the median value of A (5.5)
    - NaN values in column B should be replaced with the median value of B (60.0)
    - Column C and D should remain unchanged
    """
    # Setup the engineer with median imputation for columns A and B
    engineer = MyEngineer(
        missing_values_cols=['A', 'B'],
        impute_method=['median', 'median'],
        scale_cols=None
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # Check if NaN values are replaced with medians
    assert np.isnan(result['A']).sum() == 0
    assert np.isnan(result['B']).sum() == 0
    
    # Calculate expected medians
    median_A = sample_data['A'].median()
    median_B = sample_data['B'].median()
    
    # Check imputed values (the indices where NaNs were)
    assert result.loc[2, 'A'] == median_A
    assert result.loc[6, 'A'] == median_A
    assert result.loc[0, 'B'] == median_B
    assert result.loc[5, 'B'] == median_B

def test_standard_scaling(sample_data):
    """
    Test that StandardScaler correctly standardizes the data.
    
    Expected result:
    - Scaled column A should have mean ≈ 0 and std ≈ 1
    - Scaled column B should have mean ≈ 0 and std ≈ 1
    - Columns C and D should remain unchanged
    """
    # First impute missing values to avoid issues with scaling
    engineer = MyEngineer(
        missing_values_cols=['A', 'B'],
        impute_method=['median', 'median'],
        scale_cols=['A', 'B'],
        scale_method=['standard', 'standard']
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # Check that scaled columns have mean ≈ 0 and std ≈ 1
    assert_almost_equal(result['A'].mean(), 0.0, decimal=1)
    assert_almost_equal(result['A'].std(), 1.0, decimal=1)
    assert_almost_equal(result['B'].mean(), 0.0, decimal=1)
    assert_almost_equal(result['B'].std(), 1.0, decimal=1)
    
    # Check that column C remains unchanged
    assert_array_equal(result['C'].values, sample_data['C'].values)

def test_minmax_scaling(sample_data):
    """
    Test that MinMaxScaler correctly scales the data to the range [0, 1].
    
    Expected result:
    - Scaled column A should have min = 0 and max = 1
    - Scaled column B should have min = 0 and max = 1
    - Columns C and D should remain unchanged
    """
    # First impute missing values to avoid issues with scaling
    engineer = MyEngineer(
        missing_values_cols=['A', 'B'],
        impute_method=['median', 'median'],
        scale_cols=['A', 'B'],
        scale_method=['minmax', 'minmax']
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # Check that scaled columns have min = 0 and max = 1
    assert_almost_equal(result['A'].min(), 0.0, decimal=1)
    assert_almost_equal(result['A'].max(), 1.0, decimal=1)
    assert_almost_equal(result['B'].min(), 0.0, decimal=1)
    assert_almost_equal(result['B'].max(), 1.0, decimal=1)
    
    # Check that column C remains unchanged
    assert_array_equal(result['C'].values, sample_data['C'].values)

# ====================== 2. Pipeline Process Integrity ======================

def test_pipeline_vs_manual_steps(sample_data):
    """
    Test that the pipeline result is equivalent to applying transformations manually step by step.
    
    Expected result:
    - The output from the pipeline should be identical to the output from applying
      imputation and scaling manually in sequence
    """
    # Define the engineer with the pipeline
    engineer = MyEngineer(
        missing_values_cols=['A', 'B'],
        impute_method=['median', 'median'],
        scale_cols=['A', 'B'],
        scale_method=['standard', 'standard']
    )
    
    # Get the pipeline and its components
    pipeline = engineer.construct()
    imputer = pipeline.named_steps['imputer']
    scaler = pipeline.named_steps['scaler']
    
    # Apply pipeline
    pipeline.fit(sample_data)
    pipeline_result = pipeline.transform(sample_data)
    
    # Apply steps manually
    imputer.fit(sample_data)
    imputed_data = imputer.transform(sample_data)
    scaler.fit(imputed_data)
    manual_result = scaler.transform(imputed_data)
    
    # Compare results
    pd.testing.assert_frame_equal(pipeline_result, manual_result)

def test_output_dataframe_preservation(sample_data):
    """
    Test that the pipeline preserves the DataFrame format throughout the transformations.
    
    Expected result:
    - The output should be a pandas DataFrame
    - The output should have the same column names as the input (in some order)
    """
    engineer = MyEngineer(
        missing_values_cols=['A', 'B'],
        impute_method=['median', 'median'],
        scale_cols=['A', 'B'],
        scale_method=['standard', 'standard']
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # Check output type
    assert isinstance(result, pd.DataFrame)
    
    # Check column preservation
    assert set(result.columns) == set(sample_data.columns)

# ====================== 3. ColumnTransformer Column Assignment ======================

def test_column_targeting(sample_data):
    """
    Test that transformations only affect the specified columns.
    
    Expected result:
    - Only columns A and B should be transformed
    - Columns C and D should remain completely unchanged
    """
    engineer = MyEngineer(
        missing_values_cols=['A', 'B'],
        impute_method=['median', 'median'],
        scale_cols=['A', 'B'],
        scale_method=['standard', 'standard']
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # Check that columns A and B are transformed (mean ≈ 0, std ≈ 1)
    assert_almost_equal(result['A'].mean(), 0.0, decimal=1)
    assert_almost_equal(result['B'].mean(), 0.0, decimal=1)
    
    # Check that columns C and D remain unchanged
    assert_array_equal(result['C'].values, sample_data['C'].values)
    assert_array_equal(result['D'].values, sample_data['D'].values)

def test_mixed_transformations(sample_data):
    """
    Test that different transformation methods can be applied to different columns.
    
    Expected result:
    - Column A should be standardized (mean ≈ 0, std ≈ 1)
    - Column B should be min-max scaled (min = 0, max = 1)
    - Column C should remain unchanged
    """
    engineer = MyEngineer(
        missing_values_cols=['A', 'B'],
        impute_method=['median', 'median'],
        scale_cols=['A', 'B'],
        scale_method=['standard', 'minmax']
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # Check standardization of column A
    assert_almost_equal(result['A'].mean(), 0.0, decimal=1)
    assert_almost_equal(result['A'].std(), 1.0, decimal=1)
    
    # Check min-max scaling of column B
    assert_almost_equal(result['B'].min(), 0.0, decimal=1)
    assert_almost_equal(result['B'].max(), 1.0, decimal=1)
    
    # Check that column C remains unchanged
    assert_array_equal(result['C'].values, sample_data['C'].values)


if __name__ == "__main__":
    import sys
    # Create sample data directly (not using the fixture)
    np.random.seed(42)
    test_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
        'B': [np.nan, 20, 30, 40, 50, np.nan, 70, 80, 90, 100],
        'C': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
        'D': ['cat', 'dog', 'cat', 'fish', 'dog', 'cat', 'fish', 'dog', 'cat', 'dog']
    })
    
    # Run Core Transformer Functionality tests
    print("\n=== Core Transformer Functionality ===")
    print("Testing median imputation...")
    test_median_imputation(test_data)
    print("✓ Median imputation test passed")
    
    print("\nTesting standard scaling...")
    test_standard_scaling(test_data)
    print("✓ Standard scaling test passed")
    
    print("\nTesting min-max scaling...")
    test_minmax_scaling(test_data)
    print("✓ Min-max scaling test passed")
    
    # Run Pipeline Process Integrity tests
    print("\n=== Pipeline Process Integrity ===")
    print("Testing pipeline vs manual steps...")
    test_pipeline_vs_manual_steps(test_data)
    print("✓ Pipeline vs manual steps test passed")
    
    print("\nTesting DataFrame preservation...")
    test_output_dataframe_preservation(test_data)
    print("✓ DataFrame preservation test passed")
    
    # Run ColumnTransformer Column Assignment tests
    print("\n=== ColumnTransformer Column Assignment ===")
    print("Testing column targeting...")
    test_column_targeting(test_data)
    print("✓ Column targeting test passed")
    
    print("\nTesting mixed transformations...")
    test_mixed_transformations(test_data)
    print("✓ Mixed transformations test passed")
    
    print("\nAll tests passed successfully!")
    
    # Alternative: use pytest runner
    # import pytest
    # pytest.main(["-xvs", __file__])

