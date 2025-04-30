import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mymodels.data_engineer import MyEngineer


"""
logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)
"""



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

# ====================== 4. Categorical Encoding Tests ======================

def test_ordinal_encoding(sample_data):
    """
    Test that ordinal encoding correctly transforms categorical data.
    
    Expected result:
    - Column D should be transformed to ordinal encoded values
    - Each unique category should have a unique ordinal value
    - All other columns should remain unchanged
    """
    # Setup the engineer with ordinal encoding for column D
    engineer = MyEngineer(
        cat_features=['D'],
        encode_method=['ordinal'],
        missing_values_cols=None,
        scale_cols=None
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # Check that column D has been transformed
    assert result['D'].dtype == np.int64 \
        or result['D'].dtype == np.int32 \
        or result['D'].dtype == np.float64
    
    # Check that each unique category has a unique ordinal value
    unique_categories = sample_data['D'].unique()
    encoded_values = []
    
    for category in unique_categories:
        # Find indices where the original data has this category
        indices = sample_data[sample_data['D'] == category].index
        # Get the encoded value for the first occurrence
        encoded_value = result.loc[indices[0], 'D']
        # Store the encoded value
        encoded_values.append(encoded_value)
        
        # Check that all occurrences of this category have the same encoded value
        for idx in indices:
            assert result.loc[idx, 'D'] == encoded_value
    
    # Check that each category has a unique encoded value
    assert len(set(encoded_values)) == len(unique_categories)
    
    # Check that columns A, B, and C remain unchanged
    assert_array_equal(result['A'].values, sample_data['A'].values)
    assert_array_equal(result['B'].values, sample_data['B'].values)
    assert_array_equal(result['C'].values, sample_data['C'].values)

def test_onehot_encoding(sample_data):
    """
    Test that OneHot encoding correctly transforms categorical data.
    
    Expected result:
    - Column D should be transformed to multiple one-hot encoded columns
    - Each unique category should have its own column
    - One-hot encoded columns should have correct naming format
    - All other columns should remain unchanged
    - New unseen categories should be handled gracefully
    """
    # Setup the engineer with OneHot encoding for column D
    engineer = MyEngineer(
        cat_features=['D'],
        encode_method=['onehot'],
        missing_values_cols=None,
        scale_cols=None
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # 1. Test correct transformation
    # Get unique categories in the original data
    unique_categories = sample_data['D'].unique()
    
    # Check that one-hot columns are created (one per category)
    # Column naming pattern may vary, so we need to find them first
    onehot_columns = [col for col in result.columns if 'D' in col and col != 'D']
    assert len(onehot_columns) == len(unique_categories)
    
    # Verify each row has exactly one 1 and the rest 0s in the one-hot encoded columns
    for idx in sample_data.index:
        onehot_values = [result.loc[idx, col] for col in onehot_columns]
        assert sum(onehot_values) == 1.0
        assert all(val == 0.0 or val == 1.0 for val in onehot_values)
    
    # 2. Test correct column naming
    # Depending on how OneHotEncoder names columns, we need to check the pattern
    # For scikit-learn's OneHotEncoder with handle_unknown='ignore', the pattern is usually
    # something like 'encode_D_cat', 'encode_D_dog', etc.
    
    # Find corresponding one-hot column for each category
    for category in unique_categories:
        # Find all rows with this category
        category_indices = sample_data[sample_data['D'] == category].index
        
        # For each one-hot column, check if it correctly represents this category
        for col in onehot_columns:
            # If this column represents this category, it should have 1 for all category_indices
            if all(result.loc[idx, col] == 1.0 for idx in category_indices):
                # Check if column name contains the category name
                # This is a loose check as naming conventions can vary
                assert col.startswith('D_')
                break
    
    # 3. Test handling of unseen categories
    # Create a new DataFrame with an unseen category
    new_data = pd.DataFrame({
        'A': [11, 12],
        'B': [110, 120],
        'C': [115, 125],
        'D': ['bird', 'snake']  # 'bird' and 'snake' were not in the training data
    })
    
    # Transform the new data (no exception should be raised)
    new_result = pipeline.transform(new_data)
    
    # Verify the new result has all the expected columns
    assert set(new_result.columns) == set(result.columns)
    
    # Check that all one-hot encoded columns for the new categories are 0
    # This is expected behavior when handle_unknown='ignore'
    for idx in new_data.index:
        onehot_values = [new_result.loc[idx, col] for col in onehot_columns]
        # All should be zero since these categories weren't seen during training
        assert all(val == 0.0 for val in onehot_values)
    
    # Check that columns A, B, and C remain unchanged in the new data
    assert_array_equal(new_result['A'].values, new_data['A'].values)
    assert_array_equal(new_result['B'].values, new_data['B'].values)
    assert_array_equal(new_result['C'].values, new_data['C'].values)

def test_binary_encoding(sample_data):
    """
    Test that Binary encoding correctly transforms categorical data.
    
    Expected result:
    - Column D should be transformed to multiple binary encoded columns
    - Binary encoded columns should have naming format of column_name + number
    - All other columns should remain unchanged
    - The encoder should handle unseen categories gracefully
    """
    # Setup the engineer with Binary encoding for column D
    engineer = MyEngineer(
        cat_features=['D'],
        encode_method=['binary'],
        missing_values_cols=None,
        scale_cols=None
    )
    
    pipeline = engineer.construct()
    pipeline.fit(sample_data)
    result = pipeline.transform(sample_data)
    
    # 1. Test correct transformation
    # Get unique categories in the original data
    unique_categories = sample_data['D'].unique()
    
    # Find binary encoded columns
    binary_columns = [col for col in result.columns if 'D_' in col]
    
    # Check that binary columns are created
    # For n unique categories, we need at most log2(n) bits rounded up
    import math
    expected_bits = math.ceil(math.log2(len(unique_categories)))
    assert len(binary_columns) >= 1  # Should have at least one binary column
    
    # 2. Test correct column naming
    # Binary encoder typically creates columns named as original_column_0, original_column_1, etc.
    for i in range(len(binary_columns)):
        # At least one column should match the pattern D_0, D_1, etc.
        col_match = False
        for col in binary_columns:
            if col == f'D_{i}':
                col_match = True
                break
        assert col_match, f"Expected to find column named 'D_{i}'"
    
    # 3. Check that each category has a unique binary representation
    category_encodings = {}
    for category in unique_categories:
        # Find indices where this category appears
        indices = sample_data[sample_data['D'] == category].index
        if len(indices) == 0:
            continue
            
        # Get the binary encoding for this category
        idx = indices[0]
        encoding = tuple(result.loc[idx, col] for col in binary_columns)
        
        # Store the encoding for this category
        category_encodings[category] = encoding
        
        # Check all instances of this category have the same encoding
        for idx in indices:
            current_encoding = tuple(result.loc[idx, col] for col in binary_columns)
            assert current_encoding == encoding, \
                f"Category {category} has inconsistent encodings"
    
    # Check that all categories have different encodings
    assert len(set(category_encodings.values())) == len(category_encodings), \
        "Different categories have the same binary encoding"
    
    # 4. Test handling of unseen categories
    # Create a new DataFrame with unseen categories
    new_data = pd.DataFrame({
        'A': [11, 12],
        'B': [110, 120],
        'C': [115, 125],
        'D': ['bird', 'snake']  # 'bird' and 'snake' were not in the training data
    })
    
    # Transform the new data (should not raise an exception)
    new_result = pipeline.transform(new_data)
    
    # Check that all expected columns are present
    assert set(new_result.columns) == set(result.columns)
    
    # Check that numeric columns A, B, and C remain unchanged
    assert_array_equal(new_result['A'].values, new_data['A'].values)
    assert_array_equal(new_result['B'].values, new_data['B'].values)
    assert_array_equal(new_result['C'].values, new_data['C'].values)
    
    # For unseen categories, check that they receive some encoding
    # The exact encoding depends on the implementation, but it should be consistent
    unseen_encodings = {}
    for category in new_data['D'].unique():
        indices = new_data[new_data['D'] == category].index
        idx = indices[0]
        encoding = tuple(new_result.loc[idx, col] for col in binary_columns)
        unseen_encodings[category] = encoding
        
        # All instances of the same unseen category should have the same encoding
        for idx in indices:
            current_encoding = tuple(new_result.loc[idx, col] for col in binary_columns)
            assert current_encoding == encoding, \
                f"Unseen category {category} has inconsistent encodings"


if __name__ == "__main__":
    import sys
    # Create sample data directly (not using the fixture)
    np.random.seed(6)
    test_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
        'B': [np.nan, 20, 30, 40, 50, np.nan, 70, 80, 90, 100],
        'C': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
        'D': ['cat', 'dog', 'cat', 'fish', 'dog', 'cat', 'fish', 'dog', 'cat', 'dog']
    })
    
    # Run tests with try-except blocks to catch and display detailed error information
    all_tests_passed = True
    
    # Run Core Transformer Functionality tests
    print("\n=== Core Transformer Functionality ===")
    
    try:
        print("Testing median imputation...")
        test_median_imputation(test_data)
        print("✓ Median imputation test passed")
    except Exception as e:
        print(f"✗ Median imputation test FAILED: {e}")
        all_tests_passed = False
    
    try:
        print("\nTesting standard scaling...")
        test_standard_scaling(test_data)
        print("✓ Standard scaling test passed")
    except Exception as e:
        print(f"✗ Standard scaling test FAILED: {e}")
        all_tests_passed = False
    
    try:
        print("\nTesting min-max scaling...")
        test_minmax_scaling(test_data)
        print("✓ Min-max scaling test passed")
    except Exception as e:
        print(f"✗ Min-max scaling test FAILED: {e}")
        all_tests_passed = False
    
    # Run Pipeline Process Integrity tests
    print("\n=== Pipeline Process Integrity ===")
    
    try:
        print("Testing pipeline vs manual steps...")
        test_pipeline_vs_manual_steps(test_data)
        print("✓ Pipeline vs manual steps test passed")
    except Exception as e:
        print(f"✗ Pipeline vs manual steps test FAILED: {e}")
        all_tests_passed = False
    
    try:
        print("\nTesting DataFrame preservation...")
        test_output_dataframe_preservation(test_data)
        print("✓ DataFrame preservation test passed")
    except Exception as e:
        print(f"✗ DataFrame preservation test FAILED: {e}")
        all_tests_passed = False
    
    # Run ColumnTransformer Column Assignment tests
    print("\n=== ColumnTransformer Column Assignment ===")
    
    try:
        print("Testing column targeting...")
        test_column_targeting(test_data)
        print("✓ Column targeting test passed")
    except Exception as e:
        print(f"✗ Column targeting test FAILED: {e}")
        all_tests_passed = False
    
    try:
        print("\nTesting mixed transformations...")
        test_mixed_transformations(test_data)
        print("✓ Mixed transformations test passed")
    except Exception as e:
        print(f"✗ Mixed transformations test FAILED: {e}")
        all_tests_passed = False
    
    # Run Categorical Encoding tests
    print("\n=== Categorical Encoding Tests ===")
    
    try:
        print("Testing ordinal encoding...")
        test_ordinal_encoding(test_data)
        print("✓ Ordinal encoding test passed")
    except Exception as e:
        print(f"✗ Ordinal encoding test FAILED: {e}")
        all_tests_passed = False
    
    try:
        print("\nTesting onehot encoding...")
        test_onehot_encoding(test_data)
        print("✓ OneHot encoding test passed")
    except Exception as e:
        print(f"✗ OneHot encoding test FAILED: {e}")
        all_tests_passed = False
    
    try:
        print("\nTesting binary encoding...")
        test_binary_encoding(test_data)
        print("✓ Binary encoding test passed")
    except Exception as e:
        print(f"✗ Binary encoding test FAILED: {e}")
        all_tests_passed = False
    
    # Print final summary
    if all_tests_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests FAILED! See above for details.")
        sys.exit(1)  # Exit with non-zero code to indicate test failure