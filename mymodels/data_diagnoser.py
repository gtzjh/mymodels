import pandas as pd


from .core import MyDataLoader
from .plotting import Plotter


class MyDataDiagnoser:
    def __init__(
            self,
            dataset: MyDataLoader,
            plotter: Plotter,
        ):
        self.diagnose_x_data = dataset.x_train
        self.diagnose_y_data = dataset.y_train.to_frame()

        self.plotter = plotter

        return None
    

    def diagnose(self, sample_k: int | float | None = None, random_state: int | None = 0):
        """Diagnose the data"""

        # Validation
        assert isinstance(self.diagnose_x_data, pd.DataFrame), \
            "x_data must be a pandas DataFrame"
        assert isinstance(self.diagnose_y_data, pd.DataFrame), \
            "y_data must be a pandas DataFrame"
        assert self.diagnose_x_data.shape[0] == self.diagnose_y_data.shape[0], \
            "x_data and y_data must have the same number of rows"
        print(f"""
=========================================================
Data diagnosis is performed on TRAINING DATASET ONLY.
=========================================================
""")

        # Sample the data
        assert sample_k is None or isinstance(sample_k, int) or isinstance(sample_k, float), \
            "sample_k must be an integer or float or None"
        if sample_k is not None:
            if isinstance(sample_k, float):
                sample_k = int(sample_k * len(self.diagnose_x_data))

            _diagnose_data = self.diagnose_x_data.merge(self.diagnose_y_data,
                                                        left_index=True,
                                                        right_index=True).sample(sample_k,
                                                                                 random_state=random_state)
            self.diagnose_x_data = _diagnose_data.iloc[:, :-1]
            self.diagnose_y_data = _diagnose_data.iloc[:, -1]


        self._diagnose_categorical_features()
        self._diagnose_numerical_features()

        return None
    

    def _diagnose_categorical_features(self):
        """Diagnose the categorical features
        """
        # Identify categorical features
        _categorical_features = []
        for col, dtype in self.diagnose_x_data.dtypes.items():
            if pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                _categorical_features.append(col)

        if len(_categorical_features) > 0:
            print(f"\nCategorical Features: {_categorical_features}")

            # Plot
            for _cat_col in _categorical_features:
                self.plotter.plot_category(
                    data=self.diagnose_x_data[_cat_col],
                    name=_cat_col,
                )
        
            # Create a table with feature statistics
            feature_stats = []
            for col in self.diagnose_x_data[_categorical_features].columns:
                total_count = len(self.diagnose_x_data[col])
                null_count = self.diagnose_x_data[col].isnull().sum()
                null_ratio = null_count / total_count
                unique_count = self.diagnose_x_data[col].nunique()
                unique_ratio = unique_count / total_count
                
                feature_stats.append({
                    'Feature Name': col,
                    'Count': total_count,
                    'Null Count': null_count,
                    'Null Ratio': f'{null_ratio:.2%}',
                    'Unique Count': unique_count,
                    'Unique Ratio': f'{unique_ratio:.2%}'
                })
            
            # Convert to DataFrame and display
            _stats_df = pd.DataFrame(feature_stats)
            print("\nFeature Statistics:")
            print(_stats_df.to_string(index=False))

        return None
    

    def _diagnose_numerical_features(self):
        """Diagnose the numerical features"""

        # Identify numerical features
        _numeric_features = []
        for col, dtype in self.diagnose_x_data.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                _numeric_features.append(col)
        
        # Visualize the distribution of each numerical feature
        if len(_numeric_features) > 0:
            print(f"\nNumerical Features: {_numeric_features}")

            # Plot the distribution
            for col in _numeric_features:
                self.plotter.plot_data_distribution(
                    data=self.diagnose_x_data[col],
                    name=col,
                )
            
            # Plot the correlations
            self.plotter.plot_correlation(
                data=self.diagnose_x_data[_numeric_features],
                name="numerical_features",
            )

            # Create a table with feature statistics
            feature_stats = []
            for col in _numeric_features:
                data = self.diagnose_x_data[col]
                total_count = len(data)
                null_count = data.isnull().sum()
                null_ratio = null_count / total_count
                
                feature_stats.append({
                    'Feature Name': col,
                    'Count': total_count,
                    'Null Count': null_count,
                    'Null Ratio': f'{null_ratio:.2%}',
                    'Min': f'{data.min():.2f}',
                    '25%': f'{data.quantile(0.25):.2f}',
                    'Median': f'{data.median():.2f}',
                    '75%': f'{data.quantile(0.75):.2f}',
                    'Max': f'{data.max():.2f}',
                    'Mean': f'{data.mean():.2f}',
                    'Std': f'{data.std():.2f}',
                    'Kurtosis': f'{data.kurtosis():.2f}',
                    'Skewness': f'{data.skew():.2f}'
                })
            
            # Convert to DataFrame and display
            _stats_df = pd.DataFrame(feature_stats)
            print("\nNumerical Features Statistics:")
            print(_stats_df.to_string(index=False))

        return None

