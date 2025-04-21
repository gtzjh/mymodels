import pandas as pd
import logging, pathlib


class MyDataDiagnoser:
    def __init__(
            self,
            x_data: pd.DataFrame,
            y_data: pd.Series,
            results_dir: str | pathlib.Path,
            show: bool = False,
            plot_format: str = "jpg",
            plot_dpi: int = 500
        ):
        self.x_data = x_data.copy()
        self.y_data = y_data.copy()
        self.results_dir = pathlib.Path(results_dir)
        self.show = show
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi

        self._check_input()


    def _check_input(self):
        assert isinstance(self.x_data, pd.DataFrame), \
            "x_data must be a pandas DataFrame"
        assert isinstance(self.y_data, pd.Series), \
            "y_data must be a pandas Series"
        assert self.x_data.shape[0] == self.y_data.shape[0], \
            "x_data and y_data must have the same number of rows"
        if not self.x_data.index.equals(self.y_data.index):
            logging.warning("x_data and y_data have the different index")

        print(f"""
=========================================================
Data diagnosis should be performed on TRAINING DATA ONLY.
=========================================================
""")
        return None
    


    def diagnose(self):
        """Diagnose the data"""
        self._describe_data()
        self._diagnose_missing_values()
        self._diagnose_categorical_features()
        self._diagnose_numerical_features()

        return None
    


    def _describe_data(self):
        """Describe the data
            - Shape of dataframe       
        """
        print("==========================================")
        print("DATA DESCRIPTION")
        print(f"\nX_train shape: {self.x_data.shape}")
        print(f"\nY_train shape: {self.y_data.shape}")
        print("==========================================")
        return None
    


    def _diagnose_missing_values(self):
        """Diagnose the missing values
        
        Returns:
            pd.DataFrame: Table with missing values information for each column
        """
        print("==========================================")
        print("MISSING VALUES DIAGNOSIS")
        print(f"\nX_train info:")
        print(self.x_data.info())

        print(f"\nY_train info:")
        print(self.y_data.info())
        print("==========================================")

        return None
    


    def _diagnose_categorical_features(self):
        """Diagnose the categorical features
            - Count unique values
        """
        _categorical_features = []
        for col, dtype in self.x_data.dtypes.items():
            if pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                _categorical_features.append(col)
        
        logging.info(f"\nCategorical features: {_categorical_features}")
        if _categorical_features:
            for _cat_col in _categorical_features:
                plot_category(
                    data=self.x_data[_cat_col],
                    name=_cat_col,
                    save_dir=self.results_dir / "data_categorical",
                    plot_format=self.plot_format,
                    plot_dpi=self.plot_dpi,
                    show=self.show
                )
            logging.info(f"\nCategorical features visualization saved to: {self.results_dir / 'data_categorical'}")


        return None
    

    def _diagnose_numerical_features(self):
        """Diagnose the numerical features
        
        This method analyzes and visualizes the numerical features in the dataset:
        1. Identifies all numerical features in the dataset
        2. Creates distribution plots for each numerical feature using _vis_data_distribution
           - Each plot shows the distribution pattern, outliers, and central tendency
           - Plots are saved to results_dir/data_distribution/
        3. Generates correlation heatmaps (Pearson and Spearman) using _vis_correlation
           - Shows relationships and dependency between numerical features
           - Highlights potentially problematic high correlations
           - Maps are saved to results_dir/data_correlation/
           
        All visualizations use the format and DPI settings specified during initialization.
        Interactive display of plots depends on the 'show' parameter setting.
        """
        _numeric_features = []
        for col, dtype in self.x_data.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                _numeric_features.append(col)
        
        logging.info(f"\nNumerical features: {_numeric_features}")
        
        # Create subdirectories for visualizations
        dist_dir = self.results_dir / "data_distribution"
        corr_dir = self.results_dir / "data_correlation"
        
        # Visualize the distribution of each numerical feature
        if _numeric_features:
            for col in _numeric_features:
                plot_data_distribution(
                    data=self.x_data[col],
                    name=col,
                    save_dir=dist_dir,
                    plot_format=self.plot_format,
                    plot_dpi=self.plot_dpi,
                    show=self.show
                )
            
            # Visualize correlations between numerical features
            plot_correlation(
                data=self.x_data[_numeric_features],
                name="numerical_features",
                save_dir=corr_dir,
                plot_format=self.plot_format,
                plot_dpi=self.plot_dpi,
                show=self.show
            )
            
            logging.info(f"\nDistribution plots saved to: {dist_dir}")
            logging.info(f"Correlation plots saved to: {corr_dir}")
        else:
            logging.info("\nNo numerical features found for visualization.")

        return None

