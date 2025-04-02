import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        return None
    

    def _describe_data(self):
        """Describe the data
            - Shape of dataframe       
        """
        return None
    


    def _diagnose_missing_values(self):
        """Diagnose the missing values
        
        Returns:
            pd.DataFrame: Table with missing values information for each column
        """
        print("==========================================")
        print("MISSING VALUES DIAGNOSIS")
        print(f"X_train info:")
        print(self.x_data.info())

        print(f"Y_train info:")
        print(self.y_data.info())
        print("==========================================")

        return None
    


    def _diagnose_categorical_features(self):
        """Diagnose the categorical features
            - Count unique values
        """
        return None
    

    def _diagnose_numerical_features(self):
        """Diagnose the numerical features"""
        _numeric_features = []
        for col, dtype in self.x_data.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                _numeric_features.append(col)
        
        return None
    

    def _vis_violin(self):
        """Visualize the violin plot"""
        return None



    """A WARNING should occur if some features are highly correlated 
    'cause that may influence the model's interpretability."""
    def _vis_correlation(self):
        """Visualize the correlation
            - Heatmap of correlation matrix
            - Spearman and Pearson correlation coefficients are both shown by default
            
        """



        return None
