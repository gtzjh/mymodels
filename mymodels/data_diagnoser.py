import pandas as pd
import logging


from .core import MyDataLoader
from .plotting import Plotter



class MyDataDiagnoser:
    def __init__(
            self,
            dataset: MyDataLoader,
            plotter: Plotter,
        ):
        self.x_data = dataset.x_train.copy()
        self.y_data = dataset.y_train.copy()

        self.plotter = plotter

        return None
    


    def diagnose(self, sample_k: int | float | None = None):
        """Diagnose the data"""

        # Validation
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
Data diagnosis is performed on TRAINING DATASET ONLY.
=========================================================
""")

        ###########################################################################################
        # Sample the data
        ###########################################################################################
        assert sample_k is None or isinstance(sample_k, int) or isinstance(sample_k, float), \
            "sample_k must be an integer or float or None"
        if sample_k is not None:
            if isinstance(sample_k, float):
                sample_k = int(sample_k * len(self._x_train))

            diagnose_data = diagnose_x_data.merge(diagnose_y_data,
                                                  left_index=True,
                                                  right_index=True).sample(sample_k,
                                                                           random_state=self.random_state)
            diagnose_x_data = diagnose_data.iloc[:, :-1]
            diagnose_y_data = diagnose_data.iloc[:, -1]
        ###########################################################################################

        self._diagnose_categorical_features()
        self._diagnose_numerical_features()

        return None
    


    def _diagnose_categorical_features(self):
        """Diagnose the categorical features
        """

        # Identify categorical features
        _categorical_features = []
        for col, dtype in self.x_data.dtypes.items():
            if pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                _categorical_features.append(col)
        
        print(f"\nCategorical features: {_categorical_features}")
        if _categorical_features:
            for _cat_col in _categorical_features:
                self.plotter.plot_category(
                    data=self.x_data[_cat_col],
                    name=_cat_col,
                )

        return None
    

    def _diagnose_numerical_features(self):
        """Diagnose the numerical features
        """
        _numeric_features = []
        for col, dtype in self.x_data.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                _numeric_features.append(col)
        
        logging.info(f"\nNumerical features: {_numeric_features}")
        
        # Visualize the distribution of each numerical feature
        if _numeric_features:
            for col in _numeric_features:
                self.plotter.plot_data_distribution(
                    data=self.x_data[col],
                    name=col,
                )
            
            # Visualize correlations between numerical features
            self.plotter.plot_correlation(
                data=self.x_data[_numeric_features],
                name="numerical_features",
            )

        return None

