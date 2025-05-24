import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class MyPredictor:
    def __init__(
            self,
            optimized_dataset,
            optimized_estimator,
            optimized_data_engineer_pipeline: Pipeline | None = None
        ):
        """A class for predicting the model.

        Args:
            optimized_dataset: MyDataLoader,
            optimized_estimator: MyEstimator,
            optimized_data_engineer_pipeline: Pipeline | None = None
        """
        self._y_mapping_dict = optimized_dataset.y_mapping_dict
        self._optimized_estimator = optimized_estimator.optimal_model_object
        self._optimized_data_engineer_pipeline = optimized_data_engineer_pipeline


    def predict(self, data: pd.DataFrame):
        """Predict the model.

        Args:
            data (pd.DataFrame): The data to predict.
            
        Returns:
            _y_pred (pd.Series): The predicted data.
        """

        assert isinstance(data, pd.DataFrame), \
            "data must be a pandas.DataFrame object"

        # Transform the data
        if self._optimized_data_engineer_pipeline is not None:
            data = self._optimized_data_engineer_pipeline.transform(data)

        # Predict the data
        # For CatBoost, we need to flatten the data if the dimension is 2
        _y_pred = pd.Series(
            np.array(self._optimized_estimator.predict(data)).flatten(),
            index = data.index
        )

        # Inverse the y mapping if needed
        if self._y_mapping_dict is not None:
            _inverse_y_mapping_dict = {v: k for k, v in self._y_mapping_dict.items()}
            _y_pred = _y_pred.map(lambda x: _inverse_y_mapping_dict.get(x, x))

        return _y_pred
