import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import category_encoders as ce



logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)



class MyEngineer:
    """Data engineering for the training validation and test set.
    
    Step:
    1. Imputation of missing values
    2. Cleaning of outliers
    3. Encoding of categorical features
    4. Data standardization or normalization

    Optional:
    -  Output the report of the data engineering process
    """
    def __init__(
        self,
        outlier_cols: list[str] | tuple[str] | None = None,
        missing_values_cols: list[str] | tuple[str] | None = None,
        impute_method: str | list[str] | tuple[str] | None = None,
        cat_features: list[str] | tuple[str] | None = None,
        encode_method: str | list[str] | tuple[str] | None = None,
        scale_cols: list[str] | tuple[str] | None = None,
        scale_method: str | list[str] | tuple[str] | None = None,
        random_state: int = 0,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        # Specify the valid imputation and scaling methods
        self.VALID_IMPUTE_METHOD = {
            "mean": SimpleImputer(missing_values=np.nan, strategy="mean"),
            "median": SimpleImputer(missing_values=np.nan, strategy="median"),
            "most_frequent": SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            "constant": SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        }
        self.VALID_ENCODE_METHOD = {
            "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            "binary": ce.BinaryEncoder(),
            "label": LabelEncoder(),
            "ordinal": ce.OrdinalEncoder(),
            "frequency": "value_counts",
            "target": ce.TargetEncoder(),
        }
        self.VALID_SCALE_METHOD = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }

        # Transform the input to a list
        if isinstance(impute_method, str):
            impute_method = [impute_method] * len(missing_values_cols)
        if isinstance(encode_method, str):
            encode_method = [encode_method] * len(cat_features)
        if isinstance(cat_features, str):
            cat_features = [cat_features] * len(cat_features)
        if isinstance(scale_method, str):
            scale_method = [scale_method] * len(scale_cols)
        
        # Check the input
        if missing_values_cols is not None:
            assert len(missing_values_cols) == len(impute_method)
        if cat_features is not None:
            assert len(cat_features) == len(encode_method)
        if scale_cols is not None:
            assert len(scale_cols) == len(scale_method)

        if missing_values_cols is not None:
            assert all(s in self.VALID_IMPUTE_METHOD.keys() for s in impute_method), \
                f"impute_method must be one of {list(self.VALID_IMPUTE_METHOD.keys())}"
        if cat_features is not None:
            assert all(s in self.VALID_ENCODE_METHOD.keys() for s in encode_method), \
                f"encode_method must be one of {list(self.VALID_ENCODE_METHOD.keys())}"
        if scale_cols is not None:
            assert all(s in self.VALID_SCALE_METHOD.keys() for s in scale_method), \
                f"scale_method must be one of {list(self.VALID_SCALE_METHOD.keys())}"
        
        self.outlier_cols = outlier_cols
        self.missing_values_cols = missing_values_cols
        self.impute_method = impute_method
        self.cat_features = cat_features
        self.encode_method = encode_method
        self.scale_cols = scale_cols
        self.scale_method = scale_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose


    

    def construct(self):
        """Construct the data engineering pipeline.
        
        Returns:
            self.pipeline: The data engineering pipeline, in `sklearn.pipeline.Pipeline` object.
        """
        outlier_cleaner_column_transformer = None
        imputer_column_transformer = None
        encoder_column_transformer = None
        scaler_column_transformer = None

        _PIPELINE_DICT = {
            "outlier_cleaner": outlier_cleaner_column_transformer,
            "imputer": imputer_column_transformer,
            "encoder": encoder_column_transformer,
            "scaler": scaler_column_transformer
        }
        if self.outlier_cols is not None:
            _PIPELINE_DICT["outlier_cleaner"] = self._outlier_cleaner()

        if self.missing_values_cols is not None:
            _PIPELINE_DICT["imputer"] = self._imputer()

        if self.cat_features is not None:
            _PIPELINE_DICT["encoder"] = self._encoder()

        if self.scale_cols is not None:
            _PIPELINE_DICT["scaler"] = self._scaler()

        # Filter out None values from the pipeline dictionary
        _PIPELINE_DICT = {k: v for k, v in _PIPELINE_DICT.items() if v is not None}
        
        # Construct the pipeline using the filtered dictionary
        return Pipeline(list(_PIPELINE_DICT.items()))



    def _outlier_cleaner(self):
        """Clean the outliers.
        
        Returns:
            _outlier_cleaner_column_transformer: A column transformer, in sklearn.compose.ColumnTransformer object.
        """
        return None



    def _imputer(self):
        """Impute the missing values.
        
        Returns:
            _imputer_column_transformer: A column transformer, in sklearn.compose.ColumnTransformer object.
        """
        _imputer_column_transformer_list = list()
        for _m_c, _i_m in zip(self.missing_values_cols, self.impute_method):
            _imputer_column_transformer_list.append(
                ("impute_" + str(_m_c), self.VALID_IMPUTE_METHOD[_i_m], [_m_c])
            )
        
        _imputer_column_transformer = ColumnTransformer(
            _imputer_column_transformer_list,
            remainder = "passthrough",
            n_jobs = 1,
            verbose = False,
            verbose_feature_names_out = False
        )

        _imputer_column_transformer.set_output(transform="pandas")
        return _imputer_column_transformer



    def _encoder(self):
        """Encode the categorical features.
        
        Returns:
            _encoder_column_transformer: A column transformer, in sklearn.compose.ColumnTransformer object.
        """
        _encoder_column_transformer_list = list()
        for _e_c, _e_m in zip(self.cat_features, self.encode_method):
            _encoder_column_transformer_list.append(
                ("encode_" + str(_e_c), self.VALID_ENCODE_METHOD[_e_m], [_e_c])
            )

        _encoder_column_transformer = ColumnTransformer(
            _encoder_column_transformer_list,
            remainder = "passthrough",
            n_jobs = 1,
            verbose = False,
            verbose_feature_names_out = False
        )
        logging.debug(f"_encoder_column_transformer:\n{_encoder_column_transformer}")
        _encoder_column_transformer.set_output(transform="pandas")
        return _encoder_column_transformer



    def _scaler(self):
        """Scale the data.
        
        Returns:
            _scaler_column_transformer: A column transformer, in sklearn.compose.ColumnTransformer object.
        """
        _scaler_column_transformer_list = list()
        for _s_c, _s_m in zip(self.scale_cols, self.scale_method):
            _scaler_column_transformer_list.append(
                ("scale_" + str(_s_c), self.VALID_SCALE_METHOD[_s_m], [_s_c])
            )
            
        _scaler_column_transformer = ColumnTransformer(
            _scaler_column_transformer_list,
            remainder = "passthrough",
            n_jobs = 1,
            verbose = False,
            verbose_feature_names_out = False
        )

        _scaler_column_transformer.set_output(transform="pandas")
        logging.debug(f"_scaler_column_transformer:\n{_scaler_column_transformer}")
        return _scaler_column_transformer
    


if __name__ == "__main__":
    # obesity demo
    data = pd.read_csv("data/obesity.csv", index_col=["id"], na_values=np.nan, encoding="utf-8")
    # logging.debug(f"data.head(10):\n{data.head(10)}")
    # logging.debug(f"data.info():\n{data.info()}")
    # logging.debug(f"data.describe():\n{data.describe()}")

    # Create a sample DataFrame
    Engineer = MyEngineer(
        missing_values_cols=None,
        impute_method=None,
        cat_features=["Gender", "CAEC", "CALC", "MTRANS"],
        encode_method="binary",
        outlier_cols=None,
        scale_cols=None,
        scale_method=None,
        random_state=0,
        n_jobs=1,
        verbose=False
    )
    eng_pipeline = Engineer.construct()

    # Fit the pipeline
    eng_pipeline.fit(data)

    # Transform the data
    transformed_data = eng_pipeline.transform(data)
    logging.debug(f"transformed_data:\n{transformed_data.info()}")


    # housing demo
    """
    data = pd.read_csv("data/housing.csv", index_col=["ID"], na_values=np.nan, encoding="utf-8")
    logging.debug(f"data.head(10):\n{data.head(10)}")
    logging.debug(f"data.info():\n{data.info()}")
    logging.debug(f"data.describe():\n{data.describe()}")

    # Create a sample DataFrame
    Engineer = MyEngineer(
        missing_values_cols=["CRIM", "ZN", "INDUS", "CHAS", "AGE", "LSTAT"],
        impute_method=["median", "median", "median", "median", "median", "median"],
        cat_features=None,
        encode_method=None,
        outlier_cols=None,
        scale_cols=["CRIM", "ZN", "INDUS", "CHAS", "AGE", "LSTAT"],
        scale_method=["standard", "standard", "standard", "minmax", "minmax", "minmax"],
        random_state=0,
        n_jobs=1,
        verbose=False
    )
    eng_pipeline = Engineer.construct()

    # Fit the pipeline
    eng_pipeline.fit(data)

    # Transform the data
    transformed_data = eng_pipeline.transform(data)
    logging.debug(f"transformed_data.describe():\n{transformed_data.describe().round(4)}")

    """
    
    


