"""Data engineering module for preprocessing machine learning data.

This module provides functions and classes for handling various data preprocessing tasks
including imputation, feature encoding, and scaling.
"""
import logging
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import category_encoders as ce


def data_engineer(
    missing_values_cols: list[str] | tuple[str] | None = None,
    impute_method: str | list[str] | tuple[str] | None = None,
    cat_features: list[str] | tuple[str] | None = None,
    encode_method: str | list[str] | tuple[str] | None = None,
    scale_cols: list[str] | tuple[str] | None = None,
    scale_method: str | list[str] | tuple[str] | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
):
    """Create a data engineering pipeline.
    
    Args:
        missing_values_cols: Columns with missing values to impute
        impute_method: Methods for imputing missing values
        cat_features: Categorical features to encode
        encode_method: Methods for encoding categorical features
        scale_cols: Columns to scale
        scale_method: Methods for scaling numeric features
        n_jobs: Number of jobs to run in parallel
        verbose: Whether to print verbose output
    
    Returns:
        A scikit-learn Pipeline object or None if no steps specified
    """
    engineer = MyEngineer(
        missing_values_cols=missing_values_cols,
        impute_method=impute_method,
        cat_features=cat_features,
        encode_method=encode_method,
        scale_cols=scale_cols,
        scale_method=scale_method,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # If all data engineering steps are None, return None
    if (missing_values_cols is None \
            and cat_features is None \
                and scale_cols is None):
        return None

    return engineer.construct()


class MyEngineer:
    """Data engineering for the training validation and test set.
    
    Step:
    1. Imputation of missing values
    3. Encoding of categorical features
    4. Data standardization or normalization

    Optional:
    -  Output the report of the data engineering process
    """
    def __init__(
        self,
        missing_values_cols: list[str] | tuple[str] | None = None,
        impute_method: str | list[str] | tuple[str] | None = None,
        cat_features: list[str] | tuple[str] | None = None,
        encode_method: str | list[str] | tuple[str] | None = None,
        scale_cols: list[str] | tuple[str] | None = None,
        scale_method: str | list[str] | tuple[str] | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        # Specify the valid imputation and scaling methods
        self.valid_impute_method = {
            "mean": SimpleImputer(missing_values=np.nan, strategy="mean"),
            "median": SimpleImputer(missing_values=np.nan, strategy="median"),
            "most_frequent": SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            "constant": SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        }
        self.valid_encode_method = {
            "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            "binary": ce.BinaryEncoder(),
            "ordinal": ce.OrdinalEncoder(),
        }
        self.valid_scale_method = {
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

        # Check if one variable is None, the other should also be None
        # For missing values and imputation
        if ((missing_values_cols is None and impute_method is not None) or
            (missing_values_cols is not None and impute_method is None)):
            raise ValueError("If missing_values_cols is None, impute_method must also be None, and vice versa.")
        
        # For categorical features and encoding
        if ((cat_features is None and encode_method is not None) or
            (cat_features is not None and encode_method is None)):
            raise ValueError("If cat_features is None, encode_method must also be None, and vice versa.")
        
        # For scaling columns and methods
        if ((scale_cols is None and scale_method is not None) or
            (scale_cols is not None and scale_method is None)):
            raise ValueError("If scale_cols is None, scale_method must also be None, and vice versa.")

        # Check the length of the input
        if missing_values_cols:
            assert len(missing_values_cols) == len(impute_method)
        if cat_features:
            assert len(cat_features) == len(encode_method)
        if scale_cols:
            assert len(scale_cols) == len(scale_method)
        
        # Check if elements in impute_method, encode_method, scale_method are valid
        if missing_values_cols:
            assert all(s in self.valid_impute_method for s in impute_method), \
                f"impute_method must be one of {list(self.valid_impute_method.keys())}"
        if cat_features:
            assert all(s in self.valid_encode_method for s in encode_method), \
                f"encode_method must be one of {list(self.valid_encode_method.keys())}"
        if scale_cols:
            assert all(s in self.valid_scale_method for s in scale_method), \
                f"scale_method must be one of {list(self.valid_scale_method.keys())}"

        self.missing_values_cols = missing_values_cols
        self.impute_method = impute_method
        self.cat_features = cat_features
        self.encode_method = encode_method
        self.scale_cols = scale_cols
        self.scale_method = scale_method
        self.n_jobs = n_jobs
        self.verbose = verbose

    def construct(self):
        """Construct the data engineering pipeline.
        
        Returns:
            self.pipeline: The data engineering pipeline, in `sklearn.pipeline.Pipeline` object.
        """
        imputer_column_transformer = None
        encoder_column_transformer = None
        scaler_column_transformer = None

        pipeline_dict = {
            "imputer": imputer_column_transformer,
            "encoder": encoder_column_transformer,
            "scaler": scaler_column_transformer
        }

        if self.missing_values_cols is not None:
            pipeline_dict["imputer"] = self._imputer()

        if self.cat_features is not None:
            pipeline_dict["encoder"] = self._encoder()

        if self.scale_cols is not None:
            pipeline_dict["scaler"] = self._scaler()

        # Filter out None values from the pipeline dictionary
        pipeline_dict = {k: v for k, v in pipeline_dict.items() if v is not None}
        
        # Construct the pipeline using the filtered dictionary
        return Pipeline([(k, v) for k, v in pipeline_dict.items()])

    def _imputer(self):
        """Impute the missing values.
        
        Returns:
            _imputer_column_transformer: 
                A column transformer, in sklearn.compose.ColumnTransformer object.
        """
        imputer_column_transformer_list = []
        for m_c, i_m in zip(self.missing_values_cols, self.impute_method):
            imputer_column_transformer_list.append(
                ("impute_" + str(m_c), self.valid_impute_method[i_m], [m_c])
            )
        
        imputer_column_transformer = ColumnTransformer(
            imputer_column_transformer_list,
            remainder="passthrough",
            n_jobs=1,
            verbose=False,
            verbose_feature_names_out=False
        )

        imputer_column_transformer.set_output(transform="pandas")
        return imputer_column_transformer

    def _encoder(self):
        """Encode the categorical features.
        
        Returns:
            _encoder_column_transformer: 
                A column transformer, in sklearn.compose.ColumnTransformer object.
        """
        encoder_column_transformer_list = []
        for e_c, e_m in zip(self.cat_features, self.encode_method):
            encoder_column_transformer_list.append(
                ("encode_" + str(e_c), self.valid_encode_method[e_m], [e_c])
            )

        encoder_column_transformer = ColumnTransformer(
            encoder_column_transformer_list,
            remainder="passthrough",
            n_jobs=1,
            verbose=False,
            verbose_feature_names_out=False
        )
        logging.debug("encoder_column_transformer:\n%s", encoder_column_transformer)
        encoder_column_transformer.set_output(transform="pandas")
        return encoder_column_transformer

    def _scaler(self):
        """Scale the data.
        
        Returns:
            _scaler_column_transformer: 
                A column transformer, in sklearn.compose.ColumnTransformer object.
        """
        scaler_column_transformer_list = []
        for s_c, s_m in zip(self.scale_cols, self.scale_method):
            scaler_column_transformer_list.append(
                ("scale_" + str(s_c), self.valid_scale_method[s_m], [s_c])
            )
        
        scaler_column_transformer = ColumnTransformer(
            scaler_column_transformer_list,
            remainder="passthrough",
            n_jobs=1,
            verbose=False,
            verbose_feature_names_out=False
        )

        scaler_column_transformer.set_output(transform="pandas")
        logging.debug("scaler_column_transformer:\n%s", scaler_column_transformer)
        return scaler_column_transformer
