import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import category_encoders as ce
import joblib
import os
import json
import numpy as np


class Encoder():
    """Category variable encoding module
    Supported methods: OneHot, Label, Target, Frequency, Binary, Ordinal

    Note:
    1. In machine learning, the encoder from the training set should also be used during the testing phase to avoid data leakage.
    2. Similarly, during cross-validation, a separate encoder should be constructed for each fold.
    """
    def __init__(self, method='onehot', target_col=None):
        """
        param method: Encoding method ['onehot','label','target','frequency','binary','ordinal']
        param target_col: Target column name required for target encoding
        """
        self.VALID_METHODS = ['onehot', 'label', 'target', 'frequency', 'binary', 'ordinal']
        
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method. Choose from {self.VALID_METHODS}")
        if method == 'target' and target_col is None:
            raise ValueError("target_col must be specified for target encoding")

        self.method = method
        self.target_col = target_col
        self.encoder = None
        self.feature_names = None


    def fit(self, X, cat_cols: list[str]|tuple[str], y=None):
        """
        Fit the encoder
        
        Parameters:
            X: DataFrame to fit
            y: Target values (used for target encoding)
            cat_cols: Category columns to encode

        Returns:
            self
        """
        # Validate input types
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(cat_cols, (list, tuple)):
            raise TypeError("cat_cols must be a list or tuple")
        if X[cat_cols].isnull().any().any():
            raise ValueError("Input contains null values. Please handle missing values before encoding.")
        if not cat_cols:
            raise ValueError("cat_cols cannot be empty")
        missing_cols = [col for col in cat_cols if col not in X]
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in dataframe")
        # Validate y type for target encoding
        if self.method == 'target':
            if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
                raise TypeError("y must be a numpy ndarray, pandas Series, or pandas DataFrame")
            # Get length of y accounting for different types
            y_length = len(y) if isinstance(y, (pd.Series, pd.DataFrame)) else y.shape[0]
            if y_length != len(X):
                raise ValueError(f"Length of y ({y_length}) does not match length of X ({len(X)})")

        # Store cat_cols for later use in transform
        self.cat_cols_ = cat_cols

        if self.method == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.encoder.fit(X[cat_cols])
            
        elif self.method == 'label':
            self.encoder = {col: LabelEncoder() for col in cat_cols}
            for col in cat_cols:
                self.encoder[col].fit(X[col])
                
        elif self.method == 'target':
            self.encoder = ce.TargetEncoder(cols=cat_cols)
            self.encoder.fit(X[cat_cols], y)
            
        elif self.method == 'frequency':
            self.encoder = {col: X[col].value_counts(normalize=True).to_dict() 
                          for col in cat_cols}
            
        elif self.method == 'binary':
            self.encoder = ce.BinaryEncoder(cols=cat_cols)
            self.encoder.fit(X[cat_cols])
            
        elif self.method == 'ordinal':
            self.encoder = ce.OrdinalEncoder(cols=cat_cols)
            self.encoder.fit(X[cat_cols])
            
        return self


    def transform(self, X):
        """Apply encoding
        Parameters:
            X: DataFrame to transform
        Returns:
            Transformed DataFrame
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not fitted. Call fit() first")
        
        if not hasattr(self, 'cat_cols_'):
            raise RuntimeError("cat_cols not found. Encoder may not be properly fitted.")

        cat_cols = self.cat_cols_
        missing_cols = [col for col in cat_cols if col not in X]
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in dataframe")

        X = X.copy()
        
        if self.method == 'onehot':
            encoded = self.encoder.transform(X[cat_cols])
            encoded_df = pd.DataFrame(encoded,
                                    columns=self.encoder.get_feature_names_out(cat_cols),
                                    index=X.index)
            return pd.concat([X.drop(cat_cols, axis=1), encoded_df], axis=1)
            
        elif self.method == 'label':
            for col in cat_cols:
                X[col] = self.encoder[col].transform(X[col])
            return X
                
        elif self.method == 'target':
            X[cat_cols] = self.encoder.transform(X[cat_cols])
            return X
            
        elif self.method == 'frequency':
            for col in cat_cols:
                frequency_map = self.encoder[col]  # Get frequency mapping dictionary for current category column
                
                # If the column is Categorical, first convert to string type
                if pd.api.types.is_categorical_dtype(X[col]):
                    current_col = X[col].astype(str)
                else:
                    current_col = X[col]
                
                # Map category values to corresponding frequencies
                mapped_frequencies = current_col.map(frequency_map)
                
                # Handle unknown categories by filling NaN with 0
                mapped_frequencies_filled = mapped_frequencies.fillna(0)
                
                # Create a new frequency encoding column
                new_col_name = col + '_freq'
                X[new_col_name] = mapped_frequencies_filled
            
            return X.drop(cat_cols, axis=1)
        
        elif self.method == 'binary':
            binary_cols = self.encoder.transform(X[cat_cols])
            return pd.concat([X.drop(cat_cols, axis=1), binary_cols], axis=1)
            
        elif self.method == 'ordinal':
            X[cat_cols] = self.encoder.transform(X[cat_cols])
            return X


    def fit_transform(self, X, cat_cols, y=None):
        """
        Fit encoder and return transformed data
        Parameters:
            X: DataFrame to fit and transform
            y: Target values (used for target encoding)
            cat_cols: Category columns to encode
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, cat_cols, y).transform(X)


    def save(self, path=None):
        """Save the encoder"""
        # Set default path using method name if not provided
        if path is None:
            path = os.path.join('encoding', f'{self.method}_encoder.pkl')
            
        if os.path.exists(path):
            import warnings
            warnings.warn(f"File {path} already exists and will be overwritten")
        
        # Ensure the target directory exists (including the encoding directory)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            joblib.dump({
                'method': self.method,
                'target_col': self.target_col,
                'encoder': self.encoder,
                'feature_names': self.feature_names
            }, path)
        except Exception as e:
            raise IOError(f"Failed to save encoder: {str(e)}")


    @classmethod
    def load(cls, path='encoding/encoder.pkl'):
        """Load the encoder"""
        data = joblib.load(path)
        obj = cls(method=data['method'], target_col=data['target_col'])
        obj.encoder = data['encoder']
        obj.feature_names = data['feature_names']
        return obj

    def validate_encoding(self, encoded_df, original_df):
        """Validate the encoding results"""
        # Check if number of rows matches
        if len(encoded_df) != len(original_df):
            raise ValueError("Number of rows changed after encoding")
        
        # Check for invalid values
        if encoded_df.isnull().any().any():
            raise ValueError("Encoding produced null values")
        
        # For onehot encoding, verify binary values
        if self.method == 'onehot':
            encoded_cols = [col for col in encoded_df.columns 
                           if col not in original_df.columns]
            if not encoded_df[encoded_cols].isin([0, 1]).all().all():
                raise ValueError("OneHot encoding produced non-binary values")


    def get_mapping(self, df, cat_cols):
        """Get encoding mapping relationships.
        Parameters
            df : pandas.DataFrame
                Original dataframe containing categorical features
            cat_cols : list
                List of categorical column names to get mapping for
        Returns
            dict :
                Dictionary containing mapping relationships between original and encoded values
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not fitted. Call fit() first")
        
        mapping = {}
        
        if self.method == 'onehot':
            for col in cat_cols:
                unique_values = df[col].unique()
                encoded = self.encoder.transform(df[cat_cols].loc[df[col].isin(unique_values)].head(len(unique_values)))
                feature_names = self.encoder.get_feature_names_out()
                col_mapping = {
                    val: dict(zip(feature_names, row))
                    for val, row in zip(unique_values, encoded)
                }
                mapping[col] = col_mapping
            
        elif self.method == 'label':
            for col in cat_cols:
                label_encoder = self.encoder[col]
                mapping[col] = dict(zip(label_encoder.classes_, 
                                      label_encoder.transform(label_encoder.classes_)))
            
        elif self.method == 'target':
            # 需要保持与fit时相同的列结构
            for col in cat_cols:
                # 创建包含所有分类列的临时数据帧
                # 其他列用第一个类别值填充，当前列用实际值
                temp_df = pd.DataFrame({
                    c: [df[c].iloc[0]] * len(df[col].unique())
                    for c in cat_cols
                })
                temp_df[col] = df[col].unique()
                
                # 获取编码结果（保持所有分类列存在）
                encoded_values = self.encoder.transform(temp_df)[col].values
                
                # 构建映射关系
                mapping[col] = {
                    self.convert_numpy_types(orig): self.convert_numpy_types(enc)
                    for orig, enc in zip(temp_df[col], encoded_values)
                }

        elif self.method == 'frequency':
            mapping = self.encoder  # frequency encoder itself stores the mapping
            
        elif self.method == 'binary':
            # For binary encoding, show binary representation for each category value
            for col in cat_cols:
                unique_values = df[col].unique()
                temp_df = pd.DataFrame({c: [unique_values[0]] * len(unique_values) for c in cat_cols})
                temp_df[col] = unique_values
                encoded = self.encoder.transform(temp_df)
                binary_cols = [c for c in encoded.columns if c.startswith(f"{col}_")]
                mapping[col] = {
                    val: dict(zip(binary_cols, row))
                    for val, row in zip(unique_values, encoded[binary_cols].values)
                }
            
        elif self.method == 'ordinal':
            # For ordinal encoding, return category to ordinal mapping
            for col in cat_cols:
                # Get unique values from the original data
                unique_values = df[col].unique()
                # Create a temporary dataframe with all category columns
                temp_df = pd.DataFrame({c: [unique_values[0]] * len(unique_values) for c in cat_cols})
                # Set the current column's values
                temp_df[col] = unique_values
                # Transform the values to get their ordinal encoding
                encoded_values = self.encoder.transform(temp_df)[col].values
                # Create mapping dictionary
                mapping[col] = dict(zip(unique_values, encoded_values))
        
        return mapping


    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {self.convert_numpy_types(key): self.convert_numpy_types(value) 
                    for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_numpy_types(item) for item in obj]
        return obj


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data.csv')
    
    # Create encoding directory if it doesn't exist
    os.makedirs('encoding', exist_ok=True)
    
    # Test all encoding methods and save mapping relationships
    cat_cols = ['x16', 'x17']
    
    # Define encoding methods to test except target encoding
    methods = ['onehot', 'label', 'frequency', 'binary', 'ordinal']
    for method in methods:
        encoder = Encoder(method=method)  # Initialize encoder
        encoder.fit(df, cat_cols=cat_cols)  # Fit data
        encoder.save()  # Save encoder
        mapping = encoder.get_mapping(df, cat_cols)  # Get mapping relationships
        mapping = encoder.convert_numpy_types(mapping)  # Convert numpy types to native Python types
        
        # Save to JSON file
        json_path = f'encoding/{method}_mapping.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
    
    # Handle target encoding separately as it requires target column
    encoder = Encoder(method='target', target_col='y')
    encoder.fit(df, y=df['y'], cat_cols=cat_cols)
    mapping = encoder.get_mapping(df, cat_cols)
    mapping = encoder.convert_numpy_types(mapping)
    
    json_path = 'encoding/target_mapping.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    
    print(f"Saved target encoding mapping to {json_path}")