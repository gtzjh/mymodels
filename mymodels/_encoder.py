import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import category_encoders as ce
import numpy as np
import json
from sklearn.model_selection import train_test_split


# Convert all NumPy types in the mapping dictionary before serialization
def convert_numpy_types(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(i) for i in obj]
    return obj



class Encoder():
    """Category variable encoding module
    Supported methods: OneHot, Label, Target, Frequency, Binary, Ordinal

    Note:
    1. In machine learning, the encoder from the training set should also be used during the testing phase to avoid data leakage.
    2. Similarly, during cross-validation, a separate encoder should be constructed for each fold.
    """
    def __init__(self, method='onehot', target_col=None):
        self.VALID_METHODS = ['onehot', 'label', 'target', 'frequency', 'binary', 'ordinal']
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method. Choose from {self.VALID_METHODS}")
        if method == 'target' and target_col is None:
            raise ValueError("target_col must be specified for target encoding")

        self.method = method
        self.target_col = target_col
        self.encoder = None
        self.X = None
        self._cat_cols = None


    def fit(self, X, cat_cols: str, y=None):
        """Fit an encoder on a list of training data"""

        assert isinstance(cat_cols, str)
        self._cat_cols = cat_cols

        # 接受list, tuple, Series, DataFrame, np.array格式，强制转为1d array，然后再回到pd.dataframe，确保类型正确
        assert isinstance(X, pd.DataFrame)
        assert len(X.columns) == 1, "Input DataFrame must be 1-dimensional"
        self.X = X

        if y is not None:
            assert isinstance(y, pd.Series)

        if self.method == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.encoder.fit(self.X)
            
        elif self.method == 'label':
            self.encoder = LabelEncoder()
            self.encoder.fit(self.X[self._cat_cols].values.ravel())
                
        elif self.method == 'target':
            self.encoder = ce.TargetEncoder(cols=cat_cols)
            self.encoder.fit(self.X, y)
            
        elif self.method == 'frequency':
            self.encoder = self.X[self._cat_cols].value_counts(normalize=True).to_dict()
            
        elif self.method == 'binary':
            self.encoder = ce.BinaryEncoder(cols=self._cat_cols)
            self.encoder.fit(self.X)
            
        elif self.method == 'ordinal':
            self.encoder = ce.OrdinalEncoder(cols=self._cat_cols)
            self.encoder.fit(self.X)
            
        return self


    def transform(self, X):
        """Apply encoding

        Parameters:
            X: 1D array or DataFrame to transform
        
        Returns:
            Transformed DataFrame in 1D array or DataFrame
        """

        if not hasattr(self, '_cat_cols'):
            raise RuntimeError("cat_cols not found. Encoder may not be properly fitted.")

        # 接受list, tuple, Series, DataFrame, np.array格式，强制转为1d array，然后再回到pd.dataframe，确保类型正确
        assert isinstance(X, pd.DataFrame)
        assert len(X.columns) == 1, "Input DataFrame must be 1-dimensional"
    
        if self.method == 'onehot':
            encoded = self.encoder.transform(X)
            encoded_df = pd.DataFrame(encoded,
                                      columns=self.encoder.get_feature_names_out([self._cat_cols]),
                                      index=X.index)
            return encoded_df
        
        elif self.method == 'label':
            encoded = self.encoder.transform(X[self._cat_cols].values.ravel())
            return pd.DataFrame(encoded,
                                columns=[self._cat_cols],
                                index=X.index)
                
        elif self.method == 'target':
            return self.encoder.transform(X)
            
        elif self.method == 'frequency':
            # 直接使用频率映射字典
            col = self._cat_cols
            
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                current_col = X[col].astype(str)
            else:
                current_col = X[col]
            
            # 直接使用self.encoder，它已经是频率映射字典
            mapped_frequencies = current_col.map(self.encoder)
            mapped_frequencies_filled = mapped_frequencies.fillna(0)
            
            new_col_name = col + '_freq'
            X[new_col_name] = mapped_frequencies_filled
            
            return X.drop(col, axis=1)
        
        elif self.method == 'binary':
            return self.encoder.transform(X)
            
        elif self.method == 'ordinal':
            return self.encoder.transform(X)
    

    def get_mapping(self):
        """Get encoding mapping relationships"""
        if not hasattr(self, '_cat_cols'):
            raise RuntimeError("cat_cols not found. Encoder may not be properly fitted.")
        
        mapping = dict()  # 使用dict()来初始化mapping字典

        if self.method == 'onehot':
            unique_values = self.X[self._cat_cols].unique()
            temp_df = pd.DataFrame({self._cat_cols: unique_values})
            encoded = self.encoder.transform(temp_df)
            feature_names = self.encoder.get_feature_names_out([self._cat_cols])
            
            mapping = {
                convert_numpy_types(val): dict(zip(feature_names, row))
                for val, row in zip(unique_values, encoded)
            }

        elif self.method == 'label':
            mapping[self._cat_cols] = dict(zip(self.encoder.classes_, 
                                               self.encoder.transform(self.encoder.classes_)))
            
        elif self.method == 'target':
            temp_df = pd.DataFrame({self._cat_cols: self.X[self._cat_cols].unique()})
            encoded_values = self.encoder.transform(temp_df)[self._cat_cols].values
            mapping = {
                convert_numpy_types(orig): convert_numpy_types(enc)
                for orig, enc in zip(temp_df[self._cat_cols], encoded_values)
            }

        elif self.method == 'frequency':
            mapping = convert_numpy_types(self.encoder)
            
        elif self.method == 'binary':
            unique_values = self.X[self._cat_cols].unique()
            temp_df = pd.DataFrame({self._cat_cols: unique_values})
            encoded = self.encoder.transform(temp_df)
            binary_cols = [c for c in encoded.columns if c.startswith(f"{self._cat_cols}_")]
            mapping = {
                convert_numpy_types(val): convert_numpy_types(dict(zip(binary_cols, row)))
                for val, row in zip(unique_values, encoded[binary_cols].values)
            }
            
        elif self.method == 'ordinal':
            unique_values = self.X[self._cat_cols].unique()
            temp_df = pd.DataFrame({self._cat_cols: unique_values})
            encoded_values = self.encoder.transform(temp_df)[self._cat_cols].values
            mapping = convert_numpy_types(dict(zip(unique_values, encoded_values)))
        
        return mapping
    


def fit_transform_multi_features(
        categorical_X: pd.DataFrame,
        encoder_methods: str | list[str] | tuple[str] | None,
        y: pd.Series | None = None
    ) -> tuple[pd.DataFrame, dict]:
    """Transform the categorical features of the dataset
    ONE BY ONE COLUMNS
    如果_encoder_methods为list或是tuple, 则它们的长度要与输入的categorical_X的列数相同
    如此确保每种类别都有对应的编码方法
    如果 _encoder_methods 为str, 则使用该编码方法对所有类别特征进行编码
    """
    if isinstance(encoder_methods, (list, tuple)):
        assert len(encoder_methods) == len(categorical_X.columns), \
            "The length of list of encoder method must be the same as the list of categorical features"
    else:
        """
        如果是只是指定一种编码方法, 则生成一个与输入X同等列数的list, 当中每个元素都是这一编码方法。
        即为每个分类特征使用相同编码方法
        """
        encoder_methods = [encoder_methods] * len(categorical_X.columns) if encoder_methods else None

    if y is not None:
        y_name = str(y.name)
    else:
        y_name = None
    
    transformed_X_list = list()
    encoder_dict = dict()
    mapping_dict = dict()
    for c, e in zip(categorical_X.columns, encoder_methods):
        encoder = Encoder(
            method=e,
            target_col=y_name
        )
        encoder.fit(
            X=categorical_X.loc[:, c].to_frame(),
            cat_cols=c,
            y=y
        )
        transformed_X = encoder.transform(X=categorical_X.loc[:, c].to_frame())
        transformed_X_list.append(transformed_X)
        encoder_dict[c] = encoder
        mapping_dict[c] = encoder.get_mapping()
        mapping_dict[c] = convert_numpy_types(mapping_dict[c])
    
    transformed_X_df = pd.concat(transformed_X_list, axis=1)
    return transformed_X_df, encoder_dict, mapping_dict


def transform_multi_features(
        categorical_X: pd.DataFrame,
        encoder_dict: dict
    ) -> pd.DataFrame:
    """Transform the categorical features of the dataset
    ONE BY ONE COLUMNS
    """
    assert list(categorical_X.columns) == list(encoder_dict.keys())

    categorical_X_list = list()
    for c in categorical_X.columns:
        transformed_X = encoder_dict[c].transform(X=categorical_X.loc[:, c].to_frame())
        categorical_X_list.append(transformed_X)
    transformed_X_df = pd.concat(categorical_X_list, axis=1)

    return transformed_X_df



if __name__ == "__main__":
    data = pd.read_csv("data/titanic/train.csv", encoding="utf-8", na_values=np.nan)
    data = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
    data = data.dropna()
    data = data.reset_index(drop=True)

    cat_cols = ["Sex", "Embarked"]

    def convert_to_category(df):
        """Convert non-numeric columns in a dataframe to category dtype"""
        df_copy = df.copy()
        non_numeric_cols = df_copy.select_dtypes(exclude=['int64', 'float64']).columns
        for col in non_numeric_cols:
            df_copy[col] = df_copy[col].astype('category')
        return df_copy
    data = convert_to_category(data)
    train, test = train_test_split(data.loc[:, :], test_size=0.3, random_state=42)
    print(train.info())


    ###################################################################################
    """Test function `trans_category` in single encode method"""
    for method in [
        "onehot", 
        "label", "target", "frequency", "binary", "ordinal"
    ]:
        transformed_train_df, encoder_dict, mapping_dict = fit_transform_multi_features(
            train.loc[:, cat_cols],
            encoder_methods = method,
            y = train["Survived"]
        )
        train_transformed = train.drop(columns=cat_cols)
        train_transformed = pd.concat([train_transformed, transformed_train_df], axis=1)
        print(train_transformed.info())

        transformed_test_df = transform_multi_features(
            test.loc[:, cat_cols],
            encoder_dict
        )
        test_transformed = test.drop(columns=cat_cols)
        test_transformed = pd.concat([test_transformed, transformed_test_df], axis=1)
        print(test_transformed.info())
        
        # Apply conversion to the entire mapping dictionary
        print(
            json.dumps(
                mapping_dict,
                indent=4,
                sort_keys=True,
                ensure_ascii=False,
                separators=(',', ': ')
            )
        )
        print("-" * 80)
        
    ###################################################################################


    ###################################################################################
    """Test function `trans_category` in multiple encode methods"""
    """
    transformed_X_df, encoder_dict, mapping_dict = fit_transform_multi_features(
        data.loc[:, cat_cols],
        encoder_methods = ["onehot", "label"],
        y = data["Survived"]
    )
    print(transformed_X_df.head(10))

    # Apply conversion to the entire mapping dictionary
    print(
        json.dumps(
            convert_numpy_types(mapping_dict),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            separators=(',', ': ')
        )
    )
    print("-" * 80)
    """
    ###################################################################################
    

    