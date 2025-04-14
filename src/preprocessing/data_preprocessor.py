import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

# Optional MissForest import with fallback to IterativeImputer
try:
    from missingpy import MissForest
    MISSFOREST_AVAILABLE = True
except ImportError:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor
    MISSFOREST_AVAILABLE = False
    print("MissForest not available. Will use IterativeImputer as fallback when missforest method is selected.")

class DataPreprocessor:
    """
    Class for data preprocessing including:
    - Missing value imputation
    - Categorical encoding
    - Feature type handling
    """
    def __init__(self):
        self.encoders = {}
        self.cat_columns = []
        self.numeric_columns = []
        self.fitted = False
        self.label_encoders = {}  # For storing label encoders used in missforest

    def _get_column_types(self, data):
        """Identify numeric and categorical columns in the dataset."""
        self.numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        self.cat_columns = data.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()

    def impute_knn(self, data, n_neighbors=5):
        """Impute missing values in numeric columns using KNN."""
        numeric_cols = data.select_dtypes(include=["number"]).columns
        if not numeric_cols.empty:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            data_numeric = data[numeric_cols]
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data_numeric),
                columns=numeric_cols,
                index=data.index
            )
            data.loc[:, numeric_cols] = data_imputed
        return data

    def impute_categorical(self, data):
        """Impute missing values in categorical columns using mode."""
        cat_cols = data.select_dtypes(include=["object", "category"]).columns
        
        for col in cat_cols:
            data[col] = data[col].astype(str)
            modes = data[col].mode()
            if len(modes) > 0:
                random_mode = np.random.choice(modes)
                data[col] = data[col].replace("nan", np.nan)
                data[col] = data[col].replace("non", np.nan)
                data[col] = data[col].fillna(random_mode)
            else:
                data[col] = data[col].fillna("Unknown")
        return data

    def impute_missforest(self, data, max_iter=10, n_estimators=100, random_state=42):
        """
        Impute missing values using MissForest or IterativeImputer as fallback.
        
        Args:
            data: pandas DataFrame
            max_iter: int, maximum number of iterations
            n_estimators: int, number of trees in random forest
            random_state: int, random state for reproducibility
        
        Returns:
            pandas DataFrame with imputed values
        """
        # Store original column order and types
        original_columns = data.columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        # Temporarily encode categorical variables for imputation
        data_encoded = data.copy()
        self.label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            # Handle missing values for fitting
            non_missing = data_encoded[col].dropna()
            le.fit(non_missing)
            self.label_encoders[col] = le
            
            # Transform non-missing values
            mask = data_encoded[col].notna()
            data_encoded.loc[mask, col] = le.transform(data_encoded.loc[mask, col])
            # Convert to float to handle NaN
            data_encoded[col] = data_encoded[col].astype(float)

        # Choose imputer based on availability
        if MISSFOREST_AVAILABLE:
            imputer = MissForest(max_iter=max_iter, 
                               n_estimators=n_estimators,
                               random_state=random_state)
        else:
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=n_estimators,
                                              random_state=random_state),
                max_iter=max_iter,
                random_state=random_state
            )

        # Perform imputation
        imputed_array = imputer.fit_transform(data_encoded)
        imputed_df = pd.DataFrame(imputed_array, columns=original_columns, index=data.index)

        # Reverse categorical encoding
        for col in categorical_columns:
            le = self.label_encoders[col]
            imputed_df[col] = le.inverse_transform(imputed_df[col].round().astype(int))

        return imputed_df

    def _onehot_encode(self, data):
        """Encode categorical variables using one-hot encoding."""
        if not self.fitted:
            data[self.cat_columns] = data[self.cat_columns].astype(str)
            
            self.encoders['onehot'] = OneHotEncoder(
                handle_unknown='ignore', 
                sparse_output=False
            )
            self.encoders['onehot'].fit(data[self.cat_columns])
            self.fitted = True

        data[self.cat_columns] = data[self.cat_columns].astype(str)
        encoded = self.encoders['onehot'].transform(data[self.cat_columns])
        
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoders['onehot'].get_feature_names_out(self.cat_columns),
            index=data.index
        )
        return pd.concat([data.drop(columns=self.cat_columns), encoded_df], axis=1)

    def _label_encode(self, data):
        """Encode categorical variables using label encoding."""
        data = data.copy()
        
        for col in self.cat_columns:
            # Пропускаем колонки, которых нет в данных
            if col not in data.columns:
                print(f"Column {col} not found in test data")
                continue
                
            if col not in self.encoders:
                le = LabelEncoder()
                le.fit(data[col])
                self.encoders[col] = le
            
            mask = data[col].isin(self.encoders[col].classes_)
            encoded_values = np.full(len(data), -1)
            encoded_values[mask] = self.encoders[col].transform(data[col][mask])
            data[col] = encoded_values.astype('int32')
        
        return data

    def _target_encode(self, data, target_col):
        """Encode categorical variables using target encoding."""
        if not self.fitted:
            self.encoders['target'] = TargetEncoder(cols=self.cat_columns)
            self.encoders['target'].fit(data[self.cat_columns], data[target_col])
            self.fitted = True
        
        data[self.cat_columns] = self.encoders['target'].transform(data[self.cat_columns])
        return data

    def impute(self, data, method="knn", **kwargs):
        """
        Impute missing values using specified method.
        
        Args:
            data: pandas DataFrame
            method: str, imputation method ('knn' or 'missforest')
            **kwargs: additional arguments for imputation method
                For KNN:
                    - n_neighbors: int
                For MissForest:
                    - max_iter: int
                    - n_estimators: int
                    - random_state: int
        """
        if method == 'knn':
            data = self.impute_knn(data, **kwargs)
            data = self.impute_categorical(data)
            return data
        elif method == 'missforest':
            return self.impute_missforest(data, **kwargs)
        else:
            raise ValueError("Unknown imputation method. Use 'knn' or 'missforest'.")
            
    def encode(self, data, method='onehot', target_col=None):
        """
        Encode categorical variables using specified method.
        
        Args:
            data: pandas DataFrame
            method: str, encoding method ('onehot', 'label', or 'target')
            target_col: str, column name for target encoding
        """
        if not self.cat_columns:  # Определяем типы колонок только если это ещё не сделано
            self._get_column_types(data)
            print("Before encoding:", self.cat_columns)
        
        if method == 'onehot':
            return self._onehot_encode(data)
        elif method == 'label':
            return self._label_encode(data)
        elif method == 'target':
            if not target_col:
                raise ValueError("For Target Encoding specify target_col")
            return self._target_encode(data, target_col)
        else:
            raise ValueError("Unknown encoding method") 