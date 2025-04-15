import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from category_encoders import LeaveOneOutEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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
        self.target_encoder_le = None
        self.lsa_vectorizer = None
        self.lsa_svd = None
        self.lsa_feature_names = None

    def _get_column_types(self, data):
        """Identify numeric and categorical columns in the dataset."""
        self.numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        self.cat_columns = data.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        print(f"Initial numeric columns: {self.numeric_columns}")
        print(f"Initial categorical columns: {self.cat_columns}")

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
        cat_cols_present = [col for col in self.cat_columns if col in data.columns]

        for col in cat_cols_present:
            data[col] = data[col].astype(str)
            modes = data[col].mode()
            if not modes.empty:
                data[col] = data[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
                mode_val = modes[0]
                data[col] = data[col].fillna(mode_val)
            else:
                data[col] = data[col].fillna("Unknown")
        return data

    def impute_missforest(self, data, max_iter=10, n_estimators=100, random_state=42):
        """
        Impute missing values using MissForest or IterativeImputer as fallback.
        """
        original_columns = data.columns
        if self.cat_columns:
            categorical_columns = [col for col in self.cat_columns if col in data.columns]
        else:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        data_encoded = data.copy()
        temp_label_encoders = {}

        for col in categorical_columns:
            le = LabelEncoder()
            non_missing = data_encoded[col].dropna().astype(str)
            if not non_missing.empty:
                le.fit(non_missing)
                temp_label_encoders[col] = le
                mask = data_encoded[col].notna()
                data_encoded.loc[mask, col] = le.transform(data_encoded.loc[mask, col].astype(str))
                data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')
            else:
                data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')

        if MISSFOREST_AVAILABLE:
            imputer = MissForest(max_iter=max_iter, n_estimators=n_estimators, random_state=random_state)
        else:
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
                max_iter=max_iter, random_state=random_state
            )

        numeric_cols_for_impute = data_encoded.select_dtypes(include=np.number).columns
        imputed_array = imputer.fit_transform(data_encoded[numeric_cols_for_impute])
        imputed_df_numeric = pd.DataFrame(imputed_array, columns=numeric_cols_for_impute, index=data.index)

        imputed_df = data.copy()
        imputed_df[numeric_cols_for_impute] = imputed_df_numeric

        for col in categorical_columns:
            if col in temp_label_encoders:
                le = temp_label_encoders[col]
                min_label, max_label = 0, len(le.classes_) - 1
                if col in imputed_df.columns:
                    imputed_int = imputed_df[col].fillna(-1).round().astype(int)
                    imputed_clipped = np.clip(imputed_int, min_label, max_label)
                    valid_mask = imputed_clipped >= 0
                    imputed_df.loc[valid_mask, col] = le.inverse_transform(imputed_clipped[valid_mask])

        return imputed_df

    def _label_encode_target(self, data, target_col):
        """Encode the target column specifically using LabelEncoder."""
        if target_col not in data.columns:
            print(f"Target column '{target_col}' not found for label encoding.")
            return data

        if self.target_encoder_le is None:
            print(f"Fitting target encoder for column: {target_col}")
            self.target_encoder_le = LabelEncoder()
            self.target_encoder_le.fit(data[target_col].astype(str))

        print(f"Transforming target column: {target_col}")
        data[target_col] = self.target_encoder_le.transform(data[target_col].astype(str))
        return data

    def _get_active_cat_cols(self, data, target_col, method):
        """Get list of categorical columns for current encoding method, excluding target unless method is 'target'."""
        if not self.cat_columns:
             print("Warning: Categorical columns not identified. Call _get_column_types first.")
             return []

        if method == 'target':
            cols = [col for col in self.cat_columns if col != target_col and col in data.columns]
        else:
            cols = [col for col in self.cat_columns if col != target_col and col in data.columns]
        return cols

    def _onehot_encode(self, data, cols_to_encode):
        """Encode specified columns using one-hot encoding."""
        if not cols_to_encode:
            return data

        data = data.copy()
        encoder_key = 'onehot'
        if encoder_key not in self.encoders:
            print(f"OHE: Fitting on {len(cols_to_encode)} columns.")
            self.encoders[encoder_key] = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )
            self.encoders[encoder_key].fit(data[cols_to_encode].astype(str))

        print(f"OHE: Transforming {len(cols_to_encode)} columns.")
        encoded = self.encoders[encoder_key].transform(data[cols_to_encode].astype(str))
        new_cols = self.encoders[encoder_key].get_feature_names_out(cols_to_encode)
        encoded_df = pd.DataFrame(encoded, columns=new_cols, index=data.index)

        cols_to_keep = self.numeric_columns + [col for col in data.columns if col not in cols_to_encode and col not in self.numeric_columns]
        cols_to_keep = [col for col in cols_to_keep if col in data.columns]

        return pd.concat([data[cols_to_keep], encoded_df], axis=1)

    def _label_encode(self, data, cols_to_encode):
        """Encode specified columns using label encoding."""
        if not cols_to_encode:
            return data

        data = data.copy()
        print(f"LabelEncoding: Processing {len(cols_to_encode)} columns.")
        for col in cols_to_encode:
            col_key = f'label_{col}'
            if col_key not in self.encoders:
                le = LabelEncoder()
                le.fit(data[col].astype(str))
                self.encoders[col_key] = le
            else:
                le = self.encoders[col_key]

            data[col] = data[col].astype(str)
            mask = data[col].isin(le.classes_)
            encoded_values = np.full(len(data), -1)
            encoded_values[mask] = le.transform(data.loc[mask, col])
            data[col] = encoded_values.astype('int32')
        return data

    def _ordinal_encode(self, data, cols_to_encode):
        """Encode specified columns using ordinal encoding."""
        if not cols_to_encode:
             return data

        data = data.copy()
        encoder_key = 'ordinal'
        print(f"OrdinalEncoding: Processing {len(cols_to_encode)} columns.")
        if encoder_key not in self.encoders:
            self.encoders[encoder_key] = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            self.encoders[encoder_key].fit(data[cols_to_encode].astype(str))

        data[cols_to_encode] = self.encoders[encoder_key].transform(data[cols_to_encode].astype(str))
        data[cols_to_encode] = data[cols_to_encode].astype('int32')
        return data

    def _leaveoneout_encode(self, data, cols_to_encode, target_col, sigma=0.05):
        """Encode specified columns using LeaveOneOut encoding."""
        if not cols_to_encode:
             print("LeaveOneOut: No columns specified for encoding.")
             return data
        if target_col not in data.columns:
             raise ValueError(f"Target column '{target_col}' not found for LeaveOneOut encoding.")

        data = data.copy()
        encoder_key = 'leaveoneout'
        cols_to_encode_present = [c for c in cols_to_encode if c in data.columns]
        if not cols_to_encode_present:
            print(f"LeaveOneOut: None of the specified columns {cols_to_encode} found in data.")
            return data

        print(f"LeaveOneOut: Using {len(cols_to_encode_present)} columns: {cols_to_encode_present} with target '{target_col}', sigma={sigma}")
        if encoder_key not in self.encoders:
            encoder = LeaveOneOutEncoder(
                cols=cols_to_encode_present,
                sigma=sigma,
                handle_unknown='value',
                handle_missing='value'
            )
            encoder.fit(data[cols_to_encode_present], data[target_col])
            self.encoders[encoder_key] = encoder
        else:
            print(f"LeaveOneOut: Using previously fitted encoder.")

        if target_col in data.columns:
            data = self.encoders[encoder_key].transform(data[cols_to_encode_present], data[target_col])
        else:
            data = self.encoders[encoder_key].transform(data[cols_to_encode_present])
        return data

    def _categorical_to_texts(self, data, cols):
        """Transform categorical data rows into texts."""
        data = data.copy()
        for col in cols:
            data[col] = data[col].astype(str)
        new_data_list = []
        for i, row in data[cols].iterrows():
            new_line = ' '.join([f'{col}_{val}' for col, val in row.items()])
            new_data_list.append(new_line)
        return new_data_list

    def _lsa_encode(self, data, cols_to_encode, n_components):
        """Encode specified columns using LSA."""
        if not cols_to_encode:
            print("LSA: No categorical columns to encode found in data.")
            return data
        data = data.copy()
        # Check fitted status by checking if vectorizer exists
        if self.lsa_vectorizer is None:
            print(f"LSA: Fitting on {len(cols_to_encode)} columns with n_components={n_components}")
            texts = self._categorical_to_texts(data, cols_to_encode)
            self.lsa_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5)
            term_doc_matrix = self.lsa_vectorizer.fit_transform(texts)
            print(f"LSA: Vectorized matrix shape: {term_doc_matrix.shape}")
            self.lsa_svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsa_embeddings = self.lsa_svd.fit_transform(term_doc_matrix)
            print(f"LSA: Embeddings shape: {lsa_embeddings.shape}")
            print(f"LSA: Explained variance ratio sum: {self.lsa_svd.explained_variance_ratio_.sum():.4f}")
            self.lsa_feature_names = [f'LSA_{i+1}' for i in range(n_components)]
            # No longer need self.lsa_fitted flag
        else:
            print(f"LSA: Transforming using fitted components on {len(cols_to_encode)} columns.")
            texts = self._categorical_to_texts(data, cols_to_encode)
            term_doc_matrix = self.lsa_vectorizer.transform(texts)
            lsa_embeddings = self.lsa_svd.transform(term_doc_matrix)

        lsa_df = pd.DataFrame(lsa_embeddings, columns=self.lsa_feature_names, index=data.index)

        # Keep all columns NOT in cols_to_encode
        cols_to_keep = [col for col in data.columns if col not in cols_to_encode]
        result_df = pd.concat([data[cols_to_keep], lsa_df], axis=1)
        return result_df

    def impute(self, data, method="knn", **kwargs):
        """
        Impute missing values using specified method.
        """
        if not self.numeric_columns and not self.cat_columns:
             self._get_column_types(data)

        if method == 'knn':
            data = self.impute_knn(data, **kwargs)
            data = self.impute_categorical(data)
            return data
        elif method == 'missforest':
             return self.impute_missforest(data, **kwargs)
        else:
            raise ValueError("Unknown imputation method. Use 'knn' or 'missforest'.")

    def encode(self, data, method='label', target_col=None, **kwargs):
        """
        Encode categorical features (excluding target unless method='target').
        Target column is encoded separately using _label_encode_target if needed.
        """
        if not self.cat_columns and not self.numeric_columns:
            print("Determining column types for encoding...")
            self._get_column_types(data)

        active_cat_cols = self._get_active_cat_cols(data, target_col, method)
        if not active_cat_cols and method != 'target':
             print(f"Encoding method '{method}': No active categorical columns found to encode (excluding target).")
             if method != 'target': return data

        print(f"Starting encoding method: {method} on columns: {active_cat_cols}")
        if method == 'onehot':
            encoded_data = self._onehot_encode(data, active_cat_cols)
        elif method == 'label':
            encoded_data = self._label_encode(data, active_cat_cols)
        elif method == 'ordinal':
             encoded_data = self._ordinal_encode(data, active_cat_cols)
        elif method == 'leaveoneout':
            if not target_col:
                raise ValueError("Target column must be specified for LeaveOneOut Encoding")
            sigma = kwargs.get('sigma', 0.05)
            encoded_data = self._leaveoneout_encode(data, active_cat_cols, target_col, sigma=sigma)
        elif method == 'lsa':
            n_components = kwargs.get('n_components', 10)
            encoded_data = self._lsa_encode(data, active_cat_cols, n_components=n_components)
        else:
            raise ValueError(f"Unknown encoding method: {method}. Use 'onehot', 'label', 'ordinal', 'leaveoneout', or 'lsa'.")

        print(f"Finished encoding with method: {method}")
        return encoded_data 