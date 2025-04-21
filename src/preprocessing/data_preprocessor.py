import pandas as pd
import numpy as np
import sys
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Optional imports for embeddings with improved error handling
GENSIM_AVAILABLE = False
try:
    import gensim
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
except ImportError as e:
    print(f"Error importing gensim: {e}")
    print("Gensim not available. Install with: pip install gensim")

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
        self.medians = {}

    def _get_column_types(self, data):
        """Identify numeric and categorical columns in the dataset."""
        self.numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        self.cat_columns = data.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    def _impute_median(self, data):
        """Impute missing values in numeric columns using the median."""
        data = data.copy()
        numeric_cols_present = [col for col in self.numeric_columns if col in data.columns]

        if not self.medians:
            for col in numeric_cols_present:
                median_val = data[col].median()
                if pd.isna(median_val):
                    median_val = 0
                self.medians[col] = median_val
                data[col] = data[col].fillna(median_val)
        else:
            for col in numeric_cols_present:
                stored_median = self.medians.get(col)
                if stored_median is not None:
                    data[col] = data[col].fillna(stored_median)
                else:
                    fallback_median = data[col].median()
                    fallback_median = fallback_median if pd.notna(fallback_median) else 0
                    data[col] = data[col].fillna(fallback_median)
        return data

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
            return data

        if self.target_encoder_le is None:
            self.target_encoder_le = LabelEncoder()
            self.target_encoder_le.fit(data[target_col].astype(str))

        data[target_col] = self.target_encoder_le.transform(data[target_col].astype(str))
        return data

    def _get_active_cat_cols(self, data, target_col, method):
        """Get list of categorical columns for current encoding method, excluding target unless method is 'target'."""
        if not self.cat_columns:
             return []

        cols = [col for col in self.cat_columns if col != target_col and col in data.columns]
        return cols

    def _onehot_encode(self, data, cols_to_encode):
        """Encode specified columns using one-hot encoding."""
        if not cols_to_encode:
            return data

        data = data.copy()
        encoder_key = 'onehot'
        if encoder_key not in self.encoders:
            self.encoders[encoder_key] = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop=None
            )
            self.encoders[encoder_key].fit(data[cols_to_encode].astype(str))

        encoder = self.encoders[encoder_key]
        encoded = encoder.transform(data[cols_to_encode].astype(str))
        new_cols = encoder.get_feature_names_out(cols_to_encode)
        encoded_df = pd.DataFrame(encoded, columns=new_cols, index=data.index)
        data_remaining = data.drop(columns=cols_to_encode)
        return pd.concat([data_remaining, encoded_df], axis=1)

    def _label_encode(self, data, cols_to_encode):
        """Encode specified columns using label encoding."""
        if not cols_to_encode:
            return data

        data = data.copy()
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
        if encoder_key not in self.encoders:
            self.encoders[encoder_key] = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            self.encoders[encoder_key].fit(data[cols_to_encode].astype(str))

        data[cols_to_encode] = self.encoders[encoder_key].transform(data[cols_to_encode].astype(str))
        data[cols_to_encode] = data[cols_to_encode].astype('int32')
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
            return data
        data = data.copy()
        if self.lsa_vectorizer is None:
            texts = self._categorical_to_texts(data, cols_to_encode)
            self.lsa_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5)
            term_doc_matrix = self.lsa_vectorizer.fit_transform(texts)
            self.lsa_svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsa_embeddings = self.lsa_svd.fit_transform(term_doc_matrix)
            self.lsa_feature_names = [f'LSA_{i+1}' for i in range(n_components)]
        else:
            texts = self._categorical_to_texts(data, cols_to_encode)
            term_doc_matrix = self.lsa_vectorizer.transform(texts)
            lsa_embeddings = self.lsa_svd.transform(term_doc_matrix)

        lsa_df = pd.DataFrame(lsa_embeddings, columns=self.lsa_feature_names, index=data.index)

        cols_to_keep = [col for col in data.columns if col not in cols_to_encode]
        result_df = pd.concat([data[cols_to_keep], lsa_df], axis=1)
        return result_df

    def _embedding_encode(self, data, cols_to_encode, n_components=100, embedding_method='word2vec'):
        """
        Encode specified columns using different embedding methods
        
        Parameters:
        -----------
        data : pandas DataFrame
            The input data to encode
        cols_to_encode : list
            List of categorical columns to encode
        n_components : int
            Dimension of embeddings or final dimension after reduction
        embedding_method : str
            Method to use: 'word2vec' or 'fasttext'
        
        Returns:
        --------
        DataFrame with embedded representations replacing categorical columns
        """
        if not cols_to_encode:
            return data
            
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim library is required for Word2Vec and FastText embeddings")
            
        data = data.copy()
        texts = self._categorical_to_texts(data, cols_to_encode)
        
        tokenized_texts = [text.split() for text in texts]
        
        embed_key = f'{embedding_method}_embedder'
        feature_key = f'{embedding_method}_feature_names'
        
        if embedding_method in ['word2vec', 'fasttext']:
            if embed_key not in self.encoders:
                if embedding_method == 'word2vec':
                    model = Word2Vec(sentences=tokenized_texts, vector_size=n_components, 
                                     window=5, min_count=1, workers=4)
                else:  # fasttext
                    model = FastText(sentences=tokenized_texts, vector_size=n_components,
                                    window=5, min_count=1, workers=4)
                
                self.encoders[embed_key] = model
            else:
                model = self.encoders[embed_key]
                
            doc_vectors = []
            for tokens in tokenized_texts:
                token_vecs = []
                for token in tokens:
                    if token in model.wv:
                        token_vecs.append(model.wv[token])
                if token_vecs:
                    doc_vectors.append(np.mean(token_vecs, axis=0))
                else:
                    doc_vectors.append(np.zeros(n_components))
                    
            embeddings = np.array(doc_vectors)
            embed_feature_names = [f'{embedding_method}_{i+1}' for i in range(n_components)]
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}. Use 'word2vec' or 'fasttext'.")
            
        setattr(self, feature_key, embed_feature_names)
        
        embed_df = pd.DataFrame(embeddings, columns=embed_feature_names, index=data.index)
        
        cols_to_keep = [col for col in data.columns if col not in cols_to_encode]
        result_df = pd.concat([data[cols_to_keep], embed_df], axis=1)
        
        return result_df

    def impute(self, data, method="knn", **kwargs):
        """
        Impute missing values using specified method.
        Supported methods: 'knn', 'missforest', 'median'
        """
        if not self.numeric_columns and not self.cat_columns:
             self._get_column_types(data)

        if method == 'knn':
            data = self.impute_knn(data, **kwargs)
            data = self.impute_categorical(data)
            result = data
        elif method == 'missforest':
             result = self.impute_missforest(data, **kwargs)
        elif method == 'median':
            data = self._impute_median(data)
            data = self.impute_categorical(data)
            result = data
        else:
            raise ValueError(f"Unknown imputation method: {method}. Use 'knn', 'missforest', or 'median'.")

        return result

    def encode(self, data, method='label', target_col=None, **kwargs):
        """
        Encode categorical features (excluding target unless method='target').
        Target column is encoded separately using _label_encode_target if needed.
        """
        if not self.cat_columns and not self.numeric_columns:
            self._get_column_types(data)

        active_cat_cols = self._get_active_cat_cols(data, target_col, method)
        if not active_cat_cols and method != 'target':
             return data

        if method == 'onehot':
            encoded_data = self._onehot_encode(data, active_cat_cols)
        elif method == 'label':
            encoded_data = self._label_encode(data, active_cat_cols)
        elif method == 'ordinal':
             encoded_data = self._ordinal_encode(data, active_cat_cols)
        elif method == 'lsa':
            n_components = kwargs.get('n_components', 10)
            encoded_data = self._lsa_encode(data, active_cat_cols, n_components=n_components)
        elif method == 'embedding':
            try:
                n_components = kwargs.get('n_components', 100)
                embedding_method = kwargs.get('embedding_method', 'word2vec')
                encoded_data = self._embedding_encode(data, active_cat_cols, 
                                                    n_components=n_components,
                                                    embedding_method=embedding_method)
            except ImportError as e:
                n_components = kwargs.get('n_components', 10)
                encoded_data = self._lsa_encode(data, active_cat_cols, n_components=n_components)
        else:
            raise ValueError(f"Unknown encoding method: {method}. Use 'onehot', 'label', 'ordinal', 'lsa', or 'embedding'.")

        return encoded_data 