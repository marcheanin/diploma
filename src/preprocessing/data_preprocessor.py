import pandas as pd
import numpy as np
import sys
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import LeaveOneOutEncoder

# Optional imports for embeddings with improved error handling
GENSIM_AVAILABLE = False
try:
    import gensim
    from gensim.models import FastText
    GENSIM_AVAILABLE = True
except ImportError as e:
    print(f"Error importing gensim: {e}")
    print("Gensim not available. Install with: pip install gensim")

# Conditional import for MissForest
try:
    from missingpy import MissForest
    MISSFOREST_AVAILABLE = True
except ImportError:
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
        self.lsa_components = {}
        self.word2vec_models = {}
        self.word2vec_dims = 50
        self.imputed_medians = {}
        self.imputed_modes = {}

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
                self.imputed_modes[col] = mode_val
            else:
                data[col] = data[col].fillna("Unknown")
                self.imputed_modes[col] = "Unknown"
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
        self.encoders[target_col] = self.target_encoder_le
        # print(f"Target column '{target_col}' was label encoded.")
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
        
        # Filter out high cardinality columns before fitting the encoder
        low_cardinality_cols = []
        skipped_cols = []
        MAX_OH_CARDINALITY = 50 # Max unique values for OneHotEncoding
        for col in cols_to_encode:
            if data[col].nunique(dropna=False) <= MAX_OH_CARDINALITY:
                low_cardinality_cols.append(col)
            else:
                skipped_cols.append(col)
        
        if not low_cardinality_cols:
            if skipped_cols:
                print(f"Warning: OneHotEncoding skipped for all specified columns due to high cardinality: {skipped_cols}. Columns to encode: {cols_to_encode}")
            return data # Return original data if no columns are suitable
        
        if skipped_cols:
            print(f"Warning: OneHotEncoding skipped for high cardinality columns: {skipped_cols}. Applied to: {low_cardinality_cols}")

        if encoder_key not in self.encoders:
            # Fit encoder only on low cardinality columns
            self.encoders[encoder_key] = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False, # Still potentially memory intensive if many low_card_cols generate many features
                drop=None
            )
            # Ensure fitting with string type to be consistent
            self.encoders[encoder_key].fit(data[low_cardinality_cols].astype(str))

        encoder = self.encoders[encoder_key]
        # Transform only low cardinality columns
        encoded = encoder.transform(data[low_cardinality_cols].astype(str))
        new_cols = encoder.get_feature_names_out(low_cardinality_cols)
        encoded_df = pd.DataFrame(encoded, columns=new_cols, index=data.index)
        
        # Drop the original low_cardinality_cols that were encoded, keep skipped_cols and other columns
        data_remaining = data.drop(columns=low_cardinality_cols)
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
                    model = gensim.models.FastText(sentences=tokenized_texts, vector_size=n_components,
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

    def _impute_categorical_with_mode(self, data, column):
        mode_val = data[column].mode()
        if not mode_val.empty:
            mode = mode_val[0]
            data[column] = data[column].fillna(mode)
            self.imputed_modes[column] = mode
            # print(f"Imputed categorical column '{column}' with mode '{mode}'.")
        else:
            fallback_mode = "__MISSING__"
            data[column] = data[column].fillna(fallback_mode)
            self.imputed_modes[column] = fallback_mode
            # print(f"Imputed categorical column '{column}' with fallback '{fallback_mode}'.")
        return data

    def _impute_numerical_with_median(self, data, column):
        median = data[column].median()
        data[column] = data[column].fillna(median)
        self.imputed_medians[column] = median
        # print(f"Imputed numerical column '{column}' with median '{median}'.")
        return data

    def impute(self, data, method='knn', **kwargs):
        # print(f"Starting imputation with method: {method}")
        data_imputed = data.copy()
        numeric_cols = data_imputed.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data_imputed.select_dtypes(exclude=np.number).columns.tolist()

        for col in categorical_cols:
            if data_imputed[col].isnull().any():
                data_imputed = self._impute_categorical_with_mode(data_imputed, col)
        
        numeric_cols_with_na = [col for col in numeric_cols if data_imputed[col].isnull().any()]
        if not numeric_cols_with_na:
            # print("No numerical columns with NAs to impute.")
            pass
        elif method == 'median':
            # print(f"Applying median imputation to numeric columns: {numeric_cols_with_na}")
            for col in numeric_cols_with_na:
                data_imputed = self._impute_numerical_with_median(data_imputed, col)
        elif method == 'knn':
            # print(f"Applying KNNImputer to numeric columns: {numeric_cols_with_na}")
            n_neighbors = kwargs.get('n_neighbors', 5)
            imputer_knn = KNNImputer(n_neighbors=n_neighbors)
            data_imputed[numeric_cols_with_na] = imputer_knn.fit_transform(data_imputed[numeric_cols_with_na])
        elif method == 'missforest':
            if MISSFOREST_AVAILABLE:
                # print(f"Applying MissForest. This might take a while...")
                mf_input_data = data_imputed.copy()
                for col in mf_input_data.columns:
                    if mf_input_data[col].dtype == 'object':
                        mf_input_data[col] = pd.Categorical(mf_input_data[col])
                if not mf_input_data.isnull().any().any():
                    # print("No NAs remaining for MissForest after categorical mode imputation.")
                    pass
                else:
                    try:
                        imputer_mf = MissForest(random_state=42, **kwargs.get('missforest_kwargs', {}))
                        imputed_values = imputer_mf.fit_transform(mf_input_data)
                        data_imputed = pd.DataFrame(imputed_values, columns=mf_input_data.columns)
                        for col in data.columns: 
                            if col in data_imputed.columns and data[col].dtype != data_imputed[col].dtype:
                                try:
                                    data_imputed[col] = data_imputed[col].astype(data[col].dtype, errors='ignore')
                                    if pd.api.types.is_numeric_dtype(data[col]) and not pd.api.types.is_numeric_dtype(data_imputed[col]):
                                         data_imputed[col] = pd.to_numeric(data_imputed[col], errors='coerce') # Coerce will turn problematic to NaN
                                except Exception as e_dtype:
                                    print(f"Warning: Could not restore dtype for column {col} after MissForest: {e_dtype}")
                        # print("MissForest imputation completed.")
                    except Exception as e_mf:
                        print(f"Error during MissForest imputation: {e_mf}.")
                        numeric_still_na = [nc for nc in numeric_cols if data_imputed[nc].isnull().any()]
                        if numeric_still_na:
                            # print("MissForest failed, falling back to IterativeImputer for remaining numeric columns.")
                            iter_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42),random_state=42,**kwargs.get('iterativeimputer_kwargs', {}))
                            data_imputed[numeric_still_na] = iter_imputer.fit_transform(data_imputed[numeric_still_na])
            else: 
                # print("MissForest not available, using IterativeImputer for numeric columns with NAs.")
                if numeric_cols_with_na:
                    iter_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42),random_state=42,**kwargs.get('iterativeimputer_kwargs', {}))
                    data_imputed[numeric_cols_with_na] = iter_imputer.fit_transform(data_imputed[numeric_cols_with_na])
        elif method not in ['median', 'knn', 'missforest']:
            raise ValueError(f"Unknown imputation method: {method}")
        
        if data_imputed.isnull().any().any():
            print("Warning: NAs still present after imputation process.")
            print(data_imputed.isnull().sum()[data_imputed.isnull().sum() > 0])
        else:
            # print("Imputation completed. No NAs remaining.")
            pass
        return data_imputed

    def encode(self, data, method='label', target_col=None, **kwargs):
        """
        Encode categorical features (excluding target unless method='target').
        Target column is encoded separately using _label_encode_target if needed.
        """
        encoded_data = data.copy() # Initialize encoded_data here

        if not self.cat_columns and not self.numeric_columns:
            self._get_column_types(encoded_data) # Use encoded_data

        active_cat_cols = self._get_active_cat_cols(encoded_data, target_col, method) # Use encoded_data
        if not active_cat_cols and method not in ['target', 'leaveoneout']: # Adjusted condition for LOOE
             # print(f"No active categorical columns to encode with method '{method}'. Returning original data.")
             return encoded_data # Return the copy

        if method == 'onehot':
            encoded_data = self._onehot_encode(encoded_data, active_cat_cols) # Pass and reassign encoded_data
        elif method == 'label':
            encoded_data = self._label_encode(encoded_data, active_cat_cols) # Pass and reassign
        elif method == 'ordinal':
             encoded_data = self._ordinal_encode(encoded_data, active_cat_cols) # Pass and reassign
        elif method == 'lsa':
            n_components = kwargs.get('n_components', 10)
            encoded_data = self._lsa_encode(encoded_data, active_cat_cols, n_components=n_components) # Pass and reassign
        elif method == 'embedding': # This is one of the word2vec blocks
            if not GENSIM_AVAILABLE:
                print("Word2Vec encoding (via 'embedding') skipped: Gensim library not available.")
                return encoded_data # Return current encoded_data

            self.word2vec_dims = kwargs.get('embedding_dim', self.word2vec_dims)
            if self.word2vec_dims <= 0:
                print("Word2Vec encoding (via 'embedding') skipped: embedding_dim must be > 0.")
                return encoded_data
            
            if not active_cat_cols:
                # print("Word2Vec (via 'embedding') skipped: No categorical columns for encoding.")
                return encoded_data

            # print(f"Applying Word2Vec (via 'embedding') with embedding_dim={self.word2vec_dims} for columns: {active_cat_cols}")
            
            processed_cols_w2v = [] # Keep track of columns successfully processed by Word2Vec
            for col in active_cat_cols:
                # Ensure column data is a list of lists of strings for Word2Vec
                # Each category value is treated as a single 'word' in its own 'sentence'
                # Convert to string and fill NA to handle all cases before .apply
                sentences = encoded_data[col].astype(str).fillna('__NULL_W2V__').apply(lambda x: [x]).tolist()
                
                if not sentences: # Should not happen if active_cat_cols is not empty, but as a safeguard
                    print(f"Skipping Word2Vec for column '{col}' due to no data/sentences.")
                    continue

                # Train Word2Vec model for the current column
                # window=1 as each category is its own context. min_count=1 to include all categories.
                # sg=1 for Skip-gram model, often better for infrequent words (categories)
                try:
                    w2v_model = Word2Vec(sentences=sentences, vector_size=self.word2vec_dims, window=1, min_count=1, workers=4, sg=1, seed=42)
                    self.word2vec_models[col] = w2v_model
                    
                    # Create embedding features for the column
                    embedding_vectors = []
                    for val_list in sentences: # val_list is like ['category_value']
                        word = val_list[0]
                        if word in w2v_model.wv:
                            embedding_vectors.append(w2v_model.wv[word])
                        else:
                            # This case should be rare with min_count=1 if word was in training sentences
                            # and not '__NULL_W2V__' if all values were null.
                            # If __NULL_W2V__ was the only word, it gets an embedding. Else, zero vector for safety.
                            embedding_vectors.append(np.zeros(self.word2vec_dims))
                    
                    embedding_df = pd.DataFrame(embedding_vectors, index=encoded_data.index)
                    embedding_df.columns = [f'{col}_w2v_{i}' for i in range(self.word2vec_dims)]
                    
                    # Concatenate new embedding features and mark original column for dropping
                    encoded_data = pd.concat([encoded_data, embedding_df], axis=1)
                    processed_cols_w2v.append(col)
                except Exception as e_w2v:
                    print(f"Error training or applying Word2Vec for column '{col}': {e_w2v}. Column will be skipped.")
            
            # Drop original categorical columns that were successfully processed by Word2Vec
            if processed_cols_w2v:
                encoded_data = encoded_data.drop(columns=processed_cols_w2v)
                print(f"Dropped original Word2Vec processed columns: {processed_cols_w2v}")
        elif method == 'leaveoneout':
            if target_col is None or target_col not in data.columns:
                raise ValueError("LeaveOneOutEncoder requires a target column specified and present in data.")
            
            # Use encoded_data for LOOE, not original 'data'
            temp_target_series = encoded_data[target_col]
            if not pd.api.types.is_numeric_dtype(temp_target_series):
                print(f"Target column '{target_col}' for LOOE is categorical. Applying temporary LabelEncoding.")
                le_target = LabelEncoder()
                temp_target_series = le_target.fit_transform(temp_target_series)

            looe = LeaveOneOutEncoder(cols=active_cat_cols, sigma=kwargs.get('sigma', 0.05))
            # Ensure fit_transform is on the correct DataFrame columns
            encoded_looe_df = pd.DataFrame(looe.fit_transform(encoded_data[active_cat_cols], temp_target_series), columns=active_cat_cols, index=encoded_data.index)
            # Update the columns in encoded_data, don't just assign the result to encoded_data directly if it only contains the encoded cols
            for col in active_cat_cols:
                encoded_data[col] = encoded_looe_df[col]
            self.encoders['leaveoneout'] = looe
        elif method == 'word2vec': # This is the second word2vec block
            if not GENSIM_AVAILABLE:
                print("Word2Vec encoding skipped: Gensim library not available.")
                return encoded_data # Return current encoded_data

            self.word2vec_dims = kwargs.get('embedding_dim', self.word2vec_dims)
            if self.word2vec_dims <= 0:
                print("Word2Vec encoding skipped: embedding_dim must be > 0.")
                return encoded_data

            if not active_cat_cols:
                # print("Word2Vec skipped: No categorical columns identified for encoding.")
                return encoded_data

            # print(f"Applying Word2Vec with embedding_dim={self.word2vec_dims} for columns: {active_cat_cols}")
            
            processed_cols_w2v = []
            for col in active_cat_cols:
                sentences = encoded_data[col].astype(str).fillna('__NULL_W2V__').apply(lambda x: [x]).tolist()
                
                if not sentences:
                    print(f"Skipping Word2Vec for column '{col}' due to no data/sentences after processing.")
                    continue
                try:
                    w2v_model = Word2Vec(sentences=sentences, vector_size=self.word2vec_dims, window=1, min_count=1, workers=4, sg=1, seed=42)
                    self.word2vec_models[col] = w2v_model
                    embedding_vectors = []
                    for val_list in sentences:
                        word = val_list[0]
                        if word in w2v_model.wv:
                            embedding_vectors.append(w2v_model.wv[word])
                        else:
                            embedding_vectors.append(np.zeros(self.word2vec_dims))
                    
                    embedding_df = pd.DataFrame(embedding_vectors, index=encoded_data.index)
                    embedding_df.columns = [f'{col}_w2v_{i}' for i in range(self.word2vec_dims)]
                    encoded_data = pd.concat([encoded_data, embedding_df], axis=1)
                    processed_cols_w2v.append(col)
                except Exception as e_w2v:
                    print(f"Error training or applying Word2Vec for column '{col}': {e_w2v}. Column will be skipped.")
            
            if processed_cols_w2v:
                encoded_data = encoded_data.drop(columns=processed_cols_w2v)
                print(f"Dropped original Word2Vec processed columns: {processed_cols_w2v}")
        else:
            raise ValueError(f"Unknown encoding method: {method}. Use 'onehot', 'label', 'ordinal', 'lsa', 'embedding', or 'leaveoneout'.")

        return encoded_data 