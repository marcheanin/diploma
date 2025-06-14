import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class OutlierRemover:
    """
    A class to remove outliers from a pandas DataFrame.
    Supported methods: 'isolation_forest', 'iqr', 'none'.
    """
    def __init__(self, method='isolation_forest', **kwargs):
        """
        Initialize the OutlierRemover.

        Args:
            method (str): Method to use for outlier detection.
                          'isolation_forest', 'iqr', or 'none'.
            **kwargs: Hyperparameters for the chosen method.
                      For 'isolation_forest': n_estimators, contamination.
                      For 'iqr': multiplier.
        """
        if method not in ['isolation_forest', 'iqr', 'none']:
            raise ValueError(f"Unsupported outlier removal method: {method}")
        
        self.method = method
        self.model = None
        self.kwargs = kwargs # Store all kwargs
        self.outlier_indices_ = None

        if self.method == 'isolation_forest':

            if_n_estimators = self.kwargs.get('n_estimators', 100)
            if_contamination = self.kwargs.get('contamination', 'auto')
            self.model = IsolationForest(n_estimators=if_n_estimators, 
                                         contamination=if_contamination, 
                                         random_state=42)
        elif self.method == 'iqr':
            self.iqr_multiplier = self.kwargs.get('multiplier', 1.5)
            pass # No model to pre-initialize for IQR, logic is in remove_outliers
        elif self.method == 'none':
            # print("Outlier removal method is 'none'. No model initialized.")
            pass
        else:
            raise ValueError(f"Unknown outlier removal method: {self.method}")

    def remove_outliers(self, data):
        """
        Remove outliers from the DataFrame based on the chosen method.
        Only numeric columns are considered for outlier detection.
        
        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with outliers removed if a removal method was chosen,
                          otherwise the original DataFrame.
        """
        if self.method == 'none' or data.empty:
            # print("Skipping outlier removal (method is 'none' or data is empty).")
            return data

        data_cleaned = data.copy()
        numeric_cols = data_cleaned.select_dtypes(include=np.number).columns

        if numeric_cols.empty:
            print("No numeric columns found to perform outlier detection. Returning original data.")
            return data

        print(f"Performing outlier removal using method: {self.method} on {len(numeric_cols)} numeric columns.")

        if self.method == 'isolation_forest':
            if self.model is None: # Should have been initialized in __init__
                print("Error: Isolation Forest model not initialized.")
                return data # or raise error
            
            # Ensure all data is finite for Isolation Forest
            data_numeric_finite = data_cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if data_numeric_finite.empty:
                print("Warning: Data became empty after handling non-finite values for Isolation Forest. Returning original data.")
                return data
            
            original_indices = data_numeric_finite.index
            outliers = self.model.fit_predict(data_numeric_finite)
            mask_inliers = outliers != -1
            
            cleaned_numeric_data = data_numeric_finite[mask_inliers]

        elif self.method == 'iqr':
            multiplier = self.kwargs.get('multiplier', 1.5) # Get from stored kwargs or default
            # print(f"Applying IQR with multiplier: {multiplier}")
            Q1 = data_cleaned[numeric_cols].quantile(0.25)
            Q3 = data_cleaned[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask_inliers = ~((data_cleaned[numeric_cols] < lower_bound) | (data_cleaned[numeric_cols] > upper_bound)).any(axis=1)
            cleaned_numeric_data = data_cleaned[mask_inliers]
            # print(f"IQR: {np.sum(~mask_inliers)} outliers removed out of {len(data_cleaned[numeric_cols])} samples.")
        else:
            # Should not happen due to __init__ check, but as a safeguard
            print(f"Unknown or unhandled outlier removal method: {self.method}. Returning original data.")
            return data
        

        if self.method == 'iqr':
            final_data = cleaned_numeric_data # For IQR, cleaned_numeric_data is the full filtered dataframe
        elif self.method == 'isolation_forest':
            if not data_cleaned.drop(columns=numeric_cols, errors='ignore').empty:

                if not data_numeric_finite.index.equals(data_cleaned[numeric_cols].index):

                     final_data = pd.concat([cleaned_numeric_data, data_cleaned.drop(columns=numeric_cols, errors='ignore').loc[cleaned_numeric_data.index]], axis=1)
                else:
     
                     final_data = pd.concat([cleaned_numeric_data, data_cleaned.drop(columns=numeric_cols, errors='ignore').loc[cleaned_numeric_data.index]], axis=1)
            else:

                final_data = cleaned_numeric_data
        else: 
            final_data = data_cleaned 

        original_shape = data.shape
        cleaned_shape = final_data.shape
        print(f"Outlier removal: {original_shape[0]} rows before, {cleaned_shape[0]} rows after.")
        
        return final_data 