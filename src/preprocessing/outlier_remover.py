import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class OutlierRemover:
    """
    A class to remove outliers from a pandas DataFrame.
    Supported methods: 'isolation_forest', 'iqr', 'none'.
    """
    def __init__(self, method='isolation_forest', contamination=0.05, iqr_multiplier=1.5, random_state=42):
        """
        Initialize the OutlierRemover.

        Args:
            method (str): Method to use for outlier detection.
                          'isolation_forest', 'iqr', or 'none'.
            contamination (float): The amount of contamination of the data set, i.e.,
                                   the proportion of outliers in the data set. 
                                   Used by Isolation Forest.
            iqr_multiplier (float): Multiplier for the IQR range. Used by IQR method.
            random_state (int): Random state for reproducibility for Isolation Forest.
        """
        if method not in ['isolation_forest', 'iqr', 'none']:
            raise ValueError(f"Unsupported outlier removal method: {method}")
        
        self.method = method
        self.contamination = contamination
        self.iqr_multiplier = iqr_multiplier
        self.random_state = random_state
        self.model = None
        self.outlier_indices_ = None

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
        if self.method == 'none':
            print("Outlier removal skipped (method='none').")
            return data

        data_cleaned = data.copy()
        numeric_cols = data_cleaned.select_dtypes(include=np.number).columns

        if numeric_cols.empty:
            print("No numeric columns found to perform outlier detection. Returning original data.")
            return data

        print(f"Performing outlier removal using method: {self.method} on {len(numeric_cols)} numeric columns.")

        if self.method == 'isolation_forest':
            self.model = IsolationForest(contamination=self.contamination, 
                                         random_state=self.random_state,
                                         n_jobs=-1)
            predictions = self.model.fit_predict(data_cleaned[numeric_cols])
            self.outlier_indices_ = data_cleaned.index[predictions == -1]
            data_cleaned = data_cleaned[predictions == 1]

        elif self.method == 'iqr':
            initial_rows = len(data_cleaned)
            outliers_found_total = 0
            for column in numeric_cols:
                Q1 = data_cleaned[column].quantile(0.25)
                Q3 = data_cleaned[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                
                column_outliers = data_cleaned[(data_cleaned[column] < lower_bound) | (data_cleaned[column] > upper_bound)]
                outliers_found_for_col = len(column_outliers)
                if outliers_found_for_col > 0:
                    pass
                
                data_cleaned = data_cleaned[(data_cleaned[column] >= lower_bound) & (data_cleaned[column] <= upper_bound)]
                outliers_found_total += (initial_rows - len(data_cleaned)) - outliers_found_total
            
            num_removed_iqr = initial_rows - len(data_cleaned)
            print(f"Removed {num_removed_iqr} rows containing outliers using IQR method across numeric columns.")

        original_shape = data.shape
        cleaned_shape = data_cleaned.shape
        print(f"Outlier removal: {original_shape[0]} rows before, {cleaned_shape[0]} rows after.")
        
        return data_cleaned 