import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

class Resampler:
    """
    Applies different resampling techniques to handle class imbalance.
    Requires features (X) and target (y) to be numeric.
    Resampling is applied ONLY to training data.
    """
    def __init__(self, method='none', random_state=42):
        """
        Initialize the resampler.

        Args:
            method (str): Resampling method. Options: 'none', 'oversample',
                          'undersample', 'smote', 'adasyn'.
            random_state (int): Random state for reproducibility.
        """
        if method not in ['none', 'oversample', 'undersample', 'smote', 'adasyn']:
            raise ValueError(f"Unknown resampling method: {method}")
        self.method = method
        self.random_state = random_state
        self.sampler = None

    def fit_resample(self, X, y):
        """
        Apply the selected resampling technique.

        Args:
            X (pd.DataFrame or np.ndarray): Features (numeric).
            y (pd.Series or np.ndarray): Target variable (numeric).

        Returns:
            tuple: (X_resampled, y_resampled) - Resampled features and target.
                   Returns original X, y if method is 'none'.
        """
        if self.method == 'none':
            return X, y

        if not isinstance(y, pd.Series):
             y_series = pd.Series(y)
        else:
             y_series = y

        if self.method == 'oversample':
            self.sampler = RandomOverSampler(random_state=self.random_state)
        elif self.method == 'undersample':
            self.sampler = RandomUnderSampler(random_state=self.random_state)
        elif self.method == 'smote':
            min_class_size = y_series.value_counts().min()
            k_neighbors = min(5, max(1, min_class_size - 1))
            self.sampler = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        elif self.method == 'adasyn':
             min_class_size = y_series.value_counts().min()
             n_neighbors = min(5, max(1, min_class_size - 1))
             self.sampler = ADASYN(random_state=self.random_state, n_neighbors=n_neighbors)

        try:
            X_input = X.values if isinstance(X, pd.DataFrame) else X
            y_input = y.values if isinstance(y, pd.Series) else y

            X_resampled, y_resampled = self.sampler.fit_resample(X_input, y_input)

            y_resampled_series = pd.Series(y_resampled)

            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                 y_resampled = pd.Series(y_resampled, name=y.name)

            return X_resampled, y_resampled

        except Exception as e:
            print(f"Error during resampling with {self.method}: {e}")
            print("Returning original data due to error.")
            return X, y 