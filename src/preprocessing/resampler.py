import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

class Resampler:
    """
    A class to handle different resampling techniques for imbalanced datasets.
    Supported methods: 'none', 'oversample' (ROS), 'undersample' (RUS), 'smote', 'adasyn'.
    """
    def __init__(self, method='none', random_state=42, **kwargs):
        """
        Initialize the Resampler.
        Args:
            method (str): The resampling method to use.
            random_state (int): Random state for reproducibility.
            **kwargs: Hyperparameters for the chosen resampling method.
                      For SMOTE/ADASYN: k_neighbors, sampling_strategy.
                      For ROS: sampling_strategy.
        """
        if method not in ['none', 'oversample', 'undersample', 'smote', 'adasyn']:
            raise ValueError(f"Unsupported resampling method: {method}")
        
        self.method = method
        self.random_state = random_state
        self.kwargs = kwargs # Store kwargs
        self.resampler = self._get_resampler()

    def _get_resampler(self):
        """Get the resampler object based on the chosen method and HPs."""
        if self.method == 'oversample':
            # ROS HPs: sampling_strategy
            ros_sampling_strategy = self.kwargs.get('sampling_strategy', 'auto')
            # print(f"ROS initialized with sampling_strategy={ros_sampling_strategy}")
            return RandomOverSampler(sampling_strategy=ros_sampling_strategy, random_state=self.random_state)
        
        elif self.method == 'undersample':
            # RUS HPs: sampling_strategy (can also be a float)
            rus_sampling_strategy = self.kwargs.get('sampling_strategy', 'auto')
            # print(f"RUS initialized with sampling_strategy={rus_sampling_strategy}")
            return RandomUnderSampler(sampling_strategy=rus_sampling_strategy, random_state=self.random_state)
        
        elif self.method == 'smote':
            # SMOTE HPs: k_neighbors, sampling_strategy
            smote_k_neighbors = self.kwargs.get('k_neighbors', 5)
            smote_sampling_strategy = self.kwargs.get('sampling_strategy', 'auto')
            # print(f"SMOTE initialized with k_neighbors={smote_k_neighbors}, sampling_strategy={smote_sampling_strategy}")
            return SMOTE(k_neighbors=smote_k_neighbors, sampling_strategy=smote_sampling_strategy, 
                         random_state=self.random_state, n_jobs=-1)
        
        elif self.method == 'adasyn':
            # ADASYN HPs: n_neighbors, sampling_strategy
            adasyn_n_neighbors = self.kwargs.get('n_neighbors', 5)
            adasyn_sampling_strategy = self.kwargs.get('sampling_strategy', 'auto')
            # print(f"ADASYN initialized with n_neighbors={adasyn_n_neighbors}, sampling_strategy={adasyn_sampling_strategy}")
            return ADASYN(n_neighbors=adasyn_n_neighbors, sampling_strategy=adasyn_sampling_strategy, 
                          random_state=self.random_state, n_jobs=-1)
        
        elif self.method == 'none':
            return None
        
        # This case should ideally be caught by __init__ validation
        raise ValueError(f"Resampling method '{self.method}' is not recognized in _get_resampler.")

    def fit_resample(self, X, y):
        """
        Apply the chosen resampling method to the data.
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Target vector.
        Returns:
            (pd.DataFrame, pd.Series): Resampled X and y.
        """
        if self.method == 'none' or self.resampler is None:
            # print("Resampling method is 'none'. No changes to data.")
            return X, y
        
        # print(f"Applying resampling method: {self.method} with HPs: {self.kwargs}")
        # Store original column names if X is a DataFrame
        original_columns = X.columns if isinstance(X, pd.DataFrame) else None
        original_index_name = X.index.name if isinstance(X, pd.DataFrame) else None

        if self.method == 'smote' or self.method == 'adasyn':
            # Ensure y is a pandas Series for nunique()
            y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
            if y_series.nunique() > 2:  # Multi-class
                # Check the sampling_strategy of the initialized SMOTE resampler
                current_strategy = self.resampler.sampling_strategy
                if isinstance(current_strategy, float):
                    adapted_strategy = 'auto'
                    print(f"Warning: SMOTE sampling_strategy '{current_strategy}' is a float but target is multi-class ({y_series.nunique()} classes).")
                    print(f"Adapting SMOTE sampling_strategy to '{adapted_strategy}' for this run.")
                    # Modify the strategy of the existing resampler instance
                    self.resampler.sampling_strategy = adapted_strategy
        
        try:
            X_resampled, y_resampled = self.resampler.fit_resample(X, y)
        except Exception as e:
            print(f"Error during {self.method} resampling with HPs {self.kwargs}: {e}")
            print("Returning original data due to resampling error.")
            return X, y
        
        # print(f"Resampling complete. Original shape: X-{X.shape}, y-{y.shape}. Resampled shape: X-{X_resampled.shape}, y-{y_resampled.shape}")

        # If X was a DataFrame, try to convert X_resampled back to DataFrame with original columns
        if original_columns is not None and isinstance(X_resampled, np.ndarray):
            X_resampled = pd.DataFrame(X_resampled, columns=original_columns)
            # Attempting to restore index is tricky as resampling changes row count and order.
            # A simple re-index might not be meaningful. For now, use default integer index.

        # Ensure y_resampled is a pd.Series if y was one
        if isinstance(y, pd.Series) and isinstance(y_resampled, np.ndarray):
            y_name = y.name if y.name else 'target'
            y_resampled = pd.Series(y_resampled, name=y_name)
            # Similarly, index for y_resampled will be the default from imblearn

        return X_resampled, y_resampled 