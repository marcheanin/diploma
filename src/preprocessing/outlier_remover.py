from sklearn.ensemble import IsolationForest

class OutlierRemover:
    """
    Class for removing outliers using Isolation Forest.
    If no features are specified, all numeric features are used.
    """
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )

    def remove_outliers(self, data, features=None):
        """
        Remove outliers from the dataset using Isolation Forest.
        
        Args:
            data: pandas DataFrame
            features: list of column names to use for outlier detection
                     (if None, all numeric features are used)
        
        Returns:
            pandas DataFrame with outliers removed
        """
        if features is None:
            features = data.select_dtypes(include=['number']).columns
        
        feature_names = list(features)
        X = data[features].values
        
        self.model.fit(X)
        preds = self.model.predict(X)
        
        data_clean = data[preds == 1].copy()
        removed = len(data) - len(data_clean)
        print(f"Removed outliers: {removed} out of {len(data)} rows")
        
        return data_clean 