from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import numpy as np

class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """
    def __init__(self, random_state=42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: int, random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        
    def train(self, data, target_column, test_size=0.2):
        """
        Train the model on the given data.
        
        Args:
            data: pandas DataFrame, the processed data
            target_column: str, name of the target column
            test_size: float, proportion of data to use for testing
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, train_preds),
                'f1': f1_score(y_train, train_preds, average='weighted'),
                'detailed_report': classification_report(y_train, train_preds)
            },
            'validation': {
                'accuracy': accuracy_score(y_val, val_preds),
                'f1': f1_score(y_val, val_preds, average='weighted'),
                'detailed_report': classification_report(y_val, val_preds)
            }
        }
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return metrics, feature_importance
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: pandas DataFrame, features to predict on
            
        Returns:
            numpy array: Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        return self.model.predict(X) 