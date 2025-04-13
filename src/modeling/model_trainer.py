from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import numpy as np
import os

class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """
    def __init__(self, random_state=42, validation_size=0.2):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: int, random state for reproducibility
            validation_size: float, size of validation split when test has no target
        """
        self.random_state = random_state
        self.validation_size = validation_size
        self.model = None
        
    def train(self, train_data, test_data, target_column):
        """
        Train the model using pre-split train and test data.
        If test data has no target column, uses part of train data for evaluation.
        
        Args:
            train_data: pandas DataFrame, training data
            test_data: pandas DataFrame, test data
            target_column: str, name of target column
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Check if test data has target column
        has_test_target = target_column in test_data.columns
        
        if has_test_target:
            # Use test data for evaluation
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # Train on full training data
            eval_name = 'test'
        else:
            print(f"\nTest data has no target column. Using {self.validation_size:.0%} of train data for evaluation.")
            # Split training data for evaluation
            train_eval, valid_data = train_test_split(
                train_data, 
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=train_data[target_column]
            )
            
            # Prepare training and evaluation sets
            X_train = train_eval.drop(columns=[target_column])
            y_train = train_eval[target_column]
            X_test = valid_data.drop(columns=[target_column])
            y_test = valid_data[target_column]
            
            eval_name = 'validation'
        
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
        eval_preds = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, train_preds),
                'f1': f1_score(y_train, train_preds, average='weighted'),
                'detailed_report': classification_report(y_train, train_preds)
            },
            eval_name: {
                'accuracy': accuracy_score(y_test, eval_preds),
                'f1': f1_score(y_test, eval_preds, average='weighted'),
                'detailed_report': classification_report(y_test, eval_preds)
            }
        }
        
        # If we have separate test data without target, make predictions for it
        if not has_test_target:
            print("\nMaking predictions for test data without target...")
            test_preds = self.model.predict(test_data)
            test_predictions = pd.DataFrame({
                'predictions': test_preds
            })
            # Note: path for saving predictions should be passed from main.py
            print("Note: Test predictions are available in the model object.")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
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