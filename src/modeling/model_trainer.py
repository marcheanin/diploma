from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

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
        
    def plot_learning_curves(self, X, y, output_path):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red', marker='o')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='red')
        
        plt.xlabel('Training Examples')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        curves_path = os.path.join(output_path, 'learning_curves.png')
        plt.savefig(curves_path)
        plt.close()
        
        return {
            'train_sizes': train_sizes,
            'train_scores': {
                'mean': train_mean,
                'std': train_std
            },
            'test_scores': {
                'mean': test_mean,
                'std': test_std
            }
        }
        
    def train(self, train_data, test_data, target_column, output_path=None, plot_learning_curves=True):
        """
        Train the model using pre-split train and test data.
        If test data has no target column, uses part of train data for evaluation.
        
        Args:
            train_data: pandas DataFrame, training data
            test_data: pandas DataFrame, test data
            target_column: str, name of target column
            output_path: str, path to save plots and results
            plot_learning_curves: bool, whether to generate and save learning curves
            
        Returns:
            tuple: (metrics, feature_importance) - Dictionary with metrics and DataFrame with feature importance
        """
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
            try:
                 # Ensure test_data has the same columns as X_train
                 test_preds = self.model.predict(test_data[X_train.columns])
                 test_predictions = pd.DataFrame({
                     'predictions': test_preds
                 }, index=test_data.index) # Preserve original index if possible
                 # Store predictions, maybe attach to metrics?
                 metrics['test_predictions'] = test_predictions
                 print(f"Test predictions generated (shape: {test_predictions.shape}).")
            except Exception as e:
                 print(f"Could not generate predictions for test data: {e}")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # --- Plot learning curves only if requested and path provided --- #
        if output_path and plot_learning_curves:
            print("\nGenerating learning curves...")
            try:
                learning_curves_data = self.plot_learning_curves(X_train, y_train, output_path)
                metrics['learning_curves'] = learning_curves_data
                print(f"Learning curves have been saved to: {os.path.join(output_path, 'learning_curves.png')}")
            except Exception as e:
                print(f"Could not generate learning curves: {e}")
        elif not plot_learning_curves:
             print("\nSkipping learning curve generation as requested.")
        # --- End plotting --- #
        
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