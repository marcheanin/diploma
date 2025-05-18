from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """
    def __init__(self, model_type='random_forest', random_state=42, validation_size=0.2):
        """
        Initialize ModelTrainer.
        
        Args:
            model_type: str, type of model to use ('random_forest', 'logistic_regression', 'gradient_boosting', 'neural_network')
            random_state: int, random state for reproducibility
            validation_size: float, size of validation split when test has no target
        """
        self.model_type = model_type
        self.random_state = random_state
        self.validation_size = validation_size
        self.model = None
        # Set random seed for TensorFlow/Keras for reproducibility if using it
        if 'tensorflow' in globals() and self.model_type == 'neural_network':
            globals()['tensorflow'].random.set_seed(self.random_state)
        
    def plot_learning_curves(self, X, y, output_path):
        # This method is for scikit-learn models
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
        plt.title('Learning Curves (Scikit-learn)')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        curves_path = os.path.join(output_path, 'sklearn_learning_curves.png')
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
        
    def _plot_keras_learning_curves(self, history, output_path):
        plt.figure(figsize=(12, 5))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
        else:
            plt.text(0.5, 0.5, 'Accuracy data not available', ha='center')


        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        if 'loss' in history.history and 'val_loss' in history.history:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
        else:
            plt.text(0.5, 0.5, 'Loss data not available', ha='center')
            
        plt.tight_layout()
        keras_curves_path = os.path.join(output_path, 'keras_learning_curves.png')
        plt.savefig(keras_curves_path)
        plt.close()
        return {'path': keras_curves_path, 'history': history.history}

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
        # Ensure test_data is not None for initial checks, even if it's empty or just for prediction structure
        if test_data is None:
             # Create a minimal DataFrame with expected columns if test_data is truly absent
             # This helps avoid errors in X_train.columns if train_data is also manipulated
            if target_column in train_data.columns:
                dummy_cols = train_data.drop(columns=[target_column]).columns
            else: # Should not happen if target_column is required for training
                dummy_cols = train_data.columns 
            test_data = pd.DataFrame(columns=dummy_cols)


        has_test_target = target_column in test_data.columns and not test_data[target_column].isnull().all()
        
        # Prepare data (common for all models first)
        _train_data = train_data.copy()
        _test_data = test_data.copy()

        # Attempt to drop 'ID' or 'id' column if it exists and is not the target
        potential_id_columns = [col for col in ['ID', 'id'] if col in _train_data.columns and col != target_column]
        if potential_id_columns:
            print(f"Dropping ID column(s): {potential_id_columns} before training.")
            _train_data = _train_data.drop(columns=potential_id_columns)
            if not _test_data.empty: # Ensure test_data is not empty before trying to drop
                 _test_data = _test_data.drop(columns=potential_id_columns, errors='ignore')

        if has_test_target:
            X_train = _train_data.drop(columns=[target_column])
            y_train = _train_data[target_column]
            X_test = _test_data.drop(columns=[target_column])
            y_test = _test_data[target_column]
            eval_name = 'test'
        else:
            if target_column not in _train_data.columns or _train_data[target_column].isnull().all():
                raise ValueError(f"Target column '{target_column}' for stratification is missing or all NaN in training data.")

            train_eval, valid_data = train_test_split(
                _train_data, 
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=_train_data[target_column]
            )
            X_train = train_eval.drop(columns=[target_column])
            y_train = train_eval[target_column]
            X_test = valid_data.drop(columns=[target_column])
            y_test = valid_data[target_column]
            eval_name = 'validation'

        # Common metrics dictionary structure
        metrics = {}
        
        if self.model_type == 'neural_network':
            # TensorFlow specific imports are at the top
            # Ensure y_train and y_test are 1D arrays
            y_train_nn = y_train.values.ravel()
            y_test_nn = y_test.values.ravel()

            num_classes = len(np.unique(y_train_nn))
            input_dim = X_train.shape[1]

            if num_classes > 2:
                y_train_keras = to_categorical(y_train_nn, num_classes=num_classes)
                y_test_keras = to_categorical(y_test_nn, num_classes=num_classes)
                loss_function = 'categorical_crossentropy'
                output_activation = 'softmax'
                output_units = num_classes
            else: # Binary classification
                y_train_keras = y_train_nn
                y_test_keras = y_test_nn
                loss_function = 'binary_crossentropy'
                output_activation = 'sigmoid'
                output_units = 1
            
            # Define Keras model
            self.model = Sequential([
                Input(shape=(input_dim,)), # Use Input layer
                Dense(128, activation='relu'), # Removed input_dim from here
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(output_units, activation=output_activation)
            ])
            
            self.model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
            
            history = self.model.fit(
                X_train, y_train_keras,
                epochs=100, # Increased epochs
                batch_size=32,
                validation_data=(X_test, y_test_keras),
                callbacks=[early_stopping],
                verbose=0 
            )
            
            train_pred_proba = self.model.predict(X_train, verbose=0)
            eval_pred_proba = self.model.predict(X_test, verbose=0)

            if num_classes > 2:
                train_preds = np.argmax(train_pred_proba, axis=1)
                eval_preds = np.argmax(eval_pred_proba, axis=1)
            else:
                train_preds = (train_pred_proba > 0.5).astype(int).ravel()
                eval_preds = (eval_pred_proba > 0.5).astype(int).ravel()

            metrics = {
                'train': {
                    'accuracy': accuracy_score(y_train_nn, train_preds),
                    'f1': f1_score(y_train_nn, train_preds, average='weighted', zero_division=0),
                    'detailed_report': classification_report(y_train_nn, train_preds, zero_division=0)
                },
                eval_name: {
                    'accuracy': accuracy_score(y_test_nn, eval_preds),
                    'f1': f1_score(y_test_nn, eval_preds, average='weighted', zero_division=0),
                    'detailed_report': classification_report(y_test_nn, eval_preds, zero_division=0)
                },
                'keras_history_path': None # Will be updated if plotting happens
            }
            feature_importance_values = np.zeros(X_train.shape[1]) # Placeholder
            if output_path and plot_learning_curves:
                try:
                    keras_curves_info = self._plot_keras_learning_curves(history, output_path)
                    metrics['keras_learning_curves'] = keras_curves_info
                except Exception as e:
                    print(f"Could not plot Keras learning curves: {e}")
                    pass
        
        else: # Scikit-learn models
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100, max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, random_state=self.random_state, n_jobs=-1
                )
            elif self.model_type == 'logistic_regression':
                self.model = LogisticRegression(
                    random_state=self.random_state, solver='liblinear', C=1.0 # Added C for regularization
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=3,
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.model.fit(X_train, y_train)
            train_preds = self.model.predict(X_train)
            eval_preds = self.model.predict(X_test)
            
            metrics = {
                'train': {
                    'accuracy': accuracy_score(y_train, train_preds),
                    'f1': f1_score(y_train, train_preds, average='weighted', zero_division=0),
                    'detailed_report': classification_report(y_train, train_preds, zero_division=0)
                },
                eval_name: {
                    'accuracy': accuracy_score(y_test, eval_preds),
                    'f1': f1_score(y_test, eval_preds, average='weighted', zero_division=0),
                    'detailed_report': classification_report(y_test, eval_preds, zero_division=0)
                }
            }
            if hasattr(self.model, 'feature_importances_'):
                feature_importance_values = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                feature_importance_values = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            else:
                feature_importance_values = np.zeros(X_train.shape[1])

            if output_path and plot_learning_curves:
                try:
                    sklearn_lc_data = self.plot_learning_curves(X_train, y_train, output_path)
                    metrics['sklearn_learning_curves'] = sklearn_lc_data
                except Exception as e:
                    print(f"Could not plot scikit-learn learning curves for {self.model_type}: {e}")
                    pass
        
        # Common part for predictions on test_data if it had no target
        if not has_test_target and test_data.shape[0] > 0 : # test_data might be the original one passed by user without target
            # Ensure test_data has the same columns as X_train (and in the same order)
            # This test_data is the one passed to the function, potentially without a target
            # Drop ID from original_test_data_for_prediction if it exists and was dropped from X_train
            if potential_id_columns:
                 original_test_data_for_prediction = test_data.drop(columns=potential_id_columns, errors='ignore')
            
            # Ensure columns are exactly the same as X_train for prediction
            original_test_data_for_prediction = original_test_data_for_prediction.reindex(columns=X_train.columns, fill_value=0)

            try:
                if self.model_type == 'neural_network':
                    test_pred_proba = self.model.predict(original_test_data_for_prediction, verbose=0)
                    if output_units > 1 : # multiclass
                        test_final_preds = np.argmax(test_pred_proba, axis=1)
                    else: # binary
                        test_final_preds = (test_pred_proba > 0.5).astype(int).ravel()
                else: # sklearn models
                    test_final_preds = self.model.predict(original_test_data_for_prediction)
                
                test_predictions_df = pd.DataFrame({'predictions': test_final_preds}, index=original_test_data_for_prediction.index)
                metrics['test_predictions'] = test_predictions_df
            except Exception as e:
                 print(f"Could not generate predictions on provided test data (no target): {e}")
                 pass

        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importance_values
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

        if self.model_type == 'neural_network':
            pred_proba = self.model.predict(X, verbose=0)
            # Check the shape of the output layer to determine if it's multi-class or binary
            # self.model.layers[-1].output_shape[-1] gives number of units in last layer
            if self.model.layers[-1].output_shape[-1] > 1: # Multiclass
                return np.argmax(pred_proba, axis=1)
            else: # Binary
                return (pred_proba > 0.5).astype(int).ravel()
        else: # Scikit-learn models
            return self.model.predict(X) 