import tensorflow as tf # Ensure TensorFlow is imported early
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, average_precision_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """
    def __init__(self, model_type='random_forest', model_hyperparameters=None, random_state=42, validation_size=0.2):
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
        self.history = None # For Keras model training history
        self.model_hyperparameters = model_hyperparameters if model_hyperparameters is not None else {}
        # print(f"ModelTrainer initialized for {self.model_type} with HPs: {self.model_hyperparameters}")
        
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
            'train_sizes': train_sizes.tolist(), # Convert to list for JSON serialization
            'train_scores': {
                'mean': train_mean.tolist(),
                'std': train_std.tolist()
            },
            'test_scores': {
                'mean': test_mean.tolist(),
                'std': test_std.tolist()
            }
        }
        
    def _plot_keras_learning_curves(self, output_path):
        if self.history and output_path:
            os.makedirs(output_path, exist_ok=True) # Ensure output_path exists
            plt.figure(figsize=(12, 4))
            
            # Plot Loss
            plt.subplot(1, 2, 1)
            if 'loss' in self.history.history:
                 plt.plot(self.history.history['loss'], label='Train Loss')
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Keras Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            if 'loss' in self.history.history or 'val_loss' in self.history.history: # Only add legend if there's data
                plt.legend()

            # Plot Accuracy (or other primary metric like AUC if available and preferred)
            plt.subplot(1, 2, 2)
            primary_metric_key, val_primary_metric_key = None, None

            # Prioritize auprc or auc for Keras plots if available
            if 'auprc' in self.history.history: # Check for specific AUPRC metric from Keras
                primary_metric_key = 'auprc'
                val_primary_metric_key = 'val_auprc' if 'val_auprc' in self.history.history else None
            elif any(k.startswith('auc') and 'val' not in k for k in self.history.history.keys()): # Generic AUC (could be ROC AUC)
                 try:
                     primary_metric_key = [k for k in self.history.history.keys() if k.startswith('auc') and 'val' not in k][0]
                     val_primary_metric_key = [k for k in self.history.history.keys() if k.startswith('val_auc') and primary_metric_key in k][0] if any(k.startswith('val_auc') and primary_metric_key in k for k in self.history.history.keys()) else None
                 except IndexError:
                     primary_metric_key = None # Fallback if no non-val AUC found
            elif 'accuracy' in self.history.history:
                primary_metric_key = 'accuracy'
                val_primary_metric_key = 'val_accuracy' if 'val_accuracy' in self.history.history else None
            
            metric_name_display = "Metric" # Default display name
            if primary_metric_key and primary_metric_key in self.history.history:
                metric_name_display = primary_metric_key.replace("_", " ").title() # e.g. "Auc" or "Accuracy"
                plt.plot(self.history.history[primary_metric_key], label=f'Train {metric_name_display}')
            if val_primary_metric_key and val_primary_metric_key in self.history.history:
                plt.plot(self.history.history[val_primary_metric_key], label=f'Validation {metric_name_display}')
            
            plt.title(f'Keras Model {metric_name_display}')
            plt.xlabel('Epochs')
            plt.ylabel(metric_name_display)
            if primary_metric_key or (val_primary_metric_key and val_primary_metric_key in self.history.history): # Ensure legend only if data plotted
                plt.legend()
            
            plot_file = os.path.join(output_path, f'{self.model_type}_keras_learning_curves.png')
            try:
                plt.savefig(plot_file)
                # print(f"Keras learning curves saved to {plot_file}")
            except Exception as e:
                print(f"Error saving Keras learning curves: {e}")
            plt.close()

    def train(self, train_data, test_data, target_column, output_path=None, plot_learning_curves=True, save_run_results=True): # Added save_run_results
        if train_data is None or train_data.empty:
            print("Error: Training data is None or empty. Aborting training.")
            return None, None

        _train_data = train_data.copy()
        _test_data = test_data.copy() if test_data is not None else pd.DataFrame()


        potential_id_cols = [col for col in _train_data.columns if col.lower() == 'id' and col != target_column]
        if potential_id_cols:
            _train_data = _train_data.drop(columns=potential_id_cols)
            if not _test_data.empty:
                _test_data = _test_data.drop(columns=potential_id_cols, errors='ignore')
        
        has_test_target = target_column in _test_data.columns and not _test_data[_test_data[target_column].notna()].empty

        X_train, y_train, X_test, y_test = None, None, None, None
        eval_set_description = "N/A"
        
        # --- Robust extraction of X and y ---
        # Handle y_train
        if target_column not in _train_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data.")
        y_train_raw = _train_data[target_column]
        if isinstance(y_train_raw, pd.DataFrame):
            print(f"Warning: y_train extracted from _train_data['{target_column}'] was a DataFrame (shape: {y_train_raw.shape}). Using its first column as the target variable.")
            y_train = y_train_raw.iloc[:, 0]
        else:
            y_train = y_train_raw
        
        # Handle X_train (ensure no target column, even if duplicated)
        X_train = _train_data.loc[:, _train_data.columns != target_column]
        # ---

        if has_test_target:
            # --- Robust extraction for X_test and y_test ---
            if target_column not in _test_data.columns:
                 # This case should ideally be caught by has_test_target, but as a safeguard:
                print(f"Warning: Target column '{target_column}' not found in test data despite has_test_target being true. Proceeding as if no test target.")
                has_test_target = False # Correct the flag
                X_test = _test_data.copy() # X_test will be all of test_data, y_test will be None
                y_test = None
            else:
                y_test_raw = _test_data[target_column]
                if isinstance(y_test_raw, pd.DataFrame):
                    print(f"Warning: y_test extracted from _test_data['{target_column}'] was a DataFrame (shape: {y_test_raw.shape}). Using its first column as the target variable.")
                    y_test = y_test_raw.iloc[:, 0]
                else:
                    y_test = y_test_raw
                X_test = _test_data.loc[:, _test_data.columns != target_column]
            # ---
            eval_set_description = f"test_set (shape: {X_test.shape})"
        
        if not has_test_target: # This block executes if originally no test target OR if test target was problematic
            if y_train is None or y_train.empty: # y_train should be populated from _train_data above
                 raise ValueError(f"Target column '{target_column}' could not be properly extracted from training data for validation split.")

            # Stratify only if target is suitable
            stratify_on = None
            if y_train.nunique() > 1 and y_train.nunique() < len(y_train): # Check after y_train is confirmed 1D
                stratify_on = y_train

            # X_train was already prepared, so we use it directly
            # y_train was already prepared
            X_temp_train, X_val, y_temp_train, y_val = train_test_split(
                X_train, # Use already prepared X_train
                y_train, # Use already prepared y_train
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=stratify_on
            )
            X_train, y_train = X_temp_train, y_temp_train # Update X_train, y_train to be the smaller training portion
            X_test, y_test = X_val, y_val # X_test, y_test are now from the validation set
            eval_set_description = f"validation_set_from_train (shape: {X_test.shape})"

        X_train_cols_final = X_train.columns.tolist() # Columns used for training

        # Ensure target is numeric (label encoded)
        def _ensure_numeric_target(y_series, series_name="target"):
            if y_series is None: return None
            if not pd.api.types.is_numeric_dtype(y_series):
                try:
                    # Attempt direct conversion if it looks like numbers stored as strings
                    y_series_numeric = pd.to_numeric(y_series, errors='coerce')
                    if not y_series_numeric.isnull().all(): # If conversion was somewhat successful
                        if not pd.api.types.is_integer_dtype(y_series_numeric):
                             return y_series_numeric.astype(int) # Convert float to int if possible
                        return y_series_numeric
                    else: # Coercion failed, try LabelEncoding as last resort
                        print(f"Warning: Target column '{series_name}' is non-numeric. Attempting LabelEncoding.")
                        le = LabelEncoder()
                        return pd.Series(le.fit_transform(y_series.astype(str)), index=y_series.index, name=y_series.name)
                except Exception as e:
                    raise ValueError(f"Target column '{series_name}' could not be converted to numeric: {e}")
            elif not pd.api.types.is_integer_dtype(y_series):
                return y_series.astype(int) # Ensure integer type if already numeric
            return y_series

        y_train = _ensure_numeric_target(y_train, "y_train")
        y_test = _ensure_numeric_target(y_test, "y_test")

        hps = self.model_hyperparameters.copy() # Work with a copy

        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=self.random_state, **hps)
        elif self.model_type == 'logistic_regression':
            solver_penalty_config = hps.pop('solver_penalty_config', None)
            if solver_penalty_config and isinstance(solver_penalty_config, dict):
                hps.update(solver_penalty_config) # Add solver and penalty from the dict

            # Default solver if not provided by HPs or config
            current_solver = hps.get('solver', 'lbfgs') 
            current_penalty = hps.get('penalty')

            # Adjust solver if incompatible with penalty
            compatible_solvers = {
                'l1': ['liblinear', 'saga'],
                'l2': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                'elasticnet': ['saga'],
                None: ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] # For penalty=None
            }
            
            # Ensure penalty is a string or None for dictionary lookup
            penalty_key = current_penalty if isinstance(current_penalty, str) else None


            if penalty_key is not None and penalty_key in compatible_solvers:
                if current_solver not in compatible_solvers[penalty_key]:
                    new_solver = compatible_solvers[penalty_key][0] # Pick the first compatible
                    print(f"Warning: Solver '{current_solver}' for Logistic Regression is not compatible with penalty '{current_penalty}'. Changing solver to '{new_solver}'.")
                    hps['solver'] = new_solver
            elif penalty_key is None and 'penalty' in hps : # Handles penalty=None explicitly if set
                 if current_solver not in compatible_solvers[None]:
                    new_solver = compatible_solvers[None][0]
                    print(f"Warning: Solver '{current_solver}' for Logistic Regression is not compatible with no penalty. Changing solver to '{new_solver}'.")
                    hps['solver'] = new_solver


            if 'l1_ratio' in hps and hps.get('penalty') != 'elasticnet':
                hps.pop('l1_ratio') # Remove l1_ratio if not using elasticnet
            if hps.get('penalty') == 'elasticnet' and 'l1_ratio' not in hps:
                 hps['l1_ratio'] = 0.5 # Default l1_ratio for elasticnet if not specified

            self.model = LogisticRegression(random_state=self.random_state, **hps)

        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=self.random_state, **hps)
        
        elif self.model_type == 'neural_network':
            input_shape = (X_train.shape[1],)
            num_classes = y_train.nunique()

            # Keras specific HPs
            epochs = hps.get('epochs', 50)
            batch_size = hps.get('batch_size', 32)
            early_stopping_patience = hps.get('early_stopping_patience', 5)
            
            # Remove these from hps before passing to _build_keras_model if they are not direct layer/optimizer params
            keras_train_hps_keys = ['epochs', 'batch_size', 'early_stopping_patience']
            build_hps = {k: v for k, v in hps.items() if k not in keras_train_hps_keys}

            self.model = self._build_keras_model(input_shape, num_classes, **build_hps)
            
            # Ensure y_train and y_test are 1D arrays for Keras
            y_train_keras = y_train.values.ravel()
            y_test_keras = y_test.values.ravel()

            # Convert to categorical if multi-class
            if num_classes > 2:
                y_train_keras = to_categorical(y_train_keras, num_classes=num_classes)
                y_test_keras = to_categorical(y_test_keras, num_classes=num_classes)
            
            callbacks_list = []
            if early_stopping_patience > 0:
                early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1)
                callbacks_list.append(early_stopping)

            print(f"Training Keras model for {epochs} epochs, batch size {batch_size}...")
            self.history = self.model.fit(
                X_train, y_train_keras,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test_keras),
                callbacks=callbacks_list,
                verbose=0 # 0 for silent, 1 for progress bar, 2 for one line per epoch.
            )
            if plot_learning_curves and output_path:
                self._plot_keras_learning_curves(output_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if self.model_type != 'neural_network':
            print(f"Training {self.model_type}...")
            self.model.fit(X_train, y_train)
            if plot_learning_curves and output_path :
                 os.makedirs(output_path, exist_ok=True) # Ensure path exists
                 self.plot_learning_curves(X_train, y_train, output_path)


        metrics = {'evaluation_set_description': eval_set_description}
        feature_importance_df = None

        print(f"Evaluating {self.model_type} on {eval_set_description}...")
        if X_test is not None and y_test is not None and not X_test.empty:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, "predict_proba") else None

            # For Keras, y_pred might be probabilities if final layer is softmax and loss is categorical_crossentropy
            # or it could be class labels if using argmax internally after predict.
            # We need class labels for classification_report.
            if self.model_type == 'neural_network':
                if y_pred.ndim > 1 and y_pred.shape[1] > 1: # Probabilities for multi-class
                    y_pred = np.argmax(y_pred, axis=1)
                # If binary and y_pred are probabilities (e.g. shape (N,1) or (N,2) with sigmoid/softmax)
                # convert to class labels 0 or 1.
                elif y_pred.ndim == 1 or y_pred.shape[1] == 1: # Binary probabilities
                    y_pred = (y_pred > 0.5).astype(int)
                
                # y_pred_proba for Keras might be the raw output of the network.
                # If num_classes == 2 and y_pred_proba has shape (N,1), use it for positive class.
                # If num_classes > 2, it should already be (N, num_classes).
                # If num_classes == 2 and y_pred_proba has shape (N,2), use proba of positive class.
                if hasattr(self.model, 'predict'): # Keras model's predict often gives probabilities directly
                    keras_probas = self.model.predict(X_test)
                    if y_train.nunique() == 2: # Binary case
                        if keras_probas.ndim > 1 and keras_probas.shape[1] == 2:
                            y_pred_proba = keras_probas[:, 1] # Probability of positive class
                        elif keras_probas.ndim == 1 or keras_probas.shape[1] == 1: # Already (N,) or (N,1)
                             y_pred_proba = keras_probas.ravel()
                        # else: y_pred_proba might already be set from a more general predict_proba call
                    else: # Multi-class, Keras predict gives (N, num_classes)
                        y_pred_proba = keras_probas


            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['f1_score_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score_micro'] = f1_score(y_test, y_pred, average='micro', zero_division=0)
            metrics['f1_score_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            num_unique_classes = y_train.nunique() # This will be 2 for binary classification

            if y_pred_proba is not None:
                if num_unique_classes == 2: # Binary classification
                    # Ensure y_pred_proba for binary is 1D array of positive class probabilities
                    if y_pred_proba.ndim == 2:
                        if y_pred_proba.shape[1] == 2:
                            y_pred_proba_binary = y_pred_proba[:, 1]
                        elif y_pred_proba.shape[1] == 1: # Should ideally not happen if positive class proba is extracted
                            y_pred_proba_binary = y_pred_proba.ravel()
                        else: # Should not happen for binary if model is correct
                            print(f"Warning: y_pred_proba has unexpected shape {y_pred_proba.shape} for binary classification. Using as is for AUPRC/ROC AUC.")
                            # This case might be problematic, but we let roc_auc_score try
                            y_pred_proba_binary = y_pred_proba 
                    else: # Assumed to be already 1D
                        y_pred_proba_binary = y_pred_proba

                    try:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba_binary)
                        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_binary)
                        metrics['auprc'] = auc(recall, precision)
                    except ValueError as e_metrics: # Handles cases like only one class present in y_test or other issues
                        print(f"Could not compute ROC AUC or AUPRC for binary case: {e_metrics}")
                        metrics['roc_auc'] = None
                        metrics['auprc'] = None
                
                else: # Multi-class classification
                    try:
                        # For multi-class, y_pred_proba should have shape (n_samples, n_classes)
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                        
                        # For AUPRC (average precision) in multi-class:
                        # Binarize y_test
                        lb = LabelBinarizer()
                        y_test_binarized = lb.fit_transform(y_test)
                        
                        # If y_test_binarized is a 1D array after binarization (e.g. only 2 classes that were not 0 and 1)
                        # It means the original classes were not [0, 1, ..., n_classes-1]
                        # We need to ensure y_test_binarized matches dimensions of y_pred_proba for average_precision_score
                        if y_test_binarized.shape[1] == 1 and y_pred_proba.shape[1] > 1 and num_unique_classes > 2 :
                             # This can happen if LabelBinarizer squeezes output for 2 classes that are not 0,1
                             # but it's a multi-class problem by num_unique_classes.
                             # Re-binarize with explicit classes if this becomes an issue.
                             # For now, we trust num_unique_classes from y_train for the overall problem type.
                             # A common case is if classes are e.g. [1, 2, 3], LB might make it 2 columns.
                             # We need to ensure y_test_binarized has n_classes columns.
                             # Re-creating with explicit classes to ensure correct shape if num_unique_classes > 2
                             encoder = LabelEncoder()
                             # Fit on y_train to get all possible classes
                             all_classes = encoder.fit(y_train).classes_ 
                             y_test_for_binarize = encoder.transform(y_test) # Ensure y_test labels are 0 to k-1

                             lb_mc = LabelBinarizer()
                             lb_mc.fit(y_test_for_binarize) # Fit on transformed labels
                             y_test_binarized_mc = lb_mc.transform(y_test_for_binarize)

                             # Handle case where LabelBinarizer creates only one column for two classes
                             if num_unique_classes == 2 and y_test_binarized_mc.shape[1] == 1 and y_pred_proba.shape[1] == 2:
                                 # This is essentially a binary case handled by the previous block,
                                 # but if it falls here, we should use the positive class probabilities.
                                 # This should ideally not be hit if the num_unique_classes == 2 path is taken.
                                 metrics['auprc'] = average_precision_score(y_test, y_pred_proba[:, 1], average='weighted')
                             elif y_test_binarized_mc.shape[1] != y_pred_proba.shape[1] and num_unique_classes > 1 :
                                 # If after explicit binarization, shapes still mismatch (and not binary case)
                                 # this implies an issue. Fallback or log error.
                                 print(f"Warning: Shape mismatch for multi-class AUPRC. y_test_binarized shape: {y_test_binarized_mc.shape}, y_pred_proba shape: {y_pred_proba.shape}. AUPRC might be incorrect.")
                                 metrics['auprc'] = None # Or some default error value
                             else:
                                 metrics['auprc'] = average_precision_score(y_test_binarized_mc, y_pred_proba, average='weighted')

                        elif y_test_binarized.shape[1] != y_pred_proba.shape[1] and num_unique_classes > 1 : # General case mismatch
                             print(f"Warning: Shape mismatch for multi-class AUPRC (initial binarization). y_test_binarized shape: {y_test_binarized.shape}, y_pred_proba shape: {y_pred_proba.shape}. AUPRC might be incorrect.")
                             metrics['auprc'] = None
                        else: # Shapes match or it's a single column binary case handled by LabelBinarizer
                             metrics['auprc'] = average_precision_score(y_test_binarized, y_pred_proba, average="weighted")

                    except ValueError as e_metrics:
                        print(f"Could not compute ROC AUC or AUPRC for multi-class case: {e_metrics}")
                        metrics['roc_auc'] = None
                        metrics['auprc'] = None
            else:
                metrics['roc_auc'] = None
                metrics['auprc'] = None

            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report_dict
            # print(f"Metrics for {self.model_type} on {eval_set_description}:")
            # for k, v in metrics.items():
            #     if k != 'classification_report': print(f"  {k}: {v}")

        if hasattr(self.model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'feature': X_train_cols_final,
            'importance': self.model.feature_importances_
            }).sort_values(by='importance', ascending=False)
        elif hasattr(self.model, 'coef_') and self.model_type == 'logistic_regression':
            # For logistic regression, coef_ might be (1, n_features) for binary or (n_classes, n_features) for multi-class
            if self.model.coef_.ndim == 1 : # Should not happen with current sklearn
                 coefs = self.model.coef_
            elif self.model.coef_.shape[0] == 1: # Binary classification
                coefs = self.model.coef_[0]
            else: # Multi-class, take the average of absolute coefficients across classes as a simple measure
                coefs = np.mean(np.abs(self.model.coef_), axis=0)
            
            feature_importance_df = pd.DataFrame({
                'feature': X_train_cols_final,
                'importance': coefs # For LogReg, these are coefficients
            }).sort_values(by='importance', key=abs, ascending=False) # Sort by absolute value of coefs
        
        # Save results if path is provided and flag is True
        if output_path and save_run_results:
            os.makedirs(output_path, exist_ok=True)
            results_summary_path = os.path.join(output_path, f'{self.model_type}_results_summary.txt')
            with open(results_summary_path, 'w') as f:
                f.write(f"Model Type: {self.model_type}\n")
                f.write(f"Hyperparameters: {self.model_hyperparameters}\n")
                f.write(f"Evaluation Set: {eval_set_description}\n\n")
                f.write("Metrics:\n")
                for key, value in metrics.items():
                    if key == 'classification_report':
                        f.write(f"  {key}:\n")
                        # Nicely format the dict report
                        for class_label, class_metrics in value.items():
                            if isinstance(class_metrics, dict):
                                f.write(f"    {class_label}:\n")
                                for metric_name, metric_value in class_metrics.items():
                                    f.write(f"      {metric_name}: {metric_value:.4f}\n")
                            else: # e.g. accuracy, macro avg, weighted avg
                                f.write(f"    {class_label}: {value[class_label]:.4f}\n") # Ensure formatting for top-level report items
                    else:
                        f.write(f"  {key}: {value}\n") # General metrics
                
                if feature_importance_df is not None:
                    f.write("\nFeature Importances:\n")
                    f.write(feature_importance_df.to_string())
            # print(f"Results summary saved to {results_summary_path}")

        return metrics, feature_importance_df
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        Args:
            X: pandas DataFrame, data to make predictions on
        Returns:
            numpy array: predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure X has the same columns as training data, in the same order
        # This is a simplified check; more robust handling might involve storing X_train_cols
        # during training and reordering/adding missing columns in X.
        # For now, we assume X is correctly preprocessed.
        
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Make probability predictions using the trained model.
        Args:
            X: pandas DataFrame, data to make predictions on
        Returns:
            numpy array: probability predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"The model {self.model_type} does not support predict_proba.")
        return self.model.predict_proba(X)

    def _build_keras_model(self, input_shape, num_classes, **kwargs):
        """
        Builds a Keras Sequential model.
        Hyperparameters for layers, dropout, learning rate, and regularization are passed via kwargs.
        """
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (64, 32)) # Default: (64, 32)
        dropout_rate = kwargs.get('dropout_rate', 0.2) # Default: 0.2
        learning_rate = kwargs.get('learning_rate', 0.001) # Default: 0.001
        l1_reg, l2_reg = kwargs.get('l1_reg', 0.00), kwargs.get('l2_reg', 0.00)
        model = Sequential([Input(shape=input_shape)])
        for units in hidden_layer_sizes:
            model.add(Dense(units, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
            if dropout_rate > 0: model.add(Dropout(dropout_rate))
        
        metrics_to_compile = ['accuracy'] # Initialize here

        if num_classes == 2: 
            model.add(Dense(1, activation='sigmoid'))
            loss_function = 'binary_crossentropy'
            metrics_to_compile.extend([tf.keras.metrics.AUC(name='roc_auc', curve='ROC'), tf.keras.metrics.AUC(name='auprc', curve='PR')])
        else: 
            model.add(Dense(num_classes, activation='softmax'))
            loss_function = 'categorical_crossentropy'
            # For multi-class, Keras AUC with multi_label=True and num_labels implies an averaging (e.g., macro) over per-class AUCs.
            metrics_to_compile.extend([
                tf.keras.metrics.AUC(name='roc_auc', curve='ROC', multi_label=True, num_labels=num_classes),
                tf.keras.metrics.AUC(name='auprc', curve='PR', multi_label=True, num_labels=num_classes)
            ])
            
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_to_compile)
        return model

# --- End of ModelTrainer class ---

# Example usage (commented out, for testing or direct script execution)
# if __name__ == '__main__':
#     # Create dummy data for testing
#     from sklearn.datasets import make_classification
#     X_dummy, y_dummy = make_classification(n_samples=200, n_features=20, n_informative=15, 
#                                            n_redundant=5, random_state=42, n_classes=2) # Binary
#     # X_dummy, y_dummy = make_classification(n_samples=200, n_features=20, n_informative=15, 
#     #                                        n_redundant=5, random_state=42, n_classes=3, n_clusters_per_class=1) # Multiclass

#     dummy_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(X_dummy.shape[1])])
#     dummy_df['target'] = y_dummy

#     train_df, test_df = train_test_split(dummy_df, test_size=0.25, random_state=42, stratify=dummy_df['target'])
    
#     output_dir = "research/dummy_dataset/model_tests"
#     os.makedirs(output_dir, exist_ok=True)

#     # Test Scikit-learn models
#     # trainer_rf = ModelTrainer(model_type='random_forest', model_hyperparameters={'n_estimators': 50, 'max_depth': 5})
#     # print("\n--- Training Random Forest ---")
#     # metrics_rf, fi_rf = trainer_rf.train(train_df, test_df, 'target', output_path=os.path.join(output_dir, 'rf'), plot_learning_curves=True)
#     # if metrics_rf: print("Random Forest Metrics:", metrics_rf['accuracy'], metrics_rf.get('auprc'))
#     # if fi_rf is not None: print("Random Forest FI:\n", fi_rf.head())

#     # trainer_lr = ModelTrainer(model_type='logistic_regression', model_hyperparameters={'C': 0.1, 'solver_penalty_config': {'penalty': 'l1', 'solver': 'liblinear'}})
#     # print("\n--- Training Logistic Regression ---")
#     # metrics_lr, fi_lr = trainer_lr.train(train_df, test_df, 'target', output_path=os.path.join(output_dir, 'lr'), plot_learning_curves=True)
#     # if metrics_lr: print("Logistic Regression Metrics:", metrics_lr['accuracy'], metrics_lr.get('auprc'))
#     # if fi_lr is not None: print("Logistic Regression Coefs:\n", fi_lr.head())

#     # Test Neural Network
#     nn_hps = {
#         'hidden_layer_sizes': (32, 16), 
#         'dropout_rate': 0.1, 
#         'learning_rate': 0.005,
#         'epochs': 10, # Keep epochs low for quick test
#         'batch_size': 16,
#         'early_stopping_patience': 3,
#         'l1_reg': 0.001,
#         'l2_reg': 0.001
#     }
#     trainer_nn = ModelTrainer(model_type='neural_network', model_hyperparameters=nn_hps)
#     print("\n--- Training Neural Network ---")
#     metrics_nn, fi_nn = trainer_nn.train(train_df, test_df, 'target', output_path=os.path.join(output_dir, 'nn'), plot_learning_curves=True)
#     if metrics_nn: print(f"Neural Network Metrics (from {metrics_nn.get('evaluation_set_description')}): Accuracy={metrics_nn.get('accuracy')}, AUPRC={metrics_nn.get('auprc')}")
#     # NN doesn't have simple feature importance like tree models or linear models
#     if fi_nn is not None: print("Neural Network FI not applicable directly.")
#     else: print("Neural Network FI not applicable directly.")
