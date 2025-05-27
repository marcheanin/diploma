import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Assuming these modules are in the same directory or PYTHONPATH is set up
from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.outlier_remover import OutlierRemover
from preprocessing.resampler import Resampler
# from utils.data_analysis import analyze_target_correlations # Was commented out

def get_dataset_name(path):
    """
    Extract dataset name from file path.
    Example: 'datasets/credit-score-classification/train.csv' -> 'credit-score-classification'
    """
    parts = os.path.normpath(path).split(os.sep)
    if 'datasets' in parts:
        datasets_index = parts.index('datasets')
        if len(parts) > datasets_index + 1:
            return parts[datasets_index + 1]
    return 'unknown_dataset'

def process_data(train_path, test_path, target_column,
                 imputation_method='knn', imputation_params=None,
                 outlier_method='isolation_forest', outlier_params=None,
                 encoding_method='label', encoding_params=None,
                 resampling_method='none', resampling_params=None,
                 scaling_method='none', scaling_params=None,
                 save_processed_data=True,
                 save_model_artifacts=True):
    """
    Process data using specified methods and their hyperparameters.
    
    Args:
        train_path: str, path to training data
        test_path: str, path to test data
        target_column: str, name of target column
        imputation_method: str, e.g., 'knn', 'median', 'missforest'
        imputation_params: dict, HPs for the imputation method
        outlier_method: str, e.g., 'isolation_forest', 'iqr', 'none'
        outlier_params: dict, HPs for the outlier removal method
        encoding_method: str, e.g., 'onehot', 'label', 'lsa', 'word2vec'
        encoding_params: dict, HPs for the encoding method
        resampling_method: str, e.g., 'none', 'oversample', 'smote'
        resampling_params: dict, HPs for the resampling method
        scaling_method: str, e.g., 'none', 'standard', 'minmax'
        scaling_params: dict, HPs for the scaling method
        save_processed_data: bool, whether to save processed data to disk
        save_model_artifacts: bool, whether to create and use research_path for model artifacts (e.g. plots)
        
    Returns:
        tuple: (train_data, test_data, research_path)
               train_data/test_data are DataFrames if save_processed_data is False,
               otherwise they are file paths.
               Returns (None, None, None) if a critical error occurs.
    """
    imputation_params = imputation_params or {}
    outlier_params = outlier_params or {}
    encoding_params = encoding_params or {}
    resampling_params = resampling_params or {}
    scaling_params = scaling_params or {}
    
    dataset_name = get_dataset_name(train_path)
    
    # Experiment name includes method names only, HPs are too verbose for dir names
    experiment_name_parts = [imputation_method, outlier_method, encoding_method, resampling_method]
    if scaling_method != 'none':
        experiment_name_parts.append(scaling_method)
    experiment_name = "_".join(experiment_name_parts)
    
    # Path for processed data files (e.g. train_processed.csv)
    results_path = os.path.join('results', dataset_name, experiment_name)
    # Path for model-related research artifacts (e.g. learning curves, result summaries)
    # This path might still be used by ModelTrainer even if directory isn't created here,
    # if ModelTrainer receives it and decides to create subdirectories itself.
    # However, not creating it here prevents empty directories if ModelTrainer also doesn't save.
    research_path = os.path.join("research", dataset_name, experiment_name)
    
    if save_processed_data:
        os.makedirs(results_path, exist_ok=True)
    
    # Only create research_path if model artifacts are to be saved by this pipeline run
    if save_model_artifacts:
        os.makedirs(research_path, exist_ok=True)
    # If save_model_artifacts is False, research_path is still constructed and passed,
    # but the directory itself is not created by process_data.
    # ModelTrainer will then decide if it wants to use this path (and potentially create it).

    # Create a unique suffix for filenames if saving, including HPs would be too long.
    # The GA evaluation will rely on directory structure + logs for HP tracking.
    # For non-GA runs (save_processed_data=True), the path itself is a good identifier.
    output_suffix = f"_processed_{experiment_name}"

    print(f"\n[Pipeline Stage - Config: {experiment_name}] Loading data...")
    loader = DataLoader(train_path=train_path, test_path=test_path)
    train_data, test_data = loader.load_data()

    if train_data is None or train_data.empty:
        print(f"Error: Training data could not be loaded from {train_path} or is empty.")
        return None, None, None

    preprocessor = DataPreprocessor()
    
    print(f"[Pipeline Stage - Config: {experiment_name}] Imputation ({imputation_method}, HPs: {imputation_params})...")
    train_data = preprocessor.impute(train_data, method=imputation_method, **imputation_params)
    if test_data is not None and not test_data.empty:
        test_data = preprocessor.impute(test_data, method=imputation_method, **imputation_params)
    print(f"[Pipeline Stage - Config: {experiment_name}] Imputation completed.")

    potential_id_columns_process = [col for col in ['ID', 'id'] if col in train_data.columns and col != target_column]
    if potential_id_columns_process:
        print(f"Dropping ID column(s) from train_data: {potential_id_columns_process} before outlier removal.")
        train_data = train_data.drop(columns=potential_id_columns_process)
        if test_data is not None and not test_data.empty:
            test_data = test_data.drop(columns=potential_id_columns_process, errors='ignore')

    print(f"[Pipeline Stage - Config: {experiment_name}] Outlier removal ({outlier_method}, HPs: {outlier_params})...")
    if outlier_method != 'none':
        train_data_before_outliers = train_data.copy()
        try:
            remover = OutlierRemover(method=outlier_method, **outlier_params) 
            features_for_removal = train_data.drop(columns=[target_column], errors='ignore')
            
            target_series_for_reconstruction = None
            if target_column in train_data.columns:
                target_series_for_reconstruction = train_data[target_column]

            cleaned_features = remover.remove_outliers(features_for_removal)
            
            if cleaned_features.empty and not features_for_removal.empty :
                print(f"Warning: Outlier removal (method: {outlier_method}) resulted in an empty dataset. Reverting.")
                train_data = train_data_before_outliers
            elif target_series_for_reconstruction is not None:
                cleaned_features_idx = cleaned_features.index
                target_series_aligned = target_series_for_reconstruction.loc[target_series_for_reconstruction.index.intersection(cleaned_features_idx)]
                cleaned_features_aligned = cleaned_features.loc[cleaned_features.index.intersection(target_series_aligned.index)]
                
                train_data = pd.concat([cleaned_features_aligned, target_series_aligned.rename(target_column)], axis=1)

            else:
                train_data = cleaned_features
        except Exception as e:
            print(f"Could not remove outliers (method: {outlier_method}): {e}. Reverting.")
            train_data = train_data_before_outliers
    else:
        print("Outlier removal skipped as method is 'none'.")
    print(f"[Pipeline Stage - Config: {experiment_name}] Outlier removal completed.")

    # --- Diagnostic: Check for duplicate columns before encoding ---
    duplicated_cols_before_encoding = train_data.columns[train_data.columns.duplicated()].tolist()
    if duplicated_cols_before_encoding:
        print(f"WARNING: Duplicate columns found in train_data BEFORE encoding: {duplicated_cols_before_encoding}")
        # Aggressively remove duplicates, keeping the first occurrence
        train_data = train_data.loc[:, ~train_data.columns.duplicated(keep='first')]
        print(f"         Duplicates removed. Columns now: {train_data.columns.tolist()}")
    if test_data is not None and not test_data.empty:
        duplicated_cols_test_before_encoding = test_data.columns[test_data.columns.duplicated()].tolist()
        if duplicated_cols_test_before_encoding:
            print(f"WARNING: Duplicate columns found in test_data BEFORE encoding: {duplicated_cols_test_before_encoding}")
            test_data = test_data.loc[:, ~test_data.columns.duplicated(keep='first')]
            print(f"         Duplicates removed from test_data. Columns now: {test_data.columns.tolist()}")
    # --- End Diagnostic ---

    print(f"[Pipeline Stage - Config: {experiment_name}] Encoding ({encoding_method}, HPs: {encoding_params})...")
    # Print column list right before encoding for detailed check
    # print(f"DEBUG: train_data columns before preprocessor.encode: {train_data.columns.tolist()}")
    # if test_data is not None and not test_data.empty:
    # print(f"DEBUG: test_data columns before preprocessor.encode: {test_data.columns.tolist()}")

    train_data = preprocessor.encode(train_data, method=encoding_method, target_col=target_column, **encoding_params)
    if test_data is not None and not test_data.empty:
        test_data = preprocessor.encode(test_data, method=encoding_method, target_col=target_column, **encoding_params)
    print(f"[Pipeline Stage - Config: {experiment_name}] Encoding completed.")
    
    print(f"[Pipeline Stage - Config: {experiment_name}] Resampling ({resampling_method}, HPs: {resampling_params})...")
    if resampling_method != 'none':
        if target_column in train_data.columns:
            if not pd.api.types.is_numeric_dtype(train_data[target_column]):
                print(f"Warning: Target column '{target_column}' is not numeric. Applying LabelEncoding before resampling.")
                train_data = preprocessor._label_encode_target(train_data, target_column)
            
            X_train_encoded = train_data.drop(columns=[target_column])
            y_train_encoded = train_data[target_column]
            
            # Pass HPs to Resampler constructor or method
            resampler = Resampler(method=resampling_method, **resampling_params)
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_encoded, y_train_encoded)
            
            train_data = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train_encoded.columns), 
                                    pd.Series(y_train_resampled, name=target_column)], axis=1)
        else:
            print(f"Warning: Target column '{target_column}' not found. Skipping resampling.")
    else:
        print("Resampling method is 'none'. Skipping resampling step.")
    print(f"[Pipeline Stage - Config: {experiment_name}] Resampling completed.")

    print(f"[Pipeline Stage - Config: {experiment_name}] Scaling ({scaling_method}, HPs: {scaling_params})...")
    if scaling_method != 'none' and scaling_method in ['standard', 'minmax']:
        if target_column not in train_data.columns:
            print(f"CRITICAL WARNING: Target column '{target_column}' not found in train_data before scaling. Skipping scaling.")
        else:
            X_train_features = train_data.drop(columns=[target_column])
            y_train_target = train_data[target_column]

            scaler_instance = None
            if scaling_method == 'standard':
                scaler_instance = StandardScaler(**scaling_params) # Pass HPs
            elif scaling_method == 'minmax':
                scaler_instance = MinMaxScaler(**scaling_params) # Pass HPs (though minmax has few common HPs)

            if scaler_instance:
                try:
                    scaled_X_train_values = scaler_instance.fit_transform(X_train_features)
                    scaled_X_train_df = pd.DataFrame(scaled_X_train_values, columns=X_train_features.columns, index=X_train_features.index)
                    train_data = pd.concat([scaled_X_train_df, y_train_target], axis=1)

                    if test_data is not None and not test_data.empty:
                        X_test_features_original = test_data.drop(columns=[target_column], errors='ignore')
                        y_test_target_original = test_data[target_column] if target_column in test_data.columns else None
                        
                        cols_to_scale_in_test = [col for col in X_train_features.columns if col in X_test_features_original.columns]
                        
                        if not cols_to_scale_in_test:
                             print("Warning: No common features to scale in test data. Test data unscaled.")
                        elif len(cols_to_scale_in_test) != len(X_train_features.columns):
                            print("Warning: Test data feature set mismatch for scaling. Scaling subset.")
                            X_test_subset_to_scale = X_test_features_original[cols_to_scale_in_test]
                            scaled_X_test_subset_values = scaler_instance.transform(X_test_subset_to_scale)
                            scaled_X_test_subset_df = pd.DataFrame(scaled_X_test_subset_values, columns=cols_to_scale_in_test, index=X_test_subset_to_scale.index)
                            
                            X_test_features_updated = X_test_features_original.copy()
                            for col in cols_to_scale_in_test:
                                X_test_features_updated[col] = scaled_X_test_subset_df[col]

                            if y_test_target_original is not None:
                                test_data = pd.concat([X_test_features_updated, y_test_target_original], axis=1)
                            else:
                                test_data = X_test_features_updated
                        else: 
                            scaled_X_test_values = scaler_instance.transform(X_test_features_original[X_train_features.columns])
                            scaled_X_test_df = pd.DataFrame(scaled_X_test_values, columns=X_train_features.columns, index=X_test_features_original.index)

                            if y_test_target_original is not None:
                                test_data = pd.concat([scaled_X_test_df, y_test_target_original], axis=1)
                            else:
                                test_data = scaled_X_test_df
                except Exception as e:
                    print(f"Error during scaling: {e}. Skipping scaling.")
    elif scaling_method != 'none':
        print(f"Warning: Unknown scaling method '{scaling_method}'. Scaling skipped.")
    print(f"[Pipeline Stage - Config: {experiment_name}] Scaling completed.")
    
    train_data_path_out = None
    test_data_path_out = None

    if save_processed_data:
        train_data_path_out = os.path.join(results_path, f'train{output_suffix}.csv')
        test_data_path_out = os.path.join(results_path, f'test{output_suffix}.csv')
        
        train_data.to_csv(train_data_path_out, index=False)
        if test_data is not None and not test_data.empty:
            test_data.to_csv(test_data_path_out, index=False)
        else:
            test_data_path_out = None 
        print(f"[Pipeline Stage - Config: {experiment_name}] Data processing complete. Processed files saved to: {results_path}")
        return train_data_path_out, test_data_path_out, research_path # Return paths
    else:
        print(f"[Pipeline Stage - Config: {experiment_name}] Data processing complete. Processed data NOT saved (GA mode).")
        return train_data, test_data, research_path # Return DataFrames

    # analyze_target_correlations was commented out, so not re-adding here for now.
    
    # Return DataFrames directly if not saving, otherwise paths
    if not save_processed_data:
        return train_data, test_data, research_path
    else:
        return train_data_path_out, test_data_path_out, research_path 