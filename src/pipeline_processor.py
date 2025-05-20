import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
                 imputation_method='knn', 
                 outlier_method='isolation_forest',
                 encoding_method='label',
                 resampling_method='none',
                 scaling_method='none',
                 imputation_kwargs=None, 
                 outlier_kwargs=None,
                 encoding_kwargs=None,
                 scaling_kwargs=None,
                 save_processed_data=True):
    """
    Process data using specified imputation, encoding, and resampling methods.
    
    Args:
        train_path: str, path to training data
        test_path: str, path to test data
        target_column: str, name of target column
        imputation_method: str, 'knn', 'median', or 'missforest'
        outlier_method: str, 'isolation_forest', 'iqr', or 'none'
        encoding_method: str, 'onehot', 'label', 'ordinal', 'leaveoneout', 'lsa', 'word2vec'
        resampling_method: str, 'none', 'oversample' (ROS), 'undersample' (RUS), 'smote', 'adasyn'
        scaling_method: str, 'none', 'standard', 'minmax'
        imputation_kwargs: dict, keyword arguments for the imputation method
        outlier_kwargs: dict, keyword arguments for the outlier removal method
        encoding_kwargs: dict, keyword arguments for the encoding method
        scaling_kwargs: dict, keyword arguments for the scaling method
        save_processed_data: bool, whether to save processed data to disk
        
    Returns:
        tuple: (train_data_path, test_data_path, research_path) or (None, None, None) if error
    """
    imputation_kwargs = imputation_kwargs or {}
    outlier_kwargs = outlier_kwargs or {}
    encoding_kwargs = encoding_kwargs or {}
    scaling_kwargs = scaling_kwargs or {}
    
    dataset_name = get_dataset_name(train_path)
    
    experiment_name_parts = [imputation_method, outlier_method, encoding_method, resampling_method]
    if scaling_method != 'none':
        experiment_name_parts.append(scaling_method)
    experiment_name = "_".join(experiment_name_parts)
    
    results_path = os.path.join('results', dataset_name, experiment_name)
    research_path = os.path.join("research", dataset_name, experiment_name) # This is base for this config
    
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(research_path, exist_ok=True)

    print(f"\n[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Loading data...")
    loader = DataLoader(train_path=train_path, test_path=test_path)
    train_data, test_data = loader.load_data()

    if train_data is None or train_data.empty:
        print(f"Error: Training data could not be loaded from {train_path} or is empty.")
        return None, None, None

    preprocessor = DataPreprocessor()
    
    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Starting imputation ({imputation_method})...")
    train_data = preprocessor.impute(train_data, method=imputation_method, **imputation_kwargs)
    if test_data is not None and not test_data.empty:
        test_data = preprocessor.impute(test_data, method=imputation_method, **imputation_kwargs)
    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Imputation completed.")

    potential_id_columns_process = [col for col in ['ID', 'id'] if col in train_data.columns and col != target_column]
    if potential_id_columns_process:
        print(f"Dropping ID column(s) from train_data: {potential_id_columns_process} before outlier removal.")
        train_data = train_data.drop(columns=potential_id_columns_process)
        if test_data is not None and not test_data.empty:
            test_data = test_data.drop(columns=potential_id_columns_process, errors='ignore')

    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Starting outlier removal ({outlier_method})...")
    if outlier_method != 'none':
        train_data_before_outliers = train_data.copy()
        try:
            remover = OutlierRemover(method=outlier_method, **outlier_kwargs) 
            features_for_removal = train_data.drop(columns=[target_column], errors='ignore')
            target_series_for_reconstruction = None
            if target_column in train_data.columns:
                target_series_for_reconstruction = train_data[target_column]

            cleaned_features = remover.remove_outliers(features_for_removal)
            
            if cleaned_features.empty and not features_for_removal.empty :
                print(f"Warning: Outlier removal (method: {outlier_method}) resulted in an empty dataset. Reverting.")
                train_data = train_data_before_outliers
            elif target_series_for_reconstruction is not None:
                # Ensure index alignment for concatenation
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
    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Outlier removal completed.")

    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Starting encoding ({encoding_method})...")
    train_data = preprocessor.encode(train_data, method=encoding_method, target_col=target_column, **encoding_kwargs)
    if test_data is not None and not test_data.empty:
        test_data = preprocessor.encode(test_data, method=encoding_method, target_col=target_column, **encoding_kwargs)
    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Encoding completed.")
    
    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Starting resampling ({resampling_method})...")
    if resampling_method != 'none':
        if target_column in train_data.columns:
            if not pd.api.types.is_numeric_dtype(train_data[target_column]):
                print(f"Warning: Target column '{target_column}' is not numeric. Applying LabelEncoding before resampling.")
                train_data = preprocessor._label_encode_target(train_data, target_column) # ensure target is numeric
            
            X_train_encoded = train_data.drop(columns=[target_column])
            y_train_encoded = train_data[target_column]
            
            resampler = Resampler(method=resampling_method)
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_encoded, y_train_encoded)
            
            train_data = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train_encoded.columns), 
                                    pd.Series(y_train_resampled, name=target_column)], axis=1)
        else:
            print(f"Warning: Target column '{target_column}' not found. Skipping resampling.")
    else:
        print("Resampling method is 'none'. Skipping resampling step.")
    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Resampling completed.")

    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Starting scaling ({scaling_method})...")
    if scaling_method != 'none' and scaling_method in ['standard', 'minmax']:
        if target_column not in train_data.columns:
            print(f"CRITICAL WARNING: Target column '{target_column}' not found in train_data before scaling. Skipping scaling.")
        else:
            X_train_features = train_data.drop(columns=[target_column])
            y_train_target = train_data[target_column]

            scaler_instance = None
            if scaling_method == 'standard':
                scaler_instance = StandardScaler(**scaling_kwargs)
            elif scaling_method == 'minmax':
                scaler_instance = MinMaxScaler(**scaling_kwargs)

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
    print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Scaling completed.")

    filename_suffix_parts = [imputation_method, outlier_method, encoding_method, resampling_method]
    if scaling_method != 'none':
        filename_suffix_parts.append(scaling_method)
    output_suffix = f"_processed_{'_'.join(filename_suffix_parts)}"
    
    processed_train_df = train_data # Keep DataFrame in memory
    processed_test_df = test_data  # Keep DataFrame in memory

    train_data_path_out = None
    test_data_path_out = None

    if save_processed_data:
        train_data_path_out = os.path.join(results_path, f'train{output_suffix}.csv')
        test_data_path_out = os.path.join(results_path, f'test{output_suffix}.csv')
        
        processed_train_df.to_csv(train_data_path_out, index=False)
        if processed_test_df is not None and not processed_test_df.empty:
            processed_test_df.to_csv(test_data_path_out, index=False)
        else:
            test_data_path_out = None # Ensure it's None if not saved
        print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Data processing complete. Processed files saved to: {results_path}")
    else:
        print(f"[Pipeline Stage - Chromosome: {experiment_name.replace('_', ',')}] Data processing complete. Processed data NOT saved to disk (GA mode).")
        # Paths will be None, but dataframes are returned

    # analyze_target_correlations was commented out, so not re-adding here for now.
    
    # Return DataFrames directly if not saving, otherwise paths
    if not save_processed_data:
        return processed_train_df, processed_test_df, research_path
    else:
        return train_data_path_out, test_data_path_out, research_path 