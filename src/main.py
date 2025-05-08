from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.outlier_remover import OutlierRemover
from preprocessing.resampler import Resampler
from modeling.model_trainer import ModelTrainer
import os
import pandas as pd
from utils.data_analysis import (
    print_missing_summary,
    print_numeric_stats,
    print_target_distribution,
    analyze_target_correlations,
    save_model_results
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
                 imputation_method='knn', encoding_method='label',
                 resampling_method='none',
                 scaling_method='none',
                 imputation_kwargs=None, encoding_kwargs=None,
                 scaling_kwargs=None):
    """
    Process data using specified imputation, encoding, and resampling methods.
    
    Args:
        train_path: str, path to training data
        test_path: str, path to test data
        target_column: str, name of target column
        imputation_method: str, 'knn' or 'missforest'
        encoding_method: str, 'onehot', 'label', 'ordinal', 'leaveoneout', 'lsa'
        resampling_method: str, 'none', 'oversample', 'undersample', 'smote', 'adasyn'
        scaling_method: str, 'none', 'standard', 'minmax'
        imputation_kwargs: dict, keyword arguments for the imputation method (e.g., {'n_neighbors': 5})
        encoding_kwargs: dict, keyword arguments for the encoding method (e.g., {'sigma': 0.05, 'n_components': 10})
        scaling_kwargs: dict, keyword arguments for the scaling method (e.g., {'feature_range': (0, 1)})
        
    Returns:
        tuple: (train_data_path, test_data_path, research_path)
    """
    # Инициализируем пустые словари, если аргументы не переданы
    imputation_kwargs = imputation_kwargs or {}
    encoding_kwargs = encoding_kwargs or {}
    scaling_kwargs = scaling_kwargs or {}
    
    # Get dataset name and create paths
    dataset_name = get_dataset_name(train_path)
    
    # Update experiment_name to include scaling_method
    experiment_name_parts = [imputation_method, encoding_method, resampling_method]
    if scaling_method != 'none':
        experiment_name_parts.append(scaling_method)
    experiment_name = "_".join(experiment_name_parts)
    
    results_path = os.path.join('results', dataset_name, experiment_name)
    research_path = os.path.join("research", dataset_name, experiment_name)
    
    # Create necessary directories
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(research_path, exist_ok=True)

    # Load data
    loader = DataLoader(train_path=train_path, test_path=test_path)
    train_data, test_data = loader.load_data()
    
    print("Train dataset size:", train_data.shape)
    if test_data is not None:
        print("Test dataset size:", test_data.shape)
        
    # --- Проверка дисбаланса классов --- #
    if target_column in train_data.columns:
        print("\n=== Target Distribution in Training Data ===")
        target_counts = train_data[target_column].value_counts(normalize=True)
        print(target_counts)
        # Дополнительно: Определим, есть ли сильный дисбаланс (например, один класс < 10%)
        if target_counts.min() < 0.1:
            print("Warning: Significant class imbalance detected.")
    else:
        print(f"Warning: Target column '{target_column}' not found, cannot check distribution.")
    # --- Конец проверки --- #

    # Print missing values statistics
    print("\n=== Missing values statistics before imputation ===")
    print_missing_summary(train_data, "train dataset")
    if test_data is not None:
        print_missing_summary(test_data, "test dataset")

    # Preprocess data
    preprocessor = DataPreprocessor()
    
    # 1. Impute missing values using imputation_kwargs
    print(f"\nUsing {imputation_method} for imputation with args: {imputation_kwargs}...")
    train_data = preprocessor.impute(train_data, method=imputation_method, **imputation_kwargs) 
    if test_data is not None:
        test_data = preprocessor.impute(test_data, method=imputation_method, **imputation_kwargs)

    # Print missing values statistics after imputation
    print("\n=== Missing values statistics after imputation ===")
    print_missing_summary(train_data, "train dataset")
    if test_data is not None:
        print_missing_summary(test_data, "test dataset")

    # 2. Outlier Removal (on imputed train_data)
    print("\n=== Outlier Removal (on imputed train data) ===")
    train_data_before_outliers = train_data.copy() # Keep a copy in case outlier removal fails
    try:
        outlier_remover = OutlierRemover(contamination=0.05) # Contamination is hardcoded as in original
        
        # Features for outlier removal: current train_data without target
        features_for_removal = train_data.drop(columns=[target_column], errors='ignore')
        
        # Store original target series to reattach after filtering by index
        target_series_for_reconstruction = None
        if target_column in train_data.columns:
            target_series_for_reconstruction = train_data[target_column]

        cleaned_features = outlier_remover.remove_outliers(features_for_removal)
        
        # Reconstruct train_data by attaching the target to the cleaned_features
        if target_series_for_reconstruction is not None and target_column in train_data.columns:
            # Align using the index of cleaned_features
            aligned_target = target_series_for_reconstruction.loc[cleaned_features.index]
            train_data = pd.concat([cleaned_features, aligned_target.rename(target_column)], axis=1)
        else:
            train_data = cleaned_features # If no target or target was not in train_data initially
            
        print(f"Train dataset size after outlier removal: {train_data.shape}")
    except Exception as e:
         print(f"Could not remove outliers (imputation: {imputation_method}): {e}")
         print("Skipping outlier removal.")
         train_data = train_data_before_outliers # Revert to data before outlier removal attempt
    # --- End Outlier Removal ---

    # Print numeric statistics before encoding (after imputation and outlier removal)
    print_numeric_stats(train_data, "before encoding (after imputation, outlier removal)")

    # 3. Encode categorical features using encoding_kwargs
    # train_data is now after imputation and outlier removal
    # test_data is after imputation
    train_data = preprocessor.encode(train_data, method=encoding_method, target_col=target_column, **encoding_kwargs)
    if test_data is not None:
        test_data = preprocessor.encode(test_data, method=encoding_method, target_col=target_column, **encoding_kwargs)
    
    # Print numeric statistics after encoding
    print_numeric_stats(train_data, "after encoding")

    # 4. Resampling (only for train_data, after imputation, outlier removal, and encoding)
    if resampling_method != 'none':
        if target_column in train_data.columns:
            # Убедимся, что целевая колонка числовая для imblearn
            if not pd.api.types.is_numeric_dtype(train_data[target_column]):
                print(f"Warning: Target column '{target_column}' is not numeric. Applying LabelEncoding before resampling.")
                train_data = preprocessor._label_encode_target(train_data, target_column)
            
            # Разделяем X и y
            X_train_encoded = train_data.drop(columns=[target_column])
            y_train_encoded = train_data[target_column]
            
            # Применяем ресемплинг
            resampler = Resampler(method=resampling_method)
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_encoded, y_train_encoded)
            
            # Собираем обратно в DataFrame
            train_data = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train_encoded.columns), 
                                    pd.Series(y_train_resampled, name=target_column)], axis=1)
            print("Resampling applied to training data.")
        else:
            print(f"Warning: Target column '{target_column}' not found in encoded train data. Skipping resampling.")
    else:
        print("Resampling method is 'none'. Skipping resampling step.")
    # --- Конец Ресемплинга --- #

    # 5. Optional Feature Scaling
    # Scaler is fit on resampled training features and applied to resampled training features and test features.
    if scaling_method != 'none' and scaling_method in ['standard', 'minmax']:
        print(f"\n=== Applying {scaling_method} scaling ===")
        
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
                    # Fit and transform training features
                    scaled_X_train_values = scaler_instance.fit_transform(X_train_features)
                    scaled_X_train_df = pd.DataFrame(scaled_X_train_values, columns=X_train_features.columns, index=X_train_features.index)
                    
                    train_data = pd.concat([scaled_X_train_df, y_train_target], axis=1)
                    print(f"Training data features scaled using {scaling_method}.")

                    if test_data is not None:
                        X_test_features_original = test_data.drop(columns=[target_column], errors='ignore')
                        y_test_target_original = test_data[target_column] if target_column in test_data.columns else None
                        
                        # Align test columns with train columns scaler was fit on
                        # Only try to scale columns that were present during fit
                        cols_to_scale_in_test = [col for col in X_train_features.columns if col in X_test_features_original.columns]
                        
                        if not cols_to_scale_in_test:
                             print("Warning: No common features to scale found in test data matching training data features. Test data remains unscaled.")
                        elif len(cols_to_scale_in_test) != len(X_train_features.columns):
                            print("Warning: Test data feature set does not exactly match training data feature set for scaling. Scaling subset of columns. Ensure this is intended.")
                            # Potentially create a subset of X_test_features to scale
                            X_test_subset_to_scale = X_test_features_original[cols_to_scale_in_test]
                            scaled_X_test_subset_values = scaler_instance.transform(X_test_subset_to_scale)
                            scaled_X_test_subset_df = pd.DataFrame(scaled_X_test_subset_values, columns=cols_to_scale_in_test, index=X_test_subset_to_scale.index)
                            
                            # Update only the scaled columns in a copy of original test features
                            X_test_features_updated = X_test_features_original.copy()
                            for col in cols_to_scale_in_test:
                                X_test_features_updated[col] = scaled_X_test_subset_df[col]

                            if y_test_target_original is not None:
                                test_data = pd.concat([X_test_features_updated, y_test_target_original], axis=1)
                            else:
                                test_data = X_test_features_updated
                            print(f"Subset of test data features scaled using {scaling_method}.")
                        else: # Columns match perfectly
                            scaled_X_test_values = scaler_instance.transform(X_test_features_original[X_train_features.columns]) # Ensure order
                            scaled_X_test_df = pd.DataFrame(scaled_X_test_values, columns=X_train_features.columns, index=X_test_features_original.index)

                            if y_test_target_original is not None:
                                test_data = pd.concat([scaled_X_test_df, y_test_target_original], axis=1)
                            else:
                                test_data = scaled_X_test_df
                            print(f"Test data features scaled using {scaling_method}.")
                except Exception as e:
                    print(f"Error during scaling: {e}. Skipping scaling for this dataset part.")
    elif scaling_method != 'none':
        print(f"Warning: Unknown scaling method '{scaling_method}' provided. Scaling will be skipped.")
    # --- End Scaling ---

    # Save processed data
    # Update output_suffix to include scaling_method
    filename_suffix_parts = [imputation_method, encoding_method, resampling_method]
    if scaling_method != 'none':
        filename_suffix_parts.append(scaling_method)
    output_suffix = f"_processed_{'_'.join(filename_suffix_parts)}"
    
    train_data_path = os.path.join(results_path, f'train{output_suffix}.csv')
    test_data_path = os.path.join(results_path, f'test{output_suffix}.csv')
    
    train_data.to_csv(train_data_path, index=False) # Use the final train_data
    if test_data is not None:
        test_data.to_csv(test_data_path, index=False)
    else: # Если test_data не было (например, при сплите), создаем пустой файл пути
        test_data_path = None

    # Analyze and save feature correlations with target
    print("\n=== Analyzing feature correlations with target variable ===")
    # train_data now holds the fully processed training data
    if target_column in train_data.columns:
        # Добавим проверку типа перед вызовом
        if pd.api.types.is_numeric_dtype(train_data[target_column]):
            analyze_target_correlations(train_data, target_column, research_path)
        else:
            print(f"Target column '{target_column}' is not numeric. Skipping correlation analysis.")
            # Опционально: можно попытаться закодировать цель здесь, если нужно
            # try:
            #     temp_train_data = preprocessor._label_encode_target(train_data.copy(), target_column)
            #     analyze_target_correlations(temp_train_data, target_column, research_path)
            # except Exception as corr_err:
            #     print(f"Could not encode target for correlation analysis: {corr_err}")
    else:
         print(f"Target column '{target_column}' not found in cleaned train data. Skipping correlation analysis.")
    
    return train_data_path, test_data_path, research_path

def train_model(train_data_path, test_data_path, target_column, research_path, plot_learning_curves=True):
    """
    Train and evaluate model on processed data.
    
    Args:
        train_data_path: str, path to processed train data CSV file
        test_data_path: str, path to processed test data CSV file
        target_column: str, name of target column
        research_path: str, path to save research results
        plot_learning_curves: bool, whether to plot learning curves
    
    Returns:
        ModelTrainer: trained model instance
    """
    print("\n=== Loading processed data ===")
    try:
        train_data = pd.read_csv(train_data_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_path}")
        return None
    
    # Handle case where test data might not exist (e.g., if split resulted in no test set)
    if test_data_path and os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
    else:
         print(f"Warning: Test data file not found or not provided ({test_data_path}). Proceeding without test data evaluation.")
         # Create a dummy empty DataFrame for test_data to avoid errors later
         # We need at least the columns expected by the model if predictions are needed
         # Best approach might be to handle this in ModelTrainer.train based on has_test_target
         # For now, let's pass None, ModelTrainer should handle it.
         # test_data = pd.DataFrame(columns=train_data.columns.drop(target_column)) # Example placeholder
         test_data = None # Let ModelTrainer handle None test_data

    print(f"Loaded train data shape: {train_data.shape}")
    if test_data is not None:
        print(f"Loaded test data shape: {test_data.shape}")
    else:
         print("No test data loaded.")
    
    print("\n=== Training and evaluating model ===")
    model_trainer = ModelTrainer()
    
    # Check if test_data is None and handle appropriately
    if test_data is None:
        # If test data is None, we cannot use it for evaluation.
        # ModelTrainer.train should handle this by splitting train data if target_column is present in test_data (which it won't be)
        # or by just training and providing train metrics.
        # Ensure target column exists in train_data for training
        if target_column not in train_data.columns:
            print(f"Error: Target column '{target_column}' not found in training data.")
            return None
        # Create a dummy test_data for the function call signature, ModelTrainer will ignore it
        # It needs columns to potentially make predictions later if target is missing
        test_data_dummy = pd.DataFrame(columns=train_data.columns.drop(target_column)) 
        metrics, feature_importance = model_trainer.train(
            train_data, test_data_dummy, target_column, 
            output_path=research_path, 
            plot_learning_curves=plot_learning_curves
        )
    else:
        # Proceed with existing logic if test_data is available
        metrics, feature_importance = model_trainer.train(
            train_data, test_data, target_column, 
            output_path=research_path, 
            plot_learning_curves=plot_learning_curves
        )
    
    # Print metrics
    print("\nTraining Metrics:")
    print(f"Accuracy: {metrics['train']['accuracy']:.4f}")
    print(f"F1 Score: {metrics['train']['f1']:.4f}")
    
    # Determine which evaluation metrics we have (test or validation)
    eval_key = 'test' if 'test' in metrics else 'validation' if 'validation' in metrics else None
    if eval_key:
        print(f"\n{eval_key.title()} Metrics:")
        print(f"Accuracy: {metrics[eval_key]['accuracy']:.4f}")
        print(f"F1 Score: {metrics[eval_key]['f1']:.4f}")
    else:
        print("\nNo evaluation metrics available (test/validation).")
    
    # Save model results
    save_model_results(metrics, feature_importance, research_path)
    
    return model_trainer

def main():
    # print("\n=== Processing Diabets Dataset ===")
    # data_path = "datasets/diabetes.csv"
    # target_column = "Outcome"

    print("\n=== Processing Credit Score Dataset ===")
    train_path = "datasets/credit-score-classification/train.csv"
    test_path = "datasets/credit-score-classification/test.csv"
    target_column = "Credit_Score"
    n_lsa_components = 20 # Задаем количество компонент для LSA
    knn_imputation_args = {'n_neighbors': 5}
    generate_learning_curves = False# <<--- Управляем построением кривых здесь

    # --- Запуск пайплайнов с разными энкодерами --- #

   

    # 3. KNN Imputation + LSA Encoding + No Resampling
    try:
        print("\n=== Processing data with KNN + LSA Encoding + No Resampling ===")
        train_lsa_path, test_lsa_path, research_lsa_path = process_data(
            train_path, test_path, target_column,
            imputation_method='median',
            encoding_method='embedding',
            resampling_method='smote', 
            scaling_method='standard',
            encoding_kwargs={'n_components': n_lsa_components, 'embedding_method': 'word2vec'}
        )
        print("\n=== Training model on LSA encoded data (No Resampling) ===")
        train_model(
            train_lsa_path, test_lsa_path, target_column,
            research_lsa_path, plot_learning_curves=generate_learning_curves
        )
    except Exception as e:
        print(f"Error processing/training with LSA Encoding: {e}")

    # # Добавьте здесь другие комбинации, если нужно, например:
    # # - MissForest + Ordinal
    # # - MissForest + Target
    # # - MissForest + LSA
    # # - KNN + Label
    # # - KNN + OneHot

if __name__ == "__main__":
    main() 