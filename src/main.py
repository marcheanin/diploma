import warnings
# Attempt to import SklearnFutureWarning, fallback to built-in FutureWarning
try:
    from sklearn.exceptions import FutureWarning as SklearnFutureWarning
except ImportError:
    # For older scikit-learn versions where sklearn.exceptions.FutureWarning might not exist
    SklearnFutureWarning = FutureWarning 

# Игнорировать конкретные FutureWarning от sklearn, которые вы видите
warnings.filterwarnings("ignore", category=SklearnFutureWarning, message=".*`BaseEstimator._check_n_features` is deprecated.*")
warnings.filterwarnings("ignore", category=SklearnFutureWarning, message=".*`BaseEstimator._check_feature_names` is deprecated.*")
warnings.filterwarnings("ignore", category=SklearnFutureWarning, message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")
# Можно добавить и другие, если появятся, или сделать фильтр более общим, но это менее рекомендуется

from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.outlier_remover import OutlierRemover
from preprocessing.resampler import Resampler
from modeling.model_trainer import ModelTrainer
import os
import pandas as pd
from utils.data_analysis import (
    # print_missing_summary, # Will reduce verbosity
    # print_numeric_stats, # Will reduce verbosity
    # print_target_distribution, # Will reduce verbosity
    # analyze_target_correlations, # Will reduce verbosity
    save_model_results
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Chromosome Definition and Decoding ---
IMPUTATION_MAP = {0: 'knn', 1: 'median', 2: 'missforest'}
OUTLIER_MAP = {0: 'none', 1: 'isolation_forest', 2: 'iqr'}
RESAMPLING_MAP = {0: 'none', 1: 'oversample', 2: 'smote', 3: 'adasyn'} # Assuming 1 is ROS
ENCODING_MAP = {0: 'onehot', 1: 'label', 2: 'lsa', 3: 'word2vec'}
SCALING_MAP = {0: 'none', 1: 'standard', 2: 'minmax'}
MODEL_MAP = {0: 'logistic_regression', 1: 'random_forest', 2: 'gradient_boosting', 3: 'neural_network'}

def decode_and_log_chromosome(chromosome):
    """Decodes a chromosome and prints its meaning."""
    print("\n--- Chromosome Definition ---")
    if len(chromosome) != 6:
        print("Error: Chromosome must have 6 genes.")
        return None

    decoded = {
        'imputation': IMPUTATION_MAP.get(chromosome[0], "Unknown Imputation"),
        'outlier_removal': OUTLIER_MAP.get(chromosome[1], "Unknown Outlier Method"),
        'resampling': RESAMPLING_MAP.get(chromosome[2], "Unknown Resampling"),
        'encoding': ENCODING_MAP.get(chromosome[3], "Unknown Encoding"),
        'scaling': SCALING_MAP.get(chromosome[4], "Unknown Scaling"),
        'model': MODEL_MAP.get(chromosome[5], "Unknown Model")
    }

    print(f"Chromosome: {chromosome}")
    print(f"  Gene 0 (Imputation): {chromosome[0]} -> {decoded['imputation']}")
    print(f"  Gene 1 (Outlier Removal): {chromosome[1]} -> {decoded['outlier_removal']}")
    print(f"  Gene 2 (Resampling): {chromosome[2]} -> {decoded['resampling']}")
    print(f"  Gene 3 (Encoding): {chromosome[3]} -> {decoded['encoding']}")
    print(f"  Gene 4 (Scaling): {chromosome[4]} -> {decoded['scaling']}")
    print(f"  Gene 5 (Model): {chromosome[5]} -> {decoded['model']}")
    print("-----------------------------\n")
    return decoded
# --- End Chromosome Definition ---

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
                 outlier_method='isolation_forest', # New parameter for outlier removal
                 encoding_method='label',
                 resampling_method='none',
                 scaling_method='none',
                 imputation_kwargs=None, 
                 outlier_kwargs=None, # New kwargs for outlier remover
                 encoding_kwargs=None,
                 scaling_kwargs=None):
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
        outlier_kwargs: dict, keyword arguments for the outlier removal method (e.g., {'contamination': 0.05, 'iqr_multiplier': 1.5})
        encoding_kwargs: dict, keyword arguments for the encoding method
        scaling_kwargs: dict, keyword arguments for the scaling method
        
    Returns:
        tuple: (train_data_path, test_data_path, research_path)
    """
    imputation_kwargs = imputation_kwargs or {}
    outlier_kwargs = outlier_kwargs or {} # Initialize new kwargs
    encoding_kwargs = encoding_kwargs or {}
    scaling_kwargs = scaling_kwargs or {}
    
    dataset_name = get_dataset_name(train_path)
    
    experiment_name_parts = [imputation_method, outlier_method, encoding_method, resampling_method]
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
    
    # if target_column in train_data.columns:
    #     # print("\n=== Target Distribution in Training Data ===")
    #     target_counts = train_data[target_column].value_counts(normalize=True)
    #     # print(target_counts)
    #     if target_counts.min() < 0.1:
    #         print("Warning: Significant class imbalance detected.")

    # print("\n=== Missing values statistics before imputation ===")
    # print_missing_summary(train_data, "train dataset")
    # if test_data is not None:
    #     print_missing_summary(test_data, "test dataset")

    # Preprocess data
    preprocessor = DataPreprocessor()
    
    # 1. Impute missing values
    # print(f"\nUsing {imputation_method} for imputation with args: {imputation_kwargs}...")
    train_data = preprocessor.impute(train_data, method=imputation_method, **imputation_kwargs)
    if test_data is not None and not test_data.empty:
        test_data = preprocessor.impute(test_data, method=imputation_method, **imputation_kwargs)

    # --- Attempt to drop 'ID' or 'id' column before outlier removal ---
    # This ensures outlier removal is not affected by a meaningless ID column.
    potential_id_columns_process = [col for col in ['ID', 'id'] if col in train_data.columns and col != target_column]
    if potential_id_columns_process:
        print(f"Dropping ID column(s) from train_data: {potential_id_columns_process} before outlier removal.")
        train_data = train_data.drop(columns=potential_id_columns_process)
        if test_data is not None and not test_data.empty:
            # Also drop from test_data if it exists, ignoring errors if column not found
            test_data = test_data.drop(columns=potential_id_columns_process, errors='ignore')
    # --- End ID drop --- 

    # print("\n=== Missing values statistics after imputation ===")
    # print_missing_summary(train_data, "train dataset")
    # if test_data is not None:
    #     print_missing_summary(test_data, "test dataset")

    # 2. Outlier Removal (on imputed train_data)
    # print(f"\n=== Outlier Removal (method: {outlier_method}) ===")
    if outlier_method != 'none':
        train_data_before_outliers = train_data.copy()
        try:
            # Pass specific outlier_kwargs to OutlierRemover
            # Default contamination for IsolationForest and multiplier for IQR are in OutlierRemover class itself
            # but can be overridden via outlier_kwargs if needed.
            remover = OutlierRemover(method=outlier_method, **outlier_kwargs) 
            
            # Outlier removal should generally not use the target column for detection
            features_for_removal = train_data.drop(columns=[target_column], errors='ignore')
            target_series_for_reconstruction = None
            if target_column in train_data.columns:
                target_series_for_reconstruction = train_data[target_column]

            cleaned_features = remover.remove_outliers(features_for_removal)
            
            if target_series_for_reconstruction is not None:
                aligned_target = target_series_for_reconstruction.loc[cleaned_features.index]
                train_data = pd.concat([cleaned_features, aligned_target.rename(target_column)], axis=1)
            else:
                train_data = cleaned_features
            # print(f"Train dataset size after outlier removal: {train_data.shape}")
        except Exception as e:
            print(f"Could not remove outliers (method: {outlier_method}): {e}")
            print("Skipping outlier removal or reverting to data before attempt.")
            train_data = train_data_before_outliers
    else:
        print("Outlier removal skipped as method is 'none'.")
    # --- End Outlier Removal ---

    # print_numeric_stats(train_data, "before encoding (after imputation, outlier removal)")

    # 3. Encode categorical features using encoding_kwargs
    # train_data is now after imputation and outlier removal
    # test_data is after imputation
    train_data = preprocessor.encode(train_data, method=encoding_method, target_col=target_column, **encoding_kwargs)
    if test_data is not None and not test_data.empty:
        test_data = preprocessor.encode(test_data, method=encoding_method, target_col=target_column, **encoding_kwargs)
    
    # print_numeric_stats(train_data, "after encoding")

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
            # print("Resampling applied to training data.")
        else:
            print(f"Warning: Target column '{target_column}' not found in encoded train data. Skipping resampling.")
    else:
        print("Resampling method is 'none'. Skipping resampling step.")
    # --- Конец Ресемплинга --- #

    # 5. Optional Feature Scaling
    # Scaler is fit on resampled training features and applied to resampled training features and test features.
    if scaling_method != 'none' and scaling_method in ['standard', 'minmax']:
        # print(f"\n=== Applying {scaling_method} scaling ===")
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
                    # print(f"Training data features scaled using {scaling_method}.")

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
                            # print(f"Subset of test data features scaled using {scaling_method}.")
                        else: # Columns match perfectly
                            scaled_X_test_values = scaler_instance.transform(X_test_features_original[X_train_features.columns]) # Ensure order
                            scaled_X_test_df = pd.DataFrame(scaled_X_test_values, columns=X_train_features.columns, index=X_test_features_original.index)

                            if y_test_target_original is not None:
                                test_data = pd.concat([scaled_X_test_df, y_test_target_original], axis=1)
                            else:
                                test_data = scaled_X_test_df
                            # print(f"Test data features scaled using {scaling_method}.")
                except Exception as e:
                    print(f"Error during scaling: {e}. Skipping scaling for this dataset part.")
    elif scaling_method != 'none':
        print(f"Warning: Unknown scaling method '{scaling_method}' provided. Scaling will be skipped.")
    # --- End Scaling ---

    # Save processed data
    # Update output_suffix to include outlier_method and scaling_method
    filename_suffix_parts = [imputation_method, outlier_method, encoding_method, resampling_method]
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
    # print("\n=== Analyzing feature correlations with target variable ===")
    # train_data now holds the fully processed training data
    if target_column in train_data.columns:
        # Добавим проверку типа перед вызовом
        if pd.api.types.is_numeric_dtype(train_data[target_column]):
            # analyze_target_correlations(train_data, target_column, research_path)
            pass
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

def train_model(train_data_path, test_data_path, target_column, research_path, model_type='random_forest', plot_learning_curves=True):
    """
    Train and evaluate model on processed data.
    
    Args:
        train_data_path: str, path to processed train data CSV file
        test_data_path: str, path to processed test data CSV file
        target_column: str, name of target column
        research_path: str, base path to save research results for this data configuration
        model_type: str, type of model to train ('random_forest', 'logistic_regression', 'gradient_boosting')
        plot_learning_curves: bool, whether to plot learning curves
    
    Returns:
        ModelTrainer: trained model instance
    """
    # Create a specific directory for this model's results
    model_specific_research_path = os.path.join(research_path, model_type)
    os.makedirs(model_specific_research_path, exist_ok=True)
    # print(f"\n=== Training model: {model_type} ===") # Reduced verbosity
    print(f"\n=== Training model: {model_type} ===")
    print(f"Results will be saved in: {model_specific_research_path}")

    # print("\n=== Loading processed data ===") # Reduced verbosity
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
         test_data = None 

    # print(f"Loaded train data shape: {train_data.shape}") # Reduced verbosity
    # if test_data is not None:
    #     print(f"Loaded test data shape: {test_data.shape}") # Reduced verbosity
    # else:
    #      print("No test data loaded.") # Reduced verbosity
    
    # print("\n=== Training and evaluating model ===") # Reduced verbosity
    model_trainer = ModelTrainer(model_type=model_type)
    
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
            output_path=model_specific_research_path, # Use model specific path
            plot_learning_curves=plot_learning_curves
        )
    else:
        # Proceed with existing logic if test_data is available
        metrics, feature_importance = model_trainer.train(
            train_data, test_data, target_column, 
            output_path=model_specific_research_path, # Use model specific path
            plot_learning_curves=plot_learning_curves
        )
    
    # Print metrics
    print(f"\nTraining Metrics for {model_type}:")
    print(f"Accuracy: {metrics['train']['accuracy']:.4f}")
    print(f"F1 Score: {metrics['train']['f1']:.4f}")
    
    # Determine which evaluation metrics we have (test or validation)
    eval_key = 'test' if 'test' in metrics else 'validation' if 'validation' in metrics else None
    if eval_key:
        print(f"\n{eval_key.title()} Metrics for {model_type}:")
        print(f"Accuracy: {metrics[eval_key]['accuracy']:.4f}")
        print(f"F1 Score: {metrics[eval_key]['f1']:.4f}")
    else:
        print("\nNo evaluation metrics available (test/validation).")
    
    # Save model results
    save_model_results(metrics, feature_importance, model_specific_research_path) # Use model specific path
    
    return model_trainer

def main():
    # print("\n=== Processing Diabets Dataset ===")
    # data_path = "datasets/diabetes.csv"
    # target_column = "Outcome"

    print("\n=== Processing kredit card dataset ===")
    train_path = "datasets/UCI_Credit_Card.csv"
    target_column = "default.payment.next.month"
    # n_lsa_components = 20 
    # knn_imputation_args = {'n_neighbors': 5}
    generate_learning_curves = False

    # --- Example Chromosome and its interpretation ---
    # example_chromosome = [0, 1, 3, 2, 1, 1] # Example: KNN, IsolationForest, ADASYN, LSA, Standard, RandomForest
    example_chromosome = [1, 1, 1, 1, 1, 3]
    decoded_pipeline = decode_and_log_chromosome(example_chromosome)

    if decoded_pipeline is None:
        print("Error decoding chromosome. Exiting.")
        return

    # Use methods from the decoded chromosome example
    current_imputation_method = decoded_pipeline['imputation'] 
    current_outlier_method = decoded_pipeline['outlier_removal']
    current_encoding_method = decoded_pipeline['encoding']
    current_resampling_method = decoded_pipeline['resampling']
    current_scaling_method = decoded_pipeline['scaling']
    # Model type from chromosome will be used in the loop

    print(f"\n--- Running Pipeline with Decoded Chromosome ---")
    print(f"Imputation: {current_imputation_method}, Outliers: {current_outlier_method}, Encoding: {current_encoding_method}, Resampling: {current_resampling_method}, Scaling: {current_scaling_method}")

    try:
        train_processed_path, test_processed_path, research_base_path = process_data(
            train_path, None,  target_column,
            imputation_method=current_imputation_method,
            outlier_method=current_outlier_method, 
            encoding_method=current_encoding_method,
            resampling_method=current_resampling_method, 
            scaling_method=current_scaling_method,
        )
        
        models_to_run = [
            MODEL_MAP[example_chromosome[5]], # Model from chromosome example
            # You can add other models here to test them on the same processed data
            # e.g., 'logistic_regression', 'gradient_boosting', 'neural_network'
        ]
        if decoded_pipeline['model'] not in models_to_run: # Ensure the chromosome model is run
             models_to_run.insert(0, decoded_pipeline['model'])
        
        # Remove duplicates if any
        models_to_run = sorted(list(set(models_to_run)))


        for model_name in models_to_run:
            print(f"\n=== Training {model_name} model on processed data ===")
            train_model(
                train_processed_path, test_processed_path, target_column,
                research_base_path, 
                model_type=model_name,
                plot_learning_curves=generate_learning_curves
            )

    except Exception as e:
        print(f"Error processing/training pipeline: {e}")

if __name__ == "__main__":
    main() 