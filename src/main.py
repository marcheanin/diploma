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
                 imputation_kwargs=None, encoding_kwargs=None):
    """
    Process data using specified imputation, encoding, and resampling methods.
    
    Args:
        train_path: str, path to training data
        test_path: str, path to test data
        target_column: str, name of target column
        imputation_method: str, 'knn' or 'missforest'
        encoding_method: str, 'onehot', 'label', 'ordinal', 'leaveoneout', 'lsa'
        resampling_method: str, 'none', 'oversample', 'undersample', 'smote', 'adasyn'
        imputation_kwargs: dict, keyword arguments for the imputation method (e.g., {'n_neighbors': 5})
        encoding_kwargs: dict, keyword arguments for the encoding method (e.g., {'sigma': 0.05, 'n_components': 10})
        
    Returns:
        tuple: (train_data_path, test_data_path, research_path)
    """
    # Инициализируем пустые словари, если аргументы не переданы
    imputation_kwargs = imputation_kwargs or {}
    encoding_kwargs = encoding_kwargs or {}
    
    # Get dataset name and create paths
    dataset_name = get_dataset_name(train_path)
    # Добавляем метод кодирования и метод ресемплинга в путь для research и results, чтобы различать эксперименты
    experiment_name = f"{imputation_method}_{encoding_method}_{resampling_method}"
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

    # Print target distribution (существующий вызов, можно оставить или убрать, если проверка выше достаточна)
    # print_target_distribution(train_data, target_col=target_column)

    # Print missing values statistics
    print("\n=== Missing values statistics before imputation ===")
    print_missing_summary(train_data, "train dataset")
    if test_data is not None:
        print_missing_summary(test_data, "test dataset")

    # Preprocess data
    preprocessor = DataPreprocessor()
    
    # Impute missing values using imputation_kwargs
    print(f"\nUsing {imputation_method} for imputation with args: {imputation_kwargs}...")
    train_data = preprocessor.impute(train_data, method=imputation_method, **imputation_kwargs) 
    if test_data is not None:
        test_data = preprocessor.impute(test_data, method=imputation_method, **imputation_kwargs)

    # Print missing values statistics after imputation
    print("\n=== Missing values statistics after imputation ===")
    print_missing_summary(train_data, "train dataset")
    if test_data is not None:
        print_missing_summary(test_data, "test dataset")

    # Print numeric statistics before encoding
    print_numeric_stats(train_data, "before encoding")

    # Encode categorical features using encoding_kwargs
    train_data = preprocessor.encode(train_data, method=encoding_method, target_col=target_column, **encoding_kwargs)
    if test_data is not None:
        test_data = preprocessor.encode(test_data, method=encoding_method, target_col=target_column, **encoding_kwargs)
    
    # Print numeric statistics after encoding
    # Примечание: Статистика будет включать не закодированную целевую переменную, если она осталась
    print_numeric_stats(train_data, "after encoding")

    # --- Ресемплинг (только для train_data) --- #
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

    # --- Удаление выбросов (применяется к train_data, которая может быть ресемплирована) --- #
    print("\n=== Outlier Removal (on potentially resampled train data) ===")
    try:
        outlier_remover = OutlierRemover(contamination=0.05)
        # OutlierRemover работает с числовыми данными, к этому моменту все должно быть числовым
        train_data_clean = outlier_remover.remove_outliers(train_data.drop(columns=[target_column], errors='ignore'))
        # Добавляем целевую колонку обратно, если она была удалена для remove_outliers
        if target_column in train_data.columns:
             train_data_clean[target_column] = train_data.loc[train_data_clean.index, target_column]
        print(f"Train dataset size after outlier removal: {train_data_clean.shape}")
    except Exception as e:
         print(f"Could not remove outliers (encoding: {encoding_method}, resampling: {resampling_method}): {e}")
         print("Skipping outlier removal.")
         train_data_clean = train_data # Используем данные без удаления выбросов
    # --- Конец Удаления выбросов --- #

    # Save processed data
    # Используем experiment_name в имени файла
    output_suffix = f"_processed_{encoding_method}_{resampling_method}" 
    train_data_path = os.path.join(results_path, f'train{output_suffix}.csv')
    test_data_path = os.path.join(results_path, f'test{output_suffix}.csv')
    
    train_data_clean.to_csv(train_data_path, index=False)
    if test_data is not None:
        test_data.to_csv(test_data_path, index=False)
    else: # Если test_data не было (например, при сплите), создаем пустой файл пути
        test_data_path = None

    # Analyze and save feature correlations with target
    print("\n=== Analyzing feature correlations with target variable ===")
    # Примечание: analyze_target_correlations теперь может выдать ошибку,
    # если целевая переменная не является числовой. 
    # Нужно либо изменить эту функцию, либо убедиться, что цель числовая ДО ее вызова.
    if target_column in train_data_clean.columns:
        # Добавим проверку типа перед вызовом
        if pd.api.types.is_numeric_dtype(train_data_clean[target_column]):
            analyze_target_correlations(train_data_clean, target_column, research_path)
        else:
            print(f"Target column '{target_column}' is not numeric. Skipping correlation analysis.")
            # Опционально: можно попытаться закодировать цель здесь, если нужно
            # try:
            #     temp_train_data = preprocessor._label_encode_target(train_data_clean.copy(), target_column)
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
            resampling_method='smote', # <---- No Resampling
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