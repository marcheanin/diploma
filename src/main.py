from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.outlier_remover import OutlierRemover
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
                 imputation_kwargs=None, encoding_kwargs=None):
    """
    Process data using specified imputation and encoding methods.
    
    Args:
        train_path: str, path to training data
        test_path: str, path to test data
        target_column: str, name of target column
        imputation_method: str, 'knn' or 'missforest'
        encoding_method: str, 'onehot', 'label', 'ordinal', 'leaveoneout', 'lsa'
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
    # Добавляем метод кодирования в путь для research и results, чтобы различать эксперименты
    experiment_name = f"{imputation_method}_{encoding_method}"
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

    # Print target distribution
    print_target_distribution(train_data, target_col=target_column)

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

    # Remove outliers from train data
    # Примечание: Если целевая переменная не числовая, OutlierRemover ее проигнорирует
    # Если она числовая, но не закодирована (0/1), OutlierRemover может ее обработать
    try:
        outlier_remover = OutlierRemover(contamination=0.05)
        # Ensure target column exists before passing to remove_outliers if it uses it implicitly
        # Currently remove_outliers only uses numeric features by default
        train_data_clean = outlier_remover.remove_outliers(train_data)
        print("\nTrain dataset size after imputation, encoding and outlier removal:", train_data_clean.shape)
    except Exception as e:
         print(f"Could not remove outliers (method: {encoding_method}): {e}")
         print("Skipping outlier removal.")
         train_data_clean = train_data # Используем данные без удаления выбросов

    # Save processed data
    # Используем experiment_name в имени файла
    output_suffix = f"_processed_{encoding_method}" 
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

def train_model(train_data_path, test_data_path, target_column, research_path):
    """
    Train and evaluate model on processed data.
    
    Args:
        train_data_path: str, path to processed train data CSV file
        test_data_path: str, path to processed test data CSV file
        target_column: str, name of target column
        research_path: str, path to save research results
    
    Returns:
        ModelTrainer: trained model instance
    """
    print("\n=== Loading processed data ===")
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    print(f"Loaded train data shape: {train_data.shape}")
    print(f"Loaded test data shape: {test_data.shape}")
    
    print("\n=== Training and evaluating model ===")
    model_trainer = ModelTrainer()
    metrics, feature_importance = model_trainer.train(train_data, test_data, target_column, output_path=research_path)
    
    # Print metrics
    print("\nTraining Metrics:")
    print(f"Accuracy: {metrics['train']['accuracy']:.4f}")
    print(f"F1 Score: {metrics['train']['f1']:.4f}")
    
    # Determine which evaluation metrics we have (test or validation)
    eval_key = 'test' if 'test' in metrics else 'validation'
    print(f"\n{eval_key.title()} Metrics:")
    print(f"Accuracy: {metrics[eval_key]['accuracy']:.4f}")
    print(f"F1 Score: {metrics[eval_key]['f1']:.4f}")
    
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
    loo_sigma = 0.05 # Параметр сглаживания для LeaveOneOut
    knn_imputation_args = {'n_neighbors': 5}

    # --- Запуск пайплайнов с разными энкодерами --- #

    # # 1. KNN Imputation + Ordinal Encoding
    # try:
    #     print("\n=== Processing data with KNN + Ordinal Encoding ===")
    #     train_ordinal_path, test_ordinal_path, research_ordinal_path = process_data(
    #         train_path, test_path, target_column,
    #         imputation_method='knn',
    #         encoding_method='ordinal',
    #         imputation_kwargs=knn_imputation_args, # Args for imputation
    #         encoding_kwargs={} # No extra args for ordinal encoding
    #     )
    #     print("\n=== Training model on Ordinal encoded data ===")
    #     train_model(train_ordinal_path, test_ordinal_path, target_column, research_ordinal_path)
    # except Exception as e:
    #     print(f"Error processing/training with Ordinal Encoding: {e}")

    # 2. KNN Imputation + LeaveOneOut Encoding
    try:
        print("\n=== Processing data with KNN + LeaveOneOut Encoding ===")
        train_loo_path, test_loo_path, research_loo_path = process_data(
            train_path, test_path, target_column,
            imputation_method='knn',
            encoding_method='leaveoneout',
            imputation_kwargs=knn_imputation_args, # Args for imputation
            encoding_kwargs={'sigma': loo_sigma} # Args for encoding
        )
        print("\n=== Training model on LeaveOneOut encoded data ===")
        train_model(train_loo_path, test_loo_path, target_column, research_loo_path)
    except Exception as e:
        print(f"Error processing/training with LeaveOneOut Encoding: {e}")

    # 3. KNN Imputation + LSA Encoding
    try:
        print("\n=== Processing data with KNN + LSA Encoding ===")
        train_lsa_path, test_lsa_path, research_lsa_path = process_data(
            train_path, test_path, target_column,
            imputation_method='knn',
            encoding_method='lsa',
            imputation_kwargs=knn_imputation_args, # Args for imputation
            encoding_kwargs={'n_components': n_lsa_components} # Args for encoding
        )
        print("\n=== Training model on LSA encoded data ===")
        train_model(train_lsa_path, test_lsa_path, target_column, research_lsa_path)
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