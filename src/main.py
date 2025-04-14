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

def process_data(train_path, test_path, target_column, imputation_method='knn', **kwargs):
    """
    Process data using specified imputation method.
    
    Args:
        train_path: str, path to training data
        test_path: str, path to test data
        target_column: str, name of target column
        imputation_method: str, 'knn' or 'missforest'
        **kwargs: additional arguments for imputation method
        
    Returns:
        tuple: (train_data_path, test_data_path, research_path)
    """
    # Get dataset name and create paths
    dataset_name = get_dataset_name(train_path)
    results_path = os.path.join('results', dataset_name)
    research_path = os.path.join("research", dataset_name)
    
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
    
    # Impute missing values
    print(f"\nUsing {imputation_method} for imputation...")
    train_data = preprocessor.impute(train_data, method=imputation_method, **kwargs)
    if test_data is not None:
        test_data = preprocessor.impute(test_data, method=imputation_method, **kwargs)

    # Print missing values statistics after imputation
    print("\n=== Missing values statistics after imputation ===")
    print_missing_summary(train_data, "train dataset")
    if test_data is not None:
        print_missing_summary(test_data, "test dataset")

    # Print numeric statistics before encoding
    print_numeric_stats(train_data, "before encoding")

    # Encode categorical variables
    train_data = preprocessor.encode(train_data, method='label', target_col=target_column)
    if test_data is not None:
        test_data = preprocessor.encode(test_data, method='label', target_col=target_column)
    
    # Print numeric statistics after encoding
    print_numeric_stats(train_data, "after encoding")

    # Remove outliers from train data
    outlier_remover = OutlierRemover(contamination=0.05)
    train_data_clean = outlier_remover.remove_outliers(train_data)
    print("\nTrain dataset size after imputation and outlier removal:", train_data_clean.shape)

    # Save processed data
    output_suffix = f"_processed_{imputation_method}"
    train_data_path = os.path.join(results_path, f'train{output_suffix}.csv')
    test_data_path = os.path.join(results_path, f'test{output_suffix}.csv')
    
    train_data_clean.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    # Analyze and save feature correlations with target
    print("\n=== Analyzing feature correlations with target variable ===")
    analyze_target_correlations(train_data_clean, target_column, research_path)
    
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

    # Process data
    print("\n=== Processing data with KNN imputation ===")
    train_data_path, test_data_path, research_path = process_data(
        train_path, test_path, target_column,
        imputation_method='knn',
        n_neighbors=5
    )
    
    # Train and evaluate model
    print("\n=== Training model ===")
    model = train_model(train_data_path, test_data_path, target_column, research_path)

if __name__ == "__main__":
    main() 