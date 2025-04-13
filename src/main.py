from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.outlier_remover import OutlierRemover
from utils.data_analysis import (
    print_missing_summary,
    print_numeric_stats,
    print_target_distribution
)

def process_data(train_path, test_path, target_column, imputation_method='knn', **kwargs):
    """
    Process data using specified imputation method.
    
    Args:
        train_path: str, path to training data
        test_path: str, path to test data
        target_column: str, name of target column
        imputation_method: str, 'knn' or 'missforest'
        **kwargs: additional arguments for imputation method
    """
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
    train_data = preprocessor.encode(train_data, method='label')
    if test_data is not None:
        test_data = preprocessor.encode(test_data, method='label')
    
    # Print numeric statistics after encoding
    print_numeric_stats(train_data, "after encoding")

    # Remove outliers from train data
    outlier_remover = OutlierRemover(contamination=0.05)
    train_data_clean = outlier_remover.remove_outliers(train_data)
    print("\nTrain dataset size after imputation and outlier removal:", train_data_clean.shape)

    # Save processed data
    output_suffix = f"_processed_{imputation_method}"
    train_data_clean.to_csv(f'datasets/train{output_suffix}.csv', index=False)
    if test_data is not None:
        test_data.to_csv(f'datasets/test{output_suffix}.csv', index=False)

def main():
    # Use the correct paths to our dataset files
    train_path = "datasets/credit-score-classification/train.csv"
    test_path = "datasets/credit-score-classification/test.csv"
    target_column = "Credit_Score"

    # Process with KNN imputation
    print("\n=== Processing with KNN imputation ===")
    process_data(train_path, test_path, target_column, 
                imputation_method='knn', 
                n_neighbors=5)

    # Process with MissForest imputation
    # print("\n=== Processing with MissForest imputation ===")
    # process_data(train_path, test_path, target_column,
    #             imputation_method='missforest',
    #             max_iter=10,
    #             n_estimators=100,
    #             random_state=42)

if __name__ == "__main__":
    main() 