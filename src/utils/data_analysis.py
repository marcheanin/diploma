def print_missing_summary(data, name):
    """
    Print statistics about missing values in the dataset.
    
    Args:
        data: pandas DataFrame to analyze
        name: str, name of the dataset
    """
    total = data.isnull().sum().sum()
    print(f"\nMissing values in {name}:")
    print(f"Total missing values: {total}")
    if total > 0:
        print("Distribution by columns:")
        missing = data.isnull().sum()
        missing = missing[missing > 0]
        for col, count in missing.items():
            print(f"- {col}: {count} missing values")

def print_numeric_stats(data, stage_name):
    """
    Print statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
        stage_name: str, name of the processing stage
    """
    numeric_cols = data.select_dtypes(include=['number']).columns
    print(f"\n=== Numeric features {stage_name} ===")
    print(f"Total numeric columns: {len(numeric_cols)}")
    
    if len(numeric_cols) > 0:
        stats = data[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
        print("Statistics:\n", stats.round(2))

def print_target_distribution(data, target_col):
    """
    Print distribution of values in the target column.
    
    Args:
        data: pandas DataFrame
        target_col: str, name of the target column
    """
    if target_col not in data.columns:
        raise ValueError(f"Column {target_col} not found in data")
    
    target_counts = data[target_col].value_counts(dropna=False)
    target_percent = data[target_col].value_counts(normalize=True, dropna=False) * 100
    
    print(f"\nTarget variable '{target_col}' distribution:")
    print("----------------------------------")
    print(f"Total records: {len(data)}")
    print("----------------------------------")
    print("Value | Count | Percentage")
    print("----------------------------------")
    
    for value, count in target_counts.items():
        percent = target_percent[value]
        print(f"{value!r:8} | {count:5} | {percent:.2f}%")
    
    if target_counts.isna().sum() > 0:
        print("\nWarning: missing values found in target variable!")

def analyze_target_correlations(data, target_col, output_path):
    """
    Calculate and save correlations between features and target variable.
    
    Args:
        data: pandas DataFrame
        target_col: str, name of the target column
        output_path: str, path to save correlation results
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate correlations with target
    correlations = data.corr()[target_col].sort_values(ascending=False)
    
    # Create a DataFrame with correlations
    corr_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    })
    
    # Save correlations to CSV
    corr_df.to_csv(f"{output_path}/feature_correlations.csv", index=False)
    
    # Create correlation plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=corr_df, x='Correlation', y='Feature')
    plt.title(f'Feature Correlations with {target_col}')
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_correlations.png")
    plt.close()
    
    print(f"\nFeature correlations with {target_col} have been saved to:")
    print(f"- CSV: {output_path}/feature_correlations.csv")
    print(f"- Plot: {output_path}/feature_correlations.png")

def save_model_results(metrics, feature_importance, output_path):
    """
    Save model evaluation results and feature importance.
    
    Args:
        metrics: dict, containing model metrics
        feature_importance: pandas DataFrame, feature importance scores
        output_path: str, path to save results
    """
    import os
    
    # Create model results directory
    model_path = os.path.join(output_path, 'model_results')
    os.makedirs(model_path, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(model_path, 'metrics.txt'), 'w') as f:
        f.write("=== Training Metrics ===\n")
        f.write(f"Accuracy: {metrics['train']['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['train']['f1']:.4f}\n")
        f.write("\nDetailed Training Report:\n")
        f.write(metrics['train']['detailed_report'])
        
        f.write("\n\n=== Validation Metrics ===\n")
        f.write(f"Accuracy: {metrics['validation']['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['validation']['f1']:.4f}\n")
        f.write("\nDetailed Validation Report:\n")
        f.write(metrics['validation']['detailed_report'])
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(model_path, 'feature_importance.csv'), index=False) 