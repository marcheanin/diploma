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