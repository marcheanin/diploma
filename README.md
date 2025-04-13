# Credit Score Classification Data Preprocessing

This project provides a comprehensive data preprocessing pipeline for credit score classification data. It includes functionality for:

- Loading and handling train/test datasets
- Missing value imputation (KNN and MissForest)
- Categorical variable encoding
- Outlier detection and removal
- Data analysis and statistics

## Project Structure

```
project/
│
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   └── outlier_remover.py
│   │
│   ├── utils/
│   │   └── data_analysis.py
│   │
│   └── main.py
│
├── datasets/
│   ├── train.csv
│   ├── test.csv
│   ├── train_processed_knn.csv
│   ├── test_processed_knn.csv
│   ├── train_processed_missforest.csv
│   └── test_processed_missforest.csv
│
├── requirements.txt
└── README.md
```

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

Note: If MissForest installation fails, the system will automatically fall back to using scikit-learn's IterativeImputer with RandomForestRegressor as a substitute.

## Usage

1. Place your datasets in the `datasets` folder:
   - `train.csv`: Training dataset
   - `test.csv`: Test dataset

2. Run the preprocessing pipeline:
   ```
   python src/main.py
   ```

The script will:
- Load your datasets
- Show data statistics and distributions
- Impute missing values using both KNN and MissForest methods
- Encode categorical variables
- Remove outliers
- Save processed datasets (separate files for each imputation method)

## Components

### DataLoader
Handles loading train and test datasets from CSV files.

### DataPreprocessor
Provides methods for:
- Missing value imputation:
  - KNN imputation for numeric features
  - Mode imputation for categorical features
  - MissForest imputation (with fallback to IterativeImputer)
- Categorical variable encoding (one-hot, label, target encoding)
- Feature type handling

### OutlierRemover
Removes outliers using Isolation Forest algorithm.

### Data Analysis Utils
Provides functions for:
- Missing value analysis
- Numeric feature statistics
- Target variable distribution analysis

## Imputation Methods

### KNN Imputation
- Uses scikit-learn's KNNImputer for numeric features
- Uses mode imputation for categorical features
- Parameters:
  - `n_neighbors`: Number of neighbors to use (default: 5)

### MissForest Imputation
- Uses missingpy's MissForest implementation
- Falls back to scikit-learn's IterativeImputer with RandomForestRegressor if missingpy is not available
- Parameters:
  - `max_iter`: Maximum number of iterations (default: 10)
  - `n_estimators`: Number of trees in random forest (default: 100)
  - `random_state`: Random state for reproducibility (default: 42)

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- category_encoders >= 2.3.0
- missingpy >= 0.2.0 (optional, for MissForest)
- joblib >= 1.1.0 (required by missingpy) 