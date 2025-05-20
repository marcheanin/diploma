import warnings
import ga_optimizer
# Attempt to import SklearnFutureWarning, fallback to built-in FutureWarning
try:
    from sklearn.exceptions import FutureWarning as SklearnFutureWarning
except ImportError:
    # For older scikit-learn versions where sklearn.exceptions.FutureWarning might not exist
    SklearnFutureWarning = FutureWarning 

# Игнорировать конкретные FutureWarning от sklearn, которые вы видите
warnings.filterwarnings("ignore", category=SklearnFutureWarning, message=".*`BaseEstimator._check_n_features` is deprecated.*")
warnings.filterwarnings("ignore", category=SklearnFutureWarning, message=".*`BaseEstimator._check_feature_names` is deprecated.*")
warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")
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
import numpy as np

# Import the moved functions
from pipeline_processor import process_data, get_dataset_name 

# --- Chromosome Definition and Decoding ---
IMPUTATION_MAP = {0: 'knn', 1: 'median', 2: 'missforest'}
OUTLIER_MAP = {0: 'none', 1: 'isolation_forest', 2: 'iqr'}
RESAMPLING_MAP = {0: 'none', 1: 'oversample', 2: 'smote', 3: 'adasyn'} # Assuming 1 is ROS
ENCODING_MAP = {0: 'onehot', 1: 'label', 2: 'lsa', 3: 'word2vec'}
SCALING_MAP = {0: 'none', 1: 'standard', 2: 'minmax'}
MODEL_MAP = {0: 'logistic_regression', 1: 'random_forest', 2: 'gradient_boosting', 3: 'neural_network'}

# --- GA Parameters ---
POPULATION_SIZE = 5 # Small for testing, increase later (e.g., 20-50)
NUM_GENERATIONS = 3  # Small for testing, increase later (e.g., 10-30)
# CROSSOVER_RATE = 0.8 (placeholder for future use)
# MUTATION_RATE = 0.1 (placeholder for future use)

GENE_MAPS_LENGTHS = {
    'imputation': len(IMPUTATION_MAP),
    'outlier_removal': len(OUTLIER_MAP),
    'resampling': len(RESAMPLING_MAP),
    'encoding': len(ENCODING_MAP),
    'scaling': len(SCALING_MAP),
    'model': len(MODEL_MAP)
}
GENE_ORDER = ['imputation', 'outlier_removal', 'resampling', 'encoding', 'scaling', 'model']

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

def train_model(train_data_path, test_data_path, target_column, research_path, model_type='random_forest', plot_learning_curves=True, save_run_results=True):
    """
    Train and evaluate model on processed data.
    Returns metrics and feature importance from the model trainer.
    """
    
    model_specific_research_path_for_saving = None
    trainer_output_path = None
    effective_plot_learning_curves = plot_learning_curves

    if save_run_results and research_path:
        model_specific_research_path_for_saving = os.path.join(research_path, model_type)
        os.makedirs(model_specific_research_path_for_saving, exist_ok=True)
        trainer_output_path = model_specific_research_path_for_saving # Trainer saves plots here
        print(f"\n=== Training model: {model_type} === (Results will be saved)")
        print(f"Results will be saved in: {model_specific_research_path_for_saving}")
    elif research_path: # Still create a path for non-saving GA runs to avoid issues if model_trainer expects it, but don't save to it from train_model
        # This case might be simplified if ModelTrainer handles output_path=None perfectly for plot saving.
        # For now, let's assume GA still provides a base research_path, but train_model won't use it for final save.
        model_specific_research_path_for_saving = os.path.join(research_path, model_type, "ga_temp_eval") # A dummy path for trainer if needed
        os.makedirs(model_specific_research_path_for_saving, exist_ok=True) # create for trainer
        trainer_output_path = model_specific_research_path_for_saving
        effective_plot_learning_curves = False # Ensure no plots are saved by trainer if save_run_results is False
        print(f"\n=== Training model: {model_type} === (GA Eval - Results not saved from train_model)")
    else: # No research_path provided or not saving
        effective_plot_learning_curves = False
        print(f"\n=== Training model: {model_type} === (Results not saved)")

    try:
        train_data = pd.read_csv(train_data_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_path}")
        return None, None # Return None for metrics and feature_importance on error
    
    if test_data_path and os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
    else:
         print(f"Warning: Test data file not found or not provided ({test_data_path}). Proceeding with validation split or without test eval.")
         test_data = None 

    model_trainer = ModelTrainer(model_type=model_type)
    
    metrics, feature_importance = model_trainer.train(
        train_data, test_data, target_column, 
        output_path=trainer_output_path, # Pass potentially None or temp path
        plot_learning_curves=effective_plot_learning_curves # Pass effective flag
    )
    
    if metrics is None:
        print(f"Model training failed for {model_type}, no metrics returned.")
        return None, None

    print(f"\nTraining Metrics for {model_type}:")
    if 'train' in metrics and metrics['train']:
        print(f"  Accuracy: {metrics['train'].get('accuracy', -1):.4f}")
        print(f"  F1 Score: {metrics['train'].get('f1', -1):.4f}")
        print(f"  PR AUC: {metrics['train'].get('pr_auc', -1):.4f}") # PR AUC for train might not be standard
    else:
        print("  Training metrics not available.")
    
    eval_key = 'test' if 'test' in metrics and metrics['test'] else 'validation' if 'validation' in metrics and metrics['validation'] else None
    if eval_key:
        print(f"\n{eval_key.title()} Metrics for {model_type}:")
        print(f"  Accuracy: {metrics[eval_key].get('accuracy', -1):.4f}")
        print(f"  F1 Score: {metrics[eval_key].get('f1', -1):.4f}")
        if 'pr_auc' in metrics[eval_key]:
             print(f"  PR AUC: {metrics[eval_key]['pr_auc']:.4f}")
        else:
            print("  PR AUC not available for evaluation set.")
    else:
        print("\nNo evaluation metrics available (test/validation).")
    
    if save_run_results and research_path and model_specific_research_path_for_saving: # Ensure path was created for saving
        # Use the path that was definitely created for saving.
        # If research_path was provided but save_run_results was False, 
        # model_specific_research_path_for_saving might point to a "ga_temp_eval" - we shouldn't save there permanently.
        # The outer if save_run_results handles this.
        final_save_path = os.path.join(research_path, model_type) # Reconstruct the proper save path if saving
        os.makedirs(final_save_path, exist_ok=True) # Ensure it exists
        save_model_results(metrics, feature_importance, final_save_path)
    
    return metrics, feature_importance # Return metrics and feature importance

def main():
    print("\n=== Processing credit card dataset (Single Run Example) ===")
    train_path = "datasets/UCI_Credit_Card.csv"
    test_path = None
    target_column = "default.payment.next.month"
    generate_learning_curves = False # Set to True if you want curves for the single run

    # --- Example Chromosome for a single pipeline run ---
    example_chromosome = [0, 1, 2, 1, 0, 2] # Median, IQR, SMOTE, Label, Standard, RandomForest
    decoded_pipeline = decode_and_log_chromosome(example_chromosome)

    if decoded_pipeline is None:
        print("Error decoding example chromosome. Exiting.")
        return

    current_imputation_method = decoded_pipeline['imputation'] 
    current_outlier_method = decoded_pipeline['outlier_removal']
    current_encoding_method = decoded_pipeline['encoding']
    current_resampling_method = decoded_pipeline['resampling']
    current_scaling_method = decoded_pipeline['scaling']
    current_model_type = decoded_pipeline['model']

    print(f"\n--- Running Single Pipeline with Decoded Chromosome: {example_chromosome} ---")
    print(f"Parameters: Imp={current_imputation_method}, Out={current_outlier_method}, Res={current_resampling_method}, Enc={current_encoding_method}, Scale={current_scaling_method}, Model={current_model_type}")

    try:
        print(f"\n[Single Run - Chromosome: {example_chromosome}] Calling process_data...")
        # Use process_data from the new module
        processed_train_path, processed_test_path, research_base_path = process_data(
            train_path, test_path, target_column,
            imputation_method=current_imputation_method,
            outlier_method=current_outlier_method, 
            encoding_method=current_encoding_method,
            resampling_method=current_resampling_method, 
            scaling_method=current_scaling_method,
        )
        
        if processed_train_path is None: # Error in process_data
            print("Data processing failed for the single run. Exiting.")
            return

        print(f"[Single Run - Chromosome: {example_chromosome}] Data processing completed. Processed train: {processed_train_path}")
        print(f"\n[Single Run - Chromosome: {example_chromosome}] Calling train_model for model: {current_model_type}...")
        # train_model now returns metrics and feature_importance
        metrics_output, ft_importance_output = train_model(
            processed_train_path, processed_test_path, target_column,
            research_base_path, # research_base_path already includes the processing steps in its name
            model_type=current_model_type,
            plot_learning_curves=generate_learning_curves,
            save_run_results=True # Explicitly True for main single run
        )

        if metrics_output:
            print(f"\n--- Single Run Final Metrics for {current_model_type} ({example_chromosome}) ---")
            eval_key_single = 'test' if 'test' in metrics_output else 'validation'
            if eval_key_single in metrics_output and metrics_output[eval_key_single]:
                pr_auc_score = metrics_output[eval_key_single].get('pr_auc', -1)
                print(f"Evaluation PR AUC: {pr_auc_score:.4f}")
            else:
                print("Evaluation PR AUC not found for the single run.")
        else:
            print("Single run model training did not produce metrics.")
        print(f"[Single Run - Chromosome: {example_chromosome}] Model training and metric display complete.")

    except Exception as e:
        print(f"Error in single pipeline run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # main()
    # To run the GA, you would call a function from ga_optimizer.py, e.g.:
    ga_optimizer.run_genetic_algorithm() # Assuming run_genetic_algorithm is defined in ga_optimizer.py 