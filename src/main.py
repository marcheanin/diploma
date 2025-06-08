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
from pipeline_processor import process_data, get_dataset_name, decode_and_log_chromosome, train_model

# Для совместимости - теперь все конфигурации и функции в pipeline_processor.py

def main():
    # print("\n=== Processing credit score classification dataset (Single Run Example with HPs) ===")
    # train_path = "datasets/credit-score-classification-manual-cleaned.csv"
    # test_path = None
    # target_column = "Credit_Score"
    train_path = "datasets/Loan_Default.csv"
    test_path = None
    target_column = "Status"
    generate_learning_curves = False 

    # train_path = "datasets/credit-score-classification/train.csv"
    # test_path = "datasets/credit-score-classification/test.csv"
    # target_column = "Credit_Score"
    # generate_learning_curves = False    

    # Chromosomes to test
   # chromosome_1 = [2, 4, 0, 0, 0, 1, 2, 4, 0, 1, 0, 3, 2, 1, 0, 1, 7, 2, 5, 1] 
    chromosome_2 = [1, 2, 2, 1, 3, 1, 1, 4, 4, 0, 1, 0, 2, 1, 0, 1, 6, 1, 0, 2]

    test_chromosomes = {
     #   "Chromosome_1_MissForest": chromosome_1,
        "Chromosome_2_y_1d_array": chromosome_2
    }

    for name, current_chromosome in test_chromosomes.items():
        print(f"\n\n=== TESTING PIPELINE FOR: {name} ===")
        print(f"Chromosome: {current_chromosome}")
        
        decoded_params = decode_and_log_chromosome(current_chromosome)

        if decoded_params is None:
            print(f"Error decoding {name}. Skipping this test run.")
            continue

        print(f"\n--- Running Pipeline with Decoded Chromosome: {name} ---")
        # Parameters are now in decoded_params

        try:
            print(f"\n[Test Run - {name}] Calling process_data...")
            processed_train_path, processed_test_path, research_base_path = process_data(
                train_path, test_path, target_column,
                imputation_method=decoded_params['imputation_method'],
                imputation_params=decoded_params['imputation_params'],
                outlier_method=decoded_params['outlier_method'],
                outlier_params=decoded_params['outlier_params'],
                encoding_method=decoded_params['encoding_method'],
                encoding_params=decoded_params['encoding_params'],
                resampling_method=decoded_params['resampling_method'],
                resampling_params=decoded_params['resampling_params'],
                scaling_method=decoded_params['scaling_method'],
                scaling_params=decoded_params['scaling_params'],
                save_processed_data=True, # For test runs, save the intermediate files
                save_model_artifacts=True # Also save model artifacts
            )
            
            if processed_train_path is None: 
                print(f"Data processing failed for {name}. Skipping model training for this run.")
                continue

            print(f"[Test Run - {name}] Data processing completed. Processed train: {processed_train_path}")
            print(f"\n[Test Run - {name}] Calling train_model for model: {decoded_params['model_type']}...")
            
            metrics_output, ft_importance_output = train_model(
                processed_train_path, processed_test_path, target_column,
                research_path=research_base_path, 
                model_type=decoded_params['model_type'],
                model_hyperparameters=decoded_params['model_params'], # Pass model HPs
                plot_learning_curves=generate_learning_curves,
                save_run_results=True 
            )

            if metrics_output:
                print(f"\n--- Test Run Final Metrics for {decoded_params['model_type']} ({name}) ---")
                auprc_score = metrics_output.get('auprc')

                if auprc_score is not None and not pd.isna(auprc_score):
                    print(f"Evaluation AUPRC: {auprc_score:.4f}")
                else:
                    accuracy_score = metrics_output.get('accuracy')
                    if accuracy_score is not None and not pd.isna(accuracy_score):
                        print(f"Evaluation AUPRC not available or is invalid. Accuracy: {accuracy_score:.4f}")
                    else:
                        print("Evaluation AUPRC and Accuracy are not available or are invalid for this test run.")
            else:
                print(f"Test run model training for {name} did not produce metrics.")
            print(f"[Test Run - {name}] Model training and metric display complete.")

        except Exception as e:
            print(f"Error in pipeline run for {name} ({current_chromosome}): {e}")
            import traceback
            traceback.print_exc()
        print(f"=== FINISHED TESTING PIPELINE FOR: {name} ===")


def run_ga_with_config():
    """Пример запуска GA с пользовательской конфигурацией"""
    from ga_optimizer import GAConfig, run_genetic_algorithm
    
    # Создаем конфигурацию для ГА
    ga_config = GAConfig(
        train_path="../datasets/diabetes.csv",
        test_path=None,
        target_column="Outcome",
        population_size=10,
        num_generations=8,
        elitism_percent=0.25,
        mutation_rate=0.1,
        tournament_size=3,
        generate_learning_curves=False
    )
    
    # Запускаем ГА с конфигурацией
    results = run_genetic_algorithm(ga_config)
    
    if results:
        print(f"\n=== GA Results Summary ===")
        print(f"Best fitness: {results['best_fitness']:.4f}")
        print(f"Best chromosome: {results['best_chromosome']}")
    
    return results


if __name__ == "__main__":
    # Можно запустить тестирование отдельных хромосом
    # main()
    
    # Или запустить ГА с новой архитектурой
    run_ga_with_config()