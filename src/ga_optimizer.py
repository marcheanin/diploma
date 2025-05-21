import os
import numpy as np
import pandas as pd # May be needed for type hints or direct use in future GA operators
import matplotlib.pyplot as plt # Added for plotting

# Imports from other project modules
from pipeline_processor import process_data, get_dataset_name # Corrected import
from main import (
    train_model, decode_and_log_chromosome, 
    IMPUTATION_MAP, OUTLIER_MAP, RESAMPLING_MAP, ENCODING_MAP, SCALING_MAP, MODEL_MAP,
    # Import all HP maps needed for GENE_CHOICES_COUNT
    HP_IMPUTATION_KNN_N_NEIGHBORS, HP_IMPUTATION_MISSFOREST_N_ESTIMATORS, HP_IMPUTATION_MISSFOREST_MAX_ITER,
    HP_OUTLIER_IF_N_ESTIMATORS, HP_OUTLIER_IF_CONTAMINATION, HP_OUTLIER_IQR_MULTIPLIER,
    HP_RESAMPLING_ROS_STRATEGY, HP_RESAMPLING_SMOTE_K_NEIGHBORS, HP_RESAMPLING_SMOTE_STRATEGY,
    HP_RESAMPLING_ADASYN_N_NEIGHBORS, HP_RESAMPLING_ADASYN_STRATEGY,
    HP_ENCODING_ONEHOT_MAX_CARDINALITY, HP_ENCODING_ONEHOT_DROP, HP_ENCODING_LSA_N_COMPONENTS,
    HP_ENCODING_LSA_NGRAM_MAX, HP_ENCODING_W2V_DIM, HP_ENCODING_W2V_WINDOW,
    HP_SCALING_STANDARD_WITH_MEAN, HP_SCALING_STANDARD_WITH_STD,
    HP_MODEL_LOGREG_C, HP_MODEL_LOGREG_PENALTY_SOLVER, HP_MODEL_LOGREG_CLASS_WEIGHT,
    HP_MODEL_RF_N_ESTIMATORS, HP_MODEL_RF_MAX_DEPTH, HP_MODEL_RF_MIN_SAMPLES_SPLIT,
    HP_MODEL_GB_N_ESTIMATORS, HP_MODEL_GB_LEARNING_RATE, HP_MODEL_GB_MAX_DEPTH,
    HP_MODEL_NN_LAYERS, HP_MODEL_NN_DROPOUT, HP_MODEL_NN_LR
)

# --- GA Parameters ---
POPULATION_SIZE = 15
NUM_GENERATIONS = 10 
# CROSSOVER_RATE = 0.8 
# MUTATION_RATE = 0.1 

# --- Gene Information for Initialization (19 Genes Total) ---
# [ImpMethod, ImpHP1, ImpHP2, OutMethod, OutHP1, OutHP2, ResMethod, ResHP1, ResHP2, 
#  EncMethod, EncHP1, EncHP2, ScaMethod, ScaHP1, ScaHP2, ModelMethod, ModHP1, ModHP2, ModHP3]

GENE_CHOICES_COUNT = [
    len(IMPUTATION_MAP), # Imputation Method (0)
    max(len(HP_IMPUTATION_KNN_N_NEIGHBORS), len(HP_IMPUTATION_MISSFOREST_N_ESTIMATORS)), # Imputation HP1 (1) (Median has no HPs here)
    len(HP_IMPUTATION_MISSFOREST_MAX_ITER), # Imputation HP2 (2) (Only MissForest uses HP2)
    
    len(OUTLIER_MAP),    # Outlier Method (3)
    max(len(HP_OUTLIER_IF_N_ESTIMATORS), len(HP_OUTLIER_IQR_MULTIPLIER)), # Outlier HP1 (4)
    len(HP_OUTLIER_IF_CONTAMINATION), # Outlier HP2 (5) (Only IF uses HP2)
    
    len(RESAMPLING_MAP), # Resampling Method (6)
    max(len(HP_RESAMPLING_ROS_STRATEGY), len(HP_RESAMPLING_SMOTE_K_NEIGHBORS), len(HP_RESAMPLING_ADASYN_N_NEIGHBORS)), # Resampling HP1 (7)
    max(len(HP_RESAMPLING_SMOTE_STRATEGY), len(HP_RESAMPLING_ADASYN_STRATEGY)), # Resampling HP2 (8)
    
    len(ENCODING_MAP),   # Encoding Method (9)
    max(len(HP_ENCODING_ONEHOT_MAX_CARDINALITY), len(HP_ENCODING_LSA_N_COMPONENTS), len(HP_ENCODING_W2V_DIM)), # Encoding HP1 (10)
    max(len(HP_ENCODING_ONEHOT_DROP), len(HP_ENCODING_LSA_NGRAM_MAX), len(HP_ENCODING_W2V_WINDOW)), # Encoding HP2 (11)
    
    len(SCALING_MAP),    # Scaling Method (12)
    len(HP_SCALING_STANDARD_WITH_MEAN),      # Scaling HP1 (13) (Only Standard uses HPs)
    len(HP_SCALING_STANDARD_WITH_STD),       # Scaling HP2 (14)
    
    len(MODEL_MAP),      # Model Method (15)
    max(len(HP_MODEL_LOGREG_C), len(HP_MODEL_RF_N_ESTIMATORS), len(HP_MODEL_GB_N_ESTIMATORS), len(HP_MODEL_NN_LAYERS)), # Model HP1 (16)
    max(len(HP_MODEL_LOGREG_PENALTY_SOLVER), len(HP_MODEL_RF_MAX_DEPTH), len(HP_MODEL_GB_LEARNING_RATE), len(HP_MODEL_NN_DROPOUT)), # Model HP2 (17)
    max(len(HP_MODEL_LOGREG_CLASS_WEIGHT), len(HP_MODEL_RF_MIN_SAMPLES_SPLIT), len(HP_MODEL_GB_MAX_DEPTH), len(HP_MODEL_NN_LR)) # Model HP3 (18)
]
# Ensure GENE_CHOICES_COUNT has 19 elements, one for each gene.
if len(GENE_CHOICES_COUNT) != 19:
    raise ValueError(f"GENE_CHOICES_COUNT should have 19 elements, but has {len(GENE_CHOICES_COUNT)}")

def initialize_individual():
    """Initializes a single random 19-gene chromosome."""
    individual = [np.random.randint(0, max_val) for max_val in GENE_CHOICES_COUNT]
    return individual

def initialize_population(pop_size):
    """Initializes a population of random chromosomes."""
    return [initialize_individual() for _ in range(pop_size)]

def evaluate_chromosome(chromosome_genes, train_path_ga, test_path_ga, target_column_ga, base_research_path_ga, gen_learning_curves_ga):
    """Evaluates a single chromosome and returns its fitness (PR AUC)."""
    # decode_and_log_chromosome now returns a dictionary of pipeline parameters
    decoded_pipeline_params = decode_and_log_chromosome(chromosome_genes) 
    
    if decoded_pipeline_params is None:
        print("Error decoding chromosome during evaluation. Assigning low fitness.")
        return -1.0 # Low fitness for errors

    # Extract parameters from the decoded dictionary
    current_imputation_method = decoded_pipeline_params['imputation_method']
    current_imputation_params = decoded_pipeline_params['imputation_params']
    current_outlier_method = decoded_pipeline_params['outlier_method']
    current_outlier_params = decoded_pipeline_params['outlier_params']
    current_resampling_method = decoded_pipeline_params['resampling_method']
    current_resampling_params = decoded_pipeline_params['resampling_params']
    current_encoding_method = decoded_pipeline_params['encoding_method']
    current_encoding_params = decoded_pipeline_params['encoding_params']
    current_scaling_method = decoded_pipeline_params['scaling_method']
    current_scaling_params = decoded_pipeline_params['scaling_params']
    current_model_type = decoded_pipeline_params['model_type']
    current_model_params = decoded_pipeline_params['model_params']

    print(f"\n--- Evaluating Pipeline from Chromosome: {chromosome_genes} ---")
    # Logging already handled by decode_and_log_chromosome, this is just a confirmation
    # print(f"Parameters: Imp={current_imputation_method}, Out={current_outlier_method}, Res={current_resampling_method}, Enc={current_encoding_method}, Scale={current_scaling_method}, Model={current_model_type}")
    # print(f"Full Decoded HPs: {decoded_pipeline_params}")

    fitness = -1.0 
    try:
        print(f"\n[GA - Chromosome: {chromosome_genes}] Calling process_data...")
        processed_train_data_df, processed_test_data_df, research_path_for_chromosome_config = process_data(
            train_path_ga, test_path_ga, target_column_ga,
            imputation_method=current_imputation_method,
            imputation_params=current_imputation_params,
            outlier_method=current_outlier_method, 
            outlier_params=current_outlier_params,
            encoding_method=current_encoding_method,
            encoding_params=current_encoding_params,
            resampling_method=current_resampling_method, 
            resampling_params=current_resampling_params,
            scaling_method=current_scaling_method,
            scaling_params=current_scaling_params,
            save_processed_data=False,
            save_model_artifacts=False
        )
        
        if processed_train_data_df is None: 
            print(f"Data processing failed for chromosome {chromosome_genes}. Skipping model training.")
            return -1.0
            
        print(f"[GA - Chromosome: {chromosome_genes}] Calling train_model for model: {current_model_type}...")
        metrics_output, ft_importance_output = train_model(
            processed_train_data_df, 
            processed_test_data_df,  
            target_column_ga,
            research_path=research_path_for_chromosome_config, 
            model_type=current_model_type,
            model_hyperparameters=current_model_params, # Pass model HPs
            plot_learning_curves=gen_learning_curves_ga,
            save_run_results=False 
        )

        if metrics_output:
            metric_source_dict = None
            eval_set_key_used = None
            fitness_metric_name = "AUPRC" # Default expected metric

            if 'test' in metrics_output and isinstance(metrics_output['test'], dict):
                metric_source_dict = metrics_output['test']
                eval_set_key_used = 'test'
            elif 'validation' in metrics_output and isinstance(metrics_output['validation'], dict):
                metric_source_dict = metrics_output['validation']
                eval_set_key_used = 'validation'
            
            if metric_source_dict: # Primary path: metrics are in a nested dict ('test' or 'validation')
                if 'auprc' in metric_source_dict:
                    fitness = metric_source_dict['auprc']
                    if fitness is None or pd.isna(fitness): 
                        print(f"Warning: AUPRC for chromosome {chromosome_genes} (eval set: {eval_set_key_used}) is None/NaN. Assigning low fitness.")
                        fitness = -0.5 
                    else:
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (AUPRC): {fitness:.4f}")
                elif 'accuracy' in metric_source_dict: # Fallback to accuracy if auprc is missing
                    fitness = metric_source_dict['accuracy'] 
                    fitness_metric_name = "Accuracy"
                    print(f"Warning: AUPRC not found in '{eval_set_key_used}' dict for chromosome {chromosome_genes}. Using Accuracy: {fitness:.4f} as fitness.")
                    if fitness is None or pd.isna(fitness):
                        fitness = -0.7 # Low fitness for NaN accuracy
                else:
                    print(f"Core fitness metrics (AUPRC, Accuracy) not found in '{eval_set_key_used}' dict for chromosome {chromosome_genes}. Keys: {metric_source_dict.keys()}. Assigning low fitness.")
                    fitness = -0.85
            else: # Fallback path: metrics might be in a flat structure directly under metrics_output
                if 'auprc' in metrics_output:
                    fitness = metrics_output['auprc']
                    eval_set_key_used = 'top-level'
                    if fitness is None or pd.isna(fitness):
                        print(f"Warning: AUPRC for chromosome {chromosome_genes} (eval set: {eval_set_key_used}) is None/NaN. Assigning low fitness.")
                        fitness = -0.51 
                    else:
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (AUPRC): {fitness:.4f}")
                elif 'accuracy' in metrics_output: # Fallback to accuracy in flat structure
                    fitness = metrics_output['accuracy']
                    fitness_metric_name = "Accuracy"
                    eval_set_key_used = 'top-level'
                    print(f"Warning: AUPRC not found at top-level for chromosome {chromosome_genes}. Using Accuracy: {fitness:.4f} as fitness.")
                    if fitness is None or pd.isna(fitness):
                        fitness = -0.71 # Low fitness for NaN accuracy at top-level
                else:
                    print(f"Neither nested eval dict nor top-level AUPRC/Accuracy found in metrics_output for chromosome {chromosome_genes}. Metric Keys: {metrics_output.keys()}. Assigning low fitness.")
                    fitness = -0.8
        else:
            print(f"Model training failed or did not produce metrics for chromosome {chromosome_genes}. Assigning low fitness.")
            fitness = -0.9 
        
        print(f"[GA - Chromosome: {chromosome_genes}] Fitness ({fitness_metric_name}) determination complete.")

    except Exception as e:
        print(f"Error evaluating chromosome {chromosome_genes}: {e}")
        # import traceback
        # traceback.print_exc()
        fitness = -1.0 
    
    # decoded_pipeline_params already contains the full structure, so we don't need to call decode again for logging.
    print(f"Final Chromosome: {chromosome_genes}, Decoded Params Used: ..., Fitness: {fitness:.4f}") # Logging already extensive
    return fitness

def run_genetic_algorithm():
    """Main function to set up and run the genetic algorithm."""
    print("\n=== Setting up GA for Credit Score Dataset ===")
    train_path = "datasets/diabetes.csv"
    test_path = None
    target_column = "Outcome"
    
    generate_learning_curves = False

    print("\n=== Starting Genetic Algorithm ===")
    ga_base_research_path = os.path.join("research", get_dataset_name(train_path), "genetic_algorithm_runs")
    os.makedirs(ga_base_research_path, exist_ok=True)

    population = initialize_population(POPULATION_SIZE) # Uses new 19-gene init
    best_overall_chromosome = None
    best_overall_fitness = -1.0 
    best_fitness_over_generations = [] 

    try: 
        for gen in range(NUM_GENERATIONS):
            print(f"\n--- Generation {gen + 1}/{NUM_GENERATIONS} ---")
            population_with_fitness = []

            for i, ind_chromosome in enumerate(population):
                print(f"\nEvaluating Individual {i+1}/{POPULATION_SIZE} in Generation {gen+1}")
                current_fitness = evaluate_chromosome(ind_chromosome, train_path, test_path, target_column, ga_base_research_path, generate_learning_curves)
                population_with_fitness.append((ind_chromosome, current_fitness))
                
                if current_fitness > best_overall_fitness:
                    best_overall_fitness = current_fitness
                    best_overall_chromosome = ind_chromosome
                    print(f"*** New best overall chromosome found in Gen {gen+1}, Ind {i+1}: {best_overall_chromosome} with fitness: {best_overall_fitness:.4f} ***")

            population_with_fitness.sort(key=lambda x: x[1], reverse=True)
            
            if not population_with_fitness: 
                print("Warning: Population is empty after evaluation.")
                break 
            
            current_gen_best_fitness = population_with_fitness[0][1]
            best_fitness_over_generations.append(current_gen_best_fitness)

            print(f"\nEnd of Generation {gen + 1}. Best fitness in this generation: {current_gen_best_fitness:.4f}")
            print(f"Best chromosome this generation: {population_with_fitness[0][0]}")
            
            if gen < NUM_GENERATIONS - 1:
                print("GA operators (selection, crossover, mutation) not yet implemented. Creating next generation based on top half and random individuals.")
                top_half_count = POPULATION_SIZE // 2
                if top_half_count == 0 and POPULATION_SIZE > 0: top_half_count = 1
                
                next_population = [item[0] for item in population_with_fitness[:top_half_count]]
                needed_random = POPULATION_SIZE - len(next_population)
                if needed_random > 0:
                    next_population.extend([initialize_individual() for _ in range(needed_random)])
                population = next_population[:POPULATION_SIZE] 
                
                if not population and POPULATION_SIZE > 0:
                    print("Warning: Next population became empty. Re-initializing.")
                    population = initialize_population(POPULATION_SIZE)
    finally: 
        if best_fitness_over_generations:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(best_fitness_over_generations) + 1), best_fitness_over_generations, marker='o', linestyle='-')
            plt.title('Best Fitness (AUPRC) vs. Generation')
            plt.xlabel('Generation')
            plt.ylabel('Best AUPRC')
            plt.xticks(range(1, len(best_fitness_over_generations) + 1))
            plt.grid(True)
            plot_save_path = os.path.join(ga_base_research_path, "ga_fitness_progression.png")
            try:
                plt.savefig(plot_save_path)
                print(f"\nFitness progression plot saved to {plot_save_path}")
            except Exception as e:
                print(f"Error saving fitness progression plot: {e}")
            plt.close()

    print("\n=== Genetic Algorithm Finished ===")
    if best_overall_chromosome:
        print(f"Best overall chromosome found across all generations: {best_overall_chromosome}")
        # decode_and_log_chromosome is called again here for the final best chromosome to display full details
        print("Details of the Best Chromosome:")
        decode_and_log_chromosome(best_overall_chromosome) 
        print(f"Best overall fitness (AUPRC): {best_overall_fitness:.4f}")
    else:
        print("No successful evaluation run or no improvement found in the GA.")

if __name__ == '__main__':
    run_genetic_algorithm() 