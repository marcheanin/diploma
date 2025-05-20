import os
import numpy as np
import pandas as pd # May be needed for type hints or direct use in future GA operators
import matplotlib.pyplot as plt # Added for plotting

# Imports from other project modules
from pipeline_processor import process_data, get_dataset_name # Corrected import
from main import train_model, decode_and_log_chromosome, IMPUTATION_MAP, OUTLIER_MAP, RESAMPLING_MAP, ENCODING_MAP, SCALING_MAP, MODEL_MAP

# --- GA Parameters ---
POPULATION_SIZE = 15 # Example: 5-20 for faster testing, 20-50+ for serious runs
NUM_GENERATIONS = 10  # Example: 3-10 for faster testing, 10-50+ for serious runs
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

def initialize_individual():
    """Initializes a single random chromosome."""
    individual = []
    for gene_key in GENE_ORDER:
        max_value = GENE_MAPS_LENGTHS[gene_key]
        individual.append(np.random.randint(0, max_value))
    return individual

def initialize_population(pop_size):
    """Initializes a population of random chromosomes."""
    return [initialize_individual() for _ in range(pop_size)]

def evaluate_chromosome(chromosome_genes, train_path_ga, test_path_ga, target_column_ga, base_research_path_ga, gen_learning_curves_ga):
    """Evaluates a single chromosome and returns its fitness (PR AUC)."""
    decoded_pipeline = decode_and_log_chromosome(chromosome_genes) # Logs the chromosome details
    if decoded_pipeline is None:
        print("Error decoding chromosome during evaluation. Assigning low fitness.")
        return -1.0 # Low fitness for errors

    current_imputation_method = decoded_pipeline['imputation'] 
    current_outlier_method = decoded_pipeline['outlier_removal']
    current_encoding_method = decoded_pipeline['encoding']
    current_resampling_method = decoded_pipeline['resampling']
    current_scaling_method = decoded_pipeline['scaling']
    current_model_type = decoded_pipeline['model']

    print(f"\n--- Evaluating Pipeline from Chromosome: {chromosome_genes} ---")
    print(f"Parameters: Imp={current_imputation_method}, Out={current_outlier_method}, Res={current_resampling_method}, Enc={current_encoding_method}, Scale={current_scaling_method}, Model={current_model_type}")

    fitness = -1.0 # Default low fitness for any failure before PR AUC extraction
    try:
        print(f"\n[GA - Chromosome: {chromosome_genes}] Calling process_data...")
        # process_data is imported from pipeline_processor
        # It will now return DataFrames when save_processed_data is False
        processed_train_data_df, processed_test_data_df, research_path_for_chromosome_config = process_data(
            train_path_ga, test_path_ga, target_column_ga,
            imputation_method=current_imputation_method,
            outlier_method=current_outlier_method, 
            encoding_method=current_encoding_method,
            resampling_method=current_resampling_method, 
            scaling_method=current_scaling_method,
            save_processed_data=False # Ensure data is not saved for GA evals
        )
        
        if processed_train_data_df is None: # Indicates an error in process_data
            print(f"Data processing failed for chromosome {chromosome_genes}. Skipping model training.")
            return -1.0
            
        # print(f"[GA - Chromosome: {chromosome_genes}] Data processing completed. Using in-memory DataFrames.")
        # research_path_for_chromosome_config is the base for this data processing setup.
        # Model results will go into a subfolder of this, specific to the model type.
        # This path is passed to train_model, which will create a model_type subfolder if saving is enabled there (but it's not for GA evals).

        print(f"[GA - Chromosome: {chromosome_genes}] Calling train_model for model: {current_model_type}...")
        # train_model is imported from main and now accepts DataFrames directly
        metrics_output, ft_importance_output = train_model(
            processed_train_data_df, # Pass DataFrame directly
            processed_test_data_df,  # Pass DataFrame directly (can be None)
            target_column_ga,
            research_path=research_path_for_chromosome_config, 
            model_type=current_model_type,
            plot_learning_curves=gen_learning_curves_ga,
            save_run_results=False # This is already False for GA evals, as set in previous steps
        )

        if metrics_output:
            eval_key = 'test' if 'test' in metrics_output and metrics_output['test'] else 'validation' if 'validation' in metrics_output and metrics_output['validation'] else None
            if eval_key and metrics_output[eval_key] and 'pr_auc' in metrics_output[eval_key]:
                fitness = metrics_output[eval_key]['pr_auc']
                if fitness is None or pd.isna(fitness): # Handle None or NaN PR AUC
                    print(f"Warning: PR AUC for chromosome {chromosome_genes} is None/NaN. Assigning low fitness.")
                    fitness = -0.5 # Assign a specific low fitness for None/NaN PR AUC
                else:
                    print(f"Model training completed for {chromosome_genes}. Fitness (PR AUC): {fitness:.4f}")
            else:
                print(f"PR AUC not found in evaluation metrics for chromosome {chromosome_genes}. Assigning low fitness.")
                fitness = -0.8 # Different low fitness if key missing
        else:
            print(f"Model training failed or did not produce metrics for chromosome {chromosome_genes}. Assigning low fitness.")
            fitness = -0.9 # Different low fitness for training failure
        
        print(f"[GA - Chromosome: {chromosome_genes}] Fitness determination complete.")

    except Exception as e:
        print(f"Error evaluating chromosome {chromosome_genes}: {e}")
        # import traceback
        # traceback.print_exc()
        fitness = -1.0 # Low fitness for any other errors during evaluation
    
    print(f"Final Chromosome: {chromosome_genes}, Decoded: {decoded_pipeline}, Fitness: {fitness:.4f}")
    return fitness

def run_genetic_algorithm():
    """Main function to set up and run the genetic algorithm."""
    print("\n=== Setting up GA for Credit Score Dataset ===")
    # These paths and target_column would typically be configurable or passed as arguments
    # train_path = "datasets/credit-score-classification/train.csv"
    # test_path = "datasets/credit-score-classification/test.csv"
    # target_column = "Credit_Score"

    train_path = "datasets/credit-score-classification/train.csv"
    test_path = "datasets/credit-score-classification/test.csv"
    target_column = "Credit_Score"
    
    generate_learning_curves = False # Keep false for GA speed

    print("\n=== Starting Genetic Algorithm ===")
    # Create a base research path for all GA runs
    # get_dataset_name is imported from pipeline_processor
    ga_base_research_path = os.path.join("research", get_dataset_name(train_path), "genetic_algorithm_runs")
    os.makedirs(ga_base_research_path, exist_ok=True)

    population = initialize_population(POPULATION_SIZE)
    best_overall_chromosome = None
    best_overall_fitness = -1.0 # Initialize with a value lower than any possible fitness
    best_fitness_over_generations = [] # Added to track best fitness per generation

    try: # Added try-finally to ensure plot is saved even on KeyboardInterrupt
        for gen in range(NUM_GENERATIONS):
            print(f"\n--- Generation {gen + 1}/{NUM_GENERATIONS} ---")
            population_with_fitness = []

            for i, ind_chromosome in enumerate(population):
                print(f"\nEvaluating Individual {i+1}/{POPULATION_SIZE} in Generation {gen+1}")
                # Each evaluation will create its own subfolder within ga_base_research_path/ga_eval/...
                # So, pass ga_base_research_path to evaluate_chromosome which then constructs finer-grained paths.
                current_fitness = evaluate_chromosome(ind_chromosome, train_path, test_path, target_column, ga_base_research_path, generate_learning_curves)
                population_with_fitness.append((ind_chromosome, current_fitness))
                
                if current_fitness > best_overall_fitness:
                    best_overall_fitness = current_fitness
                    best_overall_chromosome = ind_chromosome
                    print(f"*** New best overall chromosome found in Gen {gen+1}, Ind {i+1}: {best_overall_chromosome} with fitness: {best_overall_fitness:.4f} ***")

            population_with_fitness.sort(key=lambda x: x[1], reverse=True)
            
            if not population_with_fitness: # Should not happen if population_size > 0
                print("Warning: Population is empty after evaluation.")
                break 
            
            current_gen_best_fitness = population_with_fitness[0][1]
            best_fitness_over_generations.append(current_gen_best_fitness)

            print(f"\nEnd of Generation {gen + 1}. Best fitness in this generation: {current_gen_best_fitness:.4f}")
            print(f"Best chromosome this generation: {population_with_fitness[0][0]}")
            
            if gen < NUM_GENERATIONS - 1:
                print("GA operators (selection, crossover, mutation) not yet implemented. Creating next generation based on top half and random individuals.")
                top_half_count = POPULATION_SIZE // 2
                if top_half_count == 0 and POPULATION_SIZE > 0: top_half_count = 1 # Ensure at least one if pop size is 1
                
                next_population = [item[0] for item in population_with_fitness[:top_half_count]]
                needed_random = POPULATION_SIZE - len(next_population)
                if needed_random > 0:
                    next_population.extend([initialize_individual() for _ in range(needed_random)])
                population = next_population[:POPULATION_SIZE] # Ensure population size is maintained
                
                if not population and POPULATION_SIZE > 0:
                    print("Warning: Next population became empty. Re-initializing.")
                    population = initialize_population(POPULATION_SIZE)
    finally: # Ensure plot is generated even if GA is interrupted
        if best_fitness_over_generations:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(best_fitness_over_generations) + 1), best_fitness_over_generations, marker='o', linestyle='-')
            plt.title('Best Fitness (PR AUC) vs. Generation')
            plt.xlabel('Generation')
            plt.ylabel('Best PR AUC')
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
        decoded_best = decode_and_log_chromosome(best_overall_chromosome)
        # print(f"Decoded Best Pipeline details: {decoded_best}") # decode_and_log_chromosome already prints this
        print(f"Best overall fitness (PR AUC): {best_overall_fitness:.4f}")
    else:
        print("No successful evaluation run or no improvement found in the GA.")

if __name__ == '__main__':
    # This allows running the GA directly by executing this file.
    run_genetic_algorithm() 