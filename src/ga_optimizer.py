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
POPULATION_SIZE = 20  # Should be ideally larger for GA, e.g., 20-50
NUM_GENERATIONS = 12 # Should be ideally larger for GA, e.g., 20-100
ELITISM_PERCENT = 0.25  # Percentage of population to carry over as elite (25%)
MUTATION_RATE = 0.1     # Probability of a gene mutating
TOURNAMENT_SIZE = 3     # Size of the tournament for parent selection

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
    individual = [np.random.randint(0, max_val) if max_val > 0 else 0 for max_val in GENE_CHOICES_COUNT]
    return individual

def initialize_population(pop_size):
    """Initializes a population of random chromosomes."""
    return [initialize_individual() for _ in range(pop_size)]

def select_parent_tournament(population_with_fitness, tournament_size):
    """Selects a single parent using tournament selection.
    Args:
        population_with_fitness: List of tuples (chromosome, fitness_score).
        tournament_size: Number of individuals to participate in the tournament.
    Returns:
        A single chromosome (the winner of the tournament).
    """
    if not population_with_fitness:
        print("Warning: Attempted to select parent from empty or invalid population_with_fitness. Returning new random individual.")
        return initialize_individual()

    actual_tournament_size = min(tournament_size, len(population_with_fitness))
    if actual_tournament_size == 0:
         print("Warning: Tournament size became 0. Returning new random individual.")
         return initialize_individual()

    # Select indices for the tournament
    tournament_indices = np.random.choice(len(population_with_fitness), size=actual_tournament_size, replace=False)
    tournament_contestants = [population_with_fitness[i] for i in tournament_indices]
    
    # Sort by fitness (descending) and pick the best (highest fitness)
    winner = max(tournament_contestants, key=lambda x: x[1])
    return winner[0] # Return the chromosome of the winner

def crossover(parent1, parent2):
    """Performs single-point crossover between two parents.
    Args:
        parent1: The first parent chromosome (list of genes).
        parent2: The second parent chromosome (list of genes).
    Returns:
        Two offspring chromosomes (tuple of two lists of genes).
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same chromosome length for crossover.")
    if len(parent1) < 2: # Crossover point needs at least 1 gene on each side
        return list(parent1), list(parent2) # No crossover if too short

    # Choose a random crossover point (ensuring it's not at the very ends)
    c_point = np.random.randint(1, len(parent1)) 
    
    offspring1 = parent1[:c_point] + parent2[c_point:]
    offspring2 = parent2[:c_point] + parent1[c_point:]
    
    return offspring1, offspring2

def mutate(chromosome, mutation_rate, gene_choices_count):
    """Performs mutation on a chromosome.
    Args:
        chromosome: The chromosome to mutate (list of genes).
        mutation_rate: The probability of each gene mutating.
        gene_choices_count: List of max values for each gene, to ensure valid mutations.
    Returns:
        The mutated chromosome (list of genes).
    """
    mutated_chromosome = list(chromosome) # Work on a copy
    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            if gene_choices_count[i] > 0: # Ensure there are choices for this gene
                # Simple mutation: pick a new random valid integer for this gene
                # To ensure it's different (optional, but can be good for exploration):
                current_value = mutated_chromosome[i]
                new_value = np.random.randint(0, gene_choices_count[i])
                # If gene_choices_count[i] is 1 (only one option), it can't change.
                # If it's > 1, try to ensure the new value is different.
                if gene_choices_count[i] > 1:
                    while new_value == current_value:
                        new_value = np.random.randint(0, gene_choices_count[i])
                mutated_chromosome[i] = new_value
            # If gene_choices_count[i] is 0, it means no valid choice, so can't mutate.
            # This shouldn't happen if GENE_CHOICES_COUNT is correctly defined based on HP maps.
    return mutated_chromosome

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
            save_run_results=False # Explicitly set to False for GA runs
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
                        fitness_metric_name = "AUPRC"
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (AUPRC): {fitness:.4f}")
                elif 'average_precision_weighted' in metric_source_dict and current_model_type != 'neural_network': 
                    fitness = metric_source_dict['average_precision_weighted']
                    fitness_metric_name = "Avg_Precision_Weighted"
                    if fitness is None or pd.isna(fitness):
                        print(f"Warning: Weighted Average Precision for {chromosome_genes} (eval set: {eval_set_key_used}) is None/NaN. Assigning low fitness.")
                        fitness = -0.6
                    else:
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (Avg Precision Weighted): {fitness:.4f}")
                elif 'accuracy' in metric_source_dict: # Fallback to accuracy if other primary metrics are missing
                    fitness = metric_source_dict['accuracy'] 
                    fitness_metric_name = "Accuracy"
                    print(f"Warning: AUPRC and Avg_Precision_Weighted not found in '{eval_set_key_used}' dict for chromosome {chromosome_genes}. Using Accuracy: {fitness:.4f} as fitness.")
                    if fitness is None or pd.isna(fitness):
                        fitness = -0.7 # Low fitness for NaN accuracy
                else:
                    print(f"Core fitness metrics (AUPRC, Avg_Precision_Weighted, Accuracy) not found in '{eval_set_key_used}' dict for chromosome {chromosome_genes}. Keys: {metric_source_dict.keys()}. Assigning low fitness.")
                    fitness = -0.85
            else: # Fallback path: metrics might be in a flat structure directly under metrics_output
                if 'auprc' in metrics_output:
                    fitness = metrics_output['auprc']
                    eval_set_key_used = 'top-level'
                    fitness_metric_name = "AUPRC"
                    if fitness is None or pd.isna(fitness):
                        print(f"Warning: AUPRC for chromosome {chromosome_genes} (eval set: {eval_set_key_used}) is None/NaN. Assigning low fitness.")
                        fitness = -0.51 
                    else:
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (AUPRC): {fitness:.4f}")
                elif 'average_precision_weighted' in metrics_output and current_model_type != 'neural_network':
                    fitness = metrics_output['average_precision_weighted']
                    fitness_metric_name = "Avg_Precision_Weighted"
                    eval_set_key_used = 'top-level'
                    if fitness is None or pd.isna(fitness):
                        print(f"Warning: Weighted Average Precision for {chromosome_genes} (eval set: {eval_set_key_used}) is None/NaN. Assigning low fitness.")
                        fitness = -0.61
                    else:
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (Avg Precision Weighted): {fitness:.4f}")
                elif 'accuracy' in metrics_output: # Fallback to accuracy in flat structure
                    fitness = metrics_output['accuracy']
                    fitness_metric_name = "Accuracy"
                    eval_set_key_used = 'top-level'
                    print(f"Warning: AUPRC and Avg_Precision_Weighted not found at top-level for chromosome {chromosome_genes}. Using Accuracy: {fitness:.4f} as fitness.")
                    if fitness is None or pd.isna(fitness):
                        fitness = -0.71 # Low fitness for NaN accuracy at top-level
                else:
                    print(f"Neither nested eval dict nor top-level AUPRC/Avg_Prec_Weighted/Accuracy found in metrics_output for chromosome {chromosome_genes}. Metric Keys: {metrics_output.keys()}. Assigning low fitness.")
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
    # train_path = "datasets/credit-score-classification/train.csv"
    # test_path = "datasets/credit-score-classification/test.csv"
    # target_column = "Credit_Score"

    # train_path = "datasets/credit-score-classification-manual-cleaned.csv"
    # test_path = None
    # target_column = "Credit_Score"

    train_path = "datasets/diabetes.csv"
    test_path = None
    target_column = "Outcome"

    # --- Check for dataset existence ---
    if not os.path.exists(train_path):
        print(f"Error: Training data file not found at: {train_path}")
        return
    if test_path is not None and not os.path.exists(test_path):
        print(f"Error: Test data file not found at: {test_path}")
        return
    # ---
    
    generate_learning_curves = False

    print("\n=== Starting Genetic Algorithm ===")
    ga_base_research_path = os.path.join("research", get_dataset_name(train_path), "genetic_algorithm_runs")
    os.makedirs(ga_base_research_path, exist_ok=True)

    population = initialize_population(POPULATION_SIZE) # Uses new 19-gene init
    best_overall_chromosome = None
    best_overall_fitness = -1.0 
    best_fitness_over_generations = [] 
    all_individuals_fitness_over_generations = [] # New: To store all fitness scores per generation

    try: 
        for gen in range(NUM_GENERATIONS):
            print(f"\n--- Generation {gen + 1}/{NUM_GENERATIONS} ---")
            population_with_fitness = []
            current_generation_fitness_scores = [] # New: To store fitness scores for the current generation

            for i, ind_chromosome in enumerate(population):
                print(f"\nEvaluating Individual {i+1}/{POPULATION_SIZE} in Generation {gen+1}")
                current_fitness = evaluate_chromosome(ind_chromosome, train_path, test_path, target_column, ga_base_research_path, generate_learning_curves)
                population_with_fitness.append((ind_chromosome, current_fitness))
                current_generation_fitness_scores.append(current_fitness) # New: Store current fitness
                
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
            all_individuals_fitness_over_generations.append(current_generation_fitness_scores) # New: Add all scores for this gen

            print(f"\nEnd of Generation {gen + 1}. Best fitness in this generation: {current_gen_best_fitness:.4f}")
            print(f"Best chromosome this generation: {population_with_fitness[0][0]}")
            
            if gen < NUM_GENERATIONS - 1:
                next_population = []
                num_elite = int(POPULATION_SIZE * ELITISM_PERCENT)
                
                # Elitism: Carry over the best individuals
                if num_elite > 0 and population_with_fitness:
                    elite_individuals = [item[0] for item in population_with_fitness[:num_elite]]
                    next_population.extend(elite_individuals)
                    # print(f"Carried over {len(elite_individuals)} elite individuals.")

                # Fill the rest of the population with offspring
                num_offspring_needed = POPULATION_SIZE - len(next_population)
                offspring_generated = 0

                # Ensure population_with_fitness is not empty for parent selection
                if not population_with_fitness:
                    print("Warning: Current population is empty. Cannot select parents. Filling with random individuals.")
                    next_population.extend([initialize_individual() for _ in range(num_offspring_needed)])
                else:
                    # Generate offspring through crossover and mutation
                    while offspring_generated < num_offspring_needed:
                        # Select parents
                        parent1 = select_parent_tournament(population_with_fitness, TOURNAMENT_SIZE)
                        parent2 = select_parent_tournament(population_with_fitness, TOURNAMENT_SIZE)
                        
                        # Perform crossover to get two offspring
                        offspring1, offspring2 = crossover(parent1, parent2)
                        
                        # Mutate offspring1
                        mutated_offspring1 = mutate(offspring1, MUTATION_RATE, GENE_CHOICES_COUNT)
                        if offspring_generated < num_offspring_needed:
                            next_population.append(mutated_offspring1)
                            offspring_generated += 1
                        
                        # Mutate offspring2 (if still needed)
                        if offspring_generated < num_offspring_needed:
                            mutated_offspring2 = mutate(offspring2, MUTATION_RATE, GENE_CHOICES_COUNT)
                            next_population.append(mutated_offspring2)
                            offspring_generated += 1
                
                population = next_population[:POPULATION_SIZE] # Ensure population size is correct
                if not population and POPULATION_SIZE > 0:
                    print("Warning: Next population became empty after operators. Re-initializing with random individuals.")
                    population = initialize_population(POPULATION_SIZE)
                # print(f"Next generation created with {len(population)} individuals.")

    finally: 
        if best_fitness_over_generations:
            plt.figure(figsize=(12, 7)) # Slightly larger figure
            generations_x_axis = range(1, len(best_fitness_over_generations) + 1)

            # Plot all individual fitness scores for each generation as background
            for gen_idx, fitness_scores_in_gen in enumerate(all_individuals_fitness_over_generations):
                # Create an x-axis array for this generation, can add slight jitter for visibility if needed
                # For simplicity, plotting all at the same x-coordinate for the generation
                x_values_for_gen = [generations_x_axis[gen_idx]] * len(fitness_scores_in_gen)
                plt.scatter(x_values_for_gen, fitness_scores_in_gen, color='lightblue', alpha=0.5, s=10, label='_nolegend_') # s is size

            # Plot the best fitness line (highlighted)
            plt.plot(generations_x_axis, best_fitness_over_generations, marker='o', linestyle='-', color='blue', linewidth=2, label='Best Fitness per Generation')
            
            plt.title('GA Fitness Progression: All Individuals and Best per Generation')
            plt.xlabel('Generation')
            plt.ylabel('Fitness (AUPRC)')
            plt.xticks(generations_x_axis)
            plt.legend()
            plt.grid(True)
            plot_save_path = os.path.join(ga_base_research_path, "ga_fitness_progression_detailed.png")
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