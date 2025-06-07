import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Imports from other project modules
from pipeline_processor import process_data, get_dataset_name # Corrected import
from main import (
    train_model, decode_and_log_chromosome, 
    IMPUTATION_MAP, OUTLIER_MAP, RESAMPLING_MAP, ENCODING_MAP, SCALING_MAP, MODEL_MAP,
    HP_IMPUTATION_KNN_N_NEIGHBORS, HP_IMPUTATION_MISSFOREST_N_ESTIMATORS, HP_IMPUTATION_MISSFOREST_MAX_ITER,
    HP_OUTLIER_IF_N_ESTIMATORS, HP_OUTLIER_IF_CONTAMINATION, HP_OUTLIER_IQR_MULTIPLIER,
    HP_RESAMPLING_ROS_STRATEGY, HP_RESAMPLING_SMOTE_K_NEIGHBORS, HP_RESAMPLING_SMOTE_STRATEGY,
    HP_RESAMPLING_ADASYN_N_NEIGHBORS, HP_RESAMPLING_ADASYN_STRATEGY,
    HP_ENCODING_ONEHOT_MAX_CARDINALITY, HP_ENCODING_ONEHOT_DROP, HP_ENCODING_LSA_N_COMPONENTS,
    HP_ENCODING_LSA_NGRAM_MAX, HP_ENCODING_W2V_DIM, HP_ENCODING_W2V_WINDOW,
    HP_SCALING_STANDARD_WITH_MEAN, HP_SCALING_STANDARD_WITH_STD,
    HP_MODEL_LOGREG_C, HP_MODEL_LOGREG_PENALTY_SOLVER, HP_MODEL_LOGREG_CLASS_WEIGHT, HP_MODEL_LOGREG_MAX_ITER,
    HP_MODEL_RF_N_ESTIMATORS, HP_MODEL_RF_MAX_DEPTH, HP_MODEL_RF_MIN_SAMPLES_SPLIT, HP_MODEL_RF_MIN_SAMPLES_LEAF,
    HP_MODEL_GB_N_ESTIMATORS, HP_MODEL_GB_LEARNING_RATE, HP_MODEL_GB_MAX_DEPTH, HP_MODEL_GB_SUBSAMPLE,
    HP_MODEL_NN_LAYERS, HP_MODEL_NN_DROPOUT, HP_MODEL_NN_LR, HP_MODEL_NN_BATCH_SIZE
)

# --- GA Parameters ---
POPULATION_SIZE = 10  
NUM_GENERATIONS = 8 
ELITISM_PERCENT = 0.25 
MUTATION_RATE = 0.1   
TOURNAMENT_SIZE = 3  


GENE_CHOICES_COUNT = [
    len(IMPUTATION_MAP), # Imputation Method (0)
    max(len(HP_IMPUTATION_KNN_N_NEIGHBORS), len(HP_IMPUTATION_MISSFOREST_N_ESTIMATORS)), # Imputation HP1 (1)
    len(HP_IMPUTATION_MISSFOREST_MAX_ITER), # Imputation HP2 (2)
    
    len(OUTLIER_MAP),    # Outlier Method (3)
    max(len(HP_OUTLIER_IF_N_ESTIMATORS), len(HP_OUTLIER_IQR_MULTIPLIER)), # Outlier HP1 (4)
    len(HP_OUTLIER_IF_CONTAMINATION), # Outlier HP2 (5)
    
    len(RESAMPLING_MAP), # Resampling Method (6)
    max(len(HP_RESAMPLING_ROS_STRATEGY), len(HP_RESAMPLING_SMOTE_K_NEIGHBORS), len(HP_RESAMPLING_ADASYN_N_NEIGHBORS)), # Resampling HP1 (7)
    max(len(HP_RESAMPLING_SMOTE_STRATEGY), len(HP_RESAMPLING_ADASYN_STRATEGY)), # Resampling HP2 (8)
    
    len(ENCODING_MAP),   # Encoding Method (9)
    max(len(HP_ENCODING_ONEHOT_MAX_CARDINALITY), len(HP_ENCODING_LSA_N_COMPONENTS), len(HP_ENCODING_W2V_DIM)), # Encoding HP1 (10)
    max(len(HP_ENCODING_ONEHOT_DROP), len(HP_ENCODING_LSA_NGRAM_MAX), len(HP_ENCODING_W2V_WINDOW)), # Encoding HP2 (11)
    
    len(SCALING_MAP),    # Scaling Method (12)
    len(HP_SCALING_STANDARD_WITH_MEAN),      # Scaling HP1 (13)
    len(HP_SCALING_STANDARD_WITH_STD),       # Scaling HP2 (14)
    
    len(MODEL_MAP),      # Model Method (15)
    max(len(HP_MODEL_LOGREG_C), len(HP_MODEL_RF_N_ESTIMATORS), len(HP_MODEL_GB_N_ESTIMATORS), len(HP_MODEL_NN_LAYERS)), # Model HP1 (16)
    max(len(HP_MODEL_LOGREG_PENALTY_SOLVER), len(HP_MODEL_RF_MAX_DEPTH), len(HP_MODEL_GB_LEARNING_RATE), len(HP_MODEL_NN_DROPOUT)), # Model HP2 (17)
    max(len(HP_MODEL_LOGREG_CLASS_WEIGHT), len(HP_MODEL_RF_MIN_SAMPLES_SPLIT), len(HP_MODEL_GB_MAX_DEPTH), len(HP_MODEL_NN_LR)), # Model HP3 (18)
    max(len(HP_MODEL_LOGREG_MAX_ITER), len(HP_MODEL_RF_MIN_SAMPLES_LEAF), len(HP_MODEL_GB_SUBSAMPLE), len(HP_MODEL_NN_BATCH_SIZE)) # Model HP4 (19)
]

if len(GENE_CHOICES_COUNT) != 20:
    raise ValueError(f"GENE_CHOICES_COUNT should have 20 elements, but has {len(GENE_CHOICES_COUNT)}")

def initialize_individual():
    """Initializes a single random 20-gene chromosome."""
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
        return initialize_individual()

    actual_tournament_size = min(tournament_size, len(population_with_fitness))
    if actual_tournament_size == 0:
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
    mutated_chromosome = list(chromosome) 
    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            if gene_choices_count[i] > 0: 
                current_value = mutated_chromosome[i]
                new_value = np.random.randint(0, gene_choices_count[i])

                if gene_choices_count[i] > 1:
                    while new_value == current_value:
                        new_value = np.random.randint(0, gene_choices_count[i])
                mutated_chromosome[i] = new_value
 
    return mutated_chromosome

def evaluate_chromosome(chromosome_genes, train_path_ga, test_path_ga, target_column_ga, base_research_path_ga, gen_learning_curves_ga):
    """Evaluates a single chromosome and returns its fitness (PR AUC) and model type."""
    decoded_pipeline_params = decode_and_log_chromosome(chromosome_genes) 
    
    # Initialize model_type to None in case of early failure
    model_type_for_return = None

    if decoded_pipeline_params is None:
        return -1.0, model_type_for_return # Low fitness for errors

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
    
    model_type_for_return = current_model_type # Assign actual model type if decoding was successful

    fitness = -1.0 
    try:
        print(f"\n[GA - Chromosome: {chromosome_genes}] Calling process_data...") # Reduced verbosity
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
            print(f"Data processing failed for chromosome {chromosome_genes}. Skipping model training.") # Keep for potential error feedback
            return -1.0, model_type_for_return # Return model_type even on processing failure
            
        # print(f"[GA - Chromosome: {chromosome_genes}] Calling train_model for model: {current_model_type}...") # Reduced verbosity
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
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (AUPRC): {fitness:.4f}") 
                elif 'average_precision_weighted' in metric_source_dict and current_model_type != 'neural_network': 
                    fitness = metric_source_dict['average_precision_weighted']
                    fitness_metric_name = "Avg_Precision_Weighted"
                    if fitness is None or pd.isna(fitness):
                        print(f"Warning: Weighted Average Precision for {chromosome_genes} (eval set: {eval_set_key_used}) is None/NaN. Assigning low fitness.") # Keep for potential error feedback
                        fitness = -0.6
                    else:
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (Avg Precision Weighted): {fitness:.4f}") # Reduced verbosity
                elif 'accuracy' in metric_source_dict: # Fallback to accuracy if other primary metrics are missing
                    fitness = metric_source_dict['accuracy'] 
                    fitness_metric_name = "Accuracy"
                    print(f"Warning: AUPRC and Avg_Precision_Weighted not found in '{eval_set_key_used}' dict for chromosome {chromosome_genes}. Using Accuracy: {fitness:.4f} as fitness.") # Keep for potential error feedback
                    if fitness is None or pd.isna(fitness):
                        fitness = -0.7 # Low fitness for NaN accuracy
                else:
                    print(f"Core fitness metrics (AUPRC, Avg_Precision_Weighted, Accuracy) not found in '{eval_set_key_used}' dict for chromosome {chromosome_genes}. Keys: {metric_source_dict.keys()}. Assigning low fitness.") # Keep for potential error feedback
                    fitness = -0.85
            else: # Fallback path: metrics might be in a flat structure directly under metrics_output
                if 'auprc' in metrics_output:
                    fitness = metrics_output['auprc']
                    eval_set_key_used = 'top-level'
                    fitness_metric_name = "AUPRC"
                    if fitness is None or pd.isna(fitness):
                        print(f"Warning: AUPRC for chromosome {chromosome_genes} (eval set: {eval_set_key_used}) is None/NaN. Assigning low fitness.") # Keep for potential error feedback
                        fitness = -0.51
                    else:
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (AUPRC): {fitness:.4f}") # Reduced verbosity
                elif 'average_precision_weighted' in metrics_output and current_model_type != 'neural_network':
                    fitness = metrics_output['average_precision_weighted']
                    fitness_metric_name = "Avg_Precision_Weighted"
                    eval_set_key_used = 'top-level'
                    if fitness is None or pd.isna(fitness):
                        print(f"Warning: Weighted Average Precision for {chromosome_genes} (eval set: {eval_set_key_used}) is None/NaN. Assigning low fitness.") # Keep for potential error feedback
                        fitness = -0.61
                    else:
                        print(f"Model training completed for {chromosome_genes} (eval set: {eval_set_key_used}). Fitness (Avg Precision Weighted): {fitness:.4f}") # Reduced verbosity
                elif 'accuracy' in metrics_output: # Fallback to accuracy in flat structure
                    fitness = metrics_output['accuracy']
                    fitness_metric_name = "Accuracy"
                    eval_set_key_used = 'top-level'
                    print(f"Warning: AUPRC and Avg_Precision_Weighted not found at top-level for chromosome {chromosome_genes}. Using Accuracy: {fitness:.4f} as fitness.") # Keep for potential error feedback
                    if fitness is None or pd.isna(fitness):
                        fitness = -0.71 # Low fitness for NaN accuracy at top-level
                else:
                    print(f"Neither nested eval dict nor top-level AUPRC/Avg_Prec_Weighted/Accuracy found in metrics_output for chromosome {chromosome_genes}. Metric Keys: {metrics_output.keys()}. Assigning low fitness.") # Keep for potential error feedback
                    fitness = -0.8
        else:
            print(f"Model training failed or did not produce metrics for chromosome {chromosome_genes}. Assigning low fitness.") # Keep for potential error feedback
            fitness = -0.9 
        
        print(f"[GA - Chromosome: {chromosome_genes}] Fitness ({fitness_metric_name}) determination complete.") # Reduced verbosity

    except Exception as e:
        print(f"Error evaluating chromosome {chromosome_genes}: {e}")
        fitness = -1.0 # Ensure low fitness on any exception

    print(f"Final Chromosome: {chromosome_genes}, Decoded Model: {model_type_for_return}, Fitness: {fitness:.4f}")
    return fitness, model_type_for_return

def run_genetic_algorithm():
    """Main function to set up and run the genetic algorithm."""
    # print("\n=== Setting up GA for Credit Score Dataset ===") # Keep for context

    # train_path = "datasets/credit-score-classification/train.csv"
    # test_path = "datasets/credit-score-classification/test.csv"
    # target_column = "Credit_Score"

    # train_path = "datasets/credit-score-classification-manual-cleaned.csv"
    # test_path = None
    # target_column = "Credit_Score"

    train_path = "datasets/diabetes.csv"
    test_path = None
    target_column = "Outcome"

    # train_path = "datasets/UCI_Credit_Card.csv"
    # test_path = None
    # target_column = "default.payment.next.month"

    # train_path = "datasets/Loan_Default.csv"
    # test_path = None
    # target_column = "Status"

    # --- Check for dataset existence ---
    if not os.path.exists(train_path):
        print(f"Error: Training data file not found at: {train_path}")
        return
    if test_path is not None and not os.path.exists(test_path):
        print(f"Error: Test data file not found at: {test_path}")
        return
    # ---
    
    generate_learning_curves = False

    # print("\n=== Starting Genetic Algorithm ===") # Keep for context
    ga_base_research_path = os.path.join("research", get_dataset_name(train_path), "genetic_algorithm_runs")
    os.makedirs(ga_base_research_path, exist_ok=True)

    population = initialize_population(POPULATION_SIZE) 
    best_overall_chromosome = None
    best_overall_fitness = -1.0 
    best_fitness_over_generations = [] 
    all_individuals_details_over_generations = [] 
    
    best_fitness_per_model_type_over_generations = {
        model_name: [] for model_name in MODEL_MAP.values()
    }

    try: 
        for gen in range(NUM_GENERATIONS):
            print(f"\n--- Generation {gen + 1}/{NUM_GENERATIONS} ---") # Keep for progress tracking
            current_generation_details = [] 
            population_eval_results = [] 

            for i, ind_chromosome in enumerate(population):
                print(f"\nEvaluating Individual {i+1}/{POPULATION_SIZE} in Generation {gen+1}")
                current_fitness, current_model_type = evaluate_chromosome(ind_chromosome, train_path, test_path, target_column, ga_base_research_path, generate_learning_curves)
                population_eval_results.append((ind_chromosome, current_fitness, current_model_type))
                current_generation_details.append((current_fitness, current_model_type)) 
                
                if current_fitness > best_overall_fitness:
                    best_overall_fitness = current_fitness
                    best_overall_chromosome = ind_chromosome
                    print(f"*** New best: Gen {gen+1}, Ind {i+1}: {best_overall_chromosome} -> Fit: {best_overall_fitness:.4f} (Model: {current_model_type}) ***") # Keep for important events

            population_eval_results.sort(key=lambda x: x[1], reverse=True)
            population_with_fitness = [(item[0], item[1]) for item in population_eval_results]
            
            if not population_eval_results: 
                print("Warning: Population is empty after evaluation (population_eval_results).") # Keep for potential error feedback
                for model_name in MODEL_MAP.values():
                    best_fitness_per_model_type_over_generations[model_name].append(np.nan)
                best_fitness_over_generations.append(np.nan) # Also for overall best
                all_individuals_details_over_generations.append([]) # No individuals
                break 
            
            current_gen_best_overall_fitness = population_eval_results[0][1]
            current_gen_best_overall_model_type = population_eval_results[0][2]
            best_fitness_over_generations.append(current_gen_best_overall_fitness)
            all_individuals_details_over_generations.append(current_generation_details)

            # Find and store best fitness for each model type in this generation
            fitness_by_model_this_gen = {model_name: [] for model_name in MODEL_MAP.values()}
            for _, fitness, model_type in population_eval_results: # Iterate through sorted results
                if model_type in fitness_by_model_this_gen:
                    fitness_by_model_this_gen[model_type].append(fitness)
            
            for model_name in MODEL_MAP.values():
                if fitness_by_model_this_gen[model_name]: # If this model type was present
                    best_fitness_per_model_type_over_generations[model_name].append(max(fitness_by_model_this_gen[model_name]))
                else: # Model type not present in this generation
                    best_fitness_per_model_type_over_generations[model_name].append(np.nan) # Use NaN for missing data points

            print(f"\nEnd of Generation {gen + 1}. Best overall fitness in this generation: {current_gen_best_overall_fitness:.4f}") # Reduced verbosity
            print(f"Best chromosome this generation: {population_eval_results[0][0]} (Model: {current_gen_best_overall_model_type}, Fitness: {population_eval_results[0][1]:.4f})") # Reduced verbosity

            if gen < NUM_GENERATIONS - 1:
                next_population = []
                num_elite = int(POPULATION_SIZE * ELITISM_PERCENT)
                
                if num_elite > 0 and population_eval_results:
                    elite_individuals = [item[0] for item in population_eval_results[:num_elite]]
                    next_population.extend(elite_individuals)

                num_offspring_needed = POPULATION_SIZE - len(next_population)
                offspring_generated = 0

                if not population_with_fitness:
                    next_population.extend([initialize_individual() for _ in range(num_offspring_needed)])
                else:
                    # Generate offspring through crossover and mutation
                    while offspring_generated < num_offspring_needed:
                        # Select parents using population_with_fitness (chromosome, fitness)
                        parent1 = select_parent_tournament(population_with_fitness, TOURNAMENT_SIZE)
                        parent2 = select_parent_tournament(population_with_fitness, TOURNAMENT_SIZE)
                        
                        offspring1, offspring2 = crossover(parent1, parent2)
                        
                        mutated_offspring1 = mutate(offspring1, MUTATION_RATE, GENE_CHOICES_COUNT)
                        if offspring_generated < num_offspring_needed:
                            next_population.append(mutated_offspring1)
                            offspring_generated += 1
                        
                        if offspring_generated < num_offspring_needed:
                            mutated_offspring2 = mutate(offspring2, MUTATION_RATE, GENE_CHOICES_COUNT)
                            next_population.append(mutated_offspring2)
                            offspring_generated += 1
                
                population = next_population[:POPULATION_SIZE] 
                if not population and POPULATION_SIZE > 0:
                    population = initialize_population(POPULATION_SIZE)

    finally: 
        if best_fitness_over_generations:
            plt.figure(figsize=(15, 9)) # Adjusted figure size
            generations_x_axis = np.array(range(1, len(best_fitness_over_generations) + 1))

            model_colors = {
                'logistic_regression': 'cyan',
                'random_forest': 'green',
                'gradient_boosting': 'blue',
                'neural_network': 'purple',
                None: 'lightgrey' 
            }
            model_names_display = {
                'logistic_regression': 'Logistic Regression Best',
                'random_forest': 'Random Forest Best',
                'gradient_boosting': 'Gradient Boosting Best',
                'neural_network': 'Neural Network Best',
                None: 'Undefined/Error'
            }
            model_markers = { # Different markers for lines
                'logistic_regression': 's', # Square
                'random_forest': '^', # Triangle up
                'gradient_boosting': 'D', # Diamond
                'neural_network': 'P', # Plus (filled)
            }
            
            legend_handles_map = {} # Using a map for easier handle management

            for model_name, original_fitness_list in best_fitness_per_model_type_over_generations.items():
                if model_name in model_colors: # Ensure we have a color/marker for it
                    
                    # Create a plotting-specific list by carrying forward last known fitness for NaNs
                    plot_fitness_list_for_line = []
                    last_valid_fitness = np.nan 
                    for f_val in original_fitness_list:
                        if pd.isna(f_val): # Using pd.isna for robust NaN check
                            plot_fitness_list_for_line.append(last_valid_fitness) # Carry forward
                        else:
                            plot_fitness_list_for_line.append(f_val)
                            last_valid_fitness = f_val # Update last_valid_fitness

                    # Only plot if there was at least one actual data point for this model
                    if not all(pd.isna(f) for f in original_fitness_list):
                        # Plot the continuous line (using potentially carried-forward values)
                        line_plot, = plt.plot(generations_x_axis, plot_fitness_list_for_line, 
                                             linestyle='--', 
                                             linewidth=1.5, 
                                             color=model_colors[model_name],
                                             label=model_names_display.get(model_name, model_name)
                                             # No marker on this line plot itself
                                             )
                        legend_handles_map[model_names_display.get(model_name, model_name)] = line_plot

                        # Now, plot markers only at actual data points from original_fitness_list
                        actual_data_x_coords = [generations_x_axis[i] for i, y_val in enumerate(original_fitness_list) if not pd.isna(y_val)]
                        actual_data_y_coords = [y_val for y_val in original_fitness_list if not pd.isna(y_val)]
                        
                        if actual_data_x_coords: 
                            plt.plot(actual_data_x_coords, actual_data_y_coords, 
                                     marker=model_markers.get(model_name, '.'), 
                                     linestyle='None', # Crucial: no line for this plot, just markers
                                     color=model_colors[model_name],
                                     markersize=7
                                     )
            
            # 3. Plot the overall best fitness line (highlighted)
            best_overall_line, = plt.plot(generations_x_axis, best_fitness_over_generations, marker='o', linestyle='-', color='red', 
                                          linewidth=3, markersize=9, label='Overall Best Fitness')
            legend_handles_map['Overall Best Fitness'] = best_overall_line
            
            plt.title('GA Fitness Progression: Best per Model Type, Individuals & Overall Best', fontsize=16)
            plt.xlabel('Generation', fontsize=14)
            plt.ylabel('Fitness (AUPRC)', fontsize=14)
            plt.xticks(generations_x_axis, fontsize=12)
            plt.yticks(fontsize=12)
            
            # Create legend with a specific order
            preferred_order = ['Overall Best Fitness']
            # Add model type best lines to preferred order
            for mt_key in MODEL_MAP.values():
                display_name = model_names_display.get(mt_key)
                if display_name in legend_handles_map and display_name not in preferred_order:
                    preferred_order.append(display_name)

            final_handles = [legend_handles_map[label] for label in preferred_order if label in legend_handles_map]
            final_labels = [label for label in preferred_order if label in legend_handles_map]

            if final_handles: # Only show legend if there are items to show
                plt.legend(handles=final_handles, labels=final_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, title="Legend", title_fontsize="12")
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
            else:
                plt.tight_layout() # Standard layout if no legend
            
            plt.grid(True, linestyle=':', alpha=0.6)
            plot_save_path = os.path.join(ga_base_research_path, "ga_fitness_progression_detailed_v2.png") # New name
            try:
                plt.savefig(plot_save_path)
                print(f"\nFitness progression plot saved to {plot_save_path}") # Keep for confirmation
            except Exception as e:
                print(f"Error saving fitness progression plot: {e}") # Keep for error feedback
            plt.close()

    print("\n=== Genetic Algorithm Finished ===") # Keep for context
    if best_overall_chromosome:
        print(f"Best overall chromosome found across all generations: {best_overall_chromosome}") # Keep for results
        print("Details of the Best Chromosome:") # Keep for results
        decode_and_log_chromosome(best_overall_chromosome) 
        print(f"Best overall fitness (AUPRC): {best_overall_fitness:.4f}") # Keep for results
    else:
        print("No successful evaluation run or no improvement found in the GA.") # Keep for context

if __name__ == '__main__':
    run_genetic_algorithm() 