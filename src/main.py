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
# Step Method Maps
IMPUTATION_MAP = {0: 'knn', 1: 'median', 2: 'missforest'}
OUTLIER_MAP = {0: 'none', 1: 'isolation_forest', 2: 'iqr'}
RESAMPLING_MAP = {0: 'none', 1: 'oversample', 2: 'smote', 3: 'adasyn'}
ENCODING_MAP = {0: 'onehot', 1: 'label', 2: 'lsa', 3: 'word2vec'}
SCALING_MAP = {0: 'none', 1: 'standard', 2: 'minmax'}
MODEL_MAP = {0: 'logistic_regression', 1: 'random_forest', 2: 'gradient_boosting', 3: 'neural_network'}

# --- Hyperparameter Maps ---
# Imputation HP Maps
HP_IMPUTATION_KNN_N_NEIGHBORS = {0: 3, 1: 5, 2: 7, 3: 10} # For гп1_имп with knn
HP_IMPUTATION_MISSFOREST_N_ESTIMATORS = {0: 50, 1: 100, 2: 150} # For гп1_имп with missforest
HP_IMPUTATION_MISSFOREST_MAX_ITER = {0: 5, 1: 10, 2: 15} # For гп2_имп with missforest

# Outlier HP Maps
HP_OUTLIER_IF_N_ESTIMATORS = {0: 50, 1: 100, 2: 150} # For гп1_выбр with isolation_forest
HP_OUTLIER_IF_CONTAMINATION = {0: 'auto', 1: 0.01, 2: 0.05, 3: 0.1} # For гп2_выбр with isolation_forest
HP_OUTLIER_IQR_MULTIPLIER = {0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0} # For гп1_выбр with iqr

# Resampling HP Maps
HP_RESAMPLING_ROS_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.75} # For гп1_рес with oversample
HP_RESAMPLING_SMOTE_K_NEIGHBORS = {0: 3, 1: 5, 2: 7} # For гп1_рес with smote
HP_RESAMPLING_SMOTE_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.75} # For гп2_рес with smote
HP_RESAMPLING_ADASYN_N_NEIGHBORS = {0: 3, 1: 5, 2: 7} # For гп1_рес with adasyn
HP_RESAMPLING_ADASYN_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.75} # For гп2_рес with adasyn

# Encoding HP Maps
HP_ENCODING_ONEHOT_MAX_CARDINALITY = {0: 20, 1: 50, 2: 100} # For гп1_код with onehot
HP_ENCODING_ONEHOT_DROP = {0: None, 1: 'first'} # For гп2_код with onehot
HP_ENCODING_LSA_N_COMPONENTS = {0: 10, 1: 25, 2: 50} # For гп1_код with lsa
HP_ENCODING_LSA_NGRAM_MAX = {0: 1, 1: 2} # For гп2_код with lsa (ngram_range=(1, val))
HP_ENCODING_W2V_DIM = {0: 25, 1: 50, 2: 100} # For гп1_код with word2vec
HP_ENCODING_W2V_WINDOW = {0: 1, 1: 3, 2: 5} # For гп2_код with word2vec

# Scaling HP Maps
HP_SCALING_STANDARD_WITH_MEAN = {0: True, 1: False} # For гп1_масшт with standard
HP_SCALING_STANDARD_WITH_STD = {0: True, 1: False} # For гп2_масшт with standard

# Model HP Maps
HP_MODEL_LOGREG_C = {0: 0.01, 1: 0.1, 2: 1.0, 3: 10.0} # For гп1_мод with logistic_regression
HP_MODEL_LOGREG_PENALTY_SOLVER = { # For гп2_мод with logistic_regression
    0: {'penalty': 'l2', 'solver': 'lbfgs', 'l1_ratio': None}, 1: {'penalty': 'l1', 'solver': 'liblinear', 'l1_ratio': None},
    2: {'penalty': 'l2', 'solver': 'liblinear', 'l1_ratio': None}, 3: {'penalty': 'l1', 'solver': 'saga', 'l1_ratio': None},
    4: {'penalty': 'l2', 'solver': 'saga', 'l1_ratio': None}, 5: {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5}
}
HP_MODEL_LOGREG_CLASS_WEIGHT = {0: None, 1: 'balanced'} # For гп3_мод with logistic_regression

HP_MODEL_RF_N_ESTIMATORS = {0: 50, 1: 100, 2: 200} # For гп1_мод with random_forest
HP_MODEL_RF_MAX_DEPTH = {0: 5, 1: 10, 2: 20, 3: None} # For гп2_мод with random_forest
HP_MODEL_RF_MIN_SAMPLES_SPLIT = {0: 2, 1: 5, 2: 10} # For гп3_мод with random_forest

HP_MODEL_GB_N_ESTIMATORS = {0: 50, 1: 100, 2: 200} # For гп1_мод with gradient_boosting
HP_MODEL_GB_LEARNING_RATE = {0: 0.01, 1: 0.05, 2: 0.1} # For гп2_мод with gradient_boosting
HP_MODEL_GB_MAX_DEPTH = {0: 3, 1: 5, 2: 7} # For гп3_мод with gradient_boosting

HP_MODEL_NN_LAYERS = { # For гп1_мод with neural_network
    0: (32,), 1: (64,), 2: (128,),
    3: (64, 32), 4: (128, 64)
}
HP_MODEL_NN_DROPOUT = {0: 0.0, 1: 0.2, 2: 0.4} # For гп2_мод with neural_network
HP_MODEL_NN_LR = {0: 0.0005, 1: 0.001, 2: 0.005} # For гп3_мод with neural_network

# --- Gene Order and Structure ---
# Each preprocessing step: [method_idx, hp1_idx, hp2_idx]
# Model step: [method_idx, hp1_idx, hp2_idx, hp3_idx]
# Total genes = 5 * 3 + 4 = 19
GENE_DESCRIPTIONS = [
    "Imputation Method", "Imputation HP1", "Imputation HP2",
    "Outlier Method", "Outlier HP1", "Outlier HP2",
    "Resampling Method", "Resampling HP1", "Resampling HP2",
    "Encoding Method", "Encoding HP1", "Encoding HP2",
    "Scaling Method", "Scaling HP1", "Scaling HP2",
    "Model Method", "Model HP1", "Model HP2", "Model HP3"
]

# --- GA Parameters (placeholders, primary definition in ga_optimizer.py) ---
# POPULATION_SIZE = 5 
# NUM_GENERATIONS = 3 

# --- Chromosome Definition and Decoding ---
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
    """Decodes a 19-gene chromosome and prints its meaning, returning a structured dict."""
    print("\n--- Chromosome Definition ---")
    if len(chromosome) != 19:
        print(f"Error: Chromosome must have 19 genes, got {len(chromosome)}.")
        return None

    decoded_chromosome = {
        'chromosome_values': list(chromosome),
        'description': {},
        'pipeline_params': {}
    }

    # Helper to safely get from map or return a default string
    def safe_get(gene_val, val_map, default_prefix="Unknown"):
        return val_map.get(gene_val, f"{default_prefix} (raw_val: {gene_val})")

    # Helper to get HP value, with a note if index is out of bounds for the specific map
    def get_hp_value(hp_gene_val, hp_map, hp_name, method_name):
        val = hp_map.get(hp_gene_val)
        if val is None:
            # print(f"  Warning: HP gene value {hp_gene_val} for {hp_name} of {method_name} is out of defined range. Method may use default or ignore.")
            return None 
        return val

    # 1. Imputation (Genes 0, 1, 2)
    imp_method_idx, imp_hp1_idx, imp_hp2_idx = chromosome[0], chromosome[1], chromosome[2]
    imp_method = IMPUTATION_MAP.get(imp_method_idx, "Unknown Imputation")
    decoded_chromosome['description']['imputation'] = f"{imp_method} (Method Idx: {imp_method_idx})"
    decoded_chromosome['pipeline_params']['imputation_method'] = imp_method
    imp_params = {}
    if imp_method == 'knn':
        imp_params['n_neighbors'] = get_hp_value(imp_hp1_idx, HP_IMPUTATION_KNN_N_NEIGHBORS, "n_neighbors", imp_method)
        decoded_chromosome['description']['imputation_hp1'] = f"n_neighbors: {imp_params.get('n_neighbors')} (HP1 Idx: {imp_hp1_idx})"
        decoded_chromosome['description']['imputation_hp2'] = f"Unused (HP2 Idx: {imp_hp2_idx})"
    elif imp_method == 'missforest':
        imp_params['n_estimators'] = get_hp_value(imp_hp1_idx, HP_IMPUTATION_MISSFOREST_N_ESTIMATORS, "n_estimators", imp_method)
        imp_params['max_iter'] = get_hp_value(imp_hp2_idx, HP_IMPUTATION_MISSFOREST_MAX_ITER, "max_iter", imp_method)
        decoded_chromosome['description']['imputation_hp1'] = f"n_estimators: {imp_params.get('n_estimators')} (HP1 Idx: {imp_hp1_idx})"
        decoded_chromosome['description']['imputation_hp2'] = f"max_iter: {imp_params.get('max_iter')} (HP2 Idx: {imp_hp2_idx})"
    else: # median or unknown
        decoded_chromosome['description']['imputation_hp1'] = f"Unused (HP1 Idx: {imp_hp1_idx})"
        decoded_chromosome['description']['imputation_hp2'] = f"Unused (HP2 Idx: {imp_hp2_idx})"
    decoded_chromosome['pipeline_params']['imputation_params'] = {k:v for k,v in imp_params.items() if v is not None}

    # 2. Outlier Removal (Genes 3, 4, 5)
    out_method_idx, out_hp1_idx, out_hp2_idx = chromosome[3], chromosome[4], chromosome[5]
    out_method = OUTLIER_MAP.get(out_method_idx, "Unknown Outlier")
    decoded_chromosome['description']['outlier_removal'] = f"{out_method} (Method Idx: {out_method_idx})"
    decoded_chromosome['pipeline_params']['outlier_method'] = out_method
    out_params = {}
    if out_method == 'isolation_forest':
        out_params['n_estimators'] = get_hp_value(out_hp1_idx, HP_OUTLIER_IF_N_ESTIMATORS, "n_estimators", out_method)
        out_params['contamination'] = get_hp_value(out_hp2_idx, HP_OUTLIER_IF_CONTAMINATION, "contamination", out_method)
        decoded_chromosome['description']['outlier_hp1'] = f"n_estimators: {out_params.get('n_estimators')} (HP1 Idx: {out_hp1_idx})"
        decoded_chromosome['description']['outlier_hp2'] = f"contamination: {out_params.get('contamination')} (HP2 Idx: {out_hp2_idx})"
    elif out_method == 'iqr':
        out_params['multiplier'] = get_hp_value(out_hp1_idx, HP_OUTLIER_IQR_MULTIPLIER, "multiplier", out_method)
        decoded_chromosome['description']['outlier_hp1'] = f"multiplier: {out_params.get('multiplier')} (HP1 Idx: {out_hp1_idx})"
        decoded_chromosome['description']['outlier_hp2'] = f"Unused (HP2 Idx: {out_hp2_idx})"
    else: # none or unknown
        decoded_chromosome['description']['outlier_hp1'] = f"Unused (HP1 Idx: {out_hp1_idx})"
        decoded_chromosome['description']['outlier_hp2'] = f"Unused (HP2 Idx: {out_hp2_idx})"
    decoded_chromosome['pipeline_params']['outlier_params'] = {k:v for k,v in out_params.items() if v is not None}

    # 3. Resampling (Genes 6, 7, 8)
    res_method_idx, res_hp1_idx, res_hp2_idx = chromosome[6], chromosome[7], chromosome[8]
    res_method = RESAMPLING_MAP.get(res_method_idx, "Unknown Resampling")
    decoded_chromosome['description']['resampling'] = f"{res_method} (Method Idx: {res_method_idx})"
    decoded_chromosome['pipeline_params']['resampling_method'] = res_method
    res_params = {}
    if res_method == 'oversample': # ROS
        res_params['sampling_strategy'] = get_hp_value(res_hp1_idx, HP_RESAMPLING_ROS_STRATEGY, "sampling_strategy_ros", res_method)
        decoded_chromosome['description']['resampling_hp1'] = f"sampling_strategy: {res_params.get('sampling_strategy')} (HP1 Idx: {res_hp1_idx})"
        decoded_chromosome['description']['resampling_hp2'] = f"Unused (HP2 Idx: {res_hp2_idx})"
    elif res_method == 'smote':
        res_params['k_neighbors'] = get_hp_value(res_hp1_idx, HP_RESAMPLING_SMOTE_K_NEIGHBORS, "k_neighbors", res_method)
        res_params['sampling_strategy'] = get_hp_value(res_hp2_idx, HP_RESAMPLING_SMOTE_STRATEGY, "sampling_strategy_smote", res_method)
        decoded_chromosome['description']['resampling_hp1'] = f"k_neighbors: {res_params.get('k_neighbors')} (HP1 Idx: {res_hp1_idx})"
        decoded_chromosome['description']['resampling_hp2'] = f"sampling_strategy: {res_params.get('sampling_strategy')} (HP2 Idx: {res_hp2_idx})"
    elif res_method == 'adasyn':
        res_params['n_neighbors'] = get_hp_value(res_hp1_idx, HP_RESAMPLING_ADASYN_N_NEIGHBORS, "n_neighbors", res_method)
        res_params['sampling_strategy'] = get_hp_value(res_hp2_idx, HP_RESAMPLING_ADASYN_STRATEGY, "sampling_strategy_adasyn", res_method)
        decoded_chromosome['description']['resampling_hp1'] = f"n_neighbors: {res_params.get('n_neighbors')} (HP1 Idx: {res_hp1_idx})"
        decoded_chromosome['description']['resampling_hp2'] = f"sampling_strategy: {res_params.get('sampling_strategy')} (HP2 Idx: {res_hp2_idx})"
    else: # none or unknown
        decoded_chromosome['description']['resampling_hp1'] = f"Unused (HP1 Idx: {res_hp1_idx})"
        decoded_chromosome['description']['resampling_hp2'] = f"Unused (HP2 Idx: {res_hp2_idx})"
    decoded_chromosome['pipeline_params']['resampling_params'] = {k:v for k,v in res_params.items() if v is not None}

    # 4. Encoding (Genes 9, 10, 11)
    enc_method_idx, enc_hp1_idx, enc_hp2_idx = chromosome[9], chromosome[10], chromosome[11]
    enc_method = ENCODING_MAP.get(enc_method_idx, "Unknown Encoding")
    decoded_chromosome['description']['encoding'] = f"{enc_method} (Method Idx: {enc_method_idx})"
    decoded_chromosome['pipeline_params']['encoding_method'] = enc_method
    enc_params = {}
    if enc_method == 'onehot':
        enc_params['max_cardinality_threshold'] = get_hp_value(enc_hp1_idx, HP_ENCODING_ONEHOT_MAX_CARDINALITY, "max_cardinality", enc_method)
        enc_params['drop'] = get_hp_value(enc_hp2_idx, HP_ENCODING_ONEHOT_DROP, "drop", enc_method)
        decoded_chromosome['description']['encoding_hp1'] = f"max_cardinality_threshold: {enc_params.get('max_cardinality_threshold')} (HP1 Idx: {enc_hp1_idx})"
        decoded_chromosome['description']['encoding_hp2'] = f"drop_strategy: {enc_params.get('drop')} (HP2 Idx: {enc_hp2_idx})"
    elif enc_method == 'lsa':
        enc_params['n_components'] = get_hp_value(enc_hp1_idx, HP_ENCODING_LSA_N_COMPONENTS, "n_components_lsa", enc_method)
        ngram_max = get_hp_value(enc_hp2_idx, HP_ENCODING_LSA_NGRAM_MAX, "ngram_max_lsa", enc_method)
        if ngram_max is not None: enc_params['ngram_range'] = (1, ngram_max)
        decoded_chromosome['description']['encoding_hp1'] = f"n_components: {enc_params.get('n_components')} (HP1 Idx: {enc_hp1_idx})"
        decoded_chromosome['description']['encoding_hp2'] = f"ngram_range_max: {ngram_max} (HP2 Idx: {enc_hp2_idx})"
    elif enc_method == 'word2vec':
        enc_params['embedding_dim'] = get_hp_value(enc_hp1_idx, HP_ENCODING_W2V_DIM, "embedding_dim_w2v", enc_method)
        enc_params['window'] = get_hp_value(enc_hp2_idx, HP_ENCODING_W2V_WINDOW, "window_w2v", enc_method)
        decoded_chromosome['description']['encoding_hp1'] = f"vector_size: {enc_params.get('embedding_dim')} (HP1 Idx: {enc_hp1_idx})"
        decoded_chromosome['description']['encoding_hp2'] = f"window: {enc_params.get('window')} (HP2 Idx: {enc_hp2_idx})"
    else: # label or unknown
        decoded_chromosome['description']['encoding_hp1'] = f"Unused (HP1 Idx: {enc_hp1_idx})"
        decoded_chromosome['description']['encoding_hp2'] = f"Unused (HP2 Idx: {enc_hp2_idx})"
    # Keep drop=None for onehot, filter others if None
    filtered_enc_params = {}
    for k,v in enc_params.items():
        if v is not None or (enc_method == 'onehot' and k == 'drop'):
             filtered_enc_params[k] = v
    decoded_chromosome['pipeline_params']['encoding_params'] = filtered_enc_params

    # 5. Scaling (Genes 12, 13, 14)
    sca_method_idx, sca_hp1_idx, sca_hp2_idx = chromosome[12], chromosome[13], chromosome[14]
    sca_method = SCALING_MAP.get(sca_method_idx, "Unknown Scaling")
    decoded_chromosome['description']['scaling'] = f"{sca_method} (Method Idx: {sca_method_idx})"
    decoded_chromosome['pipeline_params']['scaling_method'] = sca_method
    sca_params = {}
    if sca_method == 'standard':
        sca_params['with_mean'] = get_hp_value(sca_hp1_idx, HP_SCALING_STANDARD_WITH_MEAN, "with_mean", sca_method)
        sca_params['with_std'] = get_hp_value(sca_hp2_idx, HP_SCALING_STANDARD_WITH_STD, "with_std", sca_method)
        decoded_chromosome['description']['scaling_hp1'] = f"with_mean: {sca_params.get('with_mean')} (HP1 Idx: {sca_hp1_idx})"
        decoded_chromosome['description']['scaling_hp2'] = f"with_std: {sca_params.get('with_std')} (HP2 Idx: {sca_hp2_idx})"
    else: # none, minmax, or unknown
        decoded_chromosome['description']['scaling_hp1'] = f"Unused (HP1 Idx: {sca_hp1_idx})"
        decoded_chromosome['description']['scaling_hp2'] = f"Unused (HP2 Idx: {sca_hp2_idx})"
    decoded_chromosome['pipeline_params']['scaling_params'] = {k:v for k,v in sca_params.items() if v is not None}

    # 6. Model (Genes 15, 16, 17, 18)
    model_method_idx, model_hp1_idx, model_hp2_idx, model_hp3_idx = chromosome[15], chromosome[16], chromosome[17], chromosome[18]
    model_type = MODEL_MAP.get(model_method_idx, "Unknown Model")
    decoded_chromosome['description']['model'] = f"{model_type} (Method Idx: {model_method_idx})"
    decoded_chromosome['pipeline_params']['model_type'] = model_type
    model_params = {}
    if model_type == 'logistic_regression':
        model_params['C'] = get_hp_value(model_hp1_idx, HP_MODEL_LOGREG_C, "C", model_type)
        # The solver_penalty_config is directly the dictionary from HP_MODEL_LOGREG_PENALTY_SOLVER
        model_params['solver_penalty_config'] = get_hp_value(model_hp2_idx, HP_MODEL_LOGREG_PENALTY_SOLVER, "solver/penalty", model_type)
        model_params['class_weight'] = get_hp_value(model_hp3_idx, HP_MODEL_LOGREG_CLASS_WEIGHT, "class_weight", model_type)
        decoded_chromosome['description']['model_hp1'] = f"C: {model_params.get('C')} (HP1 Idx: {model_hp1_idx})"
        decoded_chromosome['description']['model_hp2'] = f"Solver/Penalty Config: {model_params.get('solver_penalty_config')} (HP2 Idx: {model_hp2_idx})"
        decoded_chromosome['description']['model_hp3'] = f"Class Weight: {model_params.get('class_weight')} (HP3 Idx: {model_hp3_idx})"
    elif model_type == 'random_forest':
        model_params['n_estimators'] = get_hp_value(model_hp1_idx, HP_MODEL_RF_N_ESTIMATORS, "n_estimators_rf", model_type)
        model_params['max_depth'] = get_hp_value(model_hp2_idx, HP_MODEL_RF_MAX_DEPTH, "max_depth_rf", model_type)
        model_params['min_samples_split'] = get_hp_value(model_hp3_idx, HP_MODEL_RF_MIN_SAMPLES_SPLIT, "min_samples_split_rf", model_type)
        decoded_chromosome['description']['model_hp1'] = f"n_estimators: {model_params.get('n_estimators')} (HP1 Idx: {model_hp1_idx})"
        decoded_chromosome['description']['model_hp2'] = f"max_depth: {model_params.get('max_depth')} (HP2 Idx: {model_hp2_idx})"
        decoded_chromosome['description']['model_hp3'] = f"min_samples_split: {model_params.get('min_samples_split')} (HP3 Idx: {model_hp3_idx})"
    elif model_type == 'gradient_boosting':
        model_params['n_estimators'] = get_hp_value(model_hp1_idx, HP_MODEL_GB_N_ESTIMATORS, "n_estimators_gb", model_type)
        model_params['learning_rate'] = get_hp_value(model_hp2_idx, HP_MODEL_GB_LEARNING_RATE, "learning_rate_gb", model_type)
        model_params['max_depth'] = get_hp_value(model_hp3_idx, HP_MODEL_GB_MAX_DEPTH, "max_depth_gb", model_type)
        decoded_chromosome['description']['model_hp1'] = f"n_estimators: {model_params.get('n_estimators')} (HP1 Idx: {model_hp1_idx})"
        decoded_chromosome['description']['model_hp2'] = f"learning_rate: {model_params.get('learning_rate')} (HP2 Idx: {model_hp2_idx})"
        decoded_chromosome['description']['model_hp3'] = f"max_depth: {model_params.get('max_depth')} (HP3 Idx: {model_hp3_idx})"
    elif model_type == 'neural_network':
        model_params['hidden_layer_sizes'] = get_hp_value(model_hp1_idx, HP_MODEL_NN_LAYERS, "hidden_layer_sizes_nn", model_type)
        model_params['dropout_rate'] = get_hp_value(model_hp2_idx, HP_MODEL_NN_DROPOUT, "dropout_rate_nn", model_type)
        model_params['learning_rate'] = get_hp_value(model_hp3_idx, HP_MODEL_NN_LR, "learning_rate_nn", model_type)
        # Keras specific HPs from problem description (epochs, batch_size, early_stopping_patience, l1_reg, l2_reg)
        # These are not in the current HP_MODEL_NN maps directly from chromosome genes 16,17,18.
        # For now, they will take defaults in ModelTrainer or need to be added to chromosome if GA is to optimize them.
        # For simplicity, we will let ModelTrainer use its defaults for these if not found in model_params.
        decoded_chromosome['description']['model_hp1'] = f"Hidden Layers: {model_params.get('hidden_layer_sizes')} (HP1 Idx: {model_hp1_idx})"
        decoded_chromosome['description']['model_hp2'] = f"Dropout Rate: {model_params.get('dropout_rate')} (HP2 Idx: {model_hp2_idx})"
        decoded_chromosome['description']['model_hp3'] = f"Learning Rate: {model_params.get('learning_rate')} (HP3 Idx: {model_hp3_idx})"
    
    # Filter out None values from model_params, except for specific cases like class_weight or solver_penalty_config if it's a dict
    final_model_params = {}
    for k, v in model_params.items():
        if v is not None:
            final_model_params[k] = v
        elif k == 'class_weight' and v is None: # Allow None for class_weight
             final_model_params[k] = None
        elif k == 'solver_penalty_config' and isinstance(v, dict): # Allow dict even if parts are None (though current map doesn't have None values in the dict)
            final_model_params[k] = v
        # For RF max_depth=None is valid
        elif model_type == 'random_forest' and k == 'max_depth' and v is None:
            final_model_params[k] = None
            
    decoded_chromosome['pipeline_params']['model_params'] = final_model_params

    print(f"Chromosome (raw): {list(chromosome)}")
    print("Decoded Chromosome Details:")
    # Optionally print the detailed description if needed for debugging
    # for i, desc_key_value_pair in enumerate(GENE_DESCRIPTIONS):
    #     gene_cat = desc_key_value_pair.split(" ")[0].lower()
    #     hp_suffix = ""
    #     if "HP1" in desc_key_value_pair: hp_suffix = "_hp1"
    #     elif "HP2" in desc_key_value_pair: hp_suffix = "_hp2"
    #     elif "HP3" in desc_key_value_pair: hp_suffix = "_hp3"
    #     dict_key = gene_cat + hp_suffix if hp_suffix else gene_cat
    #     if dict_key == "modelhp3": dict_key = "model_hp3" # consistency for last model hp

    #     print(f"  Gene {i}: {GENE_DESCRIPTIONS[i]} = {chromosome[i]} -> {decoded_chromosome['description'].get(dict_key, 'N/A')}")

    print(f"  Imputation: {decoded_chromosome['pipeline_params']['imputation_method']}, Params: {decoded_chromosome['pipeline_params']['imputation_params']}")
    print(f"  Outlier Removal: {decoded_chromosome['pipeline_params']['outlier_method']}, Params: {decoded_chromosome['pipeline_params']['outlier_params']}")
    print(f"  Resampling: {decoded_chromosome['pipeline_params']['resampling_method']}, Params: {decoded_chromosome['pipeline_params']['resampling_params']}")
    print(f"  Encoding: {decoded_chromosome['pipeline_params']['encoding_method']}, Params: {decoded_chromosome['pipeline_params']['encoding_params']}")
    print(f"  Scaling: {decoded_chromosome['pipeline_params']['scaling_method']}, Params: {decoded_chromosome['pipeline_params']['scaling_params']}")
    print(f"  Model: {decoded_chromosome['pipeline_params']['model_type']}, Params: {decoded_chromosome['pipeline_params']['model_params']}")
    print("-----------------------------\n")
    
    return decoded_chromosome['pipeline_params']

def train_model(train_data_input, test_data_input, target_column, research_path, 
                model_type='random_forest', model_hyperparameters=None, 
                plot_learning_curves=True, save_run_results=True):
    """
    Trains a model using the specified data and parameters.
    Handles data splitting if test data is missing a target.
    """
    # print(f"\n--- Training Model: {model_type} ---")
    # print(f"Target column: {target_column}")
    # print(f"Research path for this run: {research_path}")
    if model_hyperparameters:
        # print(f"Model hyperparameters received: {model_hyperparameters}")
        pass

    os.makedirs(research_path, exist_ok=True)
    
    # Create ModelTrainer instance
    trainer = ModelTrainer(model_type=model_type, model_hyperparameters=model_hyperparameters, random_state=42)
    
    metrics, feature_importance = trainer.train(
        train_data_input, 
        test_data_input, 
        target_column,
            output_path=research_path, 
            plot_learning_curves=plot_learning_curves
        )
    
    if metrics:
        # print("\n--- Model Evaluation Metrics ---")
        # for key, value in metrics.items():
        #     if key != 'classification_report':
        #         print(f"  {key}: {value}")
        # if 'classification_report' in metrics:
            # print(pd.DataFrame(metrics['classification_report']).transpose())
        
        if save_run_results:
            dataset_name_for_saving = research_path.split(os.sep)[-2] if len(research_path.split(os.sep)) > 1 else "unknown_dataset"
            run_name = research_path.split(os.sep)[-1]
            save_model_results(
                metrics,
                model_type,
                dataset_name_for_saving,
                run_name,
                base_dir="research", # Ensure this matches the structure where save_model_results expects to save
                hyperparameters=model_hyperparameters, # Save HPs used
                feature_importance_df=feature_importance
            )
    else:
        print("Model training did not produce metrics.")
    
    return metrics, feature_importance

def main():
    print("\n=== Processing credit card dataset (Single Run Example with HPs) ===")
    train_path = "datasets/UCI_Credit_Card.csv"
    test_path = None # "datasets/UCI_Credit_Card.csv" # Can use same for test to check flow
    target_column = "default.payment.next.month"
    generate_learning_curves = False 

    # Example Chromosome (19 genes):
    # Imputation: knn (0), n_neighbors=5 (1), unused (0)
    # Outlier: isolation_forest (1), n_estimators=100 (1), contamination=0.05 (2)
    # Resampling: smote (2), k_neighbors=5 (1), strategy=auto (0)
    # Encoding: label (1), unused (0), unused (0)
    # Scaling: standard (1), with_mean=True (0), with_std=True (0)
    # Model: random_forest (1), n_estimators=100 (1), max_depth=10 (1), min_samples_split=5 (1)
    example_chromosome = [
        0, 1, 0,  # Imputation: knn, n_neighbors=5
        1, 1, 2,  # Outlier: isolation_forest, n_est=100, contam=0.05
        2, 1, 0,  # Resampling: smote, k_neighbors=5, strategy=auto
        1, 0, 0,  # Encoding: label
        1, 0, 0,  # Scaling: standard, mean=True, std=True
        1, 1, 1, 1 # Model: random_forest, n_est=100, depth=10, split=5
    ]
    
    decoded_params = decode_and_log_chromosome(example_chromosome)

    if decoded_params is None:
        print("Error decoding example chromosome. Exiting.")
        return

    print(f"\n--- Running Single Pipeline with Decoded Chromosome: {example_chromosome} ---")
    # Parameters are now in decoded_params, print statements for individual methods commented out

    try:
        print(f"\n[Single Run - Chromosome: {example_chromosome}] Calling process_data...")
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
            save_processed_data=True # For single run, save the intermediate files
        )
        
        if processed_train_path is None: 
            print("Data processing failed for the single run. Exiting.")
            return

        print(f"[Single Run - Chromosome: {example_chromosome}] Data processing completed. Processed train: {processed_train_path}")
        print(f"\n[Single Run - Chromosome: {example_chromosome}] Calling train_model for model: {decoded_params['model_type']}...")
        
        metrics_output, ft_importance_output = train_model(
            processed_train_path, processed_test_path, target_column,
            research_path=research_base_path, 
            model_type=decoded_params['model_type'],
            model_hyperparameters=decoded_params['model_params'], # Pass model HPs
            plot_learning_curves=generate_learning_curves,
            save_run_results=True 
        )

        if metrics_output:
            print(f"\n--- Single Run Final Metrics for {decoded_params['model_type']} ({example_chromosome}) ---")
            eval_key_single = 'test' if 'test' in metrics_output and metrics_output['test'] else 'validation'
            # Ensure eval_key_single maps to an existing key and its value is not None
            if eval_key_single in metrics_output and metrics_output[eval_key_single]: 
                pr_auc_score = metrics_output[eval_key_single].get('pr_auc', -1)
                if pr_auc_score is not None and pr_auc_score != -1:
                    print(f"Evaluation PR AUC: {pr_auc_score:.4f}")
                else:
                    print("Evaluation PR AUC not available or is invalid for the single run.")
            else:
                print("Evaluation metrics (test/validation) missing or None for the single run.")
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