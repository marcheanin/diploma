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
HP_IMPUTATION_KNN_N_NEIGHBORS = {0: 3, 1: 5, 2: 7, 3: 10, 4: 15} # For гп1_имп with knn
HP_IMPUTATION_MISSFOREST_N_ESTIMATORS = {0: 30, 1: 50, 2: 100, 3: 150, 4: 200} # For гп1_имп with missforest
HP_IMPUTATION_MISSFOREST_MAX_ITER = {0: 5, 1: 10, 2: 15, 3: 20} # For гп2_имп with missforest

# Outlier HP Maps
HP_OUTLIER_IF_N_ESTIMATORS = {0: 30, 1: 50, 2: 100, 3: 150, 4: 200} # For гп1_выбр with isolation_forest
HP_OUTLIER_IF_CONTAMINATION = {0: 'auto', 1: 0.01, 2: 0.025, 3: 0.05, 4: 0.1, 5: 0.15} # For гп2_выбр with isolation_forest
HP_OUTLIER_IQR_MULTIPLIER = {0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0} # For гп1_выбр with iqr

# Resampling HP Maps
HP_RESAMPLING_ROS_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.6, 4: 0.75} # For гп1_рес with oversample
HP_RESAMPLING_SMOTE_K_NEIGHBORS = {0: 3, 1: 5, 2: 7, 3: 9} # For гп1_рес with smote
HP_RESAMPLING_SMOTE_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.6, 4: 0.75} # For гп2_рес with smote
HP_RESAMPLING_ADASYN_N_NEIGHBORS = {0: 3, 1: 5, 2: 7, 3: 9} # For гп1_рес with adasyn
HP_RESAMPLING_ADASYN_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.6, 4: 0.75} # For гп2_рес with adasyn

# Encoding HP Maps
HP_ENCODING_ONEHOT_MAX_CARDINALITY = {0: 10, 1: 20, 2: 50, 3: 100} # For гп1_код with onehot
HP_ENCODING_ONEHOT_DROP = {0: None, 1: 'first'} # For гп2_код with onehot
HP_ENCODING_LSA_N_COMPONENTS = {0: 5, 1: 10, 2: 25, 3: 50, 4: 75} # For гп1_код with lsa
HP_ENCODING_LSA_NGRAM_MAX = {0: 1, 1: 2, 2: 3} # For гп2_код with lsa (ngram_range=(1, val))
HP_ENCODING_W2V_DIM = {0: 25, 1: 50, 2: 75, 3: 100, 4: 150} # For гп1_код with word2vec
HP_ENCODING_W2V_WINDOW = {0: 1, 1: 2, 2: 3, 3: 5, 4: 7} # For гп2_код with word2vec

# Scaling HP Maps
HP_SCALING_STANDARD_WITH_MEAN = {0: True, 1: False} # For гп1_масшт with standard
HP_SCALING_STANDARD_WITH_STD = {0: True, 1: False} # For гп2_масшт with standard

# Model HP Maps
HP_MODEL_LOGREG_C = {0: 0.001, 1: 0.01, 2: 0.1, 3: 1.0, 4: 10.0, 5: 100.0} # HP1
HP_MODEL_LOGREG_PENALTY_SOLVER = { # HP2
    0: {'penalty': 'l2', 'solver': 'lbfgs', 'l1_ratio': None}, 1: {'penalty': 'l1', 'solver': 'liblinear', 'l1_ratio': None},
    2: {'penalty': 'l2', 'solver': 'liblinear', 'l1_ratio': None}, 3: {'penalty': 'l1', 'solver': 'saga', 'l1_ratio': None},
    4: {'penalty': 'l2', 'solver': 'saga', 'l1_ratio': None}, 5: {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5}
}
HP_MODEL_LOGREG_CLASS_WEIGHT = {0: None, 1: 'balanced'} # HP3
HP_MODEL_LOGREG_MAX_ITER = {0: 100, 1: 200, 2: 300, 3: 500} # HP4 (New)

HP_MODEL_RF_N_ESTIMATORS = {0: 25, 1: 50, 2: 100, 3: 200, 4: 300} # HP1
HP_MODEL_RF_MAX_DEPTH = {0: 5, 1: 7, 2: 10, 3: 15, 4: 20, 5: None} # HP2
HP_MODEL_RF_MIN_SAMPLES_SPLIT = {0: 2, 1: 5, 2: 10, 3: 15} # HP3
HP_MODEL_RF_MIN_SAMPLES_LEAF = {0: 1, 1: 2, 2: 5, 3: 10} # HP4 (New)

HP_MODEL_GB_N_ESTIMATORS = {0: 25, 1: 50, 2: 100, 3: 200, 4: 300} # HP1
HP_MODEL_GB_LEARNING_RATE = {0: 0.005, 1: 0.01, 2: 0.05, 3: 0.1, 4: 0.2} # HP2
HP_MODEL_GB_MAX_DEPTH = {0: 2, 1: 3, 2: 4, 3: 5, 4: 7} # HP3
HP_MODEL_GB_SUBSAMPLE = {0: 0.7, 1: 0.8, 2: 0.9, 3: 1.0} # HP4 (New)

HP_MODEL_NN_LAYERS = { # HP1
    0: (32,), 1: (64,), 2: (128,),
    3: (32, 32), 4: (64, 32), 5: (128, 64), 
    6: (64, 64), 7: (128, 64, 32)
}
HP_MODEL_NN_DROPOUT = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5} # HP2
HP_MODEL_NN_LR = {0: 0.0001, 1: 0.0005, 2: 0.001, 3: 0.005, 4: 0.01, 5: 0.05} # HP3
HP_MODEL_NN_BATCH_SIZE = {0: 16, 1: 32, 2: 64, 3: 128} # HP4 (New)


GENE_DESCRIPTIONS = [
    "Imputation Method", "Imputation HP1", "Imputation HP2",
    "Outlier Method", "Outlier HP1", "Outlier HP2",
    "Resampling Method", "Resampling HP1", "Resampling HP2",
    "Encoding Method", "Encoding HP1", "Encoding HP2",
    "Scaling Method", "Scaling HP1", "Scaling HP2",
    "Model Method", "Model HP1", "Model HP2", "Model HP3", "Model HP4" # Added Model HP4
]

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
    """Decodes a 20-gene chromosome and prints its meaning, returning a structured dict."""
    # print("\n--- Chromosome Definition ---")
    if len(chromosome) != 20:
        print(f"Error: Chromosome must have 20 genes, got {len(chromosome)}.")
        return None

    decoded_chromosome_info = { 
        'chromosome_values': list(chromosome),
        'description': {},
        'pipeline_params': {}
    }

    def safe_get(gene_val, val_map, default_prefix="Unknown"):
        return val_map.get(gene_val, f"{default_prefix} (raw_val: {gene_val})")

    def get_hp_value(hp_gene_val, hp_map, hp_name, method_name):
        val = hp_map.get(hp_gene_val)
        if val is None:
            return None 
        return val

    # 1. Imputation (Genes 0, 1, 2)
    imp_method_idx, imp_hp1_idx, imp_hp2_idx = chromosome[0], chromosome[1], chromosome[2]
    imp_method = IMPUTATION_MAP.get(imp_method_idx, "Unknown Imputation")
    decoded_chromosome_info['description']['imputation'] = f"{imp_method} (Method Idx: {imp_method_idx})"
    decoded_chromosome_info['pipeline_params']['imputation_method'] = imp_method
    imp_params = {}
    if imp_method == 'knn':
        imp_params['n_neighbors'] = get_hp_value(imp_hp1_idx, HP_IMPUTATION_KNN_N_NEIGHBORS, "n_neighbors", imp_method)
        decoded_chromosome_info['description']['imputation_hp1'] = f"n_neighbors: {imp_params.get('n_neighbors')} (HP1 Idx: {imp_hp1_idx})"
        decoded_chromosome_info['description']['imputation_hp2'] = f"Unused (HP2 Idx: {imp_hp2_idx})"
    elif imp_method == 'missforest':
        imp_params['n_estimators'] = get_hp_value(imp_hp1_idx, HP_IMPUTATION_MISSFOREST_N_ESTIMATORS, "n_estimators", imp_method)
        imp_params['max_iter'] = get_hp_value(imp_hp2_idx, HP_IMPUTATION_MISSFOREST_MAX_ITER, "max_iter", imp_method)
        decoded_chromosome_info['description']['imputation_hp1'] = f"n_estimators: {imp_params.get('n_estimators')} (HP1 Idx: {imp_hp1_idx})"
        decoded_chromosome_info['description']['imputation_hp2'] = f"max_iter: {imp_params.get('max_iter')} (HP2 Idx: {imp_hp2_idx})"
    else: # median or unknown
        decoded_chromosome_info['description']['imputation_hp1'] = f"Unused (HP1 Idx: {imp_hp1_idx})"
        decoded_chromosome_info['description']['imputation_hp2'] = f"Unused (HP2 Idx: {imp_hp2_idx})"
    decoded_chromosome_info['pipeline_params']['imputation_params'] = {k:v for k,v in imp_params.items() if v is not None}

    # 2. Outlier Removal (Genes 3, 4, 5)
    out_method_idx, out_hp1_idx, out_hp2_idx = chromosome[3], chromosome[4], chromosome[5]
    out_method = OUTLIER_MAP.get(out_method_idx, "Unknown Outlier")
    decoded_chromosome_info['description']['outlier_removal'] = f"{out_method} (Method Idx: {out_method_idx})"
    decoded_chromosome_info['pipeline_params']['outlier_method'] = out_method
    out_params = {}
    if out_method == 'isolation_forest':
        out_params['n_estimators'] = get_hp_value(out_hp1_idx, HP_OUTLIER_IF_N_ESTIMATORS, "n_estimators", out_method)
        out_params['contamination'] = get_hp_value(out_hp2_idx, HP_OUTLIER_IF_CONTAMINATION, "contamination", out_method)
        decoded_chromosome_info['description']['outlier_hp1'] = f"n_estimators: {out_params.get('n_estimators')} (HP1 Idx: {out_hp1_idx})"
        decoded_chromosome_info['description']['outlier_hp2'] = f"contamination: {out_params.get('contamination')} (HP2 Idx: {out_hp2_idx})"
    elif out_method == 'iqr':
        out_params['multiplier'] = get_hp_value(out_hp1_idx, HP_OUTLIER_IQR_MULTIPLIER, "multiplier", out_method)
        decoded_chromosome_info['description']['outlier_hp1'] = f"multiplier: {out_params.get('multiplier')} (HP1 Idx: {out_hp1_idx})"
        decoded_chromosome_info['description']['outlier_hp2'] = f"Unused (HP2 Idx: {out_hp2_idx})"
    else: # none or unknown
        decoded_chromosome_info['description']['outlier_hp1'] = f"Unused (HP1 Idx: {out_hp1_idx})"
        decoded_chromosome_info['description']['outlier_hp2'] = f"Unused (HP2 Idx: {out_hp2_idx})"
    decoded_chromosome_info['pipeline_params']['outlier_params'] = {k:v for k,v in out_params.items() if v is not None}

    # 3. Resampling (Genes 6, 7, 8)
    res_method_idx, res_hp1_idx, res_hp2_idx = chromosome[6], chromosome[7], chromosome[8]
    res_method = RESAMPLING_MAP.get(res_method_idx, "Unknown Resampling")
    decoded_chromosome_info['description']['resampling'] = f"{res_method} (Method Idx: {res_method_idx})"
    decoded_chromosome_info['pipeline_params']['resampling_method'] = res_method
    res_params = {}
    if res_method == 'oversample': # ROS
        res_params['sampling_strategy'] = get_hp_value(res_hp1_idx, HP_RESAMPLING_ROS_STRATEGY, "sampling_strategy_ros", res_method)
        decoded_chromosome_info['description']['resampling_hp1'] = f"sampling_strategy: {res_params.get('sampling_strategy')} (HP1 Idx: {res_hp1_idx})"
        decoded_chromosome_info['description']['resampling_hp2'] = f"Unused (HP2 Idx: {res_hp2_idx})"
    elif res_method == 'smote':
        res_params['k_neighbors'] = get_hp_value(res_hp1_idx, HP_RESAMPLING_SMOTE_K_NEIGHBORS, "k_neighbors", res_method)
        res_params['sampling_strategy'] = get_hp_value(res_hp2_idx, HP_RESAMPLING_SMOTE_STRATEGY, "sampling_strategy_smote", res_method)
        decoded_chromosome_info['description']['resampling_hp1'] = f"k_neighbors: {res_params.get('k_neighbors')} (HP1 Idx: {res_hp1_idx})"
        decoded_chromosome_info['description']['resampling_hp2'] = f"sampling_strategy: {res_params.get('sampling_strategy')} (HP2 Idx: {res_hp2_idx})"
    elif res_method == 'adasyn':
        res_params['n_neighbors'] = get_hp_value(res_hp1_idx, HP_RESAMPLING_ADASYN_N_NEIGHBORS, "n_neighbors", res_method)
        res_params['sampling_strategy'] = get_hp_value(res_hp2_idx, HP_RESAMPLING_ADASYN_STRATEGY, "sampling_strategy_adasyn", res_method)
        decoded_chromosome_info['description']['resampling_hp1'] = f"n_neighbors: {res_params.get('n_neighbors')} (HP1 Idx: {res_hp1_idx})"
        decoded_chromosome_info['description']['resampling_hp2'] = f"sampling_strategy: {res_params.get('sampling_strategy')} (HP2 Idx: {res_hp2_idx})"
    else: # none or unknown
        decoded_chromosome_info['description']['resampling_hp1'] = f"Unused (HP1 Idx: {res_hp1_idx})"
        decoded_chromosome_info['description']['resampling_hp2'] = f"Unused (HP2 Idx: {res_hp2_idx})"
    decoded_chromosome_info['pipeline_params']['resampling_params'] = {k:v for k,v in res_params.items() if v is not None}

    # 4. Encoding (Genes 9, 10, 11)
    enc_method_idx, enc_hp1_idx, enc_hp2_idx = chromosome[9], chromosome[10], chromosome[11]
    enc_method = ENCODING_MAP.get(enc_method_idx, "Unknown Encoding")
    decoded_chromosome_info['description']['encoding'] = f"{enc_method} (Method Idx: {enc_method_idx})"
    decoded_chromosome_info['pipeline_params']['encoding_method'] = enc_method
    enc_params = {}
    if enc_method == 'onehot':
        enc_params['max_cardinality_threshold'] = get_hp_value(enc_hp1_idx, HP_ENCODING_ONEHOT_MAX_CARDINALITY, "max_cardinality", enc_method)
        enc_params['drop'] = get_hp_value(enc_hp2_idx, HP_ENCODING_ONEHOT_DROP, "drop", enc_method)
        decoded_chromosome_info['description']['encoding_hp1'] = f"max_cardinality_threshold: {enc_params.get('max_cardinality_threshold')} (HP1 Idx: {enc_hp1_idx})"
        decoded_chromosome_info['description']['encoding_hp2'] = f"drop_strategy: {enc_params.get('drop')} (HP2 Idx: {enc_hp2_idx})"
    elif enc_method == 'lsa':
        enc_params['n_components'] = get_hp_value(enc_hp1_idx, HP_ENCODING_LSA_N_COMPONENTS, "n_components_lsa", enc_method)
        ngram_max = get_hp_value(enc_hp2_idx, HP_ENCODING_LSA_NGRAM_MAX, "ngram_max_lsa", enc_method)
        if ngram_max is not None: enc_params['ngram_range'] = (1, ngram_max)
        decoded_chromosome_info['description']['encoding_hp1'] = f"n_components: {enc_params.get('n_components')} (HP1 Idx: {enc_hp1_idx})"
        decoded_chromosome_info['description']['encoding_hp2'] = f"ngram_range_max: {ngram_max} (HP2 Idx: {enc_hp2_idx})"
    elif enc_method == 'word2vec':
        enc_params['embedding_dim'] = get_hp_value(enc_hp1_idx, HP_ENCODING_W2V_DIM, "embedding_dim_w2v", enc_method)
        enc_params['window'] = get_hp_value(enc_hp2_idx, HP_ENCODING_W2V_WINDOW, "window_w2v", enc_method)
        decoded_chromosome_info['description']['encoding_hp1'] = f"vector_size: {enc_params.get('embedding_dim')} (HP1 Idx: {enc_hp1_idx})"
        decoded_chromosome_info['description']['encoding_hp2'] = f"window: {enc_params.get('window')} (HP2 Idx: {enc_hp2_idx})"
    else: # label or unknown
        decoded_chromosome_info['description']['encoding_hp1'] = f"Unused (HP1 Idx: {enc_hp1_idx})"
        decoded_chromosome_info['description']['encoding_hp2'] = f"Unused (HP2 Idx: {enc_hp2_idx})"
    filtered_enc_params = {}
    for k,v in enc_params.items():
        if v is not None or (enc_method == 'onehot' and k == 'drop'): # Ensure 'drop' is kept even if None for OHE
             filtered_enc_params[k] = v
    decoded_chromosome_info['pipeline_params']['encoding_params'] = filtered_enc_params

    # 5. Scaling (Genes 12, 13, 14)
    sca_method_idx, sca_hp1_idx, sca_hp2_idx = chromosome[12], chromosome[13], chromosome[14]
    sca_method = SCALING_MAP.get(sca_method_idx, "Unknown Scaling")
    decoded_chromosome_info['description']['scaling'] = f"{sca_method} (Method Idx: {sca_method_idx})"
    decoded_chromosome_info['pipeline_params']['scaling_method'] = sca_method
    sca_params = {}
    if sca_method == 'standard':
        sca_params['with_mean'] = get_hp_value(sca_hp1_idx, HP_SCALING_STANDARD_WITH_MEAN, "with_mean", sca_method)
        sca_params['with_std'] = get_hp_value(sca_hp2_idx, HP_SCALING_STANDARD_WITH_STD, "with_std", sca_method)
        decoded_chromosome_info['description']['scaling_hp1'] = f"with_mean: {sca_params.get('with_mean')} (HP1 Idx: {sca_hp1_idx})"
        decoded_chromosome_info['description']['scaling_hp2'] = f"with_std: {sca_params.get('with_std')} (HP2 Idx: {sca_hp2_idx})"
    else: # none, minmax, or unknown
        decoded_chromosome_info['description']['scaling_hp1'] = f"Unused (HP1 Idx: {sca_hp1_idx})"
        decoded_chromosome_info['description']['scaling_hp2'] = f"Unused (HP2 Idx: {sca_hp2_idx})"
    decoded_chromosome_info['pipeline_params']['scaling_params'] = {k:v for k,v in sca_params.items() if v is not None}

    # 6. Model (Genes 15, 16, 17, 18, 19)
    model_method_idx, model_hp1_idx, model_hp2_idx, model_hp3_idx, model_hp4_idx = chromosome[15], chromosome[16], chromosome[17], chromosome[18], chromosome[19]
    model_type = MODEL_MAP.get(model_method_idx, "Unknown Model")
    decoded_chromosome_info['description']['model'] = f"{model_type} (Method Idx: {model_method_idx})"
    decoded_chromosome_info['pipeline_params']['model_type'] = model_type
    model_params = {}
    if model_type == 'logistic_regression':
        model_params['C'] = get_hp_value(model_hp1_idx, HP_MODEL_LOGREG_C, "C", model_type)
        model_params['solver_penalty_config'] = get_hp_value(model_hp2_idx, HP_MODEL_LOGREG_PENALTY_SOLVER, "solver/penalty", model_type)
        model_params['class_weight'] = get_hp_value(model_hp3_idx, HP_MODEL_LOGREG_CLASS_WEIGHT, "class_weight", model_type)
        model_params['max_iter'] = get_hp_value(model_hp4_idx, HP_MODEL_LOGREG_MAX_ITER, "max_iter", model_type)
        decoded_chromosome_info['description']['model_hp1'] = f"C: {model_params.get('C')} (HP1 Idx: {model_hp1_idx})"
        decoded_chromosome_info['description']['model_hp2'] = f"Solver/Penalty Config: {model_params.get('solver_penalty_config')} (HP2 Idx: {model_hp2_idx})"
        decoded_chromosome_info['description']['model_hp3'] = f"Class Weight: {model_params.get('class_weight')} (HP3 Idx: {model_hp3_idx})"
        decoded_chromosome_info['description']['model_hp4'] = f"Max Iter: {model_params.get('max_iter')} (HP4 Idx: {model_hp4_idx})"
    elif model_type == 'random_forest':
        model_params['n_estimators'] = get_hp_value(model_hp1_idx, HP_MODEL_RF_N_ESTIMATORS, "n_estimators_rf", model_type)
        model_params['max_depth'] = get_hp_value(model_hp2_idx, HP_MODEL_RF_MAX_DEPTH, "max_depth_rf", model_type)
        model_params['min_samples_split'] = get_hp_value(model_hp3_idx, HP_MODEL_RF_MIN_SAMPLES_SPLIT, "min_samples_split_rf", model_type)
        model_params['min_samples_leaf'] = get_hp_value(model_hp4_idx, HP_MODEL_RF_MIN_SAMPLES_LEAF, "min_samples_leaf_rf", model_type)
        decoded_chromosome_info['description']['model_hp1'] = f"n_estimators: {model_params.get('n_estimators')} (HP1 Idx: {model_hp1_idx})"
        decoded_chromosome_info['description']['model_hp2'] = f"max_depth: {model_params.get('max_depth')} (HP2 Idx: {model_hp2_idx})"
        decoded_chromosome_info['description']['model_hp3'] = f"min_samples_split: {model_params.get('min_samples_split')} (HP3 Idx: {model_hp3_idx})"
        decoded_chromosome_info['description']['model_hp4'] = f"min_samples_leaf: {model_params.get('min_samples_leaf')} (HP4 Idx: {model_hp4_idx})"
    elif model_type == 'gradient_boosting':
        model_params['n_estimators'] = get_hp_value(model_hp1_idx, HP_MODEL_GB_N_ESTIMATORS, "n_estimators_gb", model_type)
        model_params['learning_rate'] = get_hp_value(model_hp2_idx, HP_MODEL_GB_LEARNING_RATE, "learning_rate_gb", model_type)
        model_params['max_depth'] = get_hp_value(model_hp3_idx, HP_MODEL_GB_MAX_DEPTH, "max_depth_gb", model_type)
        model_params['subsample'] = get_hp_value(model_hp4_idx, HP_MODEL_GB_SUBSAMPLE, "subsample_gb", model_type)
        decoded_chromosome_info['description']['model_hp1'] = f"n_estimators: {model_params.get('n_estimators')} (HP1 Idx: {model_hp1_idx})"
        decoded_chromosome_info['description']['model_hp2'] = f"learning_rate: {model_params.get('learning_rate')} (HP2 Idx: {model_hp2_idx})"
        decoded_chromosome_info['description']['model_hp3'] = f"max_depth: {model_params.get('max_depth')} (HP3 Idx: {model_hp3_idx})"
        decoded_chromosome_info['description']['model_hp4'] = f"subsample: {model_params.get('subsample')} (HP4 Idx: {model_hp4_idx})"
    elif model_type == 'neural_network':
        model_params['hidden_layer_sizes'] = get_hp_value(model_hp1_idx, HP_MODEL_NN_LAYERS, "hidden_layer_sizes_nn", model_type)
        model_params['dropout_rate'] = get_hp_value(model_hp2_idx, HP_MODEL_NN_DROPOUT, "dropout_rate_nn", model_type)
        model_params['learning_rate'] = get_hp_value(model_hp3_idx, HP_MODEL_NN_LR, "learning_rate_nn", model_type)
        model_params['batch_size'] = get_hp_value(model_hp4_idx, HP_MODEL_NN_BATCH_SIZE, "batch_size_nn", model_type)
        decoded_chromosome_info['description']['model_hp1'] = f"Hidden Layers: {model_params.get('hidden_layer_sizes')} (HP1 Idx: {model_hp1_idx})"
        decoded_chromosome_info['description']['model_hp2'] = f"Dropout Rate: {model_params.get('dropout_rate')} (HP2 Idx: {model_hp2_idx})"
        decoded_chromosome_info['description']['model_hp3'] = f"Learning Rate: {model_params.get('learning_rate')} (HP3 Idx: {model_hp3_idx})"
        decoded_chromosome_info['description']['model_hp4'] = f"Batch Size: {model_params.get('batch_size')} (HP4 Idx: {model_hp4_idx})"
    
    final_model_params = {}
    for k, v in model_params.items():
        if v is not None: # Keep if not None
            final_model_params[k] = v
        elif k == 'class_weight' and v is None: # Explicitly keep class_weight=None if it's the decoded value
             final_model_params[k] = None
        elif k == 'solver_penalty_config' and isinstance(v, dict): # Keep dict for solver/penalty
            final_model_params[k] = v
        elif model_type == 'random_forest' and k == 'max_depth' and v is None: # Keep max_depth=None for RF
            final_model_params[k] = None
            
    decoded_chromosome_info['pipeline_params']['model_params'] = final_model_params

    print(f"Chromosome (raw): {list(chromosome)}")
    print("Decoded Chromosome Details:")
    print(f"  Imputation: {decoded_chromosome_info['pipeline_params']['imputation_method']}, Params: {decoded_chromosome_info['pipeline_params']['imputation_params']}")
    print(f"  Outlier Removal: {decoded_chromosome_info['pipeline_params']['outlier_method']}, Params: {decoded_chromosome_info['pipeline_params']['outlier_params']}")
    print(f"  Resampling: {decoded_chromosome_info['pipeline_params']['resampling_method']}, Params: {decoded_chromosome_info['pipeline_params']['resampling_params']}")
    print(f"  Encoding: {decoded_chromosome_info['pipeline_params']['encoding_method']}, Params: {decoded_chromosome_info['pipeline_params']['encoding_params']}")
    print(f"  Scaling: {decoded_chromosome_info['pipeline_params']['scaling_method']}, Params: {decoded_chromosome_info['pipeline_params']['scaling_params']}")
    print(f"  Model: {decoded_chromosome_info['pipeline_params']['model_type']}, Params: {decoded_chromosome_info['pipeline_params']['model_params']}")
    print("-----------------------------\n")
    
    return decoded_chromosome_info['pipeline_params']

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

    if (save_run_results or plot_learning_curves) and research_path:
        os.makedirs(research_path, exist_ok=True)
    
    # Load data if paths are provided
    current_train_data = train_data_input
    if isinstance(train_data_input, str):
        try:
            current_train_data = pd.read_csv(train_data_input)
            # print(f"Loaded training data from path: {train_data_input}")
        except Exception as e:
            print(f"Error loading training data from path {train_data_input}: {e}")
            return None, None # Cannot proceed without training data
    
    current_test_data = test_data_input
    if isinstance(test_data_input, str):
        try:
            current_test_data = pd.read_csv(test_data_input)
            # print(f"Loaded test data from path: {test_data_input}")
        except FileNotFoundError:
            # print(f"Test data file not found at {test_data_input}. Proceeding without test data (will use validation split from train).")
            current_test_data = pd.DataFrame() # Ensure it's an empty DF, not None
        except Exception as e:
            print(f"Error loading test data from path {test_data_input}: {e}")
            current_test_data = pd.DataFrame() 
    elif current_test_data is None: # If None was passed directly (not a path)
        current_test_data = pd.DataFrame()


    # Create ModelTrainer instance
    trainer = ModelTrainer(model_type=model_type, model_hyperparameters=model_hyperparameters, random_state=42)
    
    metrics, feature_importance = trainer.train(
        current_train_data, 
        current_test_data, 
        target_column,
            output_path=research_path, 
            plot_learning_curves=plot_learning_curves,
            save_run_results=save_run_results # Pass the flag through
        )    
    return metrics, feature_importance

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
   # chromosome_1 = [2, 4, 0, 0, 0, 1, 2, 4, 0, 1, 0, 3, 2, 1, 0, 1, 7, 2, 5, 1] # MissForest TypeError
    chromosome_2 = [0, 2, 1, 2, 3, 3, 1, 3, 0, 1, 0, 2, 2, 0, 0, 0, 7, 1, 0, 1] # y should be 1d array

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


if __name__ == "__main__":
    # main()
    ga_optimizer.run_genetic_algorithm()