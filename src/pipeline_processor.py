import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Assuming these modules are in the same directory or PYTHONPATH is set up
from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.outlier_remover import OutlierRemover
from preprocessing.resampler import Resampler
from modeling.model_trainer import ModelTrainer


# ==========================================
# CHROMOSOME CONFIGURATION AND DECODING
# ==========================================

class ChromosomeConfig:
    """
    Конфигурация для декодирования 20-генных хромосом в параметры ML-пайплайна
    """
    
    # --- Method Maps ---
    IMPUTATION_MAP = {0: 'knn', 1: 'median', 2: 'missforest'}
    OUTLIER_MAP = {0: 'none', 1: 'isolation_forest', 2: 'iqr'}
    RESAMPLING_MAP = {0: 'none', 1: 'oversample', 2: 'smote', 3: 'adasyn'}
    ENCODING_MAP = {0: 'onehot', 1: 'label', 2: 'lsa', 3: 'word2vec'}
    SCALING_MAP = {0: 'none', 1: 'standard', 2: 'minmax'}
    MODEL_MAP = {0: 'logistic_regression', 1: 'random_forest', 2: 'gradient_boosting', 3: 'neural_network'}
    
    # --- Hyperparameter Maps ---
    # Imputation HP Maps
    HP_IMPUTATION_KNN_N_NEIGHBORS = {0: 3, 1: 5, 2: 7, 3: 10, 4: 15}
    HP_IMPUTATION_MISSFOREST_N_ESTIMATORS = {0: 30, 1: 50, 2: 100, 3: 150, 4: 200}
    HP_IMPUTATION_MISSFOREST_MAX_ITER = {0: 5, 1: 10, 2: 15, 3: 20}
    
    # Outlier HP Maps
    HP_OUTLIER_IF_N_ESTIMATORS = {0: 30, 1: 50, 2: 100, 3: 150, 4: 200}
    HP_OUTLIER_IF_CONTAMINATION = {0: 'auto', 1: 0.01, 2: 0.025, 3: 0.05, 4: 0.1, 5: 0.15}
    HP_OUTLIER_IQR_MULTIPLIER = {0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0}
    
    # Resampling HP Maps
    HP_RESAMPLING_ROS_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.6, 4: 0.75}
    HP_RESAMPLING_SMOTE_K_NEIGHBORS = {0: 3, 1: 5, 2: 7, 3: 9}
    HP_RESAMPLING_SMOTE_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.6, 4: 0.75}
    HP_RESAMPLING_ADASYN_N_NEIGHBORS = {0: 3, 1: 5, 2: 7, 3: 9}
    HP_RESAMPLING_ADASYN_STRATEGY = {0: 'auto', 1: 'minority', 2: 0.5, 3: 0.6, 4: 0.75}
    
    # Encoding HP Maps
    HP_ENCODING_ONEHOT_MAX_CARDINALITY = {0: 10, 1: 20, 2: 50, 3: 100}
    HP_ENCODING_ONEHOT_DROP = {0: None, 1: 'first'}
    HP_ENCODING_LSA_N_COMPONENTS = {0: 5, 1: 10, 2: 25, 3: 50, 4: 75}
    HP_ENCODING_LSA_NGRAM_MAX = {0: 1, 1: 2, 2: 3}
    HP_ENCODING_W2V_DIM = {0: 25, 1: 50, 2: 75, 3: 100, 4: 150}
    HP_ENCODING_W2V_WINDOW = {0: 1, 1: 2, 2: 3, 3: 5, 4: 7}
    
    # Scaling HP Maps
    HP_SCALING_STANDARD_WITH_MEAN = {0: True, 1: False}
    HP_SCALING_STANDARD_WITH_STD = {0: True, 1: False}
    
    # Model HP Maps
    HP_MODEL_LOGREG_C = {0: 0.001, 1: 0.01, 2: 0.1, 3: 1.0, 4: 10.0, 5: 100.0}
    HP_MODEL_LOGREG_PENALTY_SOLVER = {
        0: {'penalty': 'l2', 'solver': 'lbfgs', 'l1_ratio': None}, 
        1: {'penalty': 'l1', 'solver': 'liblinear', 'l1_ratio': None},
        2: {'penalty': 'l2', 'solver': 'liblinear', 'l1_ratio': None}, 
        3: {'penalty': 'l1', 'solver': 'saga', 'l1_ratio': None},
        4: {'penalty': 'l2', 'solver': 'saga', 'l1_ratio': None}, 
        5: {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5}
    }
    HP_MODEL_LOGREG_CLASS_WEIGHT = {0: None, 1: 'balanced'}
    HP_MODEL_LOGREG_MAX_ITER = {0: 100, 1: 200, 2: 300, 3: 500}
    
    HP_MODEL_RF_N_ESTIMATORS = {0: 25, 1: 50, 2: 100, 3: 200, 4: 300}
    HP_MODEL_RF_MAX_DEPTH = {0: 5, 1: 7, 2: 10, 3: 15, 4: 20, 5: None}
    HP_MODEL_RF_MIN_SAMPLES_SPLIT = {0: 2, 1: 5, 2: 10, 3: 15}
    HP_MODEL_RF_MIN_SAMPLES_LEAF = {0: 1, 1: 2, 2: 5, 3: 10}
    
    HP_MODEL_GB_N_ESTIMATORS = {0: 25, 1: 50, 2: 100, 3: 200, 4: 300}
    HP_MODEL_GB_LEARNING_RATE = {0: 0.005, 1: 0.01, 2: 0.05, 3: 0.1, 4: 0.2}
    HP_MODEL_GB_MAX_DEPTH = {0: 2, 1: 3, 2: 4, 3: 5, 4: 7}
    HP_MODEL_GB_SUBSAMPLE = {0: 0.7, 1: 0.8, 2: 0.9, 3: 1.0}
    
    HP_MODEL_NN_LAYERS = {
        0: (32,), 1: (64,), 2: (128,),
        3: (32, 32), 4: (64, 32), 5: (128, 64), 
        6: (64, 64), 7: (128, 64, 32)
    }
    HP_MODEL_NN_DROPOUT = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}
    HP_MODEL_NN_LR = {0: 0.0001, 1: 0.0005, 2: 0.001, 3: 0.005, 4: 0.01, 5: 0.05}
    HP_MODEL_NN_BATCH_SIZE = {0: 16, 1: 32, 2: 64, 3: 128}
    
    # Gene descriptions for debugging
    GENE_DESCRIPTIONS = [
        "Imputation Method", "Imputation HP1", "Imputation HP2",
        "Outlier Method", "Outlier HP1", "Outlier HP2",
        "Resampling Method", "Resampling HP1", "Resampling HP2",
        "Encoding Method", "Encoding HP1", "Encoding HP2",
        "Scaling Method", "Scaling HP1", "Scaling HP2",
        "Model Method", "Model HP1", "Model HP2", "Model HP3", "Model HP4"
    ]
    
    @classmethod
    def get_gene_ranges(cls):
        """Возвращает количество вариантов для каждого гена (для ГА)"""
        return [
            len(cls.IMPUTATION_MAP), # Gene 0
            max(len(cls.HP_IMPUTATION_KNN_N_NEIGHBORS), len(cls.HP_IMPUTATION_MISSFOREST_N_ESTIMATORS)), # Gene 1
            len(cls.HP_IMPUTATION_MISSFOREST_MAX_ITER), # Gene 2
            
            len(cls.OUTLIER_MAP), # Gene 3
            max(len(cls.HP_OUTLIER_IF_N_ESTIMATORS), len(cls.HP_OUTLIER_IQR_MULTIPLIER)), # Gene 4
            len(cls.HP_OUTLIER_IF_CONTAMINATION), # Gene 5
            
            len(cls.RESAMPLING_MAP), # Gene 6
            max(len(cls.HP_RESAMPLING_ROS_STRATEGY), len(cls.HP_RESAMPLING_SMOTE_K_NEIGHBORS), len(cls.HP_RESAMPLING_ADASYN_N_NEIGHBORS)), # Gene 7
            max(len(cls.HP_RESAMPLING_SMOTE_STRATEGY), len(cls.HP_RESAMPLING_ADASYN_STRATEGY)), # Gene 8
            
            len(cls.ENCODING_MAP), # Gene 9
            max(len(cls.HP_ENCODING_ONEHOT_MAX_CARDINALITY), len(cls.HP_ENCODING_LSA_N_COMPONENTS), len(cls.HP_ENCODING_W2V_DIM)), # Gene 10
            max(len(cls.HP_ENCODING_ONEHOT_DROP), len(cls.HP_ENCODING_LSA_NGRAM_MAX), len(cls.HP_ENCODING_W2V_WINDOW)), # Gene 11
            
            len(cls.SCALING_MAP), # Gene 12
            len(cls.HP_SCALING_STANDARD_WITH_MEAN), # Gene 13
            len(cls.HP_SCALING_STANDARD_WITH_STD), # Gene 14
            
            len(cls.MODEL_MAP), # Gene 15
            max(len(cls.HP_MODEL_LOGREG_C), len(cls.HP_MODEL_RF_N_ESTIMATORS), len(cls.HP_MODEL_GB_N_ESTIMATORS), len(cls.HP_MODEL_NN_LAYERS)), # Gene 16
            max(len(cls.HP_MODEL_LOGREG_PENALTY_SOLVER), len(cls.HP_MODEL_RF_MAX_DEPTH), len(cls.HP_MODEL_GB_LEARNING_RATE), len(cls.HP_MODEL_NN_DROPOUT)), # Gene 17
            max(len(cls.HP_MODEL_LOGREG_CLASS_WEIGHT), len(cls.HP_MODEL_RF_MIN_SAMPLES_SPLIT), len(cls.HP_MODEL_GB_MAX_DEPTH), len(cls.HP_MODEL_NN_LR)), # Gene 18
            max(len(cls.HP_MODEL_LOGREG_MAX_ITER), len(cls.HP_MODEL_RF_MIN_SAMPLES_LEAF), len(cls.HP_MODEL_GB_SUBSAMPLE), len(cls.HP_MODEL_NN_BATCH_SIZE)) # Gene 19
        ]


class ChromosomeDecoder:
    """
    Декодировщик хромосом в параметры ML-пайплайна
    """
    
    def __init__(self):
        self.config = ChromosomeConfig()
    
    def decode_chromosome(self, chromosome, verbose=True):
        """
        Декодирует 20-генную хромосому в параметры пайплайна
        
        Args:
            chromosome: Список из 20 генов
            verbose: Выводить детали декодирования
            
        Returns:
            Dict с параметрами пайплайна или None при ошибке
        """
        if len(chromosome) != 20:
            print(f"[ChromosomeDecoder] Ошибка: хромосома должна содержать 20 генов, получено {len(chromosome)}")
            return None
        
        if verbose:
            print(f"[ChromosomeDecoder] Декодирование хромосомы: {list(chromosome)}")
        
        decoded_info = {
            'chromosome_values': list(chromosome),
            'description': {},
            'pipeline_params': {}
        }
        
        # Декодируем каждый этап пайплайна
        self._decode_imputation(chromosome, decoded_info, verbose)
        self._decode_outlier_removal(chromosome, decoded_info, verbose)
        self._decode_resampling(chromosome, decoded_info, verbose)
        self._decode_encoding(chromosome, decoded_info, verbose)
        self._decode_scaling(chromosome, decoded_info, verbose)
        self._decode_model(chromosome, decoded_info, verbose)
        
        if verbose:
            self._print_summary(decoded_info)
        
        return decoded_info
    
    def _get_hp_value(self, gene_val, hp_map):
        """Безопасное получение значения гиперпараметра"""
        return hp_map.get(gene_val)
    
    def _decode_imputation(self, chromosome, decoded_info, verbose):
        """Декодирует гены импутации (0, 1, 2)"""
        method_idx, hp1_idx, hp2_idx = chromosome[0], chromosome[1], chromosome[2]
        method = self.config.IMPUTATION_MAP.get(method_idx, "unknown")
        
        decoded_info['pipeline_params']['imputation_method'] = method
        
        params = {}
        if method == 'knn':
            params['n_neighbors'] = self._get_hp_value(hp1_idx, self.config.HP_IMPUTATION_KNN_N_NEIGHBORS)
        elif method == 'missforest':
            params['n_estimators'] = self._get_hp_value(hp1_idx, self.config.HP_IMPUTATION_MISSFOREST_N_ESTIMATORS)
            params['max_iter'] = self._get_hp_value(hp2_idx, self.config.HP_IMPUTATION_MISSFOREST_MAX_ITER)
        
        decoded_info['pipeline_params']['imputation_params'] = {k: v for k, v in params.items() if v is not None}
        
        if verbose:
            print(f"  Импутация: {method}, параметры: {decoded_info['pipeline_params']['imputation_params']}")
    
    def _decode_outlier_removal(self, chromosome, decoded_info, verbose):
        """Декодирует гены удаления выбросов (3, 4, 5)"""
        method_idx, hp1_idx, hp2_idx = chromosome[3], chromosome[4], chromosome[5]
        method = self.config.OUTLIER_MAP.get(method_idx, "unknown")
        
        decoded_info['pipeline_params']['outlier_method'] = method
        
        params = {}
        if method == 'isolation_forest':
            params['n_estimators'] = self._get_hp_value(hp1_idx, self.config.HP_OUTLIER_IF_N_ESTIMATORS)
            params['contamination'] = self._get_hp_value(hp2_idx, self.config.HP_OUTLIER_IF_CONTAMINATION)
        elif method == 'iqr':
            params['multiplier'] = self._get_hp_value(hp1_idx, self.config.HP_OUTLIER_IQR_MULTIPLIER)
        
        decoded_info['pipeline_params']['outlier_params'] = {k: v for k, v in params.items() if v is not None}
        
        if verbose:
            print(f"  Удаление выбросов: {method}, параметры: {decoded_info['pipeline_params']['outlier_params']}")
    
    def _decode_resampling(self, chromosome, decoded_info, verbose):
        """Декодирует гены ресемплинга (6, 7, 8)"""
        method_idx, hp1_idx, hp2_idx = chromosome[6], chromosome[7], chromosome[8]
        method = self.config.RESAMPLING_MAP.get(method_idx, "unknown")
        
        decoded_info['pipeline_params']['resampling_method'] = method
        
        params = {}
        if method == 'oversample':
            params['sampling_strategy'] = self._get_hp_value(hp1_idx, self.config.HP_RESAMPLING_ROS_STRATEGY)
        elif method == 'smote':
            params['k_neighbors'] = self._get_hp_value(hp1_idx, self.config.HP_RESAMPLING_SMOTE_K_NEIGHBORS)
            params['sampling_strategy'] = self._get_hp_value(hp2_idx, self.config.HP_RESAMPLING_SMOTE_STRATEGY)
        elif method == 'adasyn':
            params['n_neighbors'] = self._get_hp_value(hp1_idx, self.config.HP_RESAMPLING_ADASYN_N_NEIGHBORS)
            params['sampling_strategy'] = self._get_hp_value(hp2_idx, self.config.HP_RESAMPLING_ADASYN_STRATEGY)
        
        decoded_info['pipeline_params']['resampling_params'] = {k: v for k, v in params.items() if v is not None}
        
        if verbose:
            print(f"  Ресемплинг: {method}, параметры: {decoded_info['pipeline_params']['resampling_params']}")
    
    def _decode_encoding(self, chromosome, decoded_info, verbose):
        """Декодирует гены кодирования (9, 10, 11)"""
        method_idx, hp1_idx, hp2_idx = chromosome[9], chromosome[10], chromosome[11]
        method = self.config.ENCODING_MAP.get(method_idx, "unknown")
        
        decoded_info['pipeline_params']['encoding_method'] = method
        
        params = {}
        if method == 'onehot':
            params['max_cardinality_threshold'] = self._get_hp_value(hp1_idx, self.config.HP_ENCODING_ONEHOT_MAX_CARDINALITY)
            params['drop'] = self._get_hp_value(hp2_idx, self.config.HP_ENCODING_ONEHOT_DROP)
        elif method == 'lsa':
            params['n_components'] = self._get_hp_value(hp1_idx, self.config.HP_ENCODING_LSA_N_COMPONENTS)
            ngram_max = self._get_hp_value(hp2_idx, self.config.HP_ENCODING_LSA_NGRAM_MAX)
            if ngram_max is not None:
                params['ngram_range'] = (1, ngram_max)
        elif method == 'word2vec':
            params['embedding_dim'] = self._get_hp_value(hp1_idx, self.config.HP_ENCODING_W2V_DIM)
            params['window'] = self._get_hp_value(hp2_idx, self.config.HP_ENCODING_W2V_WINDOW)
        
        # Сохраняем параметры (включая None для drop в onehot)
        filtered_params = {}
        for k, v in params.items():
            if v is not None or (method == 'onehot' and k == 'drop'):
                filtered_params[k] = v
        
        decoded_info['pipeline_params']['encoding_params'] = filtered_params
        
        if verbose:
            print(f"  Кодирование: {method}, параметры: {decoded_info['pipeline_params']['encoding_params']}")
    
    def _decode_scaling(self, chromosome, decoded_info, verbose):
        """Декодирует гены масштабирования (12, 13, 14)"""
        method_idx, hp1_idx, hp2_idx = chromosome[12], chromosome[13], chromosome[14]
        method = self.config.SCALING_MAP.get(method_idx, "unknown")
        
        decoded_info['pipeline_params']['scaling_method'] = method
        
        params = {}
        if method == 'standard':
            params['with_mean'] = self._get_hp_value(hp1_idx, self.config.HP_SCALING_STANDARD_WITH_MEAN)
            params['with_std'] = self._get_hp_value(hp2_idx, self.config.HP_SCALING_STANDARD_WITH_STD)
        
        decoded_info['pipeline_params']['scaling_params'] = {k: v for k, v in params.items() if v is not None}
        
        if verbose:
            print(f"  Масштабирование: {method}, параметры: {decoded_info['pipeline_params']['scaling_params']}")
    
    def _decode_model(self, chromosome, decoded_info, verbose):
        """Декодирует гены модели (15, 16, 17, 18, 19)"""
        method_idx = chromosome[15]
        hp1_idx, hp2_idx, hp3_idx, hp4_idx = chromosome[16], chromosome[17], chromosome[18], chromosome[19]
        model_type = self.config.MODEL_MAP.get(method_idx, "unknown")
        
        decoded_info['pipeline_params']['model_type'] = model_type
        
        params = {}
        if model_type == 'logistic_regression':
            params['C'] = self._get_hp_value(hp1_idx, self.config.HP_MODEL_LOGREG_C)
            params['solver_penalty_config'] = self._get_hp_value(hp2_idx, self.config.HP_MODEL_LOGREG_PENALTY_SOLVER)
            params['class_weight'] = self._get_hp_value(hp3_idx, self.config.HP_MODEL_LOGREG_CLASS_WEIGHT)
            params['max_iter'] = self._get_hp_value(hp4_idx, self.config.HP_MODEL_LOGREG_MAX_ITER)
        elif model_type == 'random_forest':
            params['n_estimators'] = self._get_hp_value(hp1_idx, self.config.HP_MODEL_RF_N_ESTIMATORS)
            params['max_depth'] = self._get_hp_value(hp2_idx, self.config.HP_MODEL_RF_MAX_DEPTH)
            params['min_samples_split'] = self._get_hp_value(hp3_idx, self.config.HP_MODEL_RF_MIN_SAMPLES_SPLIT)
            params['min_samples_leaf'] = self._get_hp_value(hp4_idx, self.config.HP_MODEL_RF_MIN_SAMPLES_LEAF)
        elif model_type == 'gradient_boosting':
            params['n_estimators'] = self._get_hp_value(hp1_idx, self.config.HP_MODEL_GB_N_ESTIMATORS)
            params['learning_rate'] = self._get_hp_value(hp2_idx, self.config.HP_MODEL_GB_LEARNING_RATE)
            params['max_depth'] = self._get_hp_value(hp3_idx, self.config.HP_MODEL_GB_MAX_DEPTH)
            params['subsample'] = self._get_hp_value(hp4_idx, self.config.HP_MODEL_GB_SUBSAMPLE)
        elif model_type == 'neural_network':
            params['hidden_layer_sizes'] = self._get_hp_value(hp1_idx, self.config.HP_MODEL_NN_LAYERS)
            params['dropout_rate'] = self._get_hp_value(hp2_idx, self.config.HP_MODEL_NN_DROPOUT)
            params['learning_rate'] = self._get_hp_value(hp3_idx, self.config.HP_MODEL_NN_LR)
            params['batch_size'] = self._get_hp_value(hp4_idx, self.config.HP_MODEL_NN_BATCH_SIZE)
        
        # Фильтруем параметры, оставляя важные None значения
        final_params = {}
        for k, v in params.items():
            if v is not None:
                final_params[k] = v
            elif k in ['class_weight', 'max_depth'] and v is None:
                final_params[k] = None  # Сохраняем важные None значения
            elif k == 'solver_penalty_config' and isinstance(v, dict):
                final_params[k] = v
        
        decoded_info['pipeline_params']['model_params'] = final_params
        
        if verbose:
            print(f"  Модель: {model_type}, параметры: {decoded_info['pipeline_params']['model_params']}")
    
    def _print_summary(self, decoded_info):
        """Выводит краткое резюме декодированной хромосомы"""
        params = decoded_info['pipeline_params']
        print(f"\n[ChromosomeDecoder] Резюме конфигурации пайплайна:")
        print(f"  • Импутация: {params['imputation_method']}")
        print(f"  • Выбросы: {params['outlier_method']}")  
        print(f"  • Ресемплинг: {params['resampling_method']}")
        print(f"  • Кодирование: {params['encoding_method']}")
        print(f"  • Масштабирование: {params['scaling_method']}")
        print(f"  • Модель: {params['model_type']}")


# Глобальный декодер для совместимости
_chromosome_decoder = ChromosomeDecoder()

def decode_and_log_chromosome(chromosome, verbose=True):
    """
    Декодирует хромосому (совместимость с существующим кодом)
    
    Args:
        chromosome: 20-генная хромосома
        verbose: Выводить детали
        
    Returns:
        Dict с параметрами пайплайна
    """
    result = _chromosome_decoder.decode_chromosome(chromosome, verbose)
    if result is None:
        return None
    
    # Возвращаем только pipeline_params для совместимости со старым кодом
    return result['pipeline_params']


def decode_chromosome_full(chromosome, verbose=True):
    """
    Декодирует хромосому с полной информацией (для CLI и новых приложений)
    
    Args:
        chromosome: 20-генная хромосома
        verbose: Выводить детали
        
    Returns:
        Dict с полной информацией о хромосоме
    """
    return _chromosome_decoder.decode_chromosome(chromosome, verbose)


# ==========================================
# EXISTING PIPELINE PROCESSING FUNCTIONS  
# ==========================================

def get_dataset_name(path):
    """
    Extract dataset name from file path.
    Example: 'datasets/credit-score-classification/train.csv' -> 'credit-score-classification'
    """
    parts = os.path.normpath(path).split(os.sep)
    if 'datasets' in parts:
        datasets_index = parts.index('datasets')
        if len(parts) > datasets_index + 1:
            return parts[datasets_index + 1]
    return 'unknown_dataset'

def process_data(train_path, test_path, target_column,
                 imputation_method='knn', imputation_params=None,
                 outlier_method='isolation_forest', outlier_params=None,
                 encoding_method='label', encoding_params=None,
                 resampling_method='none', resampling_params=None,
                 scaling_method='none', scaling_params=None,
                 save_processed_data=True,
                 save_model_artifacts=True):
    """
    Process data using specified methods and their hyperparameters.
    
    Args:
        train_path: str, path to training data
        test_path: str, path to test data
        target_column: str, name of target column
        imputation_method: str, e.g., 'knn', 'median', 'missforest'
        imputation_params: dict, HPs for the imputation method
        outlier_method: str, e.g., 'isolation_forest', 'iqr', 'none'
        outlier_params: dict, HPs for the outlier removal method
        encoding_method: str, e.g., 'onehot', 'label', 'lsa', 'word2vec'
        encoding_params: dict, HPs for the encoding method
        resampling_method: str, e.g., 'none', 'oversample', 'smote'
        resampling_params: dict, HPs for the resampling method
        scaling_method: str, e.g., 'none', 'standard', 'minmax'
        scaling_params: dict, HPs for the scaling method
        save_processed_data: bool, whether to save processed data to disk
        save_model_artifacts: bool, whether to create and use research_path for model artifacts (e.g. plots)
        
    Returns:
        tuple: (train_data, test_data, research_path)
               train_data/test_data are DataFrames if save_processed_data is False,
               otherwise they are file paths.
               Returns (None, None, None) if a critical error occurs.
    """
    imputation_params = imputation_params or {}
    outlier_params = outlier_params or {}
    encoding_params = encoding_params or {}
    resampling_params = resampling_params or {}
    scaling_params = scaling_params or {}
    
    dataset_name = get_dataset_name(train_path)
    
    experiment_name_parts = [imputation_method, outlier_method, encoding_method, resampling_method]
    if scaling_method != 'none':
        experiment_name_parts.append(scaling_method)
    experiment_name = "_".join(experiment_name_parts)
    
    results_path = os.path.join('results', dataset_name, experiment_name)
    research_path = os.path.join("research", dataset_name, experiment_name)
    
    if save_processed_data:
        os.makedirs(results_path, exist_ok=True)
    
    if save_model_artifacts:
        os.makedirs(research_path, exist_ok=True)

    output_suffix = f"_processed_{experiment_name}"

    print(f"\n[Pipeline Stage - Config: {experiment_name}] Loading data...")
    loader = DataLoader(train_path=train_path, test_path=test_path)
    train_data, test_data = loader.load_data()

    if train_data is None or train_data.empty:
        print(f"Error: Training data could not be loaded from {train_path} or is empty.")
        return None, None, None

    preprocessor = DataPreprocessor()
    
    print(f"[Pipeline Stage - Config: {experiment_name}] Imputation ({imputation_method}, HPs: {imputation_params})...")
    train_data = preprocessor.impute(train_data, method=imputation_method, **imputation_params)
    if test_data is not None and not test_data.empty:
        test_data = preprocessor.impute(test_data, method=imputation_method, **imputation_params)
    print(f"[Pipeline Stage - Config: {experiment_name}] Imputation completed.")

    potential_id_columns_process = [col for col in ['ID', 'id'] if col in train_data.columns and col != target_column]
    if potential_id_columns_process:
        print(f"Dropping ID column(s) from train_data: {potential_id_columns_process} before outlier removal.")
        train_data = train_data.drop(columns=potential_id_columns_process)
        if test_data is not None and not test_data.empty:
            test_data = test_data.drop(columns=potential_id_columns_process, errors='ignore')

    print(f"[Pipeline Stage - Config: {experiment_name}] Outlier removal ({outlier_method}, HPs: {outlier_params})...")
    if outlier_method != 'none':
        train_data_before_outliers = train_data.copy()
        try:
            remover = OutlierRemover(method=outlier_method, **outlier_params) 
            features_for_removal = train_data.drop(columns=[target_column], errors='ignore')
            
            target_series_for_reconstruction = None
            if target_column in train_data.columns:
                target_series_for_reconstruction = train_data[target_column]

            cleaned_features = remover.remove_outliers(features_for_removal)
            
            if cleaned_features.empty and not features_for_removal.empty :
                print(f"Warning: Outlier removal (method: {outlier_method}) resulted in an empty dataset. Reverting.")
                train_data = train_data_before_outliers
            elif target_series_for_reconstruction is not None:
                cleaned_features_idx = cleaned_features.index
                target_series_aligned = target_series_for_reconstruction.loc[target_series_for_reconstruction.index.intersection(cleaned_features_idx)]
                cleaned_features_aligned = cleaned_features.loc[cleaned_features.index.intersection(target_series_aligned.index)]
                
                train_data = pd.concat([cleaned_features_aligned, target_series_aligned.rename(target_column)], axis=1)

            else:
                train_data = cleaned_features
        except Exception as e:
            print(f"Could not remove outliers (method: {outlier_method}): {e}. Reverting.")
            train_data = train_data_before_outliers
    else:
        print("Outlier removal skipped as method is 'none'.")
    print(f"[Pipeline Stage - Config: {experiment_name}] Outlier removal completed.")

    # --- Diagnostic: Check for duplicate columns before encoding ---
    duplicated_cols_before_encoding = train_data.columns[train_data.columns.duplicated()].tolist()
    if duplicated_cols_before_encoding:
        print(f"WARNING: Duplicate columns found in train_data BEFORE encoding: {duplicated_cols_before_encoding}")
        train_data = train_data.loc[:, ~train_data.columns.duplicated(keep='first')]
        print(f"         Duplicates removed. Columns now: {train_data.columns.tolist()}")
    if test_data is not None and not test_data.empty:
        duplicated_cols_test_before_encoding = test_data.columns[test_data.columns.duplicated()].tolist()
        if duplicated_cols_test_before_encoding:
            print(f"WARNING: Duplicate columns found in test_data BEFORE encoding: {duplicated_cols_test_before_encoding}")
            test_data = test_data.loc[:, ~test_data.columns.duplicated(keep='first')]
            print(f"         Duplicates removed from test_data. Columns now: {test_data.columns.tolist()}")
    # --- End Diagnostic ---

    print(f"[Pipeline Stage - Config: {experiment_name}] Encoding ({encoding_method}, HPs: {encoding_params})...")

    train_data = preprocessor.encode(train_data, method=encoding_method, target_col=target_column, **encoding_params)
    if test_data is not None and not test_data.empty:
        test_data = preprocessor.encode(test_data, method=encoding_method, target_col=target_column, **encoding_params)
    print(f"[Pipeline Stage - Config: {experiment_name}] Encoding completed.")
    
    print(f"[Pipeline Stage - Config: {experiment_name}] Resampling ({resampling_method}, HPs: {resampling_params})...")
    if resampling_method != 'none':
        if target_column in train_data.columns:
            if not pd.api.types.is_numeric_dtype(train_data[target_column]):
                print(f"Warning: Target column '{target_column}' is not numeric. Applying LabelEncoding before resampling.")
                train_data = preprocessor._label_encode_target(train_data, target_column)
            
            X_train_encoded = train_data.drop(columns=[target_column])
            y_train_encoded = train_data[target_column]
            
            resampler = Resampler(method=resampling_method, **resampling_params)
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_encoded, y_train_encoded)
            
            train_data = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train_encoded.columns), 
                                    pd.Series(y_train_resampled, name=target_column)], axis=1)
        else:
            print(f"Warning: Target column '{target_column}' not found. Skipping resampling.")
    else:
        print("Resampling method is 'none'. Skipping resampling step.")
    print(f"[Pipeline Stage - Config: {experiment_name}] Resampling completed.")

    print(f"[Pipeline Stage - Config: {experiment_name}] Scaling ({scaling_method}, HPs: {scaling_params})...")
    if scaling_method != 'none' and scaling_method in ['standard', 'minmax']:
        if target_column not in train_data.columns:
            print(f"CRITICAL WARNING: Target column '{target_column}' not found in train_data before scaling. Skipping scaling.")
        else:
            X_train_features = train_data.drop(columns=[target_column])
            y_train_target = train_data[target_column]

            scaler_instance = None
            if scaling_method == 'standard':
                scaler_instance = StandardScaler(**scaling_params)
            elif scaling_method == 'minmax':
                scaler_instance = MinMaxScaler(**scaling_params)

            if scaler_instance:
                try:
                    scaled_X_train_values = scaler_instance.fit_transform(X_train_features)
                    scaled_X_train_df = pd.DataFrame(scaled_X_train_values, columns=X_train_features.columns, index=X_train_features.index)
                    train_data = pd.concat([scaled_X_train_df, y_train_target], axis=1)

                    if test_data is not None and not test_data.empty:
                        X_test_features_original = test_data.drop(columns=[target_column], errors='ignore')
                        y_test_target_original = test_data[target_column] if target_column in test_data.columns else None
                        
                        cols_to_scale_in_test = [col for col in X_train_features.columns if col in X_test_features_original.columns]
                        
                        if not cols_to_scale_in_test:
                             print("Warning: No common features to scale in test data. Test data unscaled.")
                        elif len(cols_to_scale_in_test) != len(X_train_features.columns):
                            print("Warning: Test data feature set mismatch for scaling. Scaling subset.")
                            X_test_subset_to_scale = X_test_features_original[cols_to_scale_in_test]
                            scaled_X_test_subset_values = scaler_instance.transform(X_test_subset_to_scale)
                            scaled_X_test_subset_df = pd.DataFrame(scaled_X_test_subset_values, columns=cols_to_scale_in_test, index=X_test_subset_to_scale.index)
                            
                            X_test_features_updated = X_test_features_original.copy()
                            for col in cols_to_scale_in_test:
                                X_test_features_updated[col] = scaled_X_test_subset_df[col]

                            if y_test_target_original is not None:
                                test_data = pd.concat([X_test_features_updated, y_test_target_original], axis=1)
                            else:
                                test_data = X_test_features_updated
                        else: 
                            scaled_X_test_values = scaler_instance.transform(X_test_features_original[X_train_features.columns])
                            scaled_X_test_df = pd.DataFrame(scaled_X_test_values, columns=X_train_features.columns, index=X_test_features_original.index)

                            if y_test_target_original is not None:
                                test_data = pd.concat([scaled_X_test_df, y_test_target_original], axis=1)
                            else:
                                test_data = scaled_X_test_df
                except Exception as e:
                    print(f"Error during scaling: {e}. Skipping scaling.")
    elif scaling_method != 'none':
        print(f"Warning: Unknown scaling method '{scaling_method}'. Scaling skipped.")
    print(f"[Pipeline Stage - Config: {experiment_name}] Scaling completed.")
    
    train_data_path_out = None
    test_data_path_out = None

    if save_processed_data:
        train_data_path_out = os.path.join(results_path, f'train{output_suffix}.csv')
        test_data_path_out = os.path.join(results_path, f'test{output_suffix}.csv')
        
        train_data.to_csv(train_data_path_out, index=False)
        if test_data is not None and not test_data.empty:
            test_data.to_csv(test_data_path_out, index=False)
        else:
            test_data_path_out = None 
        print(f"[Pipeline Stage - Config: {experiment_name}] Data processing complete. Processed files saved to: {results_path}")
        return train_data_path_out, test_data_path_out, research_path # Return paths
    else:
        print(f"[Pipeline Stage - Config: {experiment_name}] Data processing complete. Processed data NOT saved (GA mode).")
        return train_data, test_data, research_path # Return DataFrames

    # Return DataFrames directly if not saving, otherwise paths
    if not save_processed_data:
        return train_data, test_data, research_path
    else:
        return train_data_path_out, test_data_path_out, research_path

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