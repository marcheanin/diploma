"""
Универсальный сериализатор моделей для разных типов ML алгоритмов
"""

import os
import json
import pickle
import joblib
import warnings
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Подавление предупреждений TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Попытка импорта TensorFlow для Keras моделей
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, save_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Импорты для sklearn
try:
    from sklearn.base import BaseEstimator
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class UniversalModelSerializer:
    """
    Универсальный сериализатор для всех типов моделей в проекте.
    
    Автоматически выбирает оптимальный формат сериализации в зависимости от типа модели:
    - TensorFlow/Keras модели: SavedModel формат
    - Sklearn модели: joblib (оптимизированный для numpy)
    - Остальные: pickle (универсальный fallback)
    """
    
    SUPPORTED_FORMATS = ['tensorflow_savedmodel', 'joblib', 'pickle']
    
    @staticmethod
    def save_model(model: Any, save_path: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Сохраняет модель в оптимальном формате
        
        Args:
            model: Обученная модель для сохранения
            save_path: Путь к папке для сохранения
            model_type: Тип модели (если None, то автоопределение)
            
        Returns:
            Dict с информацией о сохраненной модели
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Автоопределение типа модели если не указан
        if model_type is None:
            model_type = UniversalModelSerializer._detect_model_type(model)
        
        print(f"[Serializer] Сохранение модели типа: {model_type}")
        
        model_info = {
            'model_type': model_type,
            'serialization_format': None,
            'model_path': None,
            'created_at': datetime.now().isoformat(),
            'library_versions': UniversalModelSerializer._get_library_versions()
        }
        
        try:
            if model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
                # Keras/TensorFlow модель
                model_path = os.path.join(save_path, 'keras_model')
                model.save(model_path)
                model_info['serialization_format'] = 'tensorflow_savedmodel'
                model_info['model_path'] = 'keras_model'
                print(f"[Serializer] Модель сохранена в формате TensorFlow SavedModel: {model_path}")
                
            elif model_type in ['logistic_regression', 'random_forest', 'gradient_boosting'] and SKLEARN_AVAILABLE:
                # Sklearn модели - используем joblib для оптимизации
                model_path = os.path.join(save_path, 'sklearn_model.joblib')
                joblib.dump(model, model_path, compress=3)  # Сжатие для экономии места
                model_info['serialization_format'] = 'joblib'
                model_info['model_path'] = 'sklearn_model.joblib'
                print(f"[Serializer] Модель сохранена в формате joblib: {model_path}")
                
            else:
                # Fallback для неизвестных моделей или отсутствующих библиотек
                model_path = os.path.join(save_path, 'model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                model_info['serialization_format'] = 'pickle'
                model_info['model_path'] = 'model.pkl'
                print(f"[Serializer] Модель сохранена в формате pickle: {model_path}")
                
        except Exception as e:
            print(f"[Serializer] Ошибка при сохранении модели: {e}")
            print(f"[Serializer] Используем fallback (pickle)")
            
            # Fallback к pickle при любой ошибке
            model_path = os.path.join(save_path, 'model_fallback.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            model_info['serialization_format'] = 'pickle'
            model_info['model_path'] = 'model_fallback.pkl'
            model_info['fallback_used'] = True
            model_info['original_error'] = str(e)
        
        # Сохраняем метаинформацию о модели
        info_path = os.path.join(save_path, 'model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"[Serializer] Метаинформация сохранена: {info_path}")
        return model_info
    
    @staticmethod
    def load_model(save_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Загружает модель, автоматически определяя формат
        
        Args:
            save_path: Путь к папке с сохраненной моделью
            
        Returns:
            Tuple[модель, метаинформация]
        """
        # Читаем метаинформацию
        info_path = os.path.join(save_path, 'model_info.json')
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Файл model_info.json не найден в {save_path}")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        model_path = os.path.join(save_path, model_info['model_path'])
        serialization_format = model_info['serialization_format']
        
        print(f"[Serializer] Загрузка модели из {model_path} (формат: {serialization_format})")
        
        # Проверяем совместимость версий
        UniversalModelSerializer._check_version_compatibility(model_info.get('library_versions', {}))
        
        try:
            if serialization_format == 'tensorflow_savedmodel':
                if not TENSORFLOW_AVAILABLE:
                    raise ImportError("TensorFlow не доступен для загрузки Keras модели")
                model = load_model(model_path)
                
            elif serialization_format == 'joblib':
                if not SKLEARN_AVAILABLE:
                    raise ImportError("Scikit-learn не доступен для загрузки joblib модели")
                model = joblib.load(model_path)
                
            elif serialization_format == 'pickle':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
            else:
                raise ValueError(f"Неизвестный формат сериализации: {serialization_format}")
            
            print(f"[Serializer] Модель успешно загружена")
            return model, model_info
            
        except Exception as e:
            print(f"[Serializer] Ошибка при загрузке модели: {e}")
            raise
    
    @staticmethod
    def _detect_model_type(model: Any) -> str:
        """Автоматическое определение типа модели"""
        
        # Проверяем Keras/TensorFlow модели
        if TENSORFLOW_AVAILABLE:
            try:
                if hasattr(model, 'save') and hasattr(model, 'predict'):
                    model_str = str(type(model))
                    if any(keyword in model_str.lower() for keyword in ['tensorflow', 'keras']):
                        return 'neural_network'
            except:
                pass
        
        # Проверяем sklearn модели
        if SKLEARN_AVAILABLE:
            try:
                if isinstance(model, BaseEstimator):
                    model_name = type(model).__name__.lower()
                    if 'logistic' in model_name:
                        return 'logistic_regression'
                    elif 'forest' in model_name:
                        return 'random_forest'
                    elif 'gradient' in model_name or 'gbm' in model_name:
                        return 'gradient_boosting'
                    else:
                        return 'sklearn_unknown'
            except:
                pass
        
        return 'unknown'
    
    @staticmethod
    def _get_library_versions() -> Dict[str, str]:
        """Получает версии используемых библиотек"""
        versions = {}
        
        try:
            import sklearn
            versions['sklearn'] = sklearn.__version__
        except ImportError:
            versions['sklearn'] = 'not_available'
        
        try:
            import tensorflow as tf
            versions['tensorflow'] = tf.__version__
        except ImportError:
            versions['tensorflow'] = 'not_available'
        
        try:
            import pandas as pd
            versions['pandas'] = pd.__version__
        except ImportError:
            versions['pandas'] = 'not_available'
        
        try:
            import numpy as np
            versions['numpy'] = np.__version__
        except ImportError:
            versions['numpy'] = 'not_available'
        
        return versions
    
    @staticmethod
    def _check_version_compatibility(saved_versions: Dict[str, str]) -> None:
        """Проверяет совместимость версий библиотек"""
        if not saved_versions:
            return
        
        current_versions = UniversalModelSerializer._get_library_versions()
        
        for lib, saved_version in saved_versions.items():
            if lib in current_versions and current_versions[lib] != 'not_available':
                current_version = current_versions[lib]
                if saved_version != current_version and saved_version != 'not_available':
                    warnings.warn(
                        f"Модель сохранена с {lib} {saved_version}, "
                        f"но загружается с {lib} {current_version}. "
                        f"Возможны проблемы совместимости."
                    )


def test_serializer():
    """Простой тест сериализатора"""
    print("\n=== ТЕСТ UNIVERSAL MODEL SERIALIZER ===")
    
    # Тест с простой sklearn моделью (если доступна)
    if SKLEARN_AVAILABLE:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Создаем простые тестовые данные
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        # Обучаем модель
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Тестируем сериализацию
        test_path = "test_serialization"
        print(f"Тестируем сериализацию в папку: {test_path}")
        
        model_info = UniversalModelSerializer.save_model(model, test_path)
        print(f"Сохранено: {model_info}")
        
        # Тестируем загрузку
        loaded_model, loaded_info = UniversalModelSerializer.load_model(test_path)
        print(f"Загружено: {loaded_info['model_type']}")
        
        # Проверяем, что модель работает
        predictions = loaded_model.predict(X[:5])
        print(f"Тестовые предсказания: {predictions}")
        
        print("✅ Тест пройден успешно!")
    else:
        print("❌ Sklearn не доступен для тестирования")


if __name__ == "__main__":
    test_serializer() 