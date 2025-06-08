"""
Класс готового к эксплуатации ML-пайплайна
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union

try:
    from .model_serializer import UniversalModelSerializer
except ImportError:
    from model_serializer import UniversalModelSerializer


class ProductionPipeline:
    """
    Готовый к эксплуатации ML-пайплайн
    
    Содержит:
    - Обученные препроцессоры (с состоянием)
    - Обученную модель
    - Метаданные о процессе обучения
    """
    
    def __init__(self, preprocessor_states: Dict[str, Any], model: Any, metadata: Dict[str, Any]):
        """
        Инициализация production пайплайна
        
        Args:
            preprocessor_states: Словарь с состояниями всех препроцессоров
            model: Обученная модель
            metadata: Метаданные о пайплайне
        """
        self.preprocessor_states = preprocessor_states
        self.model = model
        self.metadata = metadata
        
        print(f"[ProductionPipeline] Создан пайплайн для модели: {metadata.get('model_type', 'unknown')}")
    
    def preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет ту же предобработку, что была при обучении
        
        Args:
            raw_data: Необработанные данные
            
        Returns:
            Предобработанные данные
        """
        print(f"[ProductionPipeline] Предобработка данных (shape: {raw_data.shape})")
        
        data = raw_data.copy()
        
        # Применяем предобработку в том же порядке, что при обучении
        # Порядок важен! Он должен соответствовать порядку в pipeline_processor.py
        
        # 1. Импутация (если была применена)
        if 'imputation' in self.preprocessor_states:
            data = self._apply_imputation(data)
            print(f"[ProductionPipeline] Импутация применена")
        
        # 2. Кодирование (если было применено)
        if 'encoding' in self.preprocessor_states:
            data = self._apply_encoding(data)
            print(f"[ProductionPipeline] Кодирование применено")
        
        # 3. Масштабирование (если было применено)
        if 'scaling' in self.preprocessor_states:
            data = self._apply_scaling(data)
            print(f"[ProductionPipeline] Масштабирование применено")
        
        print(f"[ProductionPipeline] Предобработка завершена (final shape: {data.shape})")
        return data
    
    def predict(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Основная функция предсказания
        
        Args:
            raw_data: Необработанные данные
            
        Returns:
            DataFrame с результатами предсказаний
        """
        print(f"[ProductionPipeline] Начало предсказания для {len(raw_data)} записей")
        
        # Предобработка
        processed_data = self.preprocess(raw_data)
        
        # Предсказание
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
                predictions = self.model.predict(processed_data)
                results = self._format_results(raw_data, predictions, probabilities)
            else:
                predictions = self.model.predict(processed_data)
                results = self._format_results(raw_data, predictions)
            
            print(f"[ProductionPipeline] Предсказания успешно получены")
            return results
            
        except Exception as e:
            print(f"[ProductionPipeline] Ошибка при предсказании: {e}")
            raise
    
    def _apply_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применяет сохраненные параметры импутации"""
        # Пока заглушка - будет реализовано при интеграции с DataPreprocessor
        print(f"[ProductionPipeline] Импутация: {self.preprocessor_states['imputation']['method']}")
        return data
    
    def _apply_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применяет сохраненные энкодеры"""
        # Пока заглушка - будет реализовано при интеграции с DataPreprocessor
        print(f"[ProductionPipeline] Кодирование: {self.preprocessor_states['encoding']['method']}")
        return data
    
    def _apply_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применяет сохраненные скейлеры"""
        # Пока заглушка - будет реализовано при интеграции с DataPreprocessor
        print(f"[ProductionPipeline] Масштабирование: {self.preprocessor_states['scaling']['method']}")
        return data
    
    def _format_results(self, raw_data: pd.DataFrame, predictions: np.ndarray, 
                       probabilities: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Форматирует результаты в удобном виде
        
        Args:
            raw_data: Исходные данные
            predictions: Предсказания модели
            probabilities: Вероятности (если доступны)
            
        Returns:
            DataFrame с отформатированными результатами
        """
        results = raw_data.copy()
        results['prediction'] = predictions
        
        if probabilities is not None:
            if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                # Многоклассовая классификация или бинарная с вероятностями
                for i in range(probabilities.shape[1]):
                    results[f'probability_class_{i}'] = probabilities[:, i]
                
                if probabilities.shape[1] == 2:  # Бинарная классификация
                    results['credit_score'] = probabilities[:, 1]  # Вероятность положительного класса
                    results['risk_level'] = results['credit_score'].apply(
                        lambda x: 'Low' if x > 0.7 else 'Medium' if x > 0.4 else 'High'
                    )
        
        return results
    
    def save(self, save_path: str) -> str:
        """
        Сохраняет весь пайплайн
        
        Args:
            save_path: Путь к папке для сохранения
            
        Returns:
            Путь к сохраненному пайплайну
        """
        print(f"[ProductionPipeline] Сохранение пайплайна в {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        try:
            # 1. Сохраняем модель (используя универсальный сериализатор)
            model_info = UniversalModelSerializer.save_model(
                self.model, 
                save_path, 
                self.metadata.get('model_type')
            )
            
            # 2. Сохраняем состояния препроцессоров
            preprocessor_path = os.path.join(save_path, 'preprocessor_states.pkl')
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor_states, f)
            print(f"[ProductionPipeline] Препроцессоры сохранены: {preprocessor_path}")
            
            # 3. Сохраняем метаданные пайплайна
            pipeline_metadata = {
                **self.metadata,
                'model_info': model_info,
                'preprocessor_path': 'preprocessor_states.pkl',
                'pipeline_created_at': datetime.now().isoformat(),
                'pipeline_version': '1.0'
            }
            
            metadata_path = os.path.join(save_path, 'pipeline_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_metadata, f, indent=2, ensure_ascii=False, default=str)
            print(f"[ProductionPipeline] Метаданные сохранены: {metadata_path}")
            
            print(f"[ProductionPipeline] ✅ Пайплайн успешно сохранен в {save_path}")
            return save_path
            
        except Exception as e:
            print(f"[ProductionPipeline] ❌ Ошибка при сохранении: {e}")
            raise
    
    @classmethod
    def load(cls, save_path: str) -> 'ProductionPipeline':
        """
        Загружает весь пайплайн
        
        Args:
            save_path: Путь к папке с сохраненным пайплайном
            
        Returns:
            Экземпляр ProductionPipeline
        """
        print(f"[ProductionPipeline] Загрузка пайплайна из {save_path}")
        
        try:
            # 1. Загружаем метаданные
            metadata_path = os.path.join(save_path, 'pipeline_metadata.json')
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"pipeline_metadata.json не найден в {save_path}")
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 2. Загружаем модель
            model, model_info = UniversalModelSerializer.load_model(save_path)
            
            # 3. Загружаем состояния препроцессоров
            preprocessor_path = os.path.join(save_path, metadata['preprocessor_path'])
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Файл препроцессоров не найден: {preprocessor_path}")
            
            with open(preprocessor_path, 'rb') as f:
                preprocessor_states = pickle.load(f)
            
            print(f"[ProductionPipeline] ✅ Пайплайн успешно загружен")
            return cls(preprocessor_states, model, metadata)
            
        except Exception as e:
            print(f"[ProductionPipeline] ❌ Ошибка при загрузке: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о пайплайне"""
        return {
            'model_type': self.metadata.get('model_type', 'unknown'),
            'dataset': self.metadata.get('dataset_name', 'unknown'),
            'chromosome': self.metadata.get('chromosome', []),
            'performance': self.metadata.get('performance', {}),
            'created_at': self.metadata.get('pipeline_created_at', 'unknown'),
            'preprocessing_steps': list(self.preprocessor_states.keys())
        }


def test_production_pipeline():
    """Простой тест ProductionPipeline"""
    print("\n=== ТЕСТ PRODUCTION PIPELINE ===")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import pandas as pd
        
        # Создаем тестовые данные
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        df['target'] = y
        
        # Обучаем простую модель
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(df.drop('target', axis=1), df['target'])
        
        # Создаем тестовые состояния препроцессоров
        preprocessor_states = {
            'imputation': {'method': 'knn', 'params': {'n_neighbors': 5}},
            'encoding': {'method': 'label', 'params': {}},
            'scaling': {'method': 'standard', 'params': {}}
        }
        
        # Создаем метаданные
        metadata = {
            'model_type': 'random_forest',
            'dataset_name': 'test_dataset',
            'chromosome': [1, 2, 3, 4, 5],
            'performance': {'pr_auc': 0.85}
        }
        
        # Создаем пайплайн
        pipeline = ProductionPipeline(preprocessor_states, model, metadata)
        
        # Тестируем сохранение
        save_path = "test_pipeline"
        pipeline.save(save_path)
        
        # Тестируем загрузку
        loaded_pipeline = ProductionPipeline.load(save_path)
        
        # Тестируем предсказание (на заглушке)
        test_data = df.drop('target', axis=1)[:5]
        results = loaded_pipeline.predict(test_data)
        
        print(f"Результаты предсказания:")
        print(results.head())
        
        print("✅ Тест ProductionPipeline пройден успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка в тесте ProductionPipeline: {e}")


if __name__ == "__main__":
    test_production_pipeline() 