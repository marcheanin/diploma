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
    
    def __init__(self, preprocessor_states: Dict[str, Any], model: Any, metadata: Dict[str, Any]):
        self.preprocessor_states = preprocessor_states
        self.model = model
        self.metadata = metadata
        
        print(f"[ProductionPipeline] Создан пайплайн для модели: {metadata.get('model_type', 'unknown')}")
    
    def preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        print(f"[ProductionPipeline] Предобработка данных (shape: {raw_data.shape})")
        
        data = raw_data.copy()
        
        target_col = self.metadata.get('target_column')
        if target_col and target_col in data.columns:
            data = data.drop(columns=[target_col])
            print(f"[ProductionPipeline] Удалена целевая колонка '{target_col}' для предсказания")
        
        dropped_columns = self.preprocessor_states.get('dropped_columns', [])
        if dropped_columns:
            cols_to_drop = [col for col in dropped_columns if col in data.columns]
            if cols_to_drop:
                data = data.drop(columns=cols_to_drop)
                print(f"[ProductionPipeline] Удалены ID колонки: {cols_to_drop}")
        
        if 'preprocessor' in self.preprocessor_states:
            data = self._apply_imputation(data)
            print(f"[ProductionPipeline] Импутация применена")
        
        if 'preprocessor' in self.preprocessor_states:
            data = self._apply_encoding(data)
            print(f"[ProductionPipeline] Кодирование применено")
        
        if 'scaler' in self.preprocessor_states:
            data = self._apply_scaling(data)
            print(f"[ProductionPipeline] Масштабирование применено")
        
        print(f"[ProductionPipeline] Предобработка завершена (final shape: {data.shape})")
        return data
    
    def predict(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        print(f"[ProductionPipeline] Начало предсказания для {len(raw_data)} записей")
        
        processed_data = self.preprocess(raw_data)
        
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
        if 'preprocessor' not in self.preprocessor_states:
            print(f"[ProductionPipeline] Пропуск импутации - состояние не найдено")
            return data
            
        from preprocessing.data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.set_preprocessor_state(self.preprocessor_states['preprocessor'])
        
        config = self.preprocessor_states.get('processing_config', {})
        method = config.get('imputation_method', 'knn')
        params = config.get('imputation_params', {})
        
        print(f"[ProductionPipeline] Импутация: {method}")
        return preprocessor.impute(data, method=method, **params)
    
    def _apply_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'preprocessor' not in self.preprocessor_states:
            print(f"[ProductionPipeline] Пропуск кодирования - состояние не найдено")
            return data
            
        from preprocessing.data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.set_preprocessor_state(self.preprocessor_states['preprocessor'])
        
        config = self.preprocessor_states.get('processing_config', {})
        method = config.get('encoding_method', 'label')
        params = config.get('encoding_params', {})
        target_col = self.metadata.get('target_column')
        
        print(f"[ProductionPipeline] Кодирование: {method}")
        return preprocessor.encode(data, method=method, target_col=target_col, **params)
    
    def _apply_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = self.preprocessor_states.get('scaler')
        if scaler is None:
            print(f"[ProductionPipeline] Пропуск масштабирования - скейлер не найден")
            return data
            
        target_col = self.metadata.get('target_column')
        
        if target_col in data.columns:
            X_features = data.drop(columns=[target_col])
            y_target = data[target_col]
        else:
            X_features = data
            y_target = None
        
        print(f"[ProductionPipeline] Масштабирование: {self.preprocessor_states.get('scaler_method', 'unknown')}")
        
        try:
            scaled_features = scaler.transform(X_features)
            scaled_df = pd.DataFrame(scaled_features, columns=X_features.columns, index=X_features.index)
            
            if y_target is not None:
                return pd.concat([scaled_df, y_target], axis=1)
            else:
                return scaled_df
                
        except Exception as e:
            print(f"[ProductionPipeline] Ошибка масштабирования: {e}, пропускаем")
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