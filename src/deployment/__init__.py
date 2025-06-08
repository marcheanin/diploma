"""
Модуль развертывания (deployment) для системы кредитного скоринга

Содержит классы и функции для:
- Сериализации обученных пайплайнов
- Загрузки и применения моделей в продакшене
- CLI интерфейса для работы с моделями
"""

from .model_serializer import UniversalModelSerializer
from .production_pipeline import ProductionPipeline

__all__ = [
    'UniversalModelSerializer',
    'ProductionPipeline'
] 