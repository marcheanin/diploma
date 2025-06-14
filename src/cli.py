#!/usr/bin/env python3

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ga_optimizer import GAConfig, run_genetic_algorithm
from pipeline_processor import decode_chromosome_full, process_data, train_model
from deployment.production_pipeline import ProductionPipeline
from deployment.model_serializer import UniversalModelSerializer
from modeling.model_trainer import ModelTrainer


class MLPipelineCLI:
    """Командный интерфейс для ML пайплайна"""
    
    def __init__(self):
        project_root = Path.cwd()
        self.models_dir = project_root / "models"
        self.results_dir = project_root / "results" 
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
    def run_chromosome(self, args):
        print(f"🧬 Декодирование хромосомы: {args.chromosome}")
        print(f"📊 Датасет: {args.dataset}")
        print(f"🎯 Целевая переменная: {args.target}")
        
        try:
            chromosome = [int(x.strip()) for x in args.chromosome.split(',')]
            print(f"🔍 Хромосома: {chromosome}")
            
            decoded_info = decode_chromosome_full(chromosome, verbose=True)
            params = decoded_info['pipeline_params']
            
            print(f"\n⚙️ Параметры пайплайна:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            print(f"\n🔄 Обработка данных...")
            train_data, test_data, research_path, preprocessor_states = process_data(
                args.dataset, None, args.target,
                imputation_method=params['imputation_method'],
                imputation_params=params['imputation_params'],
                outlier_method=params['outlier_method'],
                outlier_params=params['outlier_params'],
                encoding_method=params['encoding_method'],
                encoding_params=params['encoding_params'],
                resampling_method=params['resampling_method'],
                resampling_params=params['resampling_params'],
                scaling_method=params['scaling_method'],
                scaling_params=params['scaling_params'],
                save_processed_data=False,
                save_model_artifacts=False
            )
            
            print(f"📊 Размер обучающих данных: {train_data.shape}")
            print(f"📊 Размер тестовых данных: {test_data.shape}")
            
            print(f"\n🚀 Обучение модели: {params['model_type']}")
            trainer = ModelTrainer(
                model_type=params['model_type'],
                model_hyperparameters=params['model_params']
            )
            
            metrics, feature_importance, trainer_dropped_cols = trainer.train(
                train_data, test_data, args.target,
                output_path=None,
                plot_learning_curves=False,
                save_run_results=False
            )
            
            if not metrics:
                print("❌ Ошибка обучения модели")
                return False
            
            # Объединяем информацию об удаленных колонках
            process_dropped_cols = preprocessor_states.get('dropped_columns', [])
            all_dropped_cols = list(set(process_dropped_cols + trainer_dropped_cols))
            preprocessor_states['dropped_columns'] = all_dropped_cols
            
            if all_dropped_cols:
                print(f"🗑️ Удалены ID колонки: {all_dropped_cols}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
            model_name = f"{Path(args.dataset).stem}_{params['model_type']}_{timestamp}"
            model_save_path = self.models_dir / model_name
            
            print(f"\n💾 Сохранение полного пайплайна...")
            
            pipeline_metadata = {
                'model_name': model_name,
                'dataset': args.dataset,
                'target_column': args.target,
                'features': list(train_data.columns[train_data.columns != args.target]),
                'chromosome': chromosome,
                'pipeline_config': params,
                'metrics': metrics,
                'model_type': params['model_type'],
                'created_at': timestamp,
                'source': 'cli_chromosome'
            }
            
            production_pipeline = ProductionPipeline(
                preprocessor_states=preprocessor_states,
                model=trainer.model,
                metadata=pipeline_metadata
            )
            
            production_pipeline.save(str(model_save_path))
            
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n📊 Метрики качества модели:")
            print(f"   📈 AUPRC: {metrics.get('auprc', 0):.4f}")
            print(f"   🎯 ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            print(f"   🎲 F1-Score: {metrics.get('f1_score', 0):.4f}")
            print(f"   ✅ Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"   🔍 Precision: {metrics.get('precision', 0):.4f}")
            print(f"   📞 Recall: {metrics.get('recall', 0):.4f}")
            
            auprc = metrics.get('auprc', metrics.get('accuracy', 0))
            print(f"\n✅ Модель успешно сохранена!")
            print(f"📁 Папка модели: {model_save_path}")
            print(f"📄 Метаданные: {metadata_path}")
            print(f"📈 Основная метрика (AUPRC): {auprc:.4f}")
            print(f"🧬 Хромосома: {chromosome}")
            
            return True
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_ga(self, args):
        print(f"🧬 Запуск генетического алгоритма")
        print(f"📊 Датасет: {args.train}")
        print(f"🎯 Целевая переменная: {args.target}")
        print(f"👥 Популяция: {args.population}")
        print(f"🔄 Поколения: {args.generations}")
        
        try:
            ga_config = GAConfig(
                train_path=args.train,
                test_path=None,
                target_column=args.target,
                population_size=args.population,
                num_generations=args.generations,
                elitism_percent=args.elitism,
                mutation_rate=args.mutation,
                tournament_size=args.tournament,
                generate_learning_curves=args.learning_curves
            )
            
            print(f"\n⚙️ Конфигурация: {ga_config}")
            
            print("\n🚀 Запуск оптимизации...")
            results = run_genetic_algorithm(ga_config)
            
            if not results or results['best_chromosome'] is None:
                print("❌ ГА не нашел оптимальное решение")
                return False
            
            best_chromosome = results['best_chromosome']
            best_fitness = results['best_fitness']
            
            print(f"\n🏆 Найдено оптимальное решение!")
            print(f"📈 Лучший фитнес: {best_fitness:.4f}")
            print(f"🧬 Лучшая хромосома: {best_chromosome}")
            
            if args.auto_save:
                print(f"\n💾 Сохранение лучшей модели...")
                
                decoded_info = decode_chromosome_full(best_chromosome, verbose=False)
                params = decoded_info['pipeline_params']
                
                train_data, test_data, research_path, preprocessor_states = process_data(
                    args.train, None, args.target,
                    imputation_method=params['imputation_method'],
                    imputation_params=params['imputation_params'],
                    outlier_method=params['outlier_method'],
                    outlier_params=params['outlier_params'],
                    encoding_method=params['encoding_method'],
                    encoding_params=params['encoding_params'],
                    resampling_method=params['resampling_method'],
                    resampling_params=params['resampling_params'],
                    scaling_method=params['scaling_method'],
                    scaling_params=params['scaling_params'],
                    save_processed_data=False,
                    save_model_artifacts=False
                )
                
                print(f"🚀 Обучение лучшей модели: {params['model_type']}")
                trainer = ModelTrainer(
                    model_type=params['model_type'],
                    model_hyperparameters=params['model_params']
                )
                
                metrics, feature_importance, trainer_dropped_cols = trainer.train(
                    train_data, test_data, args.target,
                    output_path=None,
                    plot_learning_curves=False,
                    save_run_results=False
                )
                
                # Объединяем информацию об удаленных колонках
                process_dropped_cols = preprocessor_states.get('dropped_columns', [])
                all_dropped_cols = list(set(process_dropped_cols + trainer_dropped_cols))
                preprocessor_states['dropped_columns'] = all_dropped_cols
                
                if all_dropped_cols:
                    print(f"🗑️ Удалены ID колонки: {all_dropped_cols}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{Path(args.train).stem}_GA_best_{timestamp}"
                model_save_path = self.models_dir / model_name
                
                print(f"💾 Сохранение полного пайплайна лучшей модели...")
                
                ga_metadata = {
                    'model_name': model_name,
                    'dataset': args.train,
                    'target_column': args.target,
                    'features': list(train_data.columns[train_data.columns != args.target]),
                    'chromosome': best_chromosome,
                    'pipeline_config': params,
                    'metrics': metrics,
                    'model_type': params['model_type'],
                    'ga_results': {
                        'best_fitness': best_fitness,
                        'fitness_history': results['fitness_history']
                    },
                    'created_at': timestamp,
                    'source': 'genetic_algorithm'
                }
                
                production_pipeline = ProductionPipeline(
                    preprocessor_states=preprocessor_states,
                    model=trainer.model,
                    metadata=ga_metadata
                )
                
                production_pipeline.save(str(model_save_path))
                
                metadata_path = self.models_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(ga_metadata, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"\n📊 Метрики лучшей модели:")
                print(f"   📈 AUPRC: {metrics.get('auprc', 0):.4f}")
                print(f"   🎯 ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"   🎲 F1-Score: {metrics.get('f1_score', 0):.4f}")
                print(f"   ✅ Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"   🔍 Precision: {metrics.get('precision', 0):.4f}")
                print(f"   📞 Recall: {metrics.get('recall', 0):.4f}")
                
                print(f"\n✅ Лучшая модель успешно сохранена!")
                print(f"📁 Папка модели: {model_save_path}")
                print(f"📄 Метаданные: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, args):
        print(f"🔮 Применение модели: {args.model}")
        print(f"📊 Данные: {args.data}")
        
        try:
            metadata_path = None
            if args.model.endswith('_metadata.json'):
                metadata_path = Path(args.model)
            else:
                metadata_path = self.models_dir / f"{args.model}_metadata.json"
                if not metadata_path.exists():
                    metadata_path = self.models_dir / f"{args.model.replace('.pkl', '')}_metadata.json"
            
            if not metadata_path.exists():
                print(f"❌ Метаданные модели не найдены: {args.model}")
                print(f"💡 Доступные модели можно посмотреть командой: list-models")
                return False
            
            print(f"📥 Загрузка метаданных: {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"📋 Модель: {metadata.get('model_name', 'unknown')}")
            print(f"🎯 Цель: {metadata.get('target_column', 'unknown')}")
            print(f"🤖 Тип: {metadata.get('pipeline_config', {}).get('model_type', 'unknown')}")
            
            print(f"📊 Загрузка данных: {args.data}")
            data = pd.read_csv(args.data)
            
            print(f"📋 Размер данных: {data.shape}")
            print(f"📊 Колонки: {list(data.columns)}")
            
            print(f"\n🔍 Анализ структуры данных...")
            target_column = metadata.get('target_column')
            expected_features = metadata.get('features', [])
            
            print(f"📊 Ожидаемые признаки ({len(expected_features)}): {expected_features[:5]}{'...' if len(expected_features) > 5 else ''}")
            print(f"📊 Полученные колонки ({len(data.columns)}): {list(data.columns)[:5]}{'...' if len(data.columns) > 5 else ''}")
            
            missing_features = set(expected_features) - set(data.columns)
            if missing_features:
                print(f"⚠️ Отсутствующие признаки: {list(missing_features)[:5]}{'...' if len(missing_features) > 5 else ''}")
            
            extra_features = set(data.columns) - set(expected_features) - {target_column}
            if extra_features:
                print(f"ℹ️ Дополнительные колонки: {list(extra_features)[:5]}{'...' if len(extra_features) > 5 else ''}")
            
            print(f"\n🔮 Загрузка и применение модели...")
            
            model_name = metadata.get('model_name')
            model_folder = self.models_dir / model_name
            
            if not model_folder.exists():
                print(f"❌ Папка модели не найдена: {model_folder}")
                return False
            
            pipeline_metadata_path = model_folder / 'pipeline_metadata.json'
            
            if pipeline_metadata_path.exists():
                print(f"🔧 Обнаружен полный пайплайн, загружаем ProductionPipeline...")
                try:
                    production_pipeline = ProductionPipeline.load(str(model_folder))
                    print(f"✅ Полный пайплайн успешно загружен")
                    
                    print(f"🔄 Применение полного пайплайна с предобработкой...")
                    results = production_pipeline.predict(data)
                    
                    print(f"✅ Предсказания выполнены успешно")
                    print(f"📈 Результаты предсказаний:")
                    print(f"🔢 Количество записей: {len(results)}")
                    
                    n_show = min(10, len(results))
                    for i in range(n_show):
                        pred = results.iloc[i]['prediction']
                        prob_cols = [col for col in results.columns if col.startswith('probability_class_')]
                        if prob_cols:
                            prob_str = ", ".join([f"{results.iloc[i][col]:.3f}" for col in prob_cols])
                            print(f"  [{i+1:2d}] Класс: {pred} (вероятности: [{prob_str}])")
                        else:
                            print(f"  [{i+1:2d}] Класс: {pred}")
                    
                    if len(results) > n_show:
                        print(f"  ... и еще {len(results) - n_show} записей")
                    
                    if args.output:
                        output_file = args.output
                    else:
                        project_root = Path.cwd()
                        results_dir = project_root / "results"
                        results_dir.mkdir(exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        data_path = Path(args.data)
                        data_filename = data_path.stem
                        
                        output_file = results_dir / f"predictions_{data_filename}_{model_name}_{timestamp}.csv"
                    
                    results.to_csv(output_file, index=False)
                    
                    print(f"\n💾 Результаты сохранены: {output_file}")
                    print(f"📊 Колонки результата: {list(results.columns)}")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Ошибка загрузки полного пайплайна: {e}")
                    print(f"🔄 Переключаемся на загрузку только модели...")
            
            print(f"🔧 Загружаем только модель (без предобработки)...")
            
            # Проверяем наличие модели
            model_info = metadata.get('model_info')
            if not model_info:
                print("❌ Информация о модели отсутствует в метаданных")
                return False
            
            # Загружаем модель
            try:
                model, loaded_model_info = UniversalModelSerializer.load_model(str(model_folder))
                print(f"✅ Модель успешно загружена")
                print(f"🔧 Тип: {loaded_model_info.get('model_type', 'unknown')}")
                print(f"📦 Формат: {loaded_model_info.get('serialization_format', 'unknown')}")
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
                return False
            
            # Подготавливаем данные для предсказания
            target_column = metadata.get('target_column')
            expected_features = metadata.get('features', [])
            
            # Удаляем целевую колонку если она присутствует
            prediction_data = data.copy()
            if target_column in prediction_data.columns:
                prediction_data = prediction_data.drop(columns=[target_column])
                print(f"ℹ️ Удалена целевая колонка '{target_column}' из данных для предсказания")
            
            # Проверяем соответствие признаков
            missing_features = set(expected_features) - set(prediction_data.columns)
            if missing_features:
                print(f"⚠️ Отсутствующие признаки: {list(missing_features)}")
                # Добавляем недостающие признаки с нулевыми значениями
                for feature in missing_features:
                    prediction_data[feature] = 0
                    print(f"  + Добавлен '{feature}' = 0")
            
            # Используем только ожидаемые признаки в правильном порядке
            prediction_data = prediction_data[expected_features]
            
            print(f"📊 Данные для предсказания: {prediction_data.shape}")
            
            # Выполняем предсказания
            try:
                predictions = model.predict(prediction_data)
                probabilities = model.predict_proba(prediction_data) if hasattr(model, 'predict_proba') else None
                
                # Для нейронных сетей может потребоваться специальная обработка
                model_type = loaded_model_info.get('model_type', 'unknown')
                if model_type == 'neural_network':
                    # Для нейронных сетей predictions может быть вероятностями
                    if predictions.ndim > 1 and predictions.shape[1] > 1:
                        probabilities = predictions
                        predictions = np.argmax(predictions, axis=1)
                    elif predictions.ndim == 1 or predictions.shape[1] == 1:
                        # Бинарная классификация
                        probabilities = np.column_stack([1 - predictions.ravel(), predictions.ravel()])
                        predictions = (predictions > 0.5).astype(int).ravel()
                
                # Определяем количество классов
                if probabilities is not None:
                    n_classes = probabilities.shape[1]
                else:
                    n_classes = len(np.unique(predictions))
                
                print(f"🎯 Количество классов: {n_classes}")
                print(f"✅ Предсказания выполнены успешно")
                
            except Exception as e:
                print(f"❌ Ошибка при выполнении предсказаний: {e}")
                return False
            
            print(f"📈 Результаты предсказаний:")
            print(f"🔢 Количество записей: {len(predictions)}")
            
            # Показываем первые несколько результатов
            n_show = min(10, len(predictions))
            for i in range(n_show):
                prob_str = ", ".join([f"{probabilities[i][j]:.3f}" for j in range(n_classes)])
                print(f"  [{i+1:2d}] Класс: {predictions[i]} (вероятности: [{prob_str}])")
            
            if len(predictions) > n_show:
                print(f"  ... и еще {len(predictions) - n_show} записей")
            
            # Создаем файл с результатами (всегда)
            output_data = data.copy()
            output_data['prediction'] = predictions
            
            # Добавляем колонки для вероятностей всех классов
            for class_idx in range(n_classes):
                output_data[f'probability_class_{class_idx}'] = probabilities[:, class_idx]
            
            # Определяем имя выходного файла
            if args.output:
                output_file = args.output
            else:
                # Автоматически генерируем имя файла в папке results в корне проекта  
                project_root = Path.cwd()  # текущая рабочая директория
                results_dir = project_root / "results"
                
                # Создаем папку results если её нет
                results_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = metadata.get('model_name', 'unknown')
                data_path = Path(args.data)
                data_filename = data_path.stem  # Имя файла без расширения
                
                output_file = results_dir / f"predictions_{data_filename}_{model_name}_{timestamp}.csv"
            
            # Сохраняем результаты
            output_data.to_csv(output_file, index=False)
            
            # Показываем полный путь
            full_path = os.path.abspath(str(output_file))
            print(f"\n💾 Результаты предсказаний сохранены:")
            print(f"📄 Полный путь: {full_path}")
            print(f"📊 Структура: {len(data)} записей + prediction + {n_classes} probability columns")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_models(self, args):
        """Просмотр доступных моделей"""
        print("📋 Доступные модели:")
        
        try:
            # Ищем файлы метаданных вместо .pkl файлов
            metadata_files = list(self.models_dir.glob("*_metadata.json"))
            
            if not metadata_files:
                print("❌ Модели не найдены")
                return False
            
            for metadata_path in sorted(metadata_files):
                model_name = metadata_path.stem.replace('_metadata', '')
                
                # Загружаем метаданные
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                dataset = Path(metadata.get('dataset', 'unknown')).stem
                target = metadata.get('target_column', 'unknown')
                metrics = metadata.get('metrics', {})
                auprc = metrics.get('auprc', 'N/A')
                roc_auc = metrics.get('roc_auc', 'N/A')
                f1_score = metrics.get('f1_score', 'N/A')
                accuracy = metrics.get('accuracy', 'N/A')
                created = metadata.get('created_at', 'unknown')
                source = metadata.get('source', 'manual')
                
                print(f"\n🤖 {model_name}")
                print(f"   📊 Датасет: {dataset}")
                print(f"   🎯 Цель: {target}")
                print(f"   📈 AUPRC: {auprc if auprc != 'N/A' else 'N/A'}")
                print(f"   🎯 ROC-AUC: {roc_auc if roc_auc != 'N/A' else 'N/A'}")
                print(f"   🎲 F1-Score: {f1_score if f1_score != 'N/A' else 'N/A'}")
                print(f"   ✅ Accuracy: {accuracy if accuracy != 'N/A' else 'N/A'}")
                print(f"   📅 Создан: {created}")
                print(f"   🔧 Источник: {source}")
                
                if args.verbose and 'chromosome' in metadata:
                    print(f"   🧬 Хромосома: {metadata['chromosome']}")
                
                # Проверяем наличие полной модели (папка или pkl файл)
                model_folder = self.models_dir / model_name
                pkl_path = self.models_dir / f"{model_name}.pkl"
                
                if model_folder.exists() and model_folder.is_dir():
                    # Новый формат - папка с моделью
                    try:
                        # Ищем файлы модели в папке
                        model_files = list(model_folder.glob("*.joblib")) + list(model_folder.glob("*.pkl")) + list(model_folder.glob("keras_model/"))
                        if model_files:
                            total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                            total_size += sum(sum(subf.stat().st_size for subf in f.rglob("*") if subf.is_file()) for f in model_files if f.is_dir())
                            print(f"   💾 Модель: {total_size / 1024:.1f} KB (полная)")
                        else:
                            print(f"   💾 Модель: папка без файлов модели")
                    except:
                        print(f"   💾 Модель: папка (не удалось определить размер)")
                elif pkl_path.exists():
                    # Старый формат - pkl файл
                    size = pkl_path.stat().st_size / 1024
                    print(f"   💾 Модель: {size:.1f} KB (legacy)")
                else:
                    print(f"   💾 Модель: метаданные только")
            
            print(f"\n📊 Всего моделей: {len(metadata_files)}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False


def main():
    """Главная функция CLI"""
    parser = argparse.ArgumentParser(
        description="CLI для production deployment системы ML пайплайнов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  
  # Запуск конкретной хромосомы
  python cli.py run-chromosome --chromosome "1,2,0,1,1,2,0,1,0,0,1,1,1,0,0,1,2,3,1,0" \\
                              --dataset "../datasets/diabetes.csv" \\
                              --target "Outcome"
  
  # Запуск ГА с сохранением лучшей модели
  python cli.py run-ga --dataset "../datasets/diabetes.csv" \\
                       --target "Outcome" \\
                       --population 10 \\
                       --generations 5 \\
                       --save-best
  
  # Применение модели к новым данным
  python cli.py predict --model "diabetes_logistic_regression_20250608_120000" \\
                        --data "new_data.csv" \\
                        --output "predictions.csv"
  
  # Просмотр доступных моделей
  python cli.py list-models --verbose
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда run-chromosome
    cmd_chromosome = subparsers.add_parser(
        'run-chromosome', 
        help='Запуск конкретной хромосомы с сериализацией'
    )
    cmd_chromosome.add_argument('--chromosome', required=True, 
                              help='Хромосома в формате "1,2,0,1,..." (20 генов)')
    cmd_chromosome.add_argument('--dataset', required=True,
                              help='Путь к датасету')
    cmd_chromosome.add_argument('--target', required=True,
                              help='Название целевой переменной')
    cmd_chromosome.add_argument('--learning-curves', action='store_true',
                              help='Генерировать кривые обучения')
    
    # Команда run-ga
    cmd_ga = subparsers.add_parser(
        'run-ga',
        help='Запуск генетического алгоритма'
    )
    cmd_ga.add_argument('--train', '--dataset', required=True,
                       help='Путь к датасету для обучения')
    cmd_ga.add_argument('--target', required=True,
                       help='Название целевой переменной')
    cmd_ga.add_argument('--population', type=int, default=10,
                       help='Размер популяции (по умолчанию: 10)')
    cmd_ga.add_argument('--generations', type=int, default=8,
                       help='Количество поколений (по умолчанию: 8)')
    cmd_ga.add_argument('--elitism', type=float, default=0.25,
                       help='Процент элитизма (по умолчанию: 0.25)')
    cmd_ga.add_argument('--mutation', type=float, default=0.1,
                       help='Вероятность мутации (по умолчанию: 0.1)')
    cmd_ga.add_argument('--tournament', type=int, default=3,
                       help='Размер турнира (по умолчанию: 3)')
    cmd_ga.add_argument('--learning-curves', action='store_true',
                       help='Генерировать кривые обучения')
    cmd_ga.add_argument('--auto-save', '--save-best', action='store_true',
                       help='Автоматически сохранить лучшую модель')
    
    # Команда predict
    cmd_predict = subparsers.add_parser(
        'predict',
        help='Применение модели к новым данным'
    )
    cmd_predict.add_argument('--model', required=True,
                           help='Путь к модели или имя модели')
    cmd_predict.add_argument('--data', required=True,
                           help='Путь к файлу с данными для предсказания')
    cmd_predict.add_argument('--output',
                           help='Путь для сохранения результатов')
    
    # Команда list-models
    cmd_list = subparsers.add_parser(
        'list-models',
        help='Просмотр доступных моделей'
    )
    cmd_list.add_argument('--verbose', action='store_true',
                         help='Подробная информация о моделях')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = MLPipelineCLI()
    
    # Маршрутизация команд
    if args.command == 'run-chromosome':
        success = cli.run_chromosome(args)
    elif args.command == 'run-ga':
        success = cli.run_ga(args)
    elif args.command == 'predict':
        success = cli.predict(args)
    elif args.command == 'list-models':
        success = cli.list_models(args)
    else:
        print(f"❌ Неизвестная команда: {args.command}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 