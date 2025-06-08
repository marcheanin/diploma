#!/usr/bin/env python3
"""
Быстрый тест ГА алгоритма на датасете diabetes.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ga_optimizer import GAConfig, run_genetic_algorithm

def test_ga_quick():
    print("=== Быстрый тест ГА алгоритма (новая архитектура) ===\n")
    
    try:
        # Создаем конфигурацию для быстрого теста
        test_config = GAConfig(
            train_path="../datasets/diabetes.csv",
            test_path=None,
            target_column="Outcome",
            population_size=4,  # Малая популяция
            num_generations=2,  # Всего 2 поколения
            elitism_percent=0.25,
            mutation_rate=0.1,
            tournament_size=2,  # Меньший турнир для малой популяции
            generate_learning_curves=False
        )
        
        print(f"Параметры теста:")
        print(f"  - Популяция: {test_config.population_size}")
        print(f"  - Поколения: {test_config.num_generations}")
        print(f"  - Датасет: {test_config.train_path}")
        print(f"  - Целевая переменная: {test_config.target_column}")
        print()
        
        # Запускаем ГА
        print("Запускаем ГА...")
        results = run_genetic_algorithm(test_config)
        
        if results and results['best_chromosome'] is not None:
            print(f"\n✅ Быстрый тест ГА завершен успешно!")
            print(f"Лучший фитнес: {results['best_fitness']:.4f}")
            print(f"Лучшая хромосома: {results['best_chromosome']}")
        else:
            print(f"\n⚠️ ГА завершился, но не найдено оптимальное решение")
        
    except Exception as e:
        print(f"❌ Ошибка в тесте ГА: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_ga_quick() 