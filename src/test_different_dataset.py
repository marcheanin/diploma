#!/usr/bin/env python3
"""
Тест гибкости новой архитектуры ГА с другим датасетом
"""

from ga_optimizer import GAConfig, run_genetic_algorithm

def test_different_dataset():
    print("=== Тест ГА с датасетом Loan_Default ===\n")
    
    # Создаем конфигурацию для другого датасета
    config = GAConfig(
        train_path="../datasets/credit-score-classification-manual-cleaned.csv",
        target_column="Credit_Score", 
        population_size=4,
        num_generations=3,
        elitism_percent=0.3,
        mutation_rate=0.2,
        tournament_size=2
    )
    
    print(f"Конфигурация: {config}")
    print()
    
    try:
        results = run_genetic_algorithm(config)
        
        if results and results['best_chromosome'] is not None:
            print(f"\n✅ Тест с другим датасетом успешен!")
            print(f"Лучший фитнес: {results['best_fitness']:.4f}")
            print(f"Конфигурация передана корректно")
        else:
            print(f"\n⚠️ ГА завершился, но не найдено решение")
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_different_dataset() 