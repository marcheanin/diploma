#!/usr/bin/env python3
"""
Тест рефакторинга декодирования хромосом
"""

from pipeline_processor import decode_and_log_chromosome

def test_refactor():
    print("=== Тест рефакторинга декодирования хромосом ===\n")
    
    # Тестовая хромосома
    test_chromosome = [1, 2, 0, 1, 3, 2, 0, 1, 4, 2, 1, 0, 1, 0, 1, 2, 3, 1, 0, 2]
    
    print("Тестируем новый декодер:")
    result = decode_and_log_chromosome(test_chromosome, verbose=True)
    
    print("\n=== Результат ===")
    print(f"Модель: {result['model_type']}")
    print(f"Импутация: {result['imputation_method']}")
    print(f"Кодирование: {result['encoding_method']}")
    print(f"Масштабирование: {result['scaling_method']}")
    
    print("\n✅ Рефакторинг успешно завершен!")
    return result

if __name__ == "__main__":
    test_refactor() 