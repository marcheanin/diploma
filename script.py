import os

# Имя файла, в который будет собран весь код
OUTPUT_FILENAME = "project_code_dump.txt"

# Директории и файлы для обработки
SRC_DIRECTORY = "src"
REQUIREMENTS_FILE = "requirements.txt"


def aggregate_project_files():
    """
    Собирает содержимое всех .py файлов из директории src и requirements.txt
    в один текстовый файл.
    """
    files_to_include = []

    # 1. Находим все .py файлы в директории src и ее подпапках
    if os.path.isdir(SRC_DIRECTORY):
        for root, _, filenames in os.walk(SRC_DIRECTORY):
            for filename in filenames:
                if filename.endswith('.py'):
                    # Добавляем полный путь к файлу
                    files_to_include.append(os.path.join(root, filename))
    else:
        print(f"Предупреждение: Директория '{SRC_DIRECTORY}' не найдена.")

    # 2. Сортируем файлы для последовательного вывода
    files_to_include.sort()
    
    # 3. Добавляем requirements.txt в начало списка, если он существует
    if os.path.exists(REQUIREMENTS_FILE):
        files_to_include.insert(0, REQUIREMENTS_FILE)
    else:
        print(f"Предупреждение: Файл '{REQUIREMENTS_FILE}' не найден.")

    # 4. Открываем выходной файл для записи
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            print(f"Создан файл '{OUTPUT_FILENAME}'. Начинаем сборку...")

            for filepath in files_to_include:
                if ".ipynb_checkpoints" in filepath:
                    continue
                
                # Нормализуем путь для корректного отображения на любой ОС
                normalized_path = os.path.normpath(filepath)
                
                # Записываем заголовок с именем файла
                outfile.write(f"========== FILE: {normalized_path} ==========\n\n")
                
                try:
                    # Открываем и читаем содержимое исходного файла
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                    
                    # Добавляем пустые строки после содержимого файла для лучшей читаемости
                    outfile.write("\n\n\n")
                    print(f" - Добавлен файл: {normalized_path}")

                except Exception as e:
                    error_message = f"!!! ОШИБКА ЧТЕНИЯ ФАЙЛА: {e} !!!\n\n\n"
                    outfile.write(error_message)
                    print(f" - Ошибка при чтении файла {normalized_path}: {e}")

        print(f"\nСборка завершена. Все файлы ({len(files_to_include)} шт.) сохранены в '{OUTPUT_FILENAME}'.")

    except IOError as e:
        print(f"Не удалось создать или записать в файл '{OUTPUT_FILENAME}'. Ошибка: {e}")


if __name__ == "__main__":
    aggregate_project_files()
