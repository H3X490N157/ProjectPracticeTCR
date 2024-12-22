import os
import subprocess

# Папки с данными
input_dir = "/home/dmitryshatalov/Документы/Cyrillic/Test/FullDS"  # Путь к папке с изображениями
output_dir = "/home/dmitryshatalov/Документы/Cyrillic/Test/Result"  # Путь к папке для сохранения результатов

# Создаем папку для результатов, если её нет
os.makedirs(output_dir, exist_ok=True)

# Проходим по каждому изображению в папке
for filename in os.listdir(input_dir):
    # Проверяем, является ли файл изображением
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0])

        # Запускаем Tesseract
        subprocess.run(["tesseract", input_path, output_path, "-l", "rus"])  # Укажите нужный язык
        print(f"Обработано: {filename}")

print("Обработка завершена. Результаты находятся в", output_dir)
