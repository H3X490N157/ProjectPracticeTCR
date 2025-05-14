import pytesseract
from PIL import Image
import os

TESSERACT_PATH = r'/home/dmitryshatalov/Documents/tesseract'
TESSDATA_DIR = r'/home/dmitryshatalov/Documents/tesseract-main/only_rus_best' #здесь лежит ТОЛЬКО наша модель rus_best, конфиг требует ссылку на папку с моделями, а не на конкретную модель
#может, буду фиксить позже
MODEL_NAME = 'rus_best'



pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
n = int(input())
for i in range(1, n + 1):
    input_file = f'/home/dmitryshatalov/Documents/tesseract-main/photos/photo{i}.jpeg'
    output_file = f'home/dmitryshatalov/Documents/tesseract-main/results/result{i}.txt'
    try:
        text = pytesseract.image_to_string(
            Image.open(input_file),
            lang=MODEL_NAME,
            config=f'--tessdata-dir "{TESSDATA_DIR}" --psm 6'
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
    except FileNotFoundError:
        print(f"Файл не найден ааааааа")
    except Exception as e:
        print(f"Какая-то другая ошибка")