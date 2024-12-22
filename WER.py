from file.py import true_text
# true_text - переменная из файла, содержащая эталонную разметку текста
# file.py должен быть заменён на другой в зависимости от использования. На гит я его не выгружаю - Шаталов Д. от 15 декабря
from file2.py import got_text
# got_text - переменная из файла, содержащая инфу от тессеракта, которую нужно сравнить с эталоном
# file2.py - файл, сливающий в свою переменную got_text данные из множества текстовых файлов, полученных через работу тессеракта от tester.py. На гит грузить пока не буду - Шаталов Д. от 15 декабря; очень забавно, если это кто-то читает из "внешних"
from jiwer import wer, cer

def calculate_wer_cer(reference, hypothesis):
    """
    Вычисляет WER (Word Error Rate) и CER (Character Error Rate) между двумя текстами.

    :param reference: Оригинальный текст (строка)
    :param hypothesis: Распознанный текст (строка)
    :return: Словарь с WER и CER
    """
    wer_score = wer(reference, hypothesis)
    cer_score = cer(reference, hypothesis)

    return {
        'WER': wer_score,
        'CER': cer_score
    }

if __name__ == "__main__":

    # Расчет WER и CER
    results = calculate_wer_cer(true_text, got_text)

    # Вывод результатов
    print("Результаты:")
    print(f"WER: {results['WER']:.2%}")
    print(f"CER: {results['CER']:.2%}")
