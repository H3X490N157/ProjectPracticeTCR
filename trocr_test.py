from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Загрузка модели и процессора
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# Открытие изображения
image = Image.open("/home/jenyagutyra/projects/images.png").convert("RGB")

# Обработка изображения
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

# Распознавание текста
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Распознанный текст: {text}")