from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from PIL import Image
import torch

# Загрузка данных
dataset = load_from_disk("/home/jenyagutyra/projects/python")

# Загрузка модели и процессора
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# Подготовка данных
def preprocess_data(example):
    image = Image.open(example['image']).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
    labels = processor.tokenizer(example['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=128).input_ids[0]
    return {"pixel_values": pixel_values, "labels": labels}

encoded_dataset = dataset.map(preprocess_data)

# Создание data collator
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model, padding=True)

# Настройки обучения
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
)

# Инициализация тренера
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

# Запуск обучения
trainer.train()
