from datasets import load_from_disk

# Загрузка датасета с вашего пути
dataset = load_from_disk("/home/jenyagutrya/projects/python")

# Просмотр информации о датасете
print(dataset)
print(dataset.column_names)