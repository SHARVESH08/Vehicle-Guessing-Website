from datasets import load_dataset
import os

dataset = load_dataset("Unit293/car_models_3887", split="train")

base_dir = "car_dataset"
os.makedirs(base_dir, exist_ok=True)

for i, item in enumerate(dataset):
    label = item["label"]
    image = item["image"]

    class_dir = os.path.join(base_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    image.save(os.path.join(class_dir, f"{i}.jpg"))


from datasets import load_dataset

dataset = load_dataset("Unit293/car_models_3887", split="train")

print(dataset)
print(dataset.features)
    