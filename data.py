from datasets import Dataset
import json

def get_dataset():

    print(" --- Loading dataset...")
    with open("./dataset/combined_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    return Dataset.from_dict(dataset)
