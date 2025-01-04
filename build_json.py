import re
import json
from sklearn.model_selection import train_test_split
import os
from utils.annotate_sentence import *
from config.entities import *

os.system('cls')

keywords_to_annotated = [keyword for entity in entities for keyword in entity["keywords"]]
data = []

# Load JSON data from a file
with open("./datasets/datasets.json", "r") as file:
    datasets = json.load(file)

# Remove duplicates
datasets = list(set(datasets))

# Write the result to a JSON file
with open("./datasets/datasets.json", "w") as f:
    json.dump(datasets, f, indent=4)

# Create training data in spaCy NER format
for sentence in datasets:
    annotations = annotate_sentence(sentence, keywords_to_annotated)
    if annotations["entities"]:
        data.append((sentence, annotations))

with open("./datasets/all.json", "w") as f:
    json.dump(data, f, indent=4)

with open("./datasets/all.json", "r") as f:
    data = json.load(f)

if len(data) > 0:
    # First split: training + validation vs. test
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    with open("./datasets/train_data.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open("./datasets/val_data.json", "w") as f:
        json.dump(val_data, f, indent=4)
    with open("./datasets/test_data.json", "w") as f:
        json.dump(test_data, f, indent=4)

    # Summary of the split
    print("Training set size:", len(train_data))
    print("Validation set size:", len(val_data))
    print("Test set size:", len(test_data))
    print("Total:", len(train_data) + len(val_data) + len(test_data))
    if len(keywords_to_annotated) > 0:
        print('All un-annotated keywords: ' + ', '.join(keywords_to_annotated))
else:
    print('No data found')
