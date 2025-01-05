import re
import json
from sklearn.model_selection import train_test_split
import os
from utils.annotate_sentence import *
from config.entities import *

os.system('cls')

keywords_to_annotated = {
    'menus_': [keyword for entity in entities[0:6] for keyword in entity["keywords"]],
    'symptoms_': entities[6]["keywords"]
}
entities_to_annotated = {
    'menus_': entities[0:6],
    'symptoms_': [entities[6]]
}
print(entities_to_annotated['menus_'])
def build_datasets(prefix = ''):
    data = []
    keywords_to_annotate = [*keywords_to_annotated[prefix]]

    # Load JSON data from a file
    with open("./datasets/" + prefix + "datasets.json", "r") as file:
        datasets = json.load(file)

    # Remove duplicates
    datasets = list(set(datasets))

    # Write the result to a JSON file
    with open("./datasets/" + prefix + "datasets.json", "w") as f:
        json.dump(datasets, f, indent=4)

    # Create training data in spaCy NER format
    for sentence in datasets:
        annotations = annotate_sentence(entities_to_annotated[prefix], sentence, keywords_to_annotate)
        if annotations["entities"]:
            data.append((sentence, annotations))

    with open("./datasets/" + prefix + "all.json", "w") as f:
        json.dump(data, f, indent=4)

    with open("./datasets/" + prefix + "all.json", "r") as f:
        data = json.load(f)

    if len(data) > 0:
        # First split: training + validation vs. test
        train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

        with open("./datasets/" + prefix + "train_data.json", "w") as f:
            json.dump(train_data, f, indent=4)
        with open("./datasets/" + prefix + "val_data.json", "w") as f:
            json.dump(val_data, f, indent=4)
        with open("./datasets/" + prefix + "test_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        # Summary of the split
        print("Training set size:", len(train_data))
        print("Validation set size:", len(val_data))
        print("Test set size:", len(test_data))
        print("Total:", len(train_data) + len(val_data) + len(test_data))
        if len(keywords_to_annotate) > 0:
            print('All un-annotated keywords: ' + ', '.join(keywords_to_annotate))
    else:
        print('No data found')

# Menus
build_datasets('menus_')

# Symptoms
build_datasets('symptoms_')