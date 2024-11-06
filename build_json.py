import re
import json
from datasets import datasets
from sklearn.model_selection import train_test_split

# Input data: Labels and keywords
labels_with_keywords = [
    {
        "label": "ITEM_CATEGORY",
        "keywords": ["dish", "food", "drink", "meal"] 
    },
    {
        "label": "FLAVOR_TYPE",
        "keywords": ["spicy", "sweet", "savory", "sour", "bitter"]
    },
    {
        "label": "DIET_TYPE",
        "keywords": ["vegan", "vegetarian"]
    },
    {
        "label": "MEAL_TYPE",
        "keywords": ["lunch", "snack","breakfast", "dinner", "treat"]
    },
    {
        "label": "TEMPERATURE",
        "keywords": ["cold", "warm", "hot", "normal"]
    },
    {
        "label": "ALLERGY_TYPE",
        "keywords": ["gluten", "dairy", "seafood", "egg", "soy", "nut"]
    }
]

# Function to find keywords in a sentence and label them
def annotate_sentence(sentence, labels_with_keywords):
    entities = []

    # Iterate over each label and its associated keywords
    for label_data in labels_with_keywords:
        label = label_data["label"]
        keywords = label_data["keywords"]
        
        # Check each keyword in the sentence
        for keyword in keywords:
            # Use regular expression to find exact match of keyword
            match = re.search(r'\b' + re.escape(keyword) + r'\b', sentence)
            if match:
                start_idx = match.start()
                end_idx = match.end()
                entities.append((start_idx, end_idx, label))
    
    return {"entities": entities}

# Create training data in spaCy NER format
data = []

for sentence in datasets:
    annotations = annotate_sentence(sentence, labels_with_keywords)
    if annotations["entities"]:
        data.append((sentence, annotations))

# Write the result to a JSON file
with open("./datasets/all.json", "w") as f:
    json.dump(data, f, indent=4)

with open("./datasets/all.json", "r") as f:
    data = json.load(f)

# First split: training + validation vs. test
train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Second split: training vs. validation
train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

with open("./datasets/train_data.json", "w") as f:
    json.dump(train_data, f)
with open("./datasets/val_data.json", "w") as f:
    json.dump(val_data, f)
with open("./datasets/test_data.json", "w") as f:
    json.dump(test_data, f)

# Summary of the split
print("Training set size:", len(train_data))
print("Validation set size:", len(val_data))
print("Test set size:", len(test_data))