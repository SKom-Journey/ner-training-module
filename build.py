import re
import json
from spacy.training.example import Example
from datasets import datasets

# Input data: Labels and keywords
labels_with_keywords = [
    {
        "label": "ITEM_CATEGORY",
        "keywords": ["dishes", "dish", "food", "foods", "eat", "drink", "drinks", "meal", "meals", "thirsty", "thirst", "hungry", "hunger"] 
    },
    {
        "label": "FLAVOR_TYPE",
        "keywords": ["spicy", "sweet", "savory"]
    },
    {
        "label": "DIET_TYPE",
        "keywords": ["vegan", "vegetarian"]
    },
    {
        "label": "MEAL_TYPE",
        "keywords": ["lunch", "lunches", "snack", "snacks", "breakfast", "breakfasts", "dinner", "treat", "dinners",]
    },
    {
        "label": "TEMPERATURE",
        "keywords": ["cold", "warm", "hot"]
    },
    {
        "label": "ALLERGY_TYPE",
        "keywords": ["gluten", "dairy", "seafood", "egg"]
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
training_data = []

for sentence in datasets:
    annotations = annotate_sentence(sentence, labels_with_keywords)
    if annotations["entities"]:
        training_data.append((sentence, annotations))

# Write the result to a JSON file
output_file = "./training.json"
with open(output_file, "w") as f:
    json.dump(training_data, f, indent=4)

print(f"Data saved to {output_file}")