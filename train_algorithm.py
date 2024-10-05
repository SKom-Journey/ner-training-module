import re
import json
from spacy.training.example import Example

# Input data: Labels and keywords
labels_with_keywords = [
    {
        "label": "ITEM_CATEGORY",
        "keywords": ["dish", "food", "foods", "eat", "drink", "drinks", "meal", "meals"] 
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
        "keywords": ["lunch", "snack", "breakfast", "dinner", "treat"]
    },
    {
        "label": "TEMPERATURE",
        "keywords": ["cold", "warm", "hot"]
    }
]

# Text data to be processed
text_data = [
    "I like spicy food",
    "I'm craving something spicy",
    "I prefer dishes that are sweet",
    "I want something savory",
    "I'm looking for a bitter option",
    "I want to drink something cold",
    "I'm following a vegan diet",
    "I like my food to be spicy",
    "Can you recommend something sweet?",
    "I feel like eating spicy",
    "I usually enjoy mild food",
    "I'm looking for something savory",
    "I love when my drink is cold.",
    "The best dishes are always spicy.",
    "My favorite foods are usually spicy.",
    "I prefer my drinks to be sweet.",
    "I'm in the mood for something sweet for my drink.",
    "Can you find me something vegetarian?",
    "What vegan dish would you recommend?",
    "I usually go for vegetarian meals.",
    "I'm interested in trying something vegan.",
    "What vegan option do you have for me?",
    "I want to try a vegetarian dish today.",
    "Do you have anything sweet on the menu?",
    "I'm in the mood for a treat.",
    "I feel like having a snack.",
    "What's a good meal to try?",
    "I prefer drinks that are hot.",
    "Can you suggest something warm for me?",
    "Do you have any lunch recommendations?"
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

for sentence in text_data:
    annotations = annotate_sentence(sentence, labels_with_keywords)
    if annotations["entities"]:
        training_data.append((sentence, annotations))

# Write the result to a JSON file
output_file = "./datasets/training.json"
with open(output_file, "w") as f:
    json.dump(training_data, f, indent=4)

print(f"Data saved to {output_file}")