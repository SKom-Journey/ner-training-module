from noun_template import noun_templates
from utils.find_entity_indices import find_entity_indices
from string import Template
import json

# Fill This With Your Data
noun = input("Noun: ")

# Label
label = input("Label: ")

# Result array for sentences and spaCy annotations
training_data = []

# Iterate through templates and substitute 'ADJECTIVE'
for template in noun_templates:
    t = Template(template)
    sentence = t.safe_substitute(NOUN=noun)

    # Find the start and end index of the adjective in the sentence using word boundaries
    start_idx, end_idx = find_entity_indices(sentence, noun)

    # Append the sentence and annotations in spaCy NER format if indices are valid
    if start_idx != -1:
        training_data.append((sentence, {"entities": [(start_idx, end_idx, label)]}))

# Write the result to a JSON file
output_file = "../datasets/noun.json"
with open(output_file, "w") as f:
    json.dump(training_data, f, indent=4)

print(f"Data saved to {output_file}")