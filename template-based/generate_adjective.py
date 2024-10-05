from adjective_template import adjective_templates
from utils.find_entity_indices import find_entity_indices
from string import Template
import json

# Fill This With Your Data
adjective = input("Adjective: ")

# Label
label = input("Label: ")

# Result array for sentences and spaCy annotations
training_data = []

# Iterate through templates and substitute 'ADJECTIVE'
for template in adjective_templates:
    t = Template(template)
    sentence = t.safe_substitute(ADJECTIVE=adjective)

    # Find the start and end index of the adjective in the sentence using word boundaries
    start_idx, end_idx = find_entity_indices(sentence, adjective)

    # Append the sentence and annotations in spaCy NER format if indices are valid
    if start_idx != -1:
        training_data.append((sentence, {"entities": [(start_idx, end_idx, label)]}))

# Write the result to a JSON file
output_file = "../datasets/adjective.json"
with open(output_file, "w") as f:
    json.dump(training_data, f, indent=4)

print(f"Data saved to {output_file}")