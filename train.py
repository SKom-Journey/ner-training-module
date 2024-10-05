import spacy
from spacy.training import Example
import random
import json

# Load the base model
nlp = spacy.load("en_core_web_sm")

# Get the NER pipeline component
ner = nlp.get_pipe("ner")

# Add NER pipeline if it's not already in the model
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Load the training data from the JSON file
with open("datasets/adjective.json", "r") as f:
    training_data = json.load(f)

# Add labels to the NER pipeline
for _, annotations in training_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipelines to only focus on NER
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

# Start training the NER model
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.begin_training()
    
    # Loop over the training data multiple times (epochs)
    for iteration in range(20):  # 20 iterations
        print(f"Starting iteration {iteration}")
        random.shuffle(training_data)
        
        losses = {}
        for text, annotations in training_data:
            # Create training Example
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)

            # Update the model with the example
            nlp.update([example], losses=losses, drop=0.35)
        print("Losses", losses)

# Save the trained model
nlp.to_disk("out/restaurant_ner_recommendation")
