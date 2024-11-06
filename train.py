import spacy
from spacy.training import Example
import random
import json

# Load the base model
nlp = spacy.load("en_core_web_sm")

# Get the NER pipeline component, or add if not present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Load the datasets
with open("./datasets/train_data.json", "r") as f:
    train_data = json.load(f)
with open("./datasets/val_data.json", "r") as f:
    val_data = json.load(f)

# Add labels to the NER pipeline based on the training data
for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipelines to only focus on NER
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

# Training loop with validation after each epoch
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.begin_training()
    
    for iteration in range(20):  # Number of epochs
        print(f"Starting iteration {iteration + 1}")
        
        # Shuffle training data
        random.shuffle(train_data)
        
        # Track losses
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], losses=losses, drop=0.35)
        print("Losses:", losses)

        # Evaluate on validation set
        val_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in val_data]
        val_scores = nlp.evaluate(val_examples)
        print(f"Validation scores at iteration {iteration + 1}:", val_scores)

# Save the trained model
nlp.to_disk("out/restaurant_ner_recommendation")
