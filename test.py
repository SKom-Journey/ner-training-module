import spacy
import os

# Load the trained NER model
nlp_ner = spacy.load("out/restaurant_ner_recommendation")

os.system('cls')

# Test it with some new sentences
doc = nlp_ner(input('Text: ').replace("?", ""))

for ent in doc.ents:
    print(ent.text, ent.label_)