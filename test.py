import spacy

# Load the trained NER model
nlp_ner = spacy.load("out/restaurant_ner_recommendation")

# Test it with some new sentences
doc = nlp_ner("Can you recommend somehting for dairy allergy?")

for ent in doc.ents:
    print(ent.text, ent.label_)