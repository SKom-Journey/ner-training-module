import spacy
from spacy.training import biluo_tags_to_offsets

# Load the trained NER model
nlp_ner = spacy.load("out/restaurant_ner_recommendation")

# Test it with some new sentences
doc = nlp_ner("""Spicy Vegan Lentil Soup. This hearty vegan lentil soup is packed with spicy flavor and perfect for a cozy lunch or dinner. Lentils are simmered in a rich tomato and garlic broth, seasoned with savory cumin and a dash of spicy chili powder. It's served hot, making it an ideal choice for cooler weather. The soup is also fully vegan, ensuring that no animal products are used, making it a safe option for those following a vegan diet.  """)

for ent in doc.ents:
    print(ent.text, ent.label_)

# # Print each token
# for token in doc:
#     print(token.text)

# # Print the BiLBO tags
# for ent in doc.ents:
#     label = ent.label_
#     start_index = ent.start_char
#     end_index = ent.end_char
#     biluo_tag = f"B-{label}" if start_index == 0 else f"L-{label}" if end_index == len(ent.text) else f"I-{label}"
#     print(ent.text, biluo_tag)