import json

with open("./datasets/menus_all.json", "r") as f:
    menus_data = json.load(f)

with open("./datasets/menus_all.json", "r") as f:
    symptoms_data = json.load(f)

for text, annots in menus_data:
    examples.append(Example.from_dict(nlp.make_doc(text), annots))