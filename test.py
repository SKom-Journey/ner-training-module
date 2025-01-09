import spacy
import os
import json 

# Load the trained NER model
nlp_ner = spacy.load("out/restaurant_ner_recommendation")

# prefix = 'menus_'

# with open("./datasets/" + prefix + "all.json", "r") as f:
#     data = json.load(f)
#     d = []
#     for da in data:
#         d.append(da[0])
#     with open("./datasets/raw_" + prefix + "all.json", "w") as f:
#         json.dump(d, f, indent=4)

while True:
    os.system('cls')

    doc = nlp_ner(input('Text: ').replace("?", ""))

    print([(w.text, w.pos_) for w in doc])

    # if len(doc.ents) == 0:
    #     print('No entities found')

    for ent in doc.ents:
        print(ent.text, ent.label_)

    input('\nPress enter... ')