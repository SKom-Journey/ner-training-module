import json

symptoms_data = []

with open("./datasets/menus_train_data.json", "r") as f:
    menus_data = json.load(f)
with open("./datasets/menus_val_data.json", "r") as f:
    menus_data.extend(json.load(f))

with open("./datasets/symptoms_train_data.json", "r") as f:
    symptoms_data.extend(json.load(f)) 
with open("./datasets/symptoms_val_data.json", "r") as f:
    symptoms_data.extend(json.load(f))


result = {}

print(len(menus_data))
print(len(symptoms_data))

for text, annots in menus_data:
    detected_list = []
    for a in annots:
        for i in annots[a]:
            detected_list.append(i)

            if result.get(i[2]) is None:
                result[i[2]] = 0

            if i[2] not in detected_list:
                result[i[2]] += 1
        # result[a][0] = 1

print(result)