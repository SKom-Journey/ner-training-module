import spacy
import json
from spacy.training.example import Example
import matplotlib.pyplot as plt

# Load the trained model
nlp = spacy.load("out/restaurant_ner_recommendation")

# Load and format test data from JSON
with open("./datasets/test_data.json", "r") as f:
    test_data = json.load(f)

# Prepare data for evaluation
examples = []
for item in test_data:
    text = item[0]  # The sentence text
    entities = {"entities": [(start, end, label) for start, end, label in item[1]["entities"]]}
    doc = nlp.make_doc(text)
    examples.append(Example.from_dict(doc, entities))

# Evaluate the model
scores = nlp.evaluate(examples)
print(scores)

# Extract metrics for each entity
entity_types = scores["ents_per_type"].keys()
precision = [scores["ents_per_type"][ent]["p"] for ent in entity_types]
recall = [scores["ents_per_type"][ent]["r"] for ent in entity_types]
f1_score = [scores["ents_per_type"][ent]["f"] for ent in entity_types]

# Plot the scores
x = range(len(entity_types))  # X-axis labels based on entity types

plt.figure(figsize=(10, 6))

# Plot precision, recall, and F1 score
plt.plot(x, precision, label="Precision", marker="o")
plt.plot(x, recall, label="Recall", marker="o")
plt.plot(x, f1_score, label="F1 Score", marker="o")

# Set x-axis labels and title
plt.xticks(x, entity_types, rotation=45)
plt.xlabel("Entity Type")
plt.ylabel("Score")
plt.title("Entity Recognition Evaluation Metrics")
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()