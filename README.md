# Install All Dependencies
- `pip install -r requirements.txt`

# How to Generate the Model
- `python build_json.py` (build train, val and test set)
- `python train.py` (train the model)
- The model will generated to `out/restaurant_ner_recommendation`

# Evaluate
- `python evaluate.py` (with graph, precision, f-1 and recall)

# Testing
- `python test.py` (input your own text to see how the model behave)

# Scraping
- `python scrap.py`
- Result will generated to `out/scrap.json`

# Entity Table

| **Entity**          | **Description**                                | **Keywords**                                                                   |
|----------------------|-----------------------------------------------|-------------------------------------------------------------------------------|
| **ITEM_CATEGORY**    | Differentiate between foods and drinks        | dish, food, drink, meal                                                       |
| **MEAL_TYPE**        | Differentiate between meal levels             | treat, snack, breakfast, lunch, dinner                                        |
| **FLAVOR_TYPE**      | Differentiate between flavors                 | spicy, sweet, savory, sour, bitter                                            |
| **DIET_TYPE**        | Differentiate between diets                   | vegan, vegetarian                                                             |
| **ALLERGY_TYPE**     | Differentiate between allergies               | gluten, dairy, soy, seafood, egg, nut                                         |
| **TEMPERATURE**      | Differentiate between food and drink temperature | cold, hot, warm, normal                                                       |
| **ALLERGY_SYMPTOM**  | Differentiate allergy symptoms                | itch, rash, nausea, vomit, diarrhea, sneeze, stomachache, cough, difficulty swallowing, abdominal pain |
