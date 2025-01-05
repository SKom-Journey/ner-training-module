import re

def annotate_sentence(entities, sentence, keywords_to_annotated):
    annotated_entities = []

    for label_data in entities:
        label = label_data["label"]
        keywords = label_data["keywords"]

        for keyword in keywords:
            match = re.search(r'\b' + re.escape(keyword) + r'\b', sentence)
            if match:
                if keyword in keywords_to_annotated:
                    keywords_to_annotated.remove(keyword)

                start_idx = match.start()
                end_idx = match.end()
                annotated_entities.append((start_idx, end_idx, label))
    
    return {"entities": annotated_entities}