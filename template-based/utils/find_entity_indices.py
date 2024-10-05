# Function to get the correct start and end character positions for the entity
def find_entity_indices(sentence, entity):
    words = sentence.split()  # Split sentence into words
    current_pos = 0
    for word in words:
        # Check if the word matches the entity
        if word == entity:
            start_idx = current_pos
            end_idx = start_idx + len(entity)
            return start_idx, end_idx
        # Add the length of the word and a space (for the next word's position)
        current_pos += len(word) + 1  # 1 is for the space
    return -1, -1  # Return invalid indices if not found