from models.MenuData import MenuData 
import json
import os
from typing import List

def write_scrap_data_to_dataset(data: List[MenuData]):
    file_path = os.path.join(os.getcwd(), 'datasets/datasets.json')
    try:
        # Convert list of MenuData objects to JSON
        datasets = []
        for d in data:
            datasets.extend(d.to_dataset())

        json_data = json.dumps(datasets, indent=4)
        
        # Write JSON to file
        with open(file_path, 'w') as file:
            file.write(json_data)
    except Exception as e:
        print("Error writing to file:", e)
