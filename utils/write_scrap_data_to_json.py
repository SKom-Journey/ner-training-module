from models.MenuData import MenuData 
import json
import os
from typing import List

def write_scrap_data_to_json(data: List[MenuData]):
    file_path = os.path.join(os.getcwd(), 'datasets/scrap.json')
    try:
        # Convert list of MenuData objects to JSON
        json_data = json.dumps([d.to_dict() for d in data], indent=4)

        # Write JSON to file
        with open(file_path, 'w') as file:
            file.write(json_data)

        print(f"Total Scraped: {len(data)}")
        print(f"Output written at: {file_path}")
    except Exception as e:
        print("Error writing to file:", e)