from models.MenuData import MenuData 
import json
import os
from typing import List
from utils.write_scrap_data_to_dataset import *

def write_scrap_data_to_json(data, prefix = ''):
    file_path = os.path.join(os.getcwd(), 'datasets/' + prefix + 'scrap.json')
    try:
        # Convert list of MenuData objects to JSON
        json_data = json.dumps([d.to_dict() for d in data], indent=4)
        # Write JSON to file
        with open(file_path, 'w') as file:
            file.write(json_data)

        # Dumps to datasets
        write_scrap_data_to_dataset(data, prefix)

        print(f"Total Scraped: {len(data)}")
        print(f"Output written at: {file_path}")
    except Exception as e:
        print("Error writing to file:", e)