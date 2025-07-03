from db.driver import MemgraphDriver
from models.models import Activity
import csv
import os
from typing import List, Tuple
import requests
import json

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'datasets',
    'activities2.csv',
)

# 'google_maps_attractions.csv'

def insert_activity_into_db(**kwargs):
    db = MemgraphDriver()
    
    # Filter out None values and create the SET clause dynamically
    valid_params = {k: v for k, v in kwargs.items() if v is not None}
    
    # Build the SET clause dynamically based on available parameters
    set_clauses = []
    for key in valid_params.keys():
        if key != 'id':  # Skip id as it's used in MERGE
            set_clauses.append(f"a.{key} = ${key}")
    
    set_clause = ",\n            ".join(set_clauses)
    
    query = f"""
        MERGE (a:Activity {{id: $id}})
        SET {set_clause}
        RETURN a
    """
    
    result = db.execute_query(query, valid_params)
    return result


def load_activities_from_csv(file_path: str):
   with open(dataset_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            # Convert empty strings to None and ensure all values are properly typed
            processed_row = {}
            for k, v in row.items():
                if v == '':
                    processed_row[k] = None
                else:
                    # Try to convert to appropriate type
                    try:
                        if k in ['cost_min', 'cost_max', 'duration_minutes', 'group_size_min', 
                               'group_size_max', 'vendor_rating', 'booking_volume']:
                            processed_row[k] = int(v)
                        elif k in ['start_coordinates']:
                            # Assuming coordinates are stored as string in format "lat,lng"
                            lat, lng = map(float, v.split(','))
                            processed_row[k] = {'lat': lat, 'lng': lng}
                        elif k in ['tags', 'seasonality', 'languages_offered', 'accessibility_features']:
                            # Assuming these are comma-separated lists
                            processed_row[k] = [item.strip() for item in v.split(',')]
                        else:
                            processed_row[k] = v
                    except (ValueError, TypeError):
                        processed_row[k] = v
            rows.append(processed_row)
        return rows

if __name__ == "__main__":
    data = load_activities_from_csv(dataset_path)
    for row in data:
        node = insert_activity_into_db(**row)
        breakpoint()
        # activity = infer_properties_for_row(row)