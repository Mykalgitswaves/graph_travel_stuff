
from .driver import MemgraphDriver
import csv
from models.models import (
    TripDataRow,
)
# 
# nodes
# 

# 1) Raw CSV row

# 
# Queries
# 
def load_trip_node_into_db(db, row: TripDataRow):
    query = """
        // 1) Create/merge nodes
        MERGE (trav:Traveler { id: randomUUID(), name: $traveler_name })
        ON CREATE SET
            trav.age = toInteger($traveler_age),
            trav.gender = $traveler_gender,
            trav.nationality = $traveler_nationality
        MERGE (trip:Trip { id: $trip_id })
        ON CREATE SET
            trip.startDate = date($trip_start_date),
            trip.endDate   = date($trip_start_date),
            trip.duration  = toInteger($trip_duration_days),
            trip.accommodationCost   = toFloat($trip_accommodation_cost),
            trip.transportationCost  = toFloat($trip_transporation_cost)
        MERGE (dest:Destination { name: $destination })
        MERGE (acc:Accommodation { type: $accomodation_type })
        MERGE (trans:Transportation { type: $transportation_type })

        // 2) Create relationships
        MERGE (trav)-[:TOOK]->(trip)
        MERGE (trip)-[:AT_DESTINATION]->(dest)
        MERGE (trip)-[:STAYED_IN]->(acc)
        MERGE (trip)-[:TRAVELED_BY]->(trans);
    """

    parameters = {
        "traveler_name": row.traveler_name,
        "traveler_age": row.traveler_age,
        "traveler_gender": row.traveler_gender,
        "traveler_nationality": row.traveler_nationality,
        "trip_id": row.trip_id,
        "trip_start_date": row.start_date,
        "trip_end_date": row.end_date,
        "trip_duration_days": row.duration_days,
        "trip_accommodation_cost": row.accommodation_cost,
        "trip_transporation_cost": row.transportation_cost,
        "destination": row.destination,
        "accomodation_type": row.accommodation_type,
        "transportation_type": row.transportation_type
    }

    result = db.execute_query(query, parameters)
    return result

if __name__ == "__main__":
    # load dataset from local.
    import os

    # Get the absolute path to the current file
    current_file_path = os.path.abspath(__file__)
    print("Current file path:", current_file_path)

    # Get the absolute path to the dataset
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), 
        'datasets', 
        'travel_details_dataset.csv'
    )

    print("Dataset path:", dataset_path)

    db = MemgraphDriver()
    with open(dataset_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        index = 0
        for row in reader:
            # Validate no empty values
            lvals = list(dict.values(row))
            if any(el == ''  for el in lvals):
                breakpoint()
                print(f'skip past row {index}')
                index += 1
                continue
            else:
                trip_data = TripDataRow(**row)  
                nodes = load_trip_node_into_db(db=db, row=trip_data)
                print(f"{nodes} @ {index}")
                index += 1
            