import db.driver as MemgraphDriver
import networkx as nx
from node2vec import Node2Vec


def create_prompt_from_node_data(user_id:str):
    db = MemgraphDriver()
    query = """
        MATCH (traveler:Traveler {id: $user_id})-[:TOOK]-(trip:Trip)
        OPTIONAL MATCH (trip)-[:AT_DESTINATION]->(destination:Destination)
        OPTIONAL MATCH (trip)-[:STAYED_IN]->(accommodation:Accommodation)
        OPTIONAL MATCH (trip)-[:TRAVELED_BY]->(transport:Transportation)
        RETURN traveler, trip, destination, accommodation, transport
    """

    params = {"user_id": user_id}
    
    result = db.execute_query(query, params)
    
    if not result:
        print('no result found for this query')
        return
    
    # We don't want to add any pii to a model. 
    # Soooo we should strip all that prior to creation.
    graph_summary = []
    
    # Track if we've already added the traveler
    traveler_node_added = False

    for record in result:
        traveler = record.get("traveler")
        trip = record.get("trip")
        destination = record.get("destination")
        accommodation = record.get("accommodation")
        transport = record.get("transport")

        if traveler and not traveler_node_added:
            graph_summary.append({
                "type": "node",
                "labels": list(traveler.labels),
                "properties": {
                    k: v for k, v in traveler.items() if k not in ['email', 'name', 'traveller_name']
                }
            })
            traveler_node_added = True

        if trip:
            trip_properties = {k: get_serializable_value(v) for k, v in trip.items()}
            graph_summary.append({
                "type": "node",
                "labels": list(trip.labels),
                "properties": trip_properties
            })
            # These relationships are implied from the query shape
            if destination:
                destination_properties = {k: get_serializable_value(v) for k, v in destination.items()}
                graph_summary.append({"type": "relationship", "relationship_type": "AT_DESTINATION"})
                graph_summary.append({
                    "type": "node",
                    "labels": list(destination.labels),
                    "properties": destination_properties
                })

            if accommodation:
                accommodation_properties = {k: get_serializable_value(v) for k, v in accommodation.items()}
                graph_summary.append({"type": "relationship", "relationship_type": "STAYED_IN"})
                graph_summary.append({
                    "type": "node",
                    "labels": list(accommodation.labels),
                    "properties": accommodation_properties
                })

            if transport:
                transport_properties = {k: get_serializable_value(v) for k, v in transport.items()}
                graph_summary.append({"type": "relationship", "relationship_type": "TRAVELED_BY"})
                graph_summary.append({
                    "type": "node",
                    "labels": list(transport.labels),
                    "properties": transport_properties
                })
  
    graph_text = json.dumps({"user_data": graph_summary}, indent=2)