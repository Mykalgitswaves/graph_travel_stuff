from db.driver import MemgraphDriver
import torch
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import base64
import numpy as np
from db.mn2v import decode_embedding
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models

load_dotenv('.env.local')

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def get_traveler_embedding_nodes():
    db = MemgraphDriver()
    query = """
        MATCH (t:Traveler)
        WHERE t.b64_graph_embeddings IS NOT NULL
        RETURN t.id as traveler_id, t.b64_graph_embeddings as graph_embeddings
    """
    result = db.execute_query(query, {})
    nodes = []
    for record in result:
        # The shape (1,64) should be used since MetaPath2Vec was configured with:
        # embedding_dim=64 in the model instantiation
        nodes.append(
            models.PointStruct(
                id=record['traveler_id'],
                vector=decode_embedding(record["graph_embeddings"], "float32", "(1, 192)").flatten(),
                payload={}
            )
        )

    return nodes

# if __name__ == "__main__":
#     points = get_traveler_embedding_nodes()
#     qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

#     # qdrant_client.create_collection(
#     #     collection_name="travelers_graph",
#     #     vectors_config=models.VectorParams(size=192, distance=models.Distance.COSINE),
#     # )
#     breakpoint()

#     qdrant_client.upsert(
#         collection_name="travelers_graph",
#         points=points,
#     )
    
#     print("Done")

def create_activity_destination_relationships():
    db = MemgraphDriver()
    query = """
        MATCH (a:Activity), (d:Destination)
        WHERE 
            // Exact match
            a.location = d.name
            OR 
            // Case-insensitive match
            toLower(a.location) = toLower(d.name)
            OR
            // Match without country suffix
            toLower(a.location) = toLower(split(d.name, ',')[0])
            OR
            // Match country names
            toLower(a.location) = toLower(split(d.name, ',')[1])
            OR
            // Match without spaces and special characters
            replace(replace(toLower(a.location), ' ', ''), '-', '') = replace(replace(toLower(d.name), ' ', ''), '-', '')
        MERGE (a)-[:AT_LOCATION]->(d)
    """
    db.execute_query(query, {})


# Starting with a node_id, find the most similar nodes using cosine similarity from our earlier embeddings, 
# then find the most similar from those
def recommend_itineraries(node_id:str, collection_name:str):
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    results = qdrant_client.query_points(collection_name=collection_name, query=node_id)

    # Only use the top 3 results in order to avoid too many options.
    ids = [point.id for point in results.points][:3]

    db = MemgraphDriver()
    
    query = """
        MATCH (trav:Traveler)
        WHERE trav.id in $ids
        OPTIONAL MATCH (trav)-[:TOOK]->(trip:Trip)
        OPTIONAL MATCH (trip)-[:AT_DESTINATION]->(dest:Destination)
        OPTIONAL MATCH (trip)-[:STAYED_IN]->(acc:Accommodation)
        OPTIONAL MATCH (trip)-[:TRAVELED_BY]->(trans:Transportation)
        OPTIONAL MATCH (dest)-[:AT_LOCATION]-(activity:Activity)
        WITH trav, trip, dest, acc, trans, activity
        ORDER BY trip.startDate
        RETURN 
            trav.id as traveler_id,
            collect(DISTINCT {
                trip_id: trip.id,
                start_date: trip.startDate,
                end_date: trip.endDate,
                duration: trip.duration,
                destination: dest.name,
                accommodation: {
                    type: acc.type,
                    cost: trip.accommodationCost
                },
                transportation: {
                    type: trans.type,
                    cost: trip.transportationCost
                },
                activities: activity
            }) as itinerary
    """
    result = db.execute_query(query, {"ids": ids})
    itineraries = []

    for record in result:
        itineraries.append(record)

    return itineraries

if __name__ == "__main__":
    # create_activity_destination_relationships()
    itineraries = recommend_itineraries('3c8d5c4c-9168-485c-bd26-87a12849ab92', 'travelers_graph')
    breakpoint()