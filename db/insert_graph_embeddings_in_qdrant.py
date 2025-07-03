from db.driver import MemgraphDriver
from db.mn2v import decode_embedding
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# load_dotenv('.env.local')

# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="memgraphmikey"  

def insert_graph_embeddings_in_qdrant():
    driver = MemgraphDriver()
    query = """
        MATCH (t:Traveler)-[:TOOK]-(tr:Trip)
        MATCH (tr)-[:AT_DESTINATION]-(d:Destination)
        MATCH (tr)-[:TRAVELED_BY]-(trans:Transportation)
        RETURN t.id as user_id, 
            t.b64_graph_embeddings as embedding, 
            d.name as destination,
            trans.type as transportation,
            tr.duration as duration
    """
    result = driver.execute_query(query)
    points = []
    for record in result:
        user_id = record['user_id']
        embedding = record['embedding']
        destination = record['destination']
        transportation = record['transportation']
        duration = int(record['duration'])

        decoded_embedding = decode_embedding(embedding, 'float32', '(192,)')

        points.append(PointStruct(
            id=user_id,
            vector=decoded_embedding,
            payload={
                "user_id": user_id,
                "duration": duration,
                "destination": destination,
                "transportation": transportation,
            }
        ))
    

    breakpoint()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    if client.collection_exists(collection_name="travelers_graph"):
        client.delete_collection(collection_name="travelers_graph")
        client.create_collection(collection_name="travelers_graph", vectors_config=VectorParams(size=192, distance=Distance.COSINE))

    client.upsert(collection_name="travelers_graph", points=points)

if __name__ == "__main__":
    insert_graph_embeddings_in_qdrant()