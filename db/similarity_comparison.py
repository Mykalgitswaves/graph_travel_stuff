from db.driver import MemgraphDriver
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os


load_dotenv('.env.local')

QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv("QDRANT_URL")


ALPHA = 0.9  # embeddings similarity Important for understanding the WHY someone likes something.
BETA = 0.1   # payload/tag similarity stricter, less heuristic? 

def get_nodes_with_embeddings(user_ids:list = None):
    db = MemgraphDriver()
    
    if user_ids:
        query = """
        MATCH (t:Traveler)
        WHERE t.id IN $user_ids
        RETURN t.id as user_id, t.embedding as embedding
        """
        result = db.execute_query(query, {"user_ids": user_ids})
    else:
        query = """
            MATCH (t:Traveler)
            WHERE t.embedding IS NOT null
            RETURN t.id as user_id, t.embedding as embedding
        """

        result = db.execute_query(query)

    traveler_nodes = []

    for record in result:
        traveler_nodes.append({
            "user_id": record['user_id'],
            "embedding": record['embedding']
        })

    return traveler_nodes

# We want to provide some additional data on the users based on their relationships. 
# qDrant allows us to provide this via payload to each node. So we are decorating 
# our qdrant points with metadata about the traveler.
def get_tags_for_node(user_id:str):
    db = MemgraphDriver()
    tags = {}
    relationships = ['duration', 'destination', 'transportation_type']
    query = """
        MATCH (t:Traveler {id: $user_id})
        optional match (t)-[r:TOOK]-(trip:Trip)
        optional match (trip)-[travBy:TRAVELED_BY]-(transportation:Transportation)
        optional match (trip)-[atDest:AT_DESTINATION]-(destination:Destination)
        return t, trip, r, atDest, destination, transportation, travBy
    """
    result = db.execute_query(query, {"user_id": user_id})
    for record in result:
        tags[relationships[0]] = record['trip']['duration']
        tags[relationships[1]] = record['destination']['name']
        tags[relationships[2]] = record['transportation']['type']
    return tags

def pad_embedding(embedding, target_dim=500):
    # Convert to list of floats if needed
    if isinstance(embedding, str):
        embedding = [float(x) for x in embedding.strip('[]').split(',')]
    elif hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    
    # Ensure it's a list of floats
    embedding = [float(x) for x in embedding]
    
    # Pad with zeros if needed
    if len(embedding) < target_dim:
        padding = [0.0] * (target_dim - len(embedding))
        embedding.extend(padding)
    # Truncate if too long
    elif len(embedding) > target_dim:
        embedding = embedding[:target_dim]
    
    return embedding

def create_payload_for_point(nodes):
    points = []
    
    for node in nodes:
        try:
            payload = get_tags_for_node(user_id=node['user_id'])
            padded_embedding = pad_embedding(node['embedding'])
            point = models.PointStruct(
                id=node['user_id'],
                vector=padded_embedding,
                payload=payload,
            )

            points.append(point)
        except Exception as e:
            print(f"Error processing node{node['user_id']}: {str(e)}")
            continue
  
    return points

def calculate_payload_similarity(payload_a, payload_b):
    score = 0
    keys = ['duration', 'destination', 'transportation_type']
    
    for key in keys:
        if payload_a.get(key) == payload_b.get(key):
            score += 1

    return score / len(keys)  # Normalize to 0–1

def search_users_for_similarity_complex(client, target_embedding, reference_payload):
    # Search for similar users
    search_result = client.search(
        collection_name="travelers",
        query_vector=target_embedding,
        limit=5,
        with_payload=True  # This will include the metadata we stored
    )
    
    similar_users = []
    for hit in search_result:
        vector_score = hit.score
        payload_similarity_score = calculate_payload_similarity(hit.payload, reference_payload)
        
        final_score = ALPHA * vector_score + BETA * payload_similarity_score

        similar_users.append({
            "user_id": hit.id,
            "vector_score": vector_score,
            "payload_similarity_score": payload_similarity_score,
            "final_score": final_score,
            "payload": hit.payload
        })

    # Sort by final_score if needed
    return similar_users.sort(key=lambda x: x["final_score"], reverse=True)

def search_users_for_similarity(client, target_embedding, reference_payload):
    search_result = client.search(
        collection_name="travelers",
        query_vector=target_embedding,
        limit=5,
        with_payload=True  # This will include the metadata we stored
    )
    
    similar_users = []
    for hit in search_result:  
        similar_users.append({
            "user_id": hit.id,
            "vector_score": hit.score,
            "payload": hit.payload
        })
    breakpoint()
    # Sort by final_score if needed
    return similar_users.sort(key=lambda x: x["vector_score"], reverse=True)

if __name__ == "__main__":
    user_ids = ["b6a229b9-8b77-46fb-9612-33f5ddfaa2e5",
     "57be7d3a-8aac-46bf-ba1b-89ab280a489e",
     "4019d6ff-d5f1-4d99-a6a1-bec0644afc11",
     "92e6330d-ad13-4451-8e90-b99d91764d38",
     "87f9c9fe-cedf-4292-95b7-3bef7f822d43"
    ]
    nodes = get_nodes_with_embeddings(user_ids)
    collection_count = len(nodes)

    collection_name = "travelers"
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    collection_exists = client.collection_exists(collection_name=collection_name)

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=500, distance=models.Distance.COSINE),
        )

    payload_for_points = create_payload_for_point(nodes=nodes)
    
    client.upsert(
        collection_name=collection_name,
        points=payload_for_points
    )

    target_embedding = payload_for_points[0].vector
    
    similar_users = search_users_for_similarity(client=client, target_embedding=target_embedding)

    # client.
    # Then we make an api call to create the collection and upload the points for similarity comparison.
    # Once we have the most similar users compared, we should be able to get the most similar users in 
    # terms of previous preference and make a recommendation based on what that user does. 

