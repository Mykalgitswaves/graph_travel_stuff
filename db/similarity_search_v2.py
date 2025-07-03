from db.driver import MemgraphDriver
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, NamedVector
from dotenv import load_dotenv
import os


load_dotenv('.env.local')

QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv("QDRANT_URL")

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

def create_payload_for_point(nodes):
    points = []
    
    for node in nodes:
        breakpoint()
        try:
            payload = get_tags_for_node(user_id=node['user_id'])

            point = PointStruct(
                id=node['user_id'],
                vector={
                    "semantic_embedding": node['semantic_embedding'],
                    "feature_embedding": node['feature_embedding'],
                },
                payload=payload,
            )
            points.append(point)
        except Exception as e:
            print(f"Error processing node{node['user_id']}: {str(e)}")
            continue
  
    return points

def get_nodes_with_embeddings(user_ids:list = None):
    db = MemgraphDriver()
    if user_ids:
        query = """
        MATCH (t:Traveler)
        WHERE t.id IN $user_ids
        RETURN t.id as user_id, t.embedding as semantic_embedding, t.feature_embeddings as feature_embedding
        """
        result = db.execute_query(query, {"user_ids": user_ids})
    else:
        query = """
            MATCH (t:Traveler)
            WHERE t.embedding IS NOT null
            RETURN t.id as user_id, t.embedding as semantic_embedding, t.feature_embeddings as feature_embedding
        """

        result = db.execute_query(query)

    traveler_nodes = []

    for record in result:
        traveler_nodes.append({
            "user_id": record['user_id'],
            "semantic_embedding": record['semantic_embedding'],
            "feature_embedding": record['feature_embedding']
        })

    return traveler_nodes

def similarity_search_v2(client, feature_embedding, semantic_embedding):
    search_request = models.SearchRequest(
        queries=[
            NamedVector(
                name="semantic_embedding",
                vector=semantic_embedding,
                score_threshold=0.5
            ),
            NamedVector(
                name="feature_embedding",
                vector=feature_embedding,
                score_threshold=0.5
            )
        ],
        limit=10,
        with_payload=True
    )
    search_result = client.search(collection_name="travelers", search_request=search_request)
    similar_users = []
    for hit in search_result:
        print(hit)
        breakpoint()

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
    
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "feature_embedding": VectorParams(
                size=6,
                distance=models.Distance.COSINE,
            ),
            "semantic_embedding": VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        }
    )

    payload_for_points = create_payload_for_point(nodes=nodes)
    client.upsert(
        collection_name=collection_name,
        points=payload_for_points
    )

    target_semantic_embedding = payload_for_points[0].vector['semantic_embedding']
    target_feature_embedding = payload_for_points[0].vector['feature_embedding']
    
    similar_users = similarity_search_v2(
        client=client, 
        feature_embedding=target_feature_embedding,
        semantic_embedding=target_semantic_embedding
    )
    breakpoint()

    