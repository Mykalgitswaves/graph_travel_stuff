from db.driver import MemgraphDriver
from api.schemas import MinimalTrip, CreateTravelerForm
import pickle
from db.mn2v import decode_embedding
from qdrant_client import QdrantClient
import requests


QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="memgraphmikey"  

class db:
    def __init__(self):
        self.driver = MemgraphDriver()

    def get_trips(self):
        query = """
            MATCH (t:Traveler)-[:TOOK]-(tr:Trip)
            MATCH (tr)-[r:AT_DESTINATION]->(d:Destination)
            MATCH (tr)-[:STAYED_IN]-(acc:Accommodation)
            RETURN t, d, tr, r, acc
        """
        result =  self.driver.execute_query(query)

        traveler_trips = []
  
        for record in result:
            traveler_trips.append(
                MinimalTrip(
                    traveler_id=record["t"].get('id'),
                    name=record["t"].get('name'),
                    location=record["d"].get('name'),
                    duration=record["tr"].get('duration'),
                    accommodation=record["acc"].get('type'),
                )
            )

        return traveler_trips
    
    def find_similar_travelers(self, user_id: str):
        # Call qdrant for similarity search.
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        results = qdrant_client.query_points(collection_name='travelers_graph', query=user_id)

        similar_travelers = [result.id for result in results.points][:10]

        query = """
            MATCH (t:Traveler)
            WHERE t.id IN $similar_travelers
            MATCH (t)-[:TOOK]->(tr:Trip)
            MATCH (tr)-[r:AT_DESTINATION]->(d:Destination)
            MATCH (tr)-[:STAYED_IN]-(acc:Accommodation)
            RETURN t, d, tr, r, acc
        """

        result = self.driver.execute_query(
            query, 
            {
                'similar_travelers': similar_travelers
            }
        )

        similar_travelers = []

        for record in result:
            similar_travelers.append(
                MinimalTrip(
                    traveler_id=record["t"].get('id'),
                    name=record["t"].get('name'),
                    location=record["d"].get('name'),
                    duration=record["tr"].get('duration'),
                    accommodation=record["acc"].get('type'),
                )
            )

        return similar_travelers
    
    def find_trip_by_id(self, traveler_id: str):
        query = """
            MATCH (t:Traveler {id: $traveler_id})-[:TOOK]->(tr:Trip)
            MATCH (tr)-[r:AT_DESTINATION]->(d:Destination)
            MATCH (tr)-[:STAYED_IN]-(acc:Accommodation)
            RETURN t, d, tr, r, acc
        """

        result = self.driver.execute_query(query, {
            'traveler_id': traveler_id
        })

        for record in result:
            return MinimalTrip(
                traveler_id=record["t"].get('id'),
                name=record["t"].get('name'),
                location=record["d"].get('name'),
                duration=record["tr"].get('duration'),
                accommodation=record["acc"].get('type'),
            )
    
    def get_trip_locations(self):
        query = """
            MATCH (d:Destination)
            RETURN d.name as name
        """

        result = self.driver.execute_query(query)

        return [record.get('name') for record in result]

    def create_traveler(self, traveler: CreateTravelerForm):
        query = """
            CREATE (t:Traveler { name: $name })
            MERGE (t)-[:HAS_PASSION]-(p:Passion { passion: $passion })
            MERGE (t)-[:WANTS_TO_TAKE]->(tr:Trip { 
                with_friends: $is_solo, 
                budget_restricted: $has_budget,
                duration: $duration,
                }
            )
            MATCH (d:Destination {name: $location})
            MERGE (tr)-[:AT_DESTINATION]-(d)
        """

        passion = traveler.passion.lower()

        pasion_label = generate_tag_for_passion(passion)

        result = self.driver.execute_query(query, {
            'name': traveler.name,
            'passion': pasion_label,
            'is_solo': traveler.is_solo,
            'has_budget': traveler.has_budget,
            'duration': traveler.duration,
            'location': traveler.location,
        })

        return result
