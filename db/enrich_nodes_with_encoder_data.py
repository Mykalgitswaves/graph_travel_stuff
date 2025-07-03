from .driver import MemgraphDriver
import torch
import json
import requests
# from .queue import app
# from celery import Task
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler



def get_serializable_value(value):
    """Convert non-serializable objects to serializable formats."""
    # Check if it's a Date object from Memgraph
    if hasattr(value, '__class__') and value.__class__.__name__ == 'Date':
        # Convert Date to ISO format string
        return value.isoformat() if hasattr(value, 'isoformat') else str(value)
    return value


# Grabs a random node id and returns a chat gpt summary of it. 
# Also creates a summary of the user to be stored in a vector db instance later on.
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
    
    prompt_v2 = f"""
        You are analyzing a traveler's past trips. Based on their trip history (destinations, activities, trip duration, costs, transportation, accommodations), 
        rate them from 0 (not at all) to 10 (very much) on how well they match each of the following travel archetypes:

        - Budget Backpacker: Enjoys inexpensive travel, hostels, low-cost transportation.
        - Luxury Traveler: Prefers 5-star accommodations, fine dining, private tours.
        - Adventure Seeker: Interested in outdoor adventures, sports, thrill-seeking.
        - Cultural Explorer: Values museums, historical sites, local culture.
        - Relaxation-Focused: Seeks beach resorts, spas, wellness retreats.
        - Social Traveler: Enjoys group tours, nightlife, meeting new people.
        - Family-Oriented Traveler: Travels with family, chooses kid-friendly destinations.
        - Nature Lover: Prioritizes parks, hiking, natural scenery.
        - Urban Explorer: Loves major cities, architecture, shopping, dining.
        - Remote Retreat Enthusiast: Prefers quiet, remote, off-grid locations.
        - Surfer: Will endure traveling long distances for the ultimate wave

        ----
        Return the result as JSON, like:

        
        budget_backpacker: 4,
        luxury_traveler: 7,
        adventure_seeker: 8,
        cultural_explorer: 6,
        relaxation_focused: 5,
        social_traveler: 2,
        family_oriented_traveler: 3,
        nature_lover: 7,
        urban_explorer: 9,
        remote_retreat_enthusiast: 1
        
        ----

        Consider both travel behaviors and preferences when assigning the scores.

        ---
        {graph_text}
        ---
    """

    prompt_v3 = f"""
        Create text that  adheres to the structure listed below to summarize the travelers behavior based on graph data provided.
        --- format ---
        Traveler is a [MALE / FEMALE], aged [age] who has traveled to [X number of destinations] across [Y continents]. 
        They show the following ratings for travel archetypes:

        - Budget Backpacker: Enjoys inexpensive travel, hostels, low-cost transportation. [RATING]
        - Luxury Traveler: Prefers 5-star accommodations, fine dining, private tours. [RATING]
        - Adventure Seeker: Interested in outdoor adventures, sports, thrill-seeking. [RATING]
        - Cultural Explorer: Values museums, historical sites, local culture. [RATING]
        - Relaxation-Focused: Seeks beach resorts, spas, wellness retreats. [RATING]
        - Social Traveler: Enjoys group tours, nightlife, meeting new people. [RATING]
        - Family-Oriented Traveler: Travels with family, chooses kid-friendly destinations. [RATING]
        - Nature Lover: Prioritizes parks, hiking, natural scenery. [RATING]
        - Urban Explorer: Loves major cities, architecture, shopping, dining. [RATING]
        - Remote Retreat Enthusiast: Prefers quiet, remote, off-grid locations. [RATING]
        - Surfer: Will endure traveling long distances for the ultimate wave. [RATING]

        Finally return a paragraph in this format:[ short summary (3-4 sentences) of traveler's patterns based on their predominant archetypes with evidence from the graph data ]

        --- endformat ---

        --- graph data ---
            { graph_text }
        --- end graph data ---
    """

    prompt_v4 = f"""
        You are an expert travel behavior analyst. Based solely on the graph data provided, create a structured profile of this traveler that quantifies their travel preferences and archetypes.

        INSTRUCTIONS:
        1. Analyze the provided graph data carefully
        2. Rate each travel archetype on a scale of 0-10 (where 0=not at all, 10=extremely strong match)
        3. Base your ratings ONLY on evidence from the provided data
        4. If insufficient evidence exists for a category, assign it a rating of 0-2
        5. Include only factual observations in your summary

        REQUIRED OUTPUT FORMAT:
        Traveler is a [GENDER], aged [AGE] who has traveled to [NUMBER] destinations across [NUMBER] continents. 

        They show the following ratings for travel archetypes (0-10 scale):
        - Budget Backpacker: Enjoys inexpensive travel, hostels, low-cost transportation. [RATING] - [brief evidence if >2]
        - Luxury Traveler: Prefers 5-star accommodations, fine dining, private tours. [RATING] - [brief evidence if >2]
        - Adventure Seeker: Interested in outdoor adventures, sports, thrill-seeking. [RATING] - [brief evidence if >2]
        - Cultural Explorer: Values museums, historical sites, local culture. [RATING] - [brief evidence if >2]
        - Relaxation-Focused: Seeks beach resorts, spas, wellness retreats. [RATING] - [brief evidence if >2]
        - Social Traveler: Enjoys group tours, nightlife, meeting new people. [RATING] - [brief evidence if >2]
        - Family-Oriented Traveler: Travels with family, chooses kid-friendly destinations. [RATING] - [brief evidence if >2]
        - Nature Lover: Prioritizes parks, hiking, natural scenery. [RATING] - [brief evidence if >2]
        - Urban Explorer: Loves major cities, architecture, shopping, dining. [RATING] - [brief evidence if >2]
        - Remote Retreat Enthusiast: Prefers quiet, remote, off-grid locations. [RATING] - [brief evidence if >2]
        - Digital Nomad: Loves new places with social scenes and nature with access to good internet. [RATING] - [brief evidence if >2]

        Key travel patterns:
        [2-3 sentences identifying core patterns in their travel behavior, destinations, accommodations, transportation choices, trip duration, and spending patterns]

        GRAPH DATA:
        {graph_text}
    """

    return prompt_v3


def create_summary_of_prompt(prompt):
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    })
    res.raise_for_status()
    print(res.json()["response"])
    return res.json()["response"]

# store an embedding on the user node. 
def store_summary_on_user_node(user_id:str, summary:any):
    res = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "all-minilm",
        "prompt": summary
    })
    res.raise_for_status()
    embedding = res.json()["embedding"]
    print(embedding)
    if embedding:
        db = MemgraphDriver()
        query = """
        MATCH (t:Traveler {id: $user_id})
        SET t.embedding = $embedding
        RETURN t
        """
        user = db.execute_query(query, {"user_id":user_id , "embedding": embedding})
        print(user)

        return {
            "user_id": user_id,
            "embedding": embedding
        }



def extract_summary(text):
    # Find the last occurrence of "Summary:" followed by text
    # fallback: try to grab everything after last "Summary:"
    return text.rsplit("Summary:", 1)[-1].strip().rsplit(".\n", 1)[0]

if __name__ == "__main__":
    # user_ids = ["f68fe24e-91a2-42a5-9fa2-c0ca66178a16"] John smith 
    # user_ids = ["0fa8d56e-ee1f-4b39-a0fe-d2cc8c82b56b"] # Jane doe
    # user_ids = ["bc3b84c2-bd09-4098-b94d-74f1dd71a514"] # David lee
    user_ids = [
    "b6a229b9-8b77-46fb-9612-33f5ddfaa2e5",
     "57be7d3a-8aac-46bf-ba1b-89ab280a489e",
     "4019d6ff-d5f1-4d99-a6a1-bec0644afc11",
     "92e6330d-ad13-4451-8e90-b99d91764d38",
     "87f9c9fe-cedf-4292-95b7-3bef7f822d43",
    "87f9c9fe-cedf-4292-95b7-3bef7f822d43",
    "f68fe24e-91a2-42a5-9fa2-c0ca66178a16",
    "0fa8d56e-ee1f-4b39-a0fe-d2cc8c82b56b",
    "bc3b84c2-bd09-4098-b94d-74f1dd71a514",
    "4af7b153-5d2c-48b7-9069-dd95b210da8e",
    "a4aef068-d0a1-4ecd-98f0-978c75739de8",
    ]

    graph_text_by_id = {}
    
    for id in user_ids:
        prompt = create_prompt_from_node_data(user_id=id)
        graph_text_by_id[id] = prompt
        summary = create_summary_of_prompt(graph_text_by_id[id])
        store_summary_on_user_node(user_id=id, summary=summary)

    print(graph_text_by_id.items())
