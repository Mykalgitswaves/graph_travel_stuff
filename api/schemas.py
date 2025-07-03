from pydantic import BaseModel


class MinimalTrip(BaseModel):
    traveler_id: str
    name: str
    duration: int
    location: str
    accommodation: str

class CreateTravelerForm(BaseModel):
    favorite_trip: str
    location: str
    duration: int
    budget: int
    friends: int
    nextTrip: str
    passion: str
    decisions: str


query = """
    MATCH (t:Traveler)
    MATCH (t)-[:FAVORITE_TRIP]->(tr:Trip)
    MATCH (t)-[:WANTS_TO_TAKE]->(tr:Trip)
    MATCH (t)-[:HAS_PASSION]->(p:Passion)
    RETURN t, tr, nt, p, d
"""