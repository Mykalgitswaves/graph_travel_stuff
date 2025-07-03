from fastapi import FastAPI, Request, HTTPException
from api.crud import db
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from api.schemas import MinimalTrip, CreateTravelerForm


app = FastAPI()
database = db()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/api/v1/get_trips")
async def get_trips(request: Request) -> list[MinimalTrip]:
    traveler_trips = database.get_trips()
    
    if not traveler_trips:
        return []
    
    return traveler_trips

@app.get("/api/v1/find_similar_travelers/{user_id}")
async def find_similar_travelers(user_id: str) -> list[MinimalTrip]:
    similar_travelers = database.find_similar_travelers(user_id)

    if not similar_travelers:
        return []
    
    return similar_travelers

@app.get("/api/v1/find_trip_by_id/{trip_id}")
async def find_trip_by_id(trip_id: str) -> MinimalTrip:
    trip = database.find_trip_by_id(trip_id)

    if not trip:
        return []
    
    return trip

@app.post("/api/v1/create_traveler")
async def create_traveler(request: Request) -> MinimalTrip:
    data = await request.json()
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    traveler = CreateTravelerForm(**data)
    
    db.create_traveler(traveler)

    if not traveler:
        return []
    
@app.get("/api/v1/get_trip_locations")
async def get_trip_locations() -> list[str]:
    locations = database.get_trip_locations()

    if not locations:
        return []
    return locations