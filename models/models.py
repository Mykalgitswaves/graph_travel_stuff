from pydantic import BaseModel, Field, field_validator
from random import randint
from datetime import datetime
from typing import List, Tuple
from datetime import date


class TripDataRow(BaseModel):
    trip_id: int = Field(default_factory=lambda: randint(0, 1000000), alias="Trip ID")
    destination: str = Field(alias="Destination")
    start_date: str = Field(alias="Start date")
    end_date: str = Field(alias="End date")
    duration_days: int = Field(alias="Duration (days)")
    traveler_name: str = Field(alias="Traveler name")
    traveler_age: int = Field(alias="Traveler age")
    traveler_gender: str = Field(alias="Traveler gender")
    traveler_nationality: str = Field(alias="Traveler nationality")
    accommodation_type: str | None = Field(alias="Accommodation type")
    accommodation_cost: float | None = Field(alias="Accommodation cost")
    transportation_type: str | None = Field(alias="Transportation type")
    transportation_cost: float | None = Field(alias="Transportation cost")

    @field_validator('start_date', 'end_date', mode='before')
    def format_dates(cls, v):
        if v:
            try:
                # Parse the date and format it back to YYYY-MM-DD
                return datetime.strptime(v, "%m/%d/%Y").strftime("%Y-%m-%d")
            except ValueError:
                # If parsing fails, return the original value
                return v
        return v
    
    @field_validator('accommodation_cost', 'transportation_cost', mode='before')
    def format_currencies(cls, v):
        if v:
            try:
                if isinstance(v, float):
                    return int(v)
                elif isinstance(v, str):
                    if "," in v:
                        return int(v.replace(',', '').strip())
                    else: 
                        return int(v.strip())
            except ValueError:
                return v
        return v
    
    @field_validator('duration_days', 'traveler_age', mode='before')
    def format_duration(cls, v):
        if v:
            try:
                if isinstance(v, str):
                    return int(v)
            except ValueError:
                return v
        return v

    class Config:
        validate_by_name = True


class Traveler(BaseModel):
    name: str
    age: int
    gender: str
    nationality: str

class Destination(BaseModel):
    name: str
    coordinates: Tuple[float, float] = (None, None) #lat, long

class Accommodation(BaseModel):
    type: str

class Transportation(BaseModel):
    type: str

class Trip(BaseModel):
    id: str
    start_date: date
    end_date: date
    duration_days: int
    accommodation_cost: float
    transportation_cost: float

    # relationships
    destination: Destination
    accommodation: Accommodation
    transportation: Transportation
    travelers: List[Traveler] = []

class Activity(BaseModel): 
  id: str
  name: str
  description: str
  location: str
  start_coordinates: dict # for map-based queries
  duration_minutes: int # key for time-based filters
  type: str
  activity_level: str
  cost_min: int
  cost_max: int
  tags: list[str] # e.g., ["sunset", "nature", "wildlife", "family-friendly"]
  seasonality: list[str] # ["summer", "fall"], or a date range
  group_size_min: int
  group_size_max: int
  languages_offered: list[str] # useful for international users
  accessibility_features: list[str] # e.g., ["wheelchair_accessible"]
  vendor_rating: int # average rating from reviews
  cancellation_policy: str
  booking_volume: int # could be a rough proxy for popularity
  image_url: str
