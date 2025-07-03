from .driver import MemgraphDriver
import aiohttp
import asyncio
import time

class DestinationEnricher:
    def __init__(self, db):
        self.db = db
        self.endpoint = "https://search.mapzen.com/v1/search"

    async def fetch_coordinates(self, session, location):
        params = {"text": location, "size": 1}
        try:
            async with session.get(self.endpoint, params=params) as response:
                data = await response.json()
                coords = data["features"][0]["geometry"]["coordinates"]
                lon, lat = coords
                return location, lat, lon
        except Exception as e:
            print(f"[Error] Location: {location} – {e}")
            return None

    async def enrich(self):
        query = "MATCH (d:Destination) RETURN d.name AS location"
        locations = [record["location"] for record in self.db.execute_query(query)]

        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_coordinates(session, loc) for loc in locations]
            results = await asyncio.gather(*tasks)

        # Filter out failed lookups
        updates = [r for r in results if r is not None]

        for location, lat, lon in updates:
            update_query = """
                MATCH (d:Destination {name: $name})
                SET d.latitude = $lat, d.longitude = $lon
            """
            self.db.execute_query(update_query, {"name": location, "lat": lat, "lon": lon})




async def fetch_coordinates(session, location):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "yourappname/1.0"
    }
    
    try:

        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 429:  # Rate limit error
                print(f"Rate limited. Retrying in 1 second...")
                await asyncio.sleep(2)  # Delay before retrying
                return await fetch_coordinates(session, location)

            if response.status == 200:  # Success
                try:
                    data = await response.json()
                    if data:
                        lat = data[0].get("lat")
                        lon = data[0].get("lon")
                        return location, lat, lon
                except Exception as e:
                    print(f"Error parsing JSON for location {location}: {e}")
            else:
                print(f"Failed to fetch data for {location}. HTTP Status: {response.status}")
    except Exception as e:
        print(f"Error fetching coordinates for {location}: {e}")

    return location, None, None  # Return None if failed

async def enrich_nodes_with_coordinate():
    db = MemgraphDriver()
    query = """
        MATCH (d:Destination)
        RETURN d.name as location
    """
    result = db.execute_query(query)

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_coordinates(session, record["location"])
            for record in result
        ]
        coordinates = await asyncio.gather(*tasks)

    for location, lat, lon in coordinates:
        print(f"{location} => lat: {lat}, lon: {lon}")


async def main():
    db = MemgraphDriver()
    enricher = DestinationEnricher(db)
    await enricher.enrich()

# Run it
if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(enrich_nodes_with_coordinate())