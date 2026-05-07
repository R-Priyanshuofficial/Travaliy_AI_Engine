from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

import requests
import json
import os
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Travaily AI Planner API",
    version="2.0.0"
)


class TripRequest(BaseModel):
    source: str
    destination: str
    start_date: str
    end_date: str
    trip_type: str = Field(description="solo / couple / group")
    group_people: Optional[int] = 0
    couple_pairs: Optional[int] = 0
    trip_mood: str
    food_pref: str
    transport_pref: str
    budget: str


class RePlanRequest(BaseModel):
    old_itinerary: dict
    reason: str
    affected_date: str
    affected_day: Optional[int] = None
    current_time: Optional[str] = None
    current_location: Optional[str] = None

    replan_type: Literal[
        "this_day_only",
        "from_this_day_onward",
        "full_trip"
    ] = "from_this_day_onward"

    keep_same_places: bool = True
    include_missed_places: bool = True
    suggest_nearby_places: bool = True
    avoid_long_travel: bool = True
    special_request: Optional[str] = None


def validate_date(date_text: str):
    try:
        date_obj = datetime.strptime(date_text, "%Y-%m-%d")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if date_obj < today:
        raise HTTPException(
            status_code=400,
            detail="Past date not allowed. Please select today or future date."
        )

    return date_obj


def calculate_travelers(trip_type: str, group_people: int = 0, couple_pairs: int = 0):
    trip_type = trip_type.lower()

    if trip_type == "solo":
        return 1

    if trip_type == "couple":
        if couple_pairs <= 0:
            raise HTTPException(
                status_code=400,
                detail="Couple pairs must be greater than 0"
            )
        return couple_pairs * 2

    if trip_type == "group":
        if group_people <= 0:
            raise HTTPException(
                status_code=400,
                detail="Group people must be greater than 0"
            )
        return group_people

    raise HTTPException(
        status_code=400,
        detail="Trip type must be solo/couple/group"
    )


def api_get(url: str, params: dict):
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def get_coordinates(city: str):
    if not GEOAPIFY_API_KEY:
        return None

    data = api_get(
        "https://api.geoapify.com/v1/geocode/search",
        {
            "text": city,
            "limit": 1,
            "apiKey": GEOAPIFY_API_KEY
        }
    )

    features = data.get("features", []) if data else []

    if not features:
        return None

    props = features[0].get("properties", {})

    return {
        "city": city,
        "latitude": props.get("lat"),
        "longitude": props.get("lon"),
        "address": props.get("formatted")
    }


def get_current_weather(destination: str):
    if not OPENWEATHER_API_KEY:
        return {"error": "OpenWeather API key missing"}

    data = api_get(
        "https://api.openweathermap.org/data/2.5/weather",
        {
            "q": destination,
            "units": "metric",
            "appid": OPENWEATHER_API_KEY
        }
    )

    if not data:
        return {"error": "Weather unavailable"}

    return {
        "temperature": f"{data.get('main', {}).get('temp')}°C",
        "condition": data.get("weather", [{}])[0].get("main"),
        "description": data.get("weather", [{}])[0].get("description"),
        "humidity": f"{data.get('main', {}).get('humidity')}%",
        "wind_speed": f"{data.get('wind', {}).get('speed')} m/s"
    }


def get_weather_forecast(destination: str):
    if not OPENWEATHER_API_KEY:
        return []

    data = api_get(
        "https://api.openweathermap.org/data/2.5/forecast",
        {
            "q": destination,
            "units": "metric",
            "appid": OPENWEATHER_API_KEY
        }
    )

    forecast_list = data.get("list", []) if data else []

    return [
        {
            "date_time": item.get("dt_txt"),
            "temperature": f"{item.get('main', {}).get('temp')}°C",
            "condition": item.get("weather", [{}])[0].get("main"),
            "description": item.get("weather", [{}])[0].get("description"),
            "humidity": f"{item.get('main', {}).get('humidity')}%",
            "wind_speed": f"{item.get('wind', {}).get('speed')} m/s"
        }
        for item in forecast_list[:16]
    ]


def get_air_pollution(latitude, longitude):
    if not OPENWEATHER_API_KEY or latitude is None or longitude is None:
        return None

    data = api_get(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        {
            "lat": latitude,
            "lon": longitude,
            "appid": OPENWEATHER_API_KEY
        }
    )

    pollution = data.get("list", []) if data else []

    if not pollution:
        return None

    item = pollution[0]

    return {
        "aqi": item.get("main", {}).get("aqi"),
        "components": item.get("components", {})
    }


def get_places(latitude, longitude):
    if not GEOAPIFY_API_KEY or latitude is None or longitude is None:
        return []

    data = api_get(
        "https://api.geoapify.com/v2/places",
        {
            "categories": "tourism.sights,tourism.attraction,catering.restaurant,catering.cafe,accommodation.hotel,entertainment,museum",
            "filter": f"circle:{longitude},{latitude},10000",
            "bias": f"proximity:{longitude},{latitude}",
            "limit": 20,
            "apiKey": GEOAPIFY_API_KEY
        }
    )

    features = data.get("features", []) if data else []
    places = []

    for feature in features:
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [None, None])

        places.append({
            "id": props.get("place_id"),
            "name": props.get("name"),
            "category": props.get("categories", []),
            "address": props.get("formatted"),
            "latitude": coords[1],
            "longitude": coords[0]
        })

    return places


def get_place_details(place_id: str):
    if not GEOAPIFY_API_KEY or not place_id:
        return None

    data = api_get(
        "https://api.geoapify.com/v2/place-details",
        {
            "id": place_id,
            "apiKey": GEOAPIFY_API_KEY
        }
    )

    features = data.get("features", []) if data else []

    if not features:
        return None

    props = features[0].get("properties", {})

    return {
        "opening_hours": props.get("opening_hours"),
        "website": props.get("website"),
        "phone": props.get("phone"),
        "contact": props.get("contact"),
        "description": props.get("description")
    }


def get_route(from_lat, from_lon, to_lat, to_lon, mode: str = "drive"):
    if not GEOAPIFY_API_KEY or None in [from_lat, from_lon, to_lat, to_lon]:
        return None

    data = api_get(
        "https://api.geoapify.com/v1/routing",
        {
            "waypoints": f"{from_lat},{from_lon}|{to_lat},{to_lon}",
            "mode": mode,
            "apiKey": GEOAPIFY_API_KEY
        }
    )

    features = data.get("features", []) if data else []

    if not features:
        return None

    props = features[0].get("properties", {})

    return {
        "distance": props.get("distance"),
        "distance_km": round(props.get("distance", 0) / 1000, 2),
        "time_seconds": props.get("time"),
        "time_minutes": round(props.get("time", 0) / 60)
    }


def enrich_places_with_details_and_route(source_coordinates, places):
    enriched_places = []

    for place in places[:10]:
        details = get_place_details(place.get("id"))
        route = get_route(
            source_coordinates.get("latitude"),
            source_coordinates.get("longitude"),
            place.get("latitude"),
            place.get("longitude")
        ) if source_coordinates else None

        enriched_places.append({
            **place,
            "details": details,
            "route": route
        })

    return enriched_places


def create_prompt(data: dict):
    return f"""
You are Travaily AI Planner.

Create HIGH QUALITY MOBILE APP FRIENDLY day-wise travel itinerary JSON.

Trip Details:
Source: {data['source']}
Destination: {data['destination']}
Start Date: {data['start_date']}
End Date: {data['end_date']}
Trip Type: {data['trip_type']}
Travelers: {data['total_travelers']}
Mood: {data['trip_mood']}
Food Preference: {data['food_pref']}
Transport Preference: {data['transport_pref']}
Budget: {data['budget']}

Real API Data:
Current Weather:
{json.dumps(data['current_weather'], indent=2)}

Weather Forecast:
{json.dumps(data['weather_forecast'], indent=2)}

Air Pollution:
{json.dumps(data['air_pollution'], indent=2)}

Source Coordinates:
{json.dumps(data['source_coordinates'], indent=2)}

Destination Coordinates:
{json.dumps(data['destination_coordinates'], indent=2)}

Real Places With Details And Route:
{json.dumps(data['places'], indent=2)}

Return ONLY valid JSON.
Do not add markdown.
Do not add explanation.
Do not add text before or after JSON.

IMPORTANT:
Keep current JSON response format exactly same.
Do not add real_weather, real_places_count, source_coordinates, destination_coordinates, or api_data in final JSON.
Use API data only to improve itinerary accuracy.

Required JSON structure:

{{
  "trip_summary": {{
    "source": "Ahmedabad",
    "destination": "Manali",
    "start_date": "2025-05-20",
    "end_date": "2025-05-23",
    "total_days": 4,
    "trip_type": "couple",
    "travelers": 2,
    "budget": "medium",
    "mood": "adventure",
    "food_preference": "veg",
    "transport_preference": "cab",
    "estimated_total_cost": "₹25000",
    "best_time_to_visit": "March to June",
    "short_description": "A scenic mountain trip with local sightseeing, food, shopping and adventure."
  }},
  "days": [
    {{
      "day": 1,
      "title": "Arrival & Local Exploration",
      "date": "2025-05-20",
      "weather": "15°C Cloudy",
      "estimated_day_cost": "₹3500",
      "day_notes": "Start slow and explore nearby places.",
      "timeline": [
        {{
          "time": "08:00 AM",
          "title": "Mall Road",
          "description": "Explore local market and enjoy breakfast",
          "place": "Mall Road Manali",
          "latitude": 32.2432,
          "longitude": 77.1892,
          "open_time": "08:00 AM",
          "close_time": "10:00 PM",
          "open_close_display": "Open • 8 AM - 10 PM",
          "estimated_cost": "₹500",
          "transport_mode": "Walk",
          "travel_time": "10 mins",
          "distance": "1 km",
          "image_search_keyword": "Mall Road Manali market",
          "category": "shopping",
          "tips": "Visit early morning to avoid crowd"
        }}
      ]
    }}
  ],
  "budget_breakdown": {{
    "hotel": "₹8000",
    "food": "₹5000",
    "transport": "₹7000",
    "activities": "₹5000",
    "total": "₹25000"
  }},
  "travel_tips": [
    "Carry warm clothes",
    "Keep cash for local shops",
    "Start sightseeing early"
  ],
  "packing_list": [
    "Jacket",
    "Shoes",
    "Power bank",
    "ID proof"
  ]
}}

Rules:
- Generate itinerary for every date between start_date and end_date.
- Each day must have minimum 5 to 8 timeline items.
- Every timeline item must include open_time and close_time.
- If place does not have official timing, use practical timing like "Open 24 hours" or "Flexible".
- Use realistic opening and closing timings.
- Use logical travel order based on location and route data.
- Include breakfast, lunch, dinner, hotel/rest where suitable.
- Include latitude and longitude for each place.
- Include image_search_keyword for app image loading.
- Keep response mobile UI friendly.
- Categories must be one of:
  sightseeing, food, hotel, shopping, adventure, travel, rest
- open_close_display must be short like:
  "Open • 8 AM - 10 PM"
  "Open • 9 AM - 6 PM"
  "Flexible"
- Return ONLY JSON.
"""


def create_replan_prompt(data: dict):
    return f"""
You are Travaily AI Re-planner.

Re-plan the user's existing itinerary based on the new situation.

Return ONLY valid JSON.
Do not add markdown.
Do not add explanation.
Do not add text before or after JSON.

IMPORTANT:
Keep the SAME JSON response structure as the old itinerary.
Do not remove required fields.
Do not add extra root fields.

Re-plan Type:
{data["replan_type"]}

Rules:
- If replan_type is "this_day_only", only update the affected day.
- If replan_type is "from_this_day_onward", update affected day and all next days.
- If replan_type is "full_trip", re-plan full itinerary.
- Keep already completed/past activities unchanged if possible.
- Move missed important places to next suitable days if possible.
- Keep trip practical and user-friendly.
- Use realistic opening and closing timings.
- Keep travel order logical.
- Avoid unnecessary long travel when avoid_long_travel is true.
- Include breakfast, lunch, dinner, hotel/rest where suitable.
- Each day must have minimum 5 to 8 timeline items.
- Every timeline item must include:
  time, title, description, place, latitude, longitude,
  open_time, close_time, open_close_display,
  estimated_cost, transport_mode, travel_time,
  distance, image_search_keyword, category, tips.
- Categories must be one of:
  sightseeing, food, hotel, shopping, adventure, travel, rest.
- open_close_display must be short like:
  "Open • 8 AM - 10 PM"
  "Open • 9 AM - 6 PM"
  "Flexible"

User Situation:
Reason: {data["reason"]}
Affected Date: {data["affected_date"]}
Affected Day: {data.get("affected_day")}
Current Time: {data.get("current_time")}
Current Location: {data.get("current_location")}
Special Request: {data.get("special_request")}

Preferences:
Keep same places if possible: {data["keep_same_places"]}
Include missed places in next days: {data["include_missed_places"]}
Suggest alternative nearby places: {data["suggest_nearby_places"]}
Avoid long travel: {data["avoid_long_travel"]}

Old Itinerary:
{json.dumps(data["old_itinerary"], indent=2)}

Return updated itinerary JSON only.
"""


def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid JSON response"
        )


def generate_ai_json(prompt: str, temperature: float = 0.7):
    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json"
                )
            )

            if not response or not response.text:
                raise Exception("Empty response from AI")

            return safe_parse_json(response.text)

        except errors.ServerError as e:
            last_error = str(e)
            time.sleep(3 * (attempt + 1))

        except HTTPException:
            raise

        except Exception as e:
            last_error = str(e)
            time.sleep(2)

    raise HTTPException(
        status_code=503,
        detail=f"AI service unavailable: {last_error}"
    )


def generate_itinerary(data: dict):
    return generate_ai_json(create_prompt(data), temperature=0.7)


def generate_replanned_itinerary(data: dict):
    return generate_ai_json(create_replan_prompt(data), temperature=0.5)


@app.get("/")
def home():
    return {
        "message": "Travaily AI Planner API Running 🚀"
    }


@app.post("/generate-itinerary")
def generate_trip(request: TripRequest):
    try:
        start = validate_date(request.start_date)
        end = validate_date(request.end_date)

        if end < start:
            raise HTTPException(
                status_code=400,
                detail="End date cannot be before start date"
            )

        total_travelers = calculate_travelers(
            request.trip_type,
            request.group_people,
            request.couple_pairs
        )

        source_coordinates = get_coordinates(request.source)
        destination_coordinates = get_coordinates(request.destination)

        current_weather = get_current_weather(request.destination)
        weather_forecast = get_weather_forecast(request.destination)

        air_pollution = None
        places = []

        if destination_coordinates:
            air_pollution = get_air_pollution(
                destination_coordinates.get("latitude"),
                destination_coordinates.get("longitude")
            )

            places = get_places(
                destination_coordinates.get("latitude"),
                destination_coordinates.get("longitude")
            )

        places = enrich_places_with_details_and_route(
            source_coordinates,
            places
        )

        data = {
            "source": request.source,
            "destination": request.destination,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "trip_type": request.trip_type,
            "group_people": request.group_people,
            "couple_pairs": request.couple_pairs,
            "total_travelers": total_travelers,
            "trip_mood": request.trip_mood,
            "food_pref": request.food_pref,
            "transport_pref": request.transport_pref,
            "budget": request.budget,
            "source_coordinates": source_coordinates,
            "destination_coordinates": destination_coordinates,
            "current_weather": current_weather,
            "weather_forecast": weather_forecast,
            "air_pollution": air_pollution,
            "places": places
        }

        result = generate_itinerary(data)

        return {
            "success": True,
            "data": result
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@app.post("/replan-itinerary")
def replan_trip(request: RePlanRequest):
    try:
        validate_date(request.affected_date)

        if not request.old_itinerary:
            raise HTTPException(
                status_code=400,
                detail="Old itinerary is required"
            )

        if not request.reason.strip():
            raise HTTPException(
                status_code=400,
                detail="Re-plan reason is required"
            )

        data = {
            "old_itinerary": request.old_itinerary,
            "reason": request.reason,
            "affected_date": request.affected_date,
            "affected_day": request.affected_day,
            "current_time": request.current_time,
            "current_location": request.current_location,
            "replan_type": request.replan_type,
            "keep_same_places": request.keep_same_places,
            "include_missed_places": request.include_missed_places,
            "suggest_nearby_places": request.suggest_nearby_places,
            "avoid_long_travel": request.avoid_long_travel,
            "special_request": request.special_request
        }

        result = generate_replanned_itinerary(data)

        return {
            "success": True,
            "message": "Itinerary re-planned successfully",
            "data": result
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )