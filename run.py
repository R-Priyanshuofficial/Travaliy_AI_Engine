from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import os

# =========================================
# LOAD ENV
# =========================================

load_dotenv()

# =========================================
# GEMINI CONFIG
# =========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


client = genai.Client(api_key=GEMINI_API_KEY)

# =========================================
# FASTAPI INIT
# =========================================

app = FastAPI(
    title="Travaily AI Planner API",
    version="1.0.0"
)

# =========================================
# REQUEST MODEL
# =========================================

class TripRequest(BaseModel):
    source: str
    destination: str

    start_date: str
    end_date: str

    trip_type: str = Field(
        description="solo / couple / group"
    )

    group_people: Optional[int] = 0
    couple_pairs: Optional[int] = 0

    trip_mood: str
    food_pref: str
    transport_pref: str
    budget: str


# =========================================
# VALIDATION
# =========================================

def validate_date(date_text):
    try:
        return datetime.strptime(date_text, "%Y-%m-%d")

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Date must be YYYY-MM-DD"
        )


def calculate_travelers(
    trip_type,
    group_people=0,
    couple_pairs=0
):
    trip_type = trip_type.lower()

    if trip_type == "solo":
        return 1

    elif trip_type == "couple":
        return couple_pairs * 2

    elif trip_type == "group":
        return group_people

    else:
        raise HTTPException(
            status_code=400,
            detail="Trip type must be solo/couple/group"
        )


# =========================================
# PROMPT BUILDER
# =========================================

def create_prompt(data):

    return f"""
You are Travaily AI Planner.

Create HIGH QUALITY day-wise travel itinerary JSON.

=========================================
USER DETAILS
=========================================

Source City: {data['source']}
Destination: {data['destination']}
Start Date: {data['start_date']}
End Date: {data['end_date']}
Trip Type: {data['trip_type']}
Total Travelers: {data['total_travelers']}
Trip Mood: {data['trip_mood']}
Food Preference: {data['food_pref']}
Transport Preference: {data['transport_pref']}
Budget: {data['budget']}

=========================================
VERY IMPORTANT RULES
=========================================

1. Create MOBILE APP FRIENDLY itinerary JSON.

2. Give DAY-WISE timeline format.

3. Every activity MUST include:
   - exact timing
   - opening time
   - closing time
   - whether place is currently open during selected slot
   - visit duration
   - travel time
   - distance

4. VERY IMPORTANT:
   Many Indian temples/gardens/museums close in afternoon.

5. CHECK REALISTIC OPEN/CLOSE TIME LOGIC.

6. If place closed during suggested slot:
   - set "is_open_during_visit": false
   - suggest alternative time

7. Keep response PERFECT for mobile timeline UI.

8. Add image keywords for frontend image search.

9. Add weather notes for each day.

10. Add estimated activity cost.

11. Add best photography timing if possible.

12. Add crowd level:
   low / medium / high

13. Seasonal place validation:
   If destination closed in selected dates:
   - set is_trip_possible false
   - mention reason
   - suggest best months

14. Return ONLY VALID JSON.

=========================================
JSON FORMAT
=========================================

{{
  "is_trip_possible": true,

  "trip_summary": "",

  "weather_summary": "",

  "daily_itinerary": [
    {{
      "day": 1,
      "date": "",
      "day_title": "",
      "weather_note": "",

      "timeline": [
        {{
          "time": "08:00 AM",

          "activity_title": "",

          "place_name": "",

          "activity_type": "",

          "description": "",

          "opening_time": "",

          "closing_time": "",

          "is_open_during_visit": true,

          "alternative_time_if_closed": "",

          "visit_duration": "",

          "travel_time_from_previous_place": "",

          "distance_from_previous_place": "",

          "estimated_activity_cost": "",

          "best_photo_time": "",

          "crowd_level": "",

          "food_recommendation": "",

          "tips": "",

          "image_keywords": [
            ""
          ]
        }}
      ]
    }}
  ],

  "cost_estimation": {{
    "currency": "INR",

    "flight_per_person": "",

    "train_per_person": "",

    "trip_per_person_without_transport": "",

    "total_with_flight_per_person": "",

    "total_with_train_per_person": "",

    "full_trip_cost_all_people": ""
  }},

  "hotel_recommendation": {{
    "best_area": "",

    "hotel_type": "",

    "avg_price_per_night": ""
  }},

  "safety": {{
    "score": "",

    "note": ""
  }},

  "permit_required": {{
    "required": false,

    "details": ""
  }}
}}
"""


# =========================================
# GEMINI GENERATE
# =========================================

import time
from google.genai import errors


def generate_itinerary(data):

    prompt = create_prompt(data)

    retries = 3

    for attempt in range(retries):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            return json.loads(response.text)

        except errors.ServerError as e:

            print("Gemini overloaded:", e)

            if attempt < retries - 1:

                time.sleep(5)

            else:

                return {
                    "success": False,
                    "message": "Gemini servers are busy. Try again later."
                }

        except Exception as e:

            return {
                "success": False,
                "message": str(e)
            }

# =========================================
# API ROUTES
# =========================================

@app.get("/")
def home():
    return {
        "message": "Travaily AI Planner API Running 🚀"
    }


@app.post("/generate-itinerary")
def generate_trip(request: TripRequest):

    # -----------------------------
    # DATE VALIDATION
    # -----------------------------

    validate_date(request.start_date)
    validate_date(request.end_date)

    # -----------------------------
    # TOTAL TRAVELERS
    # -----------------------------

    total_travelers = calculate_travelers(
        request.trip_type,
        request.group_people,
        request.couple_pairs
    )

    # -----------------------------
    # PREPARE DATA
    # -----------------------------

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
        "budget": request.budget
    }

    # -----------------------------
    # GENERATE ITINERARY
    # -----------------------------

    result = generate_itinerary(data)

    return {
        "success": True,
        "data": result
    }
