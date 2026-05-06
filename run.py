from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
import json
import os
import time

# =========================================
# LOAD ENV
# =========================================

load_dotenv()

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
# CUSTOM EXCEPTION
# =========================================

class AIServiceError(Exception):
    pass

# =========================================
# REQUEST MODEL
# =========================================

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


# =========================================
# DATE VALIDATION (PAST + FORMAT)
# =========================================

def validate_date(date_text):

    try:
        date_obj = datetime.strptime(date_text, "%Y-%m-%d")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

    today = datetime.now()
    today = datetime(today.year, today.month, today.day)

    if date_obj < today:
        raise HTTPException(
            status_code=400,
            detail="Past date not allowed. Please select today or future date."
        )

    return date_obj


# =========================================
# TRAVELER CALCULATION
# =========================================

def calculate_travelers(trip_type, group_people=0, couple_pairs=0):

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

Source: {data['source']}
Destination: {data['destination']}
Start: {data['start_date']}
End: {data['end_date']}
Trip Type: {data['trip_type']}
Travelers: {data['total_travelers']}
Mood: {data['trip_mood']}
Food: {data['food_pref']}
Transport: {data['transport_pref']}
Budget: {data['budget']}

Return ONLY valid JSON.
Include timings, cost, weather, travel details.
"""


# =========================================
# SAFE JSON PARSER
# =========================================

def safe_parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid JSON response"
        )


# =========================================
# GEMINI ENGINE (RETRY + ERROR HANDLING)
# =========================================

def generate_itinerary(data):

    prompt = create_prompt(data)

    retries = 3
    last_error = None

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

            if not response or not response.text:
                raise AIServiceError("Empty response from AI")

            return safe_parse_json(response.text)

        except errors.ServerError as e:
            last_error = str(e)
            time.sleep(3 * (attempt + 1))
            continue

        except Exception as e:
            last_error = str(e)
            time.sleep(2)
            continue

    raise HTTPException(
        status_code=503,
        detail=f"AI service unavailable: {last_error}"
    )


# =========================================
# API ROUTES
# =========================================

@app.get("/")
def home():
    return {"message": "Travaily AI Planner API Running 🚀"}


@app.post("/generate-itinerary")
def generate_trip(request: TripRequest):

    try:

        # -------------------------
        # DATE VALIDATION
        # -------------------------
        start = validate_date(request.start_date)
        end = validate_date(request.end_date)

        if end < start:
            raise HTTPException(
                status_code=400,
                detail="End date cannot be before start date"
            )

        # -------------------------
        # TRAVELER COUNT
        # -------------------------
        total_travelers = calculate_travelers(
            request.trip_type,
            request.group_people,
            request.couple_pairs
        )

        # -------------------------
        # DATA PREP
        # -------------------------
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

        # -------------------------
        # GENERATE AI ITINERARY
        # -------------------------
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