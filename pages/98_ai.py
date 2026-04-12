import streamlit as st
from openai import OpenAI
import json
from datetime import datetime, timedelta
from pymongo import MongoClient
import re

# ------------------ CONFIG ------------------

# ------------------ SAFE CONFIG ------------------

if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OPENAI_API_KEY missing in Streamlit secrets")
    st.stop()

if "mongo_uri" not in st.secrets:
    st.error("❌ mongo_uri missing in Streamlit secrets")
    st.stop()

client_ai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
mongo_client = MongoClient(st.secrets["mongo_uri"])
db = mongo_client["travelaajkal"]
ai_collection = db["ai_itineraries"]

# ------------------ HELPERS ------------------

def format_places(text):
    return "-".join([x.strip().title() for x in text.split("-")])

def format_travel_line(text):
    pattern = r"([A-Za-z ]+) to ([A-Za-z ]+)"
    def repl(match):
        return f"{match.group(1)} to {match.group(2)} (Approx. 200 km | 4–5 hrs)"
    return re.sub(pattern, repl, text)

# ------------------ AI FUNCTION ------------------

def generate_daywise_ai(destinations, days, start_city, hotel_type):

    prompt = f"""
    Create a travel itinerary for India.

    Destinations: {destinations}
    Days: {days}
    Start City: {start_city}

    Rules:
    - Medium detailed professional text
    - Logical routing
    - Mention travel routes
    - Include time flow
    - Keep attractive but concise

    prompt = f"""
Create a travel itinerary for India.

Destinations: {destinations}
Days: {days}
Start City: {start_city}

Hotel Category Selected: {hotel_type}

Rules:
- Medium detailed professional text
- Logical routing
- Mention travel routes
- Include time flow
- Keep concise

Hotel Suggestion Rules:
- Suggest hotels ONLY matching selected category
- If Homestay → suggest homestays / guest houses
- If Non-AC → avoid luxury hotels
- If 3 Star → suggest mid-range hotels
- If 5 Star → suggest premium hotels
- Maintain good rating (preferably 4.0+ or best available in that category)

Output JSON:
{{
  "days":[
    {{
      "day":"Day 1",
      "plan":"paragraph",
      "stay":"city"
    }}
  ],
  "costing": {{
    "car": "₹xxxx",
    "hotel": "₹xxxx",
    "other": "₹xxxx"
  }},
  "hotel_suggestions":[
    "Hotel Name - category appropriate",
    "Hotel Name - category appropriate"
  ]
}}
    """

    response = client_ai.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return json.loads(response.choices[0].message.content)

# ------------------ UI ------------------

st.title("🤖 AI Itinerary Generator - TravelaajKal")

client_name = st.text_input("Client Name")
destinations = format_places(st.text_input("Destinations (Pune-Mumbai-Shirdi)"))
start_city = st.text_input("Start City").title()
days = st.number_input("Days", min_value=1)

start_date = st.date_input("Travel Start Date")
pax = st.number_input("Total Persons", min_value=1)

# ------------------ STAY MODE ------------------

stay_mode = st.radio("Night Stay Mode", ["AI Suggested", "Manual Selection"])

manual_stays = []
if stay_mode == "Manual Selection":
    for i in range(days):
        manual_stays.append(st.text_input(f"Day {i+1} Stay"))

# ------------------ VEHICLE ------------------

car = st.checkbox("Include Vehicle")

if car:
    car_type = st.selectbox("Vehicle Type", [
        "Sedan AC", "Sedan Non-AC",
        "Ertiga AC", "Ertiga Non-AC",
        "Innova AC", "Innova Crysta AC",
        "Tempo Traveller AC (12 Seater)",
        "Tempo Traveller Non-AC",
        "Mini Bus AC (21 Seater)",
        "Bus AC (40 Seater)",
        "Bus Non-AC"
    ])

# ------------------ HOTEL ------------------

hotel = st.checkbox("Include Hotel")

if hotel:
    hotel_type = st.selectbox("Hotel Category", [
        "Non-AC Homestay",
        "AC Homestay",
        "Standard Non-AC Hotel",
        "Standard AC Hotel",
        "3 Star AC Hotel",
        "4 Star AC Hotel",
        "5 Star Luxury Hotel"
    ])

    rooms = st.selectbox("Total Rooms", list(range(1, 31)))
    room_type = st.selectbox("Room Type", [
        "Single Occupancy", "Double Occupancy", "Triple Occupancy",
        "Quad Occupancy", "5 Sharing", "Custom"
    ])

    food_plan = st.multiselect("Food Plan", ["Breakfast", "Lunch", "Dinner"])
    food_text = "/".join(food_plan) if food_plan else "No Meals"

# ------------------ OTHER ------------------

bhasma = st.checkbox("Include Bhasmarathi")

rep_name = st.selectbox("Sales Representative", ["Arpith", "Reena", "Teena", "Kuldeep"])

cost = st.text_input("Package Cost per Person")

# ------------------ GENERATE ------------------

if st.button("Generate Final Itinerary"):

    ai_data = generate_daywise_ai(destinations, days, start_city, hotel_type if hotel else "Standard")

    ai_data = generate_daywise_ai(destinations, days, start_city, hotel_type if hotel else "Standard")

# -------- HOTEL FILTER LOGIC --------

if hotel:
    filtered_hotels = []

    for h in ai_data.get("hotel_suggestions", []):
        if "Homestay" in hotel_type and ("Homestay" in h or "Guest House" in h):
            filtered_hotels.append(h)
        elif "Non-AC" in hotel_type:
            filtered_hotels.append(h)
        elif "3 Star" in hotel_type and ("3" in h or "Hotel" in h):
            filtered_hotels.append(h)
        elif "4 Star" in hotel_type or "5 Star" in hotel_type:
            filtered_hotels.append(h)

    ai_data["hotel_suggestions"] = filtered_hotels if filtered_hotels else ai_data["hotel_suggestions"]

    text = f"Greetings from TravelAajKal,\n\n"
    text += f"*Client Name: {client_name}*\n\n"
    text += f"*Plan:-{days} Days {days-1} Nights {start_city}-{destinations} for {pax} Persons*\n\n"
    text += "*Itinerary:*\n"

    current_date = datetime.strptime(str(start_date), "%Y-%m-%d")

    for i, d in enumerate(ai_data["days"]):
        date_str = (current_date + timedelta(days=i)).strftime("%d-%b-%Y")

        plan_text = format_travel_line(d["plan"])

        text += f"\n*Day-{i+1}: {date_str}:*\n"
        text += f"{plan_text}\n"

        stay_city = d["stay"] if stay_mode == "AI Suggested" else manual_stays[i]

        if i < days-1:
            text += f"*{stay_city.title()} Night stay*\n"

    last_dest = destinations.split("-")[-1]

    text += f"\nDrop at {last_dest} Airport or Railway Station for onward journey with divine blessings.\n\n"

    # ---------------- COST ----------------

    text += f"*Package cost: ₹{cost}/Per Person*\n"

    details = []
    if car: details.append(car_type)
    if hotel: details.append(hotel_type)
    if bhasma: details.append("Bhasmarathi")

    if details:
        text += "(" + ", ".join(details) + ")\n\n"

    # ---------------- INCLUSIONS ----------------

    inc = []

    if car:
        inc += [
            f"Entire travel by {car_type}.",
            "Toll, parking, and driver bata included.",
            "Pickup and drop included."
        ]

    if hotel:
        inc.append(f"{days-1} Nights stay in hotels on {rooms} {room_type} basis with {food_text}.")

    if bhasma:
        inc += [
            f"Bhasmarathi for {pax} persons.",
            "Bhasm-Aarti pickup and drop included."
        ]

    inclusions_block = "*Inclusions:-*\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(inc)]) if inc else "*Inclusions:-*\n1. As per itinerary."

    # ---------------- EXCLUSIONS ----------------

    exclusions = "*Exclusions:-*\n" + "\n".join([
        "1. Any meals/beverages not specified – (breakfast/lunch/dinner/snacks/personal drinks).",
        "2. Entry fees for attractions/temples unless included.",
        "3. Travel insurance.",
        "4. Personal shopping/tips.",
        "5. Early check-in/late check-out if rooms unavailable.",
        "6. Natural events/roadblocks/personal itinerary changes.",
        "7. Extra sightseeing not listed."
    ])

    # ---------------- NOTES ----------------

    notes = "\n*Important Notes:-*\n" + "\n".join([
        "1. Any attractions not in itinerary will be chargeable.",
        "2. Visits subject to traffic/temple rules; closures are beyond control & non-refundable.",
        "3. Bhasm-Aarti: we provide tickets; arrival/seating beyond our control; subject to availability.",
        "4. Hotel entry as per rules; valid ID required; only married couples allowed.",
        "5. >9 yrs considered adult; <9 yrs share bed; extra bed chargeable.",
        "6. Bhasm-Aarti tickets beyond company control; if unavailable amount refunded."
    ])

    # ---------------- CANCELLATION ----------------

    cxl = """*Cancellation Policy:-*
1. 30+ days → 20% of advance deducted.
2. 15–29 days → 50% of advance deducted.
3. <15 days → No refund.
4. No refund for no-shows.
5. One-time reschedule allowed ≥15 days prior.
"""

    # ---------------- PAYMENT ----------------

    pay = f"*Payment Terms:-*\n50% advance and remaining 50% after arrival at {start_city}.\n"

    # ---------------- ACCOUNT ----------------

    acct = """For booking confirmation, please make the advance payment to the company's current account provided below.

*Company Account details:-*
Account Name: ACHALA HOLIDAYS PVT LTD
Bank: Axis Bank
Account No:
IFSC Code: UTIB0000329
MICR Code: 452211003
Branch: Ground Floor, 77, Dewas Road, Ujjain, MP 456010
"""

    # ---------------- FINAL MERGE ----------------

    text += "\n" + inclusions_block
    text += "\n\n" + exclusions
    text += notes
    text += "\n" + cxl
    text += "\n" + pay
    text += acct

    text += f"\nRegards,\n{rep_name}\nTeam TravelAajKal™️\n"

    # ---------------- SAVE ----------------

    ai_collection.insert_one({
        "client_name": client_name,
        "destinations": destinations,
        "itinerary": text,
        "created_at": datetime.now()
    })

    # ---------------- OUTPUT ----------------

    st.success("✅ Itinerary Generated & Saved")
    st.text_area("Final Itinerary", text, height=500)

    # -------- INTERNAL AI DATA --------

    st.subheader("💰 AI Costing (Internal Only)")
    st.write(ai_data.get("costing", {}))

    st.subheader("🏨 Hotel Suggestions (4.5⭐+)")
    for h in ai_data.get("hotel_suggestions", []):
        st.write(f"- {h}")
