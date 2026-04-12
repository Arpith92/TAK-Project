import streamlit as st
from openai import OpenAI
import json
from datetime import datetime, timedelta
from pymongo import MongoClient

# ------------------ CONFIG ------------------

client_ai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
mongo_client = MongoClient(st.secrets["mongo_uri"])
db = mongo_client["travelaajkal"]
ai_collection = db["ai_itineraries"]

# ------------------ AI FUNCTION ------------------

def generate_daywise_ai(destinations, days, start_city):

    prompt = f"""
    Create a travel itinerary for India.

    Destinations: {destinations}
    Days: {days}
    Start City: {start_city}

    Rules:
    - Optimize route logically
    - Cover all destinations
    - Assign correct stay city
    - Max 4–5 activities per day
    - Keep concise

    Output JSON:
    {{
      "days":[
        {{
          "day":"Day 1",
          "plan":"...",
          "stay":"city name"
        }}
      ]
    }}
    """

    response = client_ai.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return json.loads(response.choices[0].message.content)

# ------------------ UI ------------------

st.title("🤖 AI Itinerary Generator (TravelaajKal Advanced)")

client_name = st.text_input("Client Name")
destinations = st.text_input("Destinations (e.g. Pune-Mumbai-Shirdi)")
start_city = st.text_input("Start City")
days = st.number_input("Days", min_value=1)

start_date = st.date_input("Travel Start Date")
pax = st.number_input("Total Persons", min_value=1)

# Options
car = st.checkbox("Include Car")
hotel = st.checkbox("Include Hotel")
bhasma = st.checkbox("Include Bhasmarathi")

car_type = ""
hotel_type = ""

if car:
    car_type = st.selectbox("Car Type", ["Innova Crysta", "Ertiga", "Sedan"])

if hotel:
    hotel_type = st.selectbox("Hotel Category", ["3 Star", "4 Star", "5 Star"])

cost = st.text_input("Package Cost per Person")

# ------------------ GENERATE ------------------

if st.button("Generate Final Itinerary"):

    ai_data = generate_daywise_ai(destinations, days, start_city)

    itinerary_text = f"Greetings from TravelAajKal,\n\n"
    itinerary_text += f"*Client Name: {client_name}*\n\n"
    itinerary_text += f"*Plan:-{days} Days {days-1} Nights {start_city}-{destinations} for {pax} Persons*\n\n"
    itinerary_text += "*Itinerary:*\n"

    current_date = datetime.strptime(str(start_date), "%Y-%m-%d")

    for i, d in enumerate(ai_data["days"]):
        date_str = (current_date + timedelta(days=i)).strftime("%d-%b-%Y")

        itinerary_text += f"\n*Day-{i+1}: {date_str}:*\n"
        itinerary_text += f"{d['plan']}\n"

        if i < days-1:
            itinerary_text += f"*{d['stay']} Night stay*\n"

    last_dest = destinations.split("-")[-1]

    itinerary_text += f"\nDrop at {last_dest} Airport or Railway Station for onward journey with divine blessings.\n\n"

    # ---------------- PACKAGE COST ----------------

    itinerary_text += f"*Package cost: ₹{cost}/Per Person*\n"

    details_bits = []
    if car:
        details_bits.append(car_type + " car")
    if hotel:
        details_bits.append(hotel_type + " Hotel with Breakfast and Dinner")
    if bhasma:
        details_bits.append("Pandit Ji Ganesh Mantap Bhasmarathi")

    if details_bits:
        itinerary_text += "(" + ", ".join(details_bits) + ")\n\n"

    # ---------------- INCLUSIONS ----------------

    inc = []

    if car:
        inc += [
            f"Entire travel by {car_type} car.",
            "Toll, parking, and driver bata included.",
            "Pickup and drop included."
        ]

    if hotel:
        inc += [
            "Hotel stay on double sharing basis with breakfast and dinner.",
            "Standard check-in at 12 PM and check-out at 10 AM."
        ]

    if bhasma:
        inc += [
            f"Bhasmarathi for {pax} persons.",
            "Bhasm-Aarti pickup and drop included."
        ]

    inclusions_block = "*Inclusions:-*\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(inc)]) if inc else "*Inclusions:-*\n1. As per itinerary."

    # ---------------- EXCLUSIONS ----------------

    exclusions = "*Exclusions:-*\n" + "\n".join([
        "1. Any meals/beverages not specified (breakfast/lunch/dinner/snacks/personal drinks).",
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
        "3. Bhasm-Aarti: we provide tickets; arrival/seating beyond our control; cost at actuals.",
        "4. Hotel entry as per rules; valid ID required; only married couples allowed.",
        "5. >9 yrs considered adult; <9 yrs share bed; extra bed chargeable.",
        "6. Bhasm-Aarti tickets are beyond company control. If unavailable, amount will be refunded."
    ])

    # ---------------- CANCELLATION ----------------

    cxl = """*Cancellation Policy:-*
1. 30+ days → 20% of advance deducted.
2. 15–29 days → 50% of advance deducted.
3. <15 days → No refund on advance.
4. No refund for no-shows/early departures.
5. One-time reschedule allowed ≥15 days prior.
"""

    # ---------------- PAYMENT ----------------

    pay = f"*Payment Terms:-*\n50% advance and remaining 50% after arrival at {start_city}.\n"

    # ---------------- ACCOUNT ----------------

    acct = """
For booking confirmation, please make the advance payment to the company's current account provided below.

*Company Account details:-*
Account Name: ACHALA HOLIDAYS PVT LTD
Bank: Axis Bank
Account No:
IFSC Code: UTIB0000329
MICR Code: 452211003
Branch: Ground Floor, 77, Dewas Road, Ujjain, MP 456010

Found the price a little tall? Don’t worry — we are flexible like a yoga instructor!
Share your comfortable budget with our representative, and we’ll adjust accordingly.

Regards,
Team TravelAajKal™️ • Reg. Achala Holidays Pvt Ltd
Visit: www.travelaajkal.com • IG: @travelaaj_kal
DPIIT-recognized Startup • TravelAajKal® is a registered trademark.
"""

    # ---------------- FINAL MERGE ----------------

    itinerary_text += "\n" + inclusions_block
    itinerary_text += "\n\n" + exclusions
    itinerary_text += notes
    itinerary_text += "\n" + cxl
    itinerary_text += "\n" + pay
    itinerary_text += acct

    # ---------------- SAVE TO MONGO ----------------

    ai_collection.insert_one({
        "client_name": client_name,
        "destinations": destinations,
        "days": days,
        "pax": pax,
        "itinerary": itinerary_text,
        "created_at": datetime.now()
    })

    # ---------------- OUTPUT ----------------

    st.success("✅ Itinerary Generated & Saved")
    st.text_area("Final Itinerary", itinerary_text, height=500)
