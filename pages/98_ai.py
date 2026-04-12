import streamlit as st
from openai import OpenAI
import json
from datetime import datetime, timedelta

# ------------------ AI SETUP ------------------
client_ai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_daywise_ai(destination, days, start_city):
    
    prompt = f"""
    Create a travel itinerary for India.

    Destination: {destination}
    Days: {days}
    Start City: {start_city}

    Rules:
    - Day wise plan
    - Practical route
    - Short and clear

    Output JSON:
    {{
      "days":[
        {{"day":"Day 1","plan":"..."}},
        {{"day":"Day 2","plan":"..."}}
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
st.title("🤖 AI Itinerary Generator (TravelaajKal)")

client_name = st.text_input("Client Name")
destination = st.text_input("Destination")
start_city = st.text_input("Start City")
days = st.number_input("Days", min_value=1)

start_date = st.date_input("Travel Start Date")

pax = st.number_input("Total Persons", min_value=1)

# Options
car = st.checkbox("Include Car")
hotel = st.checkbox("Include Hotel")
bhasma = st.checkbox("Include Bhasmarathi")

if car:
    car_type = st.selectbox("Car Type", ["Innova Crysta", "Ertiga", "Sedan"])

if hotel:
    hotel_type = st.selectbox("Hotel Category", ["3 Star", "4 Star", "5 Star"])

cost = st.text_input("Package Cost per Person")

# ------------------ GENERATE ------------------

if st.button("Generate Final Itinerary"):

    ai_data = generate_daywise_ai(destination, days, start_city)

    text = f"Greetings from TravelAajKal,\n\n"
    text += f"*Client Name: {client_name}*\n\n"

    text += f"*Plan:-{days} Days {days-1} Nights {start_city}-{destination} for {pax} Persons*\n\n"

    text += "*Itinerary:*\n"

    current_date = datetime.strptime(str(start_date), "%Y-%m-%d")

    for i, d in enumerate(ai_data["days"]):
        date_str = (current_date + timedelta(days=i)).strftime("%d-%b-%Y")

        text += f"\n*Day-{i+1}: {date_str}:*\n"
        text += f"{d['plan']}\n"

        if i < days-1:
            text += f"*{destination} Night stay*\n"

    text += f"\nDrop at {destination} Airport or Railway Station for onward journey with divine blessings.\n\n"

    # ---------------- COST ----------------
    text += f"*Package cost:{cost}/Per Person*\n"

    if car:
        text += f"({car_type} car, "

    if hotel:
        text += f"{hotel_type} Hotel with Breakfast and Dinner, "

    if bhasma:
        text += "Pandit Ji Ganesh Mantap Bhasmarathi"

    text += ")\n\n"

    # ---------------- INCLUSIONS ----------------
    text += "*Inclusions:-*\n"

    if car:
        text += f"1. Entire travel by {car_type} car.\n"
        text += "2. Toll, parking, and driver bata included.\n"
        text += "3. Pickup and drop included.\n"

    if hotel:
        text += f"4. Hotel stay in {destination} with breakfast and dinner.\n"
        text += "5. Standard check-in/out applicable.\n"

    if bhasma:
        text += f"6. Bhasmarathi for {pax} persons.\n"
        text += "7. Bhasm-Aarti pickup/drop included.\n"

    # ---------------- EXCLUSIONS ----------------
    text += "\n*Exclusions:-*\n"
    text += "1. Personal expenses.\n"
    text += "2. Travel insurance.\n"

    # ---------------- NOTES ----------------
    text += "\n*Important Notes:-*\n"

    if car:
        text += "1. Driving allowed 6AM–10PM only.\n"

    if hotel:
        text += "2. Valid ID required for hotel check-in.\n"

    if bhasma:
        text += "3. Bhasm-Aarti subject to availability.\n"

    # ---------------- FOOTER ----------------
    text += "\n*Cancellation Policy:-*\n"
    text += "Standard policy applicable.\n\n"

    text += f"*Payment Terms:-*\n"
    text += f"50% advance and rest after arrival at {start_city}.\n\n"

    text += "Regards,\nTeam TravelAajKal\n"

    # OUTPUT
    st.text_area("Final Itinerary", text, height=500)
