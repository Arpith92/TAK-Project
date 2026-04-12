import streamlit as st
from openai import OpenAI
import json

# ✅ Initialize AI
client_ai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ AI Function
def generate_daywise_ai(destination, days, start_city, pax):
    
    prompt = f"""
    Create a travel itinerary for India.

    Destination: {destination}
    Days: {days}
    Start City: {start_city}
    Travelers: {pax}

    Rules:
    - Give only DAY WISE PLAN
    - Each day max 4–5 activities
    - Include temples, sightseeing, travel
    - Keep realistic route
    - Keep concise

    Output STRICT JSON:
    {{
        "days": [
            "Day 1: ...",
            "Day 2: ..."
        ]
    }}
    """

    try:
        response = client_ai.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {"days": [f"Error: {e}"]}


# ✅ UI
st.title("🤖 AI Itinerary Generator (Test)")

destination = st.text_input("Destination (State / City)")
days = st.number_input("Days", min_value=1, step=1)
start_city = st.text_input("Start City")
pax = st.number_input("Total Persons", min_value=1, step=1)

# ✅ Cost control (very important)
if "ai_used" not in st.session_state:
    st.session_state["ai_used"] = False

# ✅ Button
if st.button("Generate AI Plan"):

    if st.session_state["ai_used"]:
        st.warning("⚠️ Already generated once. Refresh to regenerate (cost control).")
        st.stop()

    if not destination:
        st.error("Enter destination")
        st.stop()

    with st.spinner("Generating itinerary..."):
        ai_data = generate_daywise_ai(destination, days, start_city, pax)

    st.session_state["ai_days"] = ai_data["days"]
    st.session_state["ai_used"] = True


# ✅ Display result
if "ai_days" in st.session_state:

    st.subheader("📍 Generated Plan")

    for day in st.session_state["ai_days"]:
        st.markdown(f"- {day}")
