from __future__ import annotations

from datetime import datetime, date
from typing import Optional
import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# =========================
# Page
# =========================
st.set_page_config(page_title="Direct Car Bookings", layout="wide")
st.title("🚖 Direct Car Bookings (Cash / Employee Collection)")

# =========================
# Mongo Connection
# =========================
CAND_KEYS = ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]

def _find_uri() -> Optional[str]:
    for k in CAND_KEYS:
        try:
            v = st.secrets.get(k)
        except Exception:
            v = None
        if v: return v
    for k in CAND_KEYS:
        v = os.getenv(k)
        if v: return v
    return None

@st.cache_resource
def _get_client() -> MongoClient:
    uri = _find_uri()
    if not uri:
        st.error("Mongo URI not configured in secrets.")
        st.stop()
    client = MongoClient(uri, appName="TAK_DirectCars", tz_aware=True)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        st.error(f"Mongo connect fail: {e}")
        st.stop()
    return client

db = _get_client()["TAK_DB"]
col_cars = db["direct_car_bookings"]
col_split = db["expense_splitwise"]

# =========================
# User Login (parity with other pages)
# =========================
def load_users() -> dict:
    users = st.secrets.get("users", {})
    return users if isinstance(users, dict) else {}

user = st.session_state.get("user")
if not user:
    users_map = load_users()
    st.markdown("### 🔐 Login required")
    u = st.selectbox("User", list(users_map.keys()))
    p = st.text_input("PIN", type="password")
    if st.button("Login"):
        if str(users_map.get(u)) == str(p):
            st.session_state["user"] = u
            st.rerun()
        else:
            st.error("Invalid PIN")
    st.stop()

# =========================
# Form
# =========================
with st.form("direct_car_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        when = st.date_input("Date", value=date.today())
    with c2:
        client = st.text_input("Client Name (optional)")
    with c3:
        trip = st.text_input("Trip Plan (optional)")

    c4, c5, c6 = st.columns(3)
    with c4:
        amount = st.number_input("Amount (₹)", min_value=0, step=500)
    with c5:
        car_type = st.selectbox("Car Type", ["Sedan", "Ertiga"])
    with c6:
        recv_in = st.radio("Received In", ["Company Account", "Personal Account"], horizontal=True)

    emp = None
    if recv_in == "Personal Account":
        emp = st.selectbox("Payment received by", list(load_users().keys()), index=0)

    notes = st.text_area("Notes", placeholder="Any remarks…")

    submitted = st.form_submit_button("Save booking")

if submitted:
    if amount <= 0:
        st.error("Amount must be > 0")
    else:
        doc = {
            "date": datetime.combine(when, datetime.min.time()),
            "client_name": client,
            "trip_plan": trip,
            "amount": int(amount),
            "car_type": car_type,
            "received_in": recv_in,
            "employee": emp if recv_in == "Personal Account" else "",
            "notes": notes,
            "created_by": user,
            "created_at": datetime.utcnow(),
        }
        col_cars.insert_one(doc)

        # If Personal → push to Splitwise as settlement
        if recv_in == "Personal Account" and emp:
            col_split.insert_one({
                "kind": "settlement",
                "created_at": datetime.utcnow(),
                "created_by": user,
                "date": datetime.combine(when, datetime.min.time()),
                "employee": emp,
                "amount": int(amount),
                "ref": f"Direct Car ({car_type})",
                "notes": f"Direct booking for {client or 'N/A'}",
            })

        st.success("Booking saved successfully ✅")
        st.rerun()

# =========================
# Show recent
# =========================
st.subheader("📜 Recent Direct Car Bookings")
docs = list(col_cars.find().sort("date", -1).limit(20))
if not docs:
    st.info("No bookings yet.")
else:
    df = pd.DataFrame(docs)
    if "_id" in df: df["_id"] = df["_id"].astype(str)
    st.dataframe(df[["date","client_name","trip_plan","amount","car_type","received_in","employee","notes"]],
                 use_container_width=True, hide_index=True)
