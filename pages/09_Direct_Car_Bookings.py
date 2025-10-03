# pages/09_Direct_Car_Bookings.py
from __future__ import annotations
from datetime import datetime, date
from typing import Optional
import os, pandas as pd, streamlit as st
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

st.set_page_config(page_title="Direct Car Bookings", layout="wide")
st.title("🚖 Direct Car Bookings (Cash / Employee Collection)")

# -----------------------
# Mongo Connection
# -----------------------
def _find_uri():
    for k in ["mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"]:
        try: v = st.secrets.get(k)
        except Exception: v = None
        if v: return v
    for k in ["mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"]:
        v = os.getenv(k)
        if v: return v
    return None

@st.cache_resource
def _get_client():
    uri = _find_uri()
    if not uri:
        return None
    client = MongoClient(uri, appName="TAK_DirectCars", tz_aware=True)
    try: client.admin.command("ping")
    except ServerSelectionTimeoutError: return None
    return client

# -----------------------
# Users & Login
# -----------------------
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users: return users
    try:
        try: import tomllib
        except: import tomli as tomllib
        with open(".streamlit/secrets.toml","rb") as f:
            return tomllib.load(f).get("users", {})
    except: return {}

def _login():
    if st.session_state.get("user"):
        with st.sidebar:
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out"):
                st.session_state.pop("user"); st.experimental_set_query_params()
                st.stop()
        return st.session_state["user"]

    users_map = load_users()
    if not users_map:
        st.sidebar.error("⚠️ No login config in secrets."); st.stop()

    with st.sidebar:
        st.markdown("### 🔐 Login")
        name = st.selectbox("User", list(users_map.keys()))
        pin = st.text_input("PIN", type="password")
        if st.button("Sign in"):
            if str(users_map.get(name,"")).strip() == str(pin).strip():
                st.session_state["user"] = name
                st.experimental_set_query_params()
                st.experimental_rerun()
            else: st.error("Invalid PIN"); st.stop()
    return None

user = _login()
if not user: st.stop()

# -----------------------
# DB only after login
# -----------------------
client = _get_client()
if not client:
    st.error("❌ Could not connect to MongoDB.")
    st.stop()

db = client["TAK_DB"]
col_cars = db["direct_car_bookings"]
col_split = db["expense_splitwise"]

def all_employees() -> list[str]:
    return sorted(load_users().keys())

# -----------------------
# Booking Form
# -----------------------
st.subheader("➕ Add Direct Car Booking")

with st.form("car_form", clear_on_submit=True):
    c1,c2,c3 = st.columns(3)
    with c1: when = st.date_input("Date", value=date.today())
    with c2: client = st.text_input("Client Name (optional)")
    with c3: trip = st.text_input("Trip Plan (optional)")

    c4,c5,c6 = st.columns(3)
    with c4: amount = st.number_input("Amount (₹)", min_value=0, step=500)
    with c5: car_type = st.selectbox("Car Type", ["Sedan","Ertiga"])
    with c6: recv_in = st.radio("Received In", ["Company Account","Personal Account"], horizontal=True)

    emp_list=[]
    if recv_in=="Personal Account":
        emp_list = st.multiselect("Payment received by (employee(s))", all_employees())

    notes = st.text_area("Notes", placeholder="Any remarks…")
    submitted = st.form_submit_button("💾 Save booking")

if submitted:
    if amount <= 0:
        st.error("Amount must be > 0")
    elif recv_in=="Personal Account" and not emp_list:
        st.error("Please select at least one employee")
    else:
        safe_doc = {
            "date": datetime.combine(when, datetime.min.time()),
            "client_name": str(client or ""),
            "trip_plan": str(trip or ""),
            "amount": int(amount),
            "car_type": str(car_type),
            "received_in": str(recv_in),
            "employees": [str(e) for e in (emp_list or [])],
            "notes": str(notes or ""),
            "created_by": str(user),
            "created_at": datetime.utcnow()
        }
        col_cars.insert_one(safe_doc)

        if recv_in=="Personal Account" and emp_list:
            per_emp = int(amount/len(emp_list))
            for emp in emp_list:
                col_split.insert_one({
                    "kind": "settlement",
                    "created_at": datetime.utcnow(),
                    "created_by": str(user),
                    "date": datetime.combine(when, datetime.min.time()),
                    "employee": str(emp),
                    "amount": int(per_emp),
                    "ref": f"Direct Car ({car_type})",
                    "notes": f"Booking for {client or 'N/A'}"
                })
        st.success("✅ Booking saved successfully")

# -----------------------
# Recent Bookings
# -----------------------
st.subheader("📜 Recent Direct Car Bookings")
docs = list(col_cars.find().sort("date",-1).limit(20))
if docs:
    for d in docs:
        d["_id"] = str(d.get("_id",""))
        if isinstance(d.get("date"), datetime):
            d["date"] = d["date"].strftime("%Y-%m-%d")
    df = pd.DataFrame(docs)
    st.dataframe(df[["date","client_name","trip_plan","amount","car_type","received_in","employees","notes"]],
                 use_container_width=True, hide_index=True)
else:
    st.info("No bookings yet.")

# -----------------------
# Monthly Report
# -----------------------
st.subheader("📅 Monthly Car Bookings Report")
month_choice = st.date_input("Select Month", value=date.today())

if month_choice:
    start_m = month_choice.replace(day=1)
    if start_m.month==12: end_m=start_m.replace(year=start_m.year+1,month=1,day=1)
    else: end_m=start_m.replace(month=start_m.month+1,day=1)

    rows = list(col_cars.find({"date":{"$gte":start_m,"$lt":end_m}}).sort("date",1))
    if rows:
        for r in rows:
            r["_id"] = str(r.get("_id",""))
            if isinstance(r.get("date"), datetime):
                r["date"] = r["date"].strftime("%Y-%m-%d")
        dfm = pd.DataFrame(rows)
        dfm.index = dfm.index+1; dfm.rename_axis("Sr No", inplace=True)
        show = ["date","car_type","client_name","trip_plan","amount","received_in","employees","notes"]
        for c in show: 
            if c not in dfm: dfm[c] = ""
        st.dataframe(dfm[show], use_container_width=True)

        total = int(dfm["amount"].sum())
        comp = int(dfm.loc[dfm["received_in"]=="Company Account","amount"].sum())
        pers = int(dfm.loc[dfm["received_in"]=="Personal Account","amount"].sum())
        st.markdown(f"**Total: ₹{total:,}**  \n🏦 Bank (Company): ₹{comp:,}  \n👤 Cash (Personal): ₹{pers:,}")
    else:
        st.info("No bookings for this month.")
