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
# Users & Login (same as Splitwise page)
# =========================
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    try:
        try:
            import tomllib
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, dict) and u:
            with st.sidebar:
                st.warning("Using users from repo .streamlit/secrets.toml. For production, set them in Manage app → Secrets.")
            return u
    except Exception:
        pass
    return {}

ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith", "Kuldeep"]))

def _login() -> Optional[str]:
    with st.sidebar:
        if st.session_state.get("user"):
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out"):
                st.session_state.pop("user", None)
                st.rerun()
    if st.session_state.get("user"):
        return st.session_state["user"]

    users_map = load_users()
    if not users_map:
        st.error("Login not configured yet. Add to Manage app → Secrets:\n\nmongo_uri=\"...\"\n\n[users]\nArpith=\"1234\" ...")
        st.stop()

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns(2)
    with c1:
        name = st.selectbox("User", list(users_map.keys()), key="login_user")
    with c2:
        pin = st.text_input("PIN", type="password", key="login_pin")
    if st.button("Sign in"):
        if str(users_map.get(name, "")).strip() == str(pin).strip():
            st.session_state["user"] = name
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid PIN")
            st.stop()
    return None

user = _login()
if not user:
    st.stop()

# consistent employee list from users
@st.cache_data(ttl=60)
def all_employees() -> list[str]:
    return [e for e in sorted(load_users().keys()) if e]

# =========================
# Direct Car Booking Form
# =========================
with st.form("direct_car_form", clear_on_submit=False):   # KEEP values after first click
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

    emp_list = []
    if recv_in == "Personal Account":
        emp_list = st.multiselect("Payment received by (employee(s))", all_employees())

    notes = st.text_area("Notes", placeholder="Any remarks…")

    submitted = st.form_submit_button("💾 Save booking", use_container_width=True)

if submitted:
    if amount <= 0:
        st.error("Amount must be > 0")
    elif recv_in == "Personal Account" and not emp_list:
        st.error("Please select at least one employee")
    else:
        doc = {
            "date": datetime.combine(when, datetime.min.time()),
            "client_name": client,
            "trip_plan": trip,
            "amount": int(amount),
            "car_type": car_type,
            "received_in": recv_in,
            "employees": emp_list if recv_in == "Personal Account" else [],
            "notes": notes,
            "created_by": user,
            "created_at": datetime.utcnow(),
        }
        col_cars.insert_one(doc)

        if recv_in == "Personal Account" and emp_list:
            per_emp_amount = int(amount / len(emp_list))
            for emp in emp_list:
                col_split.insert_one({
                    "kind": "settlement",
                    "created_at": datetime.utcnow(),
                    "created_by": user,
                    "date": datetime.combine(when, datetime.min.time()),
                    "employee": emp,
                    "amount": per_emp_amount,
                    "ref": f"Direct Car ({car_type})",
                    "notes": f"Direct booking for {client or 'N/A'} | Bulk entry",
                })

        st.success("Booking saved successfully ✅")
        st.rerun()   # fixed: works in new streamlit

# =========================
# Recent bookings
# =========================
st.subheader("📜 Recent Direct Car Bookings")
docs = list(col_cars.find().sort("date", -1).limit(20))
if not docs:
    st.info("No bookings yet.")
else:
    df = pd.DataFrame(docs)
    if "_id" in df: df["_id"] = df["_id"].astype(str)
    show_cols = ["date","client_name","trip_plan","amount","car_type","received_in","employees","notes"]
    for c in show_cols:
        if c not in df: df[c] = ""
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

# =========================
# Monthly Report
# =========================
st.subheader("📅 Monthly Car Bookings Report")
month_choice = st.date_input("Select Month", value=date.today())

if month_choice:
    start_m = month_choice.replace(day=1)
    if start_m.month == 12:
        end_m = start_m.replace(year=start_m.year+1, month=1, day=1)
    else:
        end_m = start_m.replace(month=start_m.month+1, day=1)

    cur = col_cars.find({"date": {"$gte": start_m, "$lt": end_m}}).sort("date", 1)
    rows = list(cur)
    if not rows:
        st.info("No bookings for this month.")
    else:
        dfm = pd.DataFrame(rows)
        dfm["_id"] = dfm["_id"].astype(str)
        dfm.reset_index(drop=True, inplace=True)
        dfm.index = dfm.index + 1
        dfm.rename_axis("Sr No", inplace=True)

        # Pick main columns
        show_cols = ["date","car_type","client_name","trip_plan","amount","received_in","employees","notes"]
        for c in show_cols:
            if c not in dfm: dfm[c] = ""
        table = dfm[show_cols]

        # Totals
        total_all = dfm["amount"].sum()
        total_cash = dfm.loc[dfm["received_in"]=="Company Account", "amount"].sum()
        total_personal = dfm.loc[dfm["received_in"]=="Personal Account", "amount"].sum()

        st.dataframe(table, use_container_width=True)

        st.markdown(f"""
        **Total this month: ₹{total_all:,}**  
        • Company Account (Bank): ₹{total_cash:,}  
        • Personal Account (Cash to Employees): ₹{total_personal:,}
        """)
