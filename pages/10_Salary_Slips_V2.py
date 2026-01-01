from __future__ import annotations

from datetime import datetime, date, timedelta, time
from typing import List, Dict
import os

import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from fpdf import FPDF


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Salary Slips (V2)", layout="wide")
st.title("🧾 Monthly Salary Slip — V2 (Tracker Aligned)")

TTL = 0   # disable cache for correctness


# =====================================================
# INCENTIVE POLICY START
# =====================================================
def load_incentive_start() -> date:
    try:
        v = st.secrets.get("INCENTIVE_START_DATE")
        if v:
            return pd.to_datetime(v).date()
    except Exception:
        pass
    return date(2025, 8, 1)

INCENTIVE_START_DATE = load_incentive_start()


# =====================================================
# MONGO CONNECTION
# =====================================================
def get_mongo_uri():
    for k in ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]:
        if st.secrets.get(k):
            return st.secrets.get(k)
        if os.getenv(k):
            return os.getenv(k)
    return None

@st.cache_resource
def get_db():
    uri = get_mongo_uri()
    if not uri:
        st.error("Mongo URI not configured")
        st.stop()
    client = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        st.error("MongoDB connection failed")
        st.stop()
    return client["TAK_DB"]

db = get_db()

col_updates = db["package_updates"]
col_split   = db["expense_splitwise"]
col_payroll = db["salary_payments"]
col_cars    = db["direct_car_bookings"]


# =====================================================
# USERS / LOGIN
# =====================================================
def load_users():
    return st.secrets.get("users", {})

ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith", "Kuldeep"]))

def login():
    if st.session_state.get("user"):
        return st.session_state["user"]

    users = load_users()
    st.subheader("🔐 Login")
    u = st.selectbox("User", list(users.keys()))
    p = st.text_input("PIN", type="password")

    if st.button("Sign in"):
        if str(users.get(u)) == str(p):
            st.session_state["user"] = u
            st.rerun()
        else:
            st.error("Invalid PIN")
    st.stop()

user = login()
is_admin = user in ADMIN_USERS


# =====================================================
# CONSTANTS
# =====================================================
SALARY_MAP = {
    "Arpith": 10000,
    "Reena": 0,
    "Kuldeep": 10000,
    "Teena": 5000,
}

FUEL_MAP = {
    "Arpith": 0,
    "Reena": 0,
    "Kuldeep": 3000,
    "Teena": 0,
}


# =====================================================
# HELPERS
# =====================================================
def month_bounds(d: date):
    first = d.replace(day=1)
    next_first = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    return first, next_first - timedelta(days=1)

def ym_key(d: date):
    return d.strftime("%Y-%m")

def to_int(x):
    try:
        return int(round(float(x)))
    except Exception:
        return 0

def money(x):
    return f"₹ {int(x):,}"


# =====================================================
# ✅ INCENTIVE LOGIC (EXACT FOLLOW-UP TRACKER)
# =====================================================
@st.cache_data(ttl=TTL, show_spinner=False)
def incentives_for_month(month_start: date) -> int:
    """
    EXACT COPY of Follow-up Tracker incentive logic:
    - confirmed packages
    - booking_date month based
    - incentive already finalized
    """

    rows = list(col_updates.find(
        {
            "status": "confirmed",
            "incentive": {"$gt": 0},
            "booking_date": {"$ne": None}
        },
        {"_id": 0, "booking_date": 1, "incentive": 1}
    ))

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df = df[df["booking_date"].notna()]

    df = df[df["booking_date"].dt.date >= INCENTIVE_START_DATE]

    target_month = month_start.strftime("%Y-%m")
    df["Month"] = df["booking_date"].dt.strftime("%Y-%m")

    df = df[df["Month"] == target_month]

    return int(df["incentive"].apply(to_int).sum())


# =====================================================
# EXPENSES / SETTLEMENTS / CASH
# =====================================================
def expenses(emp, start, end):
    rows = list(col_split.find(
        {
            "kind": "expense",
            "payer": emp,
            "date": {"$gte": datetime.combine(start, time.min),
                     "$lte": datetime.combine(end, time.max)}
        },
        {"amount": 1}
    ))
    return sum(to_int(r.get("amount")) for r in rows)

def settlements(emp, start, end):
    rows = list(col_split.find(
        {
            "kind": "settlement",
            "employee": emp,
            "date": {"$gte": datetime.combine(start, time.min),
                     "$lte": datetime.combine(end, time.max)}
        },
        {"amount": 1}
    ))
    return sum(to_int(r.get("amount")) for r in rows)

def cash_received(emp, start, end):
    rows = list(col_cars.find(
        {
            "received_in": "Personal Account",
            "date": {"$gte": datetime.combine(start, time.min),
                     "$lte": datetime.combine(end, time.max)}
        },
        {"employees": 1, "amount": 1}
    ))
    total = 0
    for r in rows:
        if emp in (r.get("employees") or []):
            total += int(to_int(r.get("amount")) / max(len(r["employees"]), 1))
    return total


# =====================================================
# SALARY CALCULATION
# =====================================================
def calc_salary(emp: str, start: date, end: date) -> Dict:
    base = SALARY_MAP.get(emp, 0)
    fuel = FUEL_MAP.get(emp, 0)
    incentive = incentives_for_month(start)

    exp = expenses(emp, start, end)
    sett = settlements(emp, start, end)
    cash = cash_received(emp, start, end)

    net_reimb = exp - sett
    net_pay = base + fuel + incentive + net_reimb - cash

    return {
        "base": base,
        "fuel": fuel,
        "incentive": incentive,
        "reimb": net_reimb,
        "cash": cash,
        "net_pay": net_pay
    }


# =====================================================
# UI
# =====================================================
month_pick = st.date_input("Salary month", value=date.today())
month_start, month_end = month_bounds(month_pick)

st.caption(f"Period: {month_start} → {month_end}")

salary = calc_salary(user, month_start, month_end)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Base", money(salary["base"]))
c2.metric("Fuel", money(salary["fuel"]))
c3.metric("Incentives", money(salary["incentive"]))
c4.metric("Reimbursement", money(salary["reimb"]))
c5.metric("Cash Received", money(salary["cash"]))
c6.metric("Net Pay", money(salary["net_pay"]))
