# pages/05_Salary_Slips.py
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

import pandas as pd
import streamlit as st
from pymongo import MongoClient

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Salary Slips", layout="wide")
st.title("🧾 Monthly Salary Slip")

# ----------------------------
# Mongo
# ----------------------------
MONGO_URI = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
db = client["TAK_DB"]
col_updates = db["package_updates"]        # incentives live here
col_split   = db["expense_splitwise"]      # splitwise-style expenses/settlements

# ----------------------------
# App config
# ----------------------------
ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith"]))
SALARY_MAP  = {"Kuldeep": 10000, "Teena": 5000}   # base salary per month
FUEL_MAP    = {"Kuldeep": 3000,  "Teena": 0}      # monthly fuel allowance
CATEGORIES  = ["Car","Hotel","Bhasmarathi","Poojan","PhotoFrame","Other"]

# ----------------------------
# Login (same pattern, no repo warning)
# ----------------------------
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    try:
        try:
            import tomllib  # 3.11+
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        return data.get("users", {}) or {}
    except Exception:
        return {}

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
        st.error("Login not configured. Add `mongo_uri` and a [users] table in **Manage app → Secrets**.")
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
            st.error("Invalid PIN"); st.stop()
    return None

user = _login()
if not user:
    st.stop()
is_admin = user in ADMIN_USERS

# ----------------------------
# Helpers
# ----------------------------
def month_bounds(d: date):
    first = d.replace(day=1)
    next_first = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    last = next_first - timedelta(days=1)
    return first, last

def _to_int(x, default=0):
    try:
        if x is None:
            return default
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def money(n: int) -> str:
    return f"₹ {int(n):,}"

# ----------------------------
# Inputs (who + which month)
# ----------------------------
view_emp = user
if is_admin:
    emp_opts = sorted(load_users().keys())
    view_emp = st.selectbox("View employee", emp_opts, index=emp_opts.index(user) if user in emp_opts else 0)

month_pick = st.date_input("Slip month", value=date.today())
month_start, month_end = month_bounds(month_pick)

st.caption(f"Period: **{month_start} → {month_end}**")

# ----------------------------
# Data fetchers
# ----------------------------
def incentives_for(emp: str, start: date, end: date) -> int:
    q = {
        "status": "confirmed",
        "rep_name": emp,
        "booking_date": {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end,   datetime.max.time()),
        }
    }
    return sum(_to_int(d.get("incentive",0)) for d in col_updates.find(q, {"incentive":1}))

def splitwise_expenses(emp: str, start: date, end: date) -> pd.DataFrame:
    q = {
        "kind": "expense",
        "payer": emp,
        "date": {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end,   datetime.max.time()),
        }
    }
    rows = list(col_split.find(q))
    data = [{
        "Date": pd.to_datetime(r.get("date")).date() if r.get("date") else None,
        "Category": r.get("category","Other"),
        "Subheader": r.get("subheader",""),
        "Amount": _to_int(r.get("amount",0)),
        "Customer": r.get("customer_name",""),
        "ACH ID": r.get("ach_id",""),
        "Notes": r.get("notes",""),
    } for r in rows]
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["Date","Category","Subheader","Amount","Customer","ACH ID","Notes"])

def settlements_paid(emp: str, start: date, end: date) -> int:
    q = {
        "kind": "settlement",
        "employee": emp,
        "date": {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end,   datetime.max.time()),
        }
    }
    return sum(_to_int(r.get("amount",0)) for r in col_split.find(q, {"amount":1}))

# ----------------------------
# Compute components
# ----------------------------
base_salary = SALARY_MAP.get(view_emp, 0)
fuel_allow  = FUEL_MAP.get(view_emp, 0)
incentives  = incentives_for(view_emp, month_start, month_end)

df_exp = splitwise_expenses(view_emp, month_start, month_end)
reimb_total = int(df_exp["Amount"].sum()) if not df_exp.empty else 0
reimb_by_cat = (
    df_exp.groupby("Category", as_index=False)["Amount"].sum()
    if not df_exp.empty else pd.DataFrame({"Category": CATEGORIES, "Amount":[0]*len(CATEGORIES)})
)
reimb_by_cat = reimb_by_cat.sort_values("Amount", ascending=False)

settled_this_month = settlements_paid(view_emp, month_start, month_end)
net_reimb = reimb_total - settled_this_month

net_pay = base_salary + fuel_allow + incentives + net_reimb

# ----------------------------
# Slip – header & totals
# ----------------------------
l, r = st.columns([2,1])
with l:
    st.markdown(f"### Salary Slip — **{view_emp}**")
    st.write({
        "Month": month_start.strftime("%B %Y"),
        "Employee": view_emp,
    })
with r:
    st.metric("Net Pay", money(net_pay))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Base salary", money(base_salary))
k2.metric("Fuel allowance", money(fuel_allow))
k3.metric("Incentives (month)", money(incentives))
k4.metric("Reimbursable expenses", money(reimb_total))
k5.metric("Less: Settlements (month)", money(settled_this_month))

st.caption(f"**Net reimbursement** this month = {money(net_reimb)}  →  **Net Pay** = Salary + Fuel + Incentives + Net reimbursement.")

st.divider()

# ----------------------------
# Reimbursement details
# ----------------------------
st.subheader("Reimbursement breakdown (by category)")
if df_exp.empty:
    st.info("No reimbursable expenses recorded this month.")
else:
    c1, c2 = st.columns([1,1])
    with c1:
        st.dataframe(reimb_by_cat.rename(columns={"Amount":"Amount (₹)"}), use_container_width=True, hide_index=True)
    with c2:
        st.dataframe(
            df_exp[["Date","Category","Subheader","Amount","Customer","ACH ID","Notes"]]
            .rename(columns={"Amount":"Amount (₹)"})
            .sort_values(["Date","Category"], ascending=[True,True]),
            use_container_width=True, hide_index=True
        )

st.divider()

# ----------------------------
# Printable view (optional)
# ----------------------------
with st.expander("Printable slip (summary)"):
    st.markdown(f"""
**Employee:** {view_emp}  
**Month:** {month_start.strftime("%B %Y")}  

- Base salary: {money(base_salary)}  
- Fuel allowance: {money(fuel_allow)}  
- Incentives (month): {money(incentives)}  
- Reimbursable expenses (month): {money(reimb_total)}  
- Less settlements (month): {money(settled_this_month)}  
- **Net Pay:** {money(net_pay)}  
""")
