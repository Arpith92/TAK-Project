# pages/05_Salary_Slips.py
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
import os

import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# =============================
# Page
# =============================
st.set_page_config(page_title="Salary Slips", layout="wide")
st.title("🧾 Monthly Salary Slip")

TTL = 60  # short cache for fast feel with fresh-ish data

# =============================
# Mongo (safe/flexible)
# =============================
CAND_KEYS = ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]

def _find_uri() -> Optional[str]:
    for k in CAND_KEYS:
        try:
            v = st.secrets.get(k)
        except Exception:
            v = None
        if v:
            return v
    for k in CAND_KEYS:
        v = os.getenv(k)
        if v:
            return v
    return None

@st.cache_resource
def _get_client() -> MongoClient:
    uri = _find_uri()
    if not uri:
        st.error(
            "Mongo connection is not configured.\n\n"
            "Add one of these in **Manage app → Settings → Secrets** (recommended: `mongo_uri`)."
        )
        st.stop()
    client = MongoClient(
        uri,
        appName="TAK_SalarySlips",
        maxPoolSize=100,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        retryWrites=True,
        tz_aware=True,
    )
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        st.error(f"Could not connect to MongoDB. Details: {e}")
        st.stop()
    return client

@st.cache_resource
def get_db():
    return _get_client()["TAK_DB"]

db = get_db()
col_updates  = db["package_updates"]     # incentives live here
col_split    = db["expense_splitwise"]   # splitwise-style expenses/settlements
col_payroll  = db["salary_payments"]     # NEW: salary/UTR records per employee-month

# =============================
# Users & login
# =============================
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    # fallback to repo .streamlit/secrets.toml for dev only
    try:
        try:
            import tomllib  # py 3.11+
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, dict) and u:
            with st.sidebar:
                st.warning(
                    "Using users from repo .streamlit/secrets.toml. "
                    "For production, set them in Manage app → Secrets."
                )
            return u
    except Exception:
        pass
    return {}

# Admins (Kuldeep included, per your policy)
ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith", "Kuldeep"]))

# Base salary & fuel rules — update here if needed
SALARY_MAP  = {
    "Arpith": 10000,   # fixed
    "Reena":  0,       # no base (incentives only)
    "Kuldeep": 10000,
    "Teena":  5000,
}
FUEL_MAP    = {
    "Arpith": 0,
    "Reena":  0,
    "Kuldeep": 3000,
    "Teena":  0,
}
CATEGORIES  = ["Car","Hotel","Bhasmarathi","Poojan","PhotoFrame","Other"]

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

# Audit
from tak_audit import audit_pageview
audit_pageview(st.session_state.get("user", "Unknown"), page="05_Salary_Slips")

# =============================
# Helpers
# =============================
def month_bounds(d: date) -> Tuple[date, date]:
    first = d.replace(day=1)
    # next month 1st
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

@st.cache_data(ttl=TTL, show_spinner=False)
def all_employees() -> List[str]:
    return sorted(load_users().keys())

# =============================
# Inputs (who + which month)
# =============================
emp_opts = all_employees()
if is_admin:
    mode = st.radio("View mode", ["Single employee", "All employees (overview)"], horizontal=True)
    if mode == "Single employee":
        view_emp = st.selectbox("View employee", emp_opts, index=(emp_opts.index(user) if user in emp_opts else 0))
    else:
        view_emp = None  # handled below
else:
    mode = "Single employee"
    view_emp = user

month_pick = st.date_input("Slip month", value=date.today())
month_start, month_end = month_bounds(month_pick)
st.caption(f"Period: **{month_start} → {month_end}**")

# =============================
# Data fetchers
# =============================
@st.cache_data(ttl=TTL, show_spinner=False)
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

@st.cache_data(ttl=TTL, show_spinner=False)
def splitwise_expenses(emp: str, start: date, end: date) -> pd.DataFrame:
    q = {
        "kind": "expense",
        "payer": emp,
        "date": {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end,   datetime.max.time()),
        }
    }
    rows = list(col_split.find(q, {"date":1,"category":1,"subheader":1,"amount":1,"customer_name":1,"ach_id":1,"notes":1}))
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

@st.cache_data(ttl=TTL, show_spinner=False)
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

# Payroll records CRUD
def _ym_key(d: date) -> str:
    return d.strftime("%Y-%m")  # e.g., 2025-08

def load_payroll_record(emp: str, month_key: str) -> dict:
    return col_payroll.find_one({"employee": emp, "month": month_key}, {"_id":0}) or {}

def save_or_update_pay(emp: str, month_key: str, *, amount: int, paid_on: date, utr: str, notes: str, components: dict):
    payload = {
        "employee": emp,
        "month": month_key,
        "amount": int(amount),
        "paid_on": datetime.combine(paid_on, datetime.min.time()) if paid_on else None,
        "utr": (utr or "").strip(),
        "notes": (notes or "").strip(),
        "components": components,   # snapshot of calc parts used for net pay (base, fuel, incentives, reimb)
        "updated_at": datetime.utcnow(),
        "updated_by": st.session_state.get("user",""),
    }
    col_payroll.update_one({"employee": emp, "month": month_key}, {"$set": payload}, upsert=True)

def load_all_payroll_for_month(month_key: str) -> List[dict]:
    return list(col_payroll.find({"month": month_key}, {"_id":0}))

# =============================
# Calculators
# =============================
def calc_components(emp: str, start: date, end: date) -> dict:
    base_salary = _to_int(SALARY_MAP.get(emp, 0))
    fuel_allow  = _to_int(FUEL_MAP.get(emp, 0))
    incentives  = incentives_for(emp, start, end)

    df_exp = splitwise_expenses(emp, start, end)
    reimb_total = int(df_exp["Amount"].sum()) if not df_exp.empty else 0
    settled_this_month = settlements_paid(emp, start, end)
    net_reimb = reimb_total - settled_this_month

    net_pay = base_salary + fuel_allow + incentives + net_reimb
    return {
        "base_salary": base_salary,
        "fuel_allow": fuel_allow,
        "incentives": incentives,
        "reimb_total": reimb_total,
        "settled_this_month": settled_this_month,
        "net_reimb": net_reimb,
        "net_pay": net_pay,
        "df_exp": df_exp,
    }

# =============================
# MODE: ALL EMPLOYEES OVERVIEW
# =============================
month_key = _ym_key(month_start)

if mode == "All employees (overview)":
    st.subheader(f"👥 Team overview — {month_start.strftime('%B %Y')}")
    rows = []
    for emp in emp_opts:
        comp = calc_components(emp, month_start, month_end)
        payrec = load_payroll_record(emp, month_key)
        rows.append({
            "Employee": emp,
            "Base": comp["base_salary"],
            "Fuel": comp["fuel_allow"],
            "Incentives": comp["incentives"],
            "Reimb": comp["net_reimb"],
            "Net Pay": comp["net_pay"],
            "Paid?": "Yes" if payrec else "No",
            "Paid on": payrec.get("paid_on"),
            "UTR": payrec.get("utr",""),
            "Amount paid": payrec.get("amount", 0),
        })
    df_team = pd.DataFrame(rows)
    if not df_team.empty:
        df_team["Paid on"] = pd.to_datetime(df_team["Paid on"]).dt.date
        show = df_team.copy()
        show[["Base","Fuel","Incentives","Reimb","Net Pay","Amount paid"]] = show[["Base","Fuel","Incentives","Reimb","Net Pay","Amount paid"]].applymap(lambda x: f"₹ {int(x):,}")
        st.dataframe(
            show.sort_values(["Paid?","Employee"], ascending=[True, True]),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No users found.")

    st.divider()

# =============================
# MODE: SINGLE EMPLOYEE
# =============================
if mode == "Single employee":
    comp = calc_components(view_emp, month_start, month_end)

    l, r = st.columns([2,1])
    with l:
        st.markdown(f"### Salary Slip — **{view_emp}**")
        st.write({
            "Month": month_start.strftime("%B %Y"),
            "Employee": view_emp,
        })
    with r:
        st.metric("Net Pay", money(comp["net_pay"]))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Base salary", money(comp["base_salary"]))
    k2.metric("Fuel allowance", money(comp["fuel_allow"]))
    k3.metric("Incentives (month)", money(comp["incentives"]))
    k4.metric("Reimbursable expenses", money(comp["reimb_total"]))
    k5.metric("Less: Settlements (month)", money(comp["settled_this_month"]))
    st.caption(f"**Net reimbursement** this month = {money(comp['net_reimb'])}  →  **Net Pay** = Base + Fuel + Incentives + Net reimbursement.")
    st.divider()

    # Reimbursement details
    st.subheader("Reimbursement details (this month)")
    df_exp = comp["df_exp"]
    if df_exp.empty:
        st.info("No reimbursable expenses recorded this month.")
    else:
        reimb_by_cat = df_exp.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
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

    # Printable summary
    with st.expander("Printable slip (summary)"):
        st.markdown(f"""
**Employee:** {view_emp}  
**Month:** {month_start.strftime("%B %Y")}  

- Base salary: {money(comp['base_salary'])}  
- Fuel allowance: {money(comp['fuel_allow'])}  
- Incentives (month): {money(comp['incentives'])}  
- Reimbursable expenses (month): {money(comp['reimb_total'])}  
- Less settlements (month): {money(comp['settled_this_month'])}  
- **Net Pay:** {money(comp['net_pay'])}  
""")

    # =============================
    # Admin: mark / update payment for this employee & month
    # =============================
    if is_admin:
        st.divider()
        st.subheader("💵 Mark/Update payment for this month")

        existing = load_payroll_record(view_emp, month_key)
        default_paid_on = pd.to_datetime(existing.get("paid_on")).date() if existing.get("paid_on") else date.today()
        default_amt     = int(existing.get("amount", comp["net_pay"]))
        default_utr     = existing.get("utr","")
        default_notes   = existing.get("notes","")

        with st.form("pay_form", clear_on_submit=False):
            p1, p2, p3 = st.columns([1,1,2])
            with p1:
                pay_amount = st.number_input("Amount paid (₹)", min_value=0, step=500, value=int(default_amt))
            with p2:
                pay_date = st.date_input("Paid on", value=default_paid_on)
            with p3:
                utr = st.text_input("UTR / Ref", value=default_utr, placeholder="UPI/NEFT reference")
            notes = st.text_area("Notes (optional)", value=default_notes, placeholder="e.g., Salary for Aug 2025")

            submitted = st.form_submit_button("💾 Save payment record")

        if submitted:
            save_or_update_pay(
                emp=view_emp,
                month_key=month_key,
                amount=int(pay_amount),
                paid_on=pay_date,
                utr=utr,
                notes=notes,
                components={
                    "base_salary": comp["base_salary"],
                    "fuel_allow": comp["fuel_allow"],
                    "incentives": comp["incentives"],
                    "net_reimb": comp["net_reimb"],
                    "net_pay": comp["net_pay"],
                    "period": {"start": str(month_start), "end": str(month_end)}
                }
            )
            st.success("Payment record saved.")
            st.rerun()

        # show current payment record if exists
        cur = load_payroll_record(view_emp, month_key)
        if cur:
            st.caption("Current payment record")
            rec = {
                "Paid on": pd.to_datetime(cur.get("paid_on")).date() if cur.get("paid_on") else None,
                "Amount paid (₹)": cur.get("amount", 0),
                "UTR / Ref": cur.get("utr",""),
                "Notes": cur.get("notes",""),
                "Updated by": cur.get("updated_by",""),
                "Updated at": pd.to_datetime(cur.get("updated_at")).strftime("%Y-%m-%d %H:%M") if cur.get("updated_at") else "",
            }
            st.write(rec)

# =============================
# Admin: Month payments table
# =============================
if is_admin:
    st.divider()
    st.subheader(f"📋 Payment records saved for {month_start.strftime('%B %Y')}")
    all_pay = load_all_payroll_for_month(month_key)
    if not all_pay:
        st.caption("No payment records saved yet for this month.")
    else:
        rows = []
        for r in all_pay:
            rows.append({
                "Employee": r.get("employee",""),
                "Paid on": pd.to_datetime(r.get("paid_on")).date() if r.get("paid_on") else None,
                "Amount paid (₹)": _to_int(r.get("amount",0)),
                "UTR / Ref": r.get("utr",""),
                "Notes": r.get("notes",""),
                "Updated by": r.get("updated_by",""),
                "Updated at": pd.to_datetime(r.get("updated_at")).strftime("%Y-%m-%d %H:%M") if r.get("updated_at") else "",
            })
        dfp = pd.DataFrame(rows)
        st.dataframe(
            dfp.sort_values(["Employee","Paid on"], na_position="last"),
            use_container_width=True, hide_index=True
        )
