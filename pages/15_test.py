# ==========================================================
# pages/05_Salary_Slips_V2.py
# FULL FEATURE VERSION – PART 1
# Incentive logic 100% aligned with Follow-up Tracker
# ==========================================================

from __future__ import annotations

from datetime import datetime, date, timedelta, time
from typing import Optional, Dict, List, Tuple
import os
import pandas as pd
import streamlit as st

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from fpdf import FPDF

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Salary Slips (V2 – Tracker Aligned)",
    layout="wide"
)
st.title("🧾 Monthly Salary Slip — V2 (Tracker Aligned)")

TTL = 0  # disable caching for correctness

# ==========================================================
# INCENTIVE POLICY (SINGLE SOURCE)
# ==========================================================
def load_incentive_start() -> date:
    try:
        v = st.secrets.get("INCENTIVE_START_DATE")
        if v:
            return pd.to_datetime(v).date()
    except Exception:
        pass
    return date(2025, 8, 1)

INCENTIVE_START_DATE = load_incentive_start()

# ==========================================================
# MONGO CONNECTION
# ==========================================================
MONGO_KEYS = ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]

def _find_mongo_uri():
    for k in MONGO_KEYS:
        if st.secrets.get(k):
            return st.secrets.get(k)
        if os.getenv(k):
            return os.getenv(k)
    return None

@st.cache_resource
def get_db():
    uri = _find_mongo_uri()
    if not uri:
        st.error("❌ Mongo URI not configured")
        st.stop()

    client = MongoClient(
        uri,
        tz_aware=True,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000
    )
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        st.error("❌ MongoDB connection failed")
        st.stop()

    return client["TAK_DB"]

db = get_db()

# ==========================================================
# COLLECTIONS
# ==========================================================
col_updates  = db["package_updates"]      # incentives
col_split    = db["expense_splitwise"]    # expenses / settlements
col_payroll  = db["salary_payments"]      # salary payments
col_att      = db["driver_attendance"]
col_adv      = db["driver_advances"]
col_cars     = db["direct_car_bookings"]

# ==========================================================
# USERS / LOGIN
# ==========================================================
def load_users():
    return st.secrets.get("users", {})

ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith", "Kuldeep"]))

def login():
    if st.session_state.get("user"):
        return st.session_state["user"]

    users = load_users()
    if not users:
        st.error("Users not configured in secrets")
        st.stop()

    st.subheader("🔐 Login")
    u = st.selectbox("User", list(users.keys()))
    p = st.text_input("PIN", type="password")

    if st.button("Sign in"):
        if str(users.get(u)).strip() == str(p).strip():
            st.session_state["user"] = u
            st.rerun()
        else:
            st.error("Invalid PIN")
    st.stop()

user = login()
is_admin = user in ADMIN_USERS

# ==========================================================
# CONSTANTS
# ==========================================================
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

DRIVERS = ["Priyansh"]

DRV_BASE = 12000
DRV_LEAVE_DED = 400
DRV_OT_ADD = 300

ORG_LOGO = ".streamlit/logo.png"
ORG_SIGN = ".streamlit/signature.png"

# ==========================================================
# HELPERS
# ==========================================================
def month_bounds(d: date) -> Tuple[date, date]:
    first = d.replace(day=1)
    next_first = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    return first, next_first - timedelta(days=1)

def ym_key(d: date) -> str:
    return d.strftime("%Y-%m")

def to_int(x, default=0):
    try:
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def money(x):
    return f"₹ {int(x):,}"

# ==========================================================
# 🔴 INCENTIVE CALCULATION (EXACT FOLLOW-UP TRACKER)
# ==========================================================
@st.cache_data(ttl=TTL, show_spinner=False)
def incentives_for_month(month_start: date) -> int:
    """
    EXACT MATCH WITH FOLLOW-UP TRACKER:
    - confirmed only
    - booking_date month based
    - incentive already computed & frozen
    - NO rep filter
    - NO dedup
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

    # policy start
    df = df[df["booking_date"].dt.date >= INCENTIVE_START_DATE]

    # month filter
    target_month = month_start.strftime("%Y-%m")
    df["Month"] = df["booking_date"].dt.strftime("%Y-%m")
    df = df[df["Month"] == target_month]

    return int(df["incentive"].apply(to_int).sum())

# ==========================================================
# EXPENSES / SETTLEMENTS / CASH
# ==========================================================
@st.cache_data(ttl=TTL, show_spinner=False)
def reimb_expenses(emp, start, end):
    rows = list(col_split.find(
        {
            "kind": "expense",
            "payer": emp,
            "date": {
                "$gte": datetime.combine(start, time.min),
                "$lte": datetime.combine(end, time.max)
            }
        },
        {"amount": 1}
    ))
    return sum(to_int(r.get("amount")) for r in rows)

@st.cache_data(ttl=TTL, show_spinner=False)
def settlements_paid(emp, start, end):
    rows = list(col_split.find(
        {
            "kind": "settlement",
            "employee": emp,
            "date": {
                "$gte": datetime.combine(start, time.min),
                "$lte": datetime.combine(end, time.max)
            }
        },
        {"amount": 1}
    ))
    return sum(to_int(r.get("amount")) for r in rows)

@st.cache_data(ttl=TTL, show_spinner=False)
def cash_received(emp, start, end):
    rows = list(col_cars.find(
        {
            "received_in": "Personal Account",
            "date": {
                "$gte": datetime.combine(start, time.min),
                "$lte": datetime.combine(end, time.max)
            }
        },
        {"employees": 1, "amount": 1}
    ))
    total = 0
    for r in rows:
        emps = r.get("employees") or []
        if emp in emps:
            total += int(to_int(r.get("amount")) / max(len(emps), 1))
    return total

# ==========================================================
# SALARY COMPONENTS
# ==========================================================
def calc_components(emp: str, start: date, end: date) -> Dict:
    base = SALARY_MAP.get(emp, 0)
    fuel = FUEL_MAP.get(emp, 0)

    incentives = incentives_for_month(start)
    reimb = reimb_expenses(emp, start, end)
    settled = settlements_paid(emp, start, end)
    cash = cash_received(emp, start, end)

    net_reimb = reimb - settled
    net_pay = base + fuel + incentives + net_reimb - cash

    return {
        "base_salary": base,
        "fuel_allow": fuel,
        "incentives": incentives,
        "reimb_total": reimb,
        "settled_this_month": settled,
        "cash_received": cash,
        "net_reimb": net_reimb,
        "net_pay": net_pay
    }

# ==========================================================
# DRIVER ATTENDANCE
# ==========================================================
@st.cache_data(ttl=TTL, show_spinner=False)
def load_driver_attendance(driver, start, end):
    cur = col_att.find(
        {"driver": driver,
         "date": {"$gte": datetime.combine(start, time.min),
                  "$lte": datetime.combine(end, time.max)}}
    )
    rows = list(cur)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

@st.cache_data(ttl=TTL, show_spinner=False)
def load_driver_advances(driver, start, end):
    rows = list(col_adv.find(
        {"driver": driver,
         "date": {"$gte": datetime.combine(start, time.min),
                  "$lte": datetime.combine(end, time.max)}},
        {"amount": 1}
    ))
    return sum(to_int(r.get("amount")) for r in rows)

def calc_driver_salary(driver, start, end):
    att = load_driver_attendance(driver, start, end)
    leave_days = int((att["status"] == "Leave").sum()) if not att.empty else 0

    ot_units = 0
    if not att.empty:
        ot_units = int(att["outstation_overnight"].sum()) \
                 + int(att["overnight_client"].sum()) \
                 + int(att["bhasmarathi"].sum())

    leave_ded = leave_days * DRV_LEAVE_DED
    overtime = ot_units * DRV_OT_ADD
    advances = load_driver_advances(driver, start, end)

    net = DRV_BASE - leave_ded + overtime - advances

    return {
        "leave_days": leave_days,
        "ot_units": ot_units,
        "leave_ded": leave_ded,
        "overtime_amt": overtime,
        "advances": advances,
        "net_pay": net
    }

# ==========================================================
# ⏭ PART 2 CONTINUES BELOW
# PDF, UI, multi-payments, admin tables, downloads
# ==========================================================
# ==========================================================
# PAYROLL STORAGE & CARRY FORWARD
# ==========================================================
def load_payroll_record(emp: str, month_key: str) -> dict:
    rec = col_payroll.find_one(
        {"employee": emp, "month": month_key},
        {"_id": 0}
    ) or {}

    if "payments" not in rec:
        amt = to_int(rec.get("amount"))
        if amt > 0:
            rec["payments"] = [{
                "date": rec.get("paid_on"),
                "amount": amt,
                "utr": rec.get("utr", "")
            }]
        else:
            rec["payments"] = []
    return rec


def save_payroll(
    emp: str,
    month_key: str,
    payments: List[dict],
    components: dict,
    allocated_prev: int = 0
):
    total_paid = sum(to_int(p["amount"]) for p in payments)
    applied_to_month = max(total_paid - allocated_prev, 0)

    payload = {
        "employee": emp,
        "month": month_key,
        "payments": payments,
        "total_paid_raw": total_paid,
        "allocated_to_previous": allocated_prev,
        "amount": applied_to_month,
        "paid": total_paid > 0,
        "components": components,
        "updated_at": datetime.utcnow(),
        "updated_by": user
    }

    col_payroll.update_one(
        {"employee": emp, "month": month_key},
        {"$set": payload},
        upsert=True
    )


def previous_pending(emp: str, month_key: str) -> int:
    rows = list(col_payroll.find(
        {"employee": emp, "month": {"$lt": month_key}},
        {"amount": 1, "components": 1}
    ))
    due = sum(to_int(r.get("components", {}).get("net_pay")) for r in rows)
    paid = sum(to_int(r.get("amount")) for r in rows)
    return max(due - paid, 0)


def allocate_to_previous(emp: str, month_key: str, amount: int) -> int:
    if amount <= 0:
        return 0

    rows = list(col_payroll.find(
        {"employee": emp, "month": {"$lt": month_key}},
        {"_id": 1, "amount": 1, "components": 1, "month": 1}
    ).sort("month", 1))

    applied = 0
    for r in rows:
        due = to_int(r["components"]["net_pay"])
        paid = to_int(r["amount"])
        gap = due - paid
        if gap <= 0:
            continue
        use = min(gap, amount - applied)
        if use <= 0:
            break
        col_payroll.update_one(
            {"_id": r["_id"]},
            {"$inc": {"amount": use}}
        )
        applied += use
        if applied >= amount:
            break
    return applied


# ==========================================================
# PDF HELPERS
# ==========================================================
FONT = "Helvetica"

class SalaryPDF(FPDF):
    def header(self):
        if ORG_LOGO and os.path.exists(ORG_LOGO):
            self.image(ORG_LOGO, 10, 8, 25)
        self.set_font(FONT, "B", 14)
        self.cell(0, 10, "TravelaajKal – Achala Holidays Pvt. Ltd.", ln=1, align="C")
        self.set_font(FONT, "", 10)
        self.cell(0, 6, "Monthly Salary Statement", ln=1, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT, "", 8)
        self.cell(0, 10, "This is a system generated document", align="C")


def build_employee_pdf(emp, month_lbl, period, comp, carry, total, payments):
    pdf = SalaryPDF()
    pdf.add_page()
    pdf.set_font(FONT, "", 10)

    pdf.cell(0, 6, f"Employee: {emp}", ln=1)
    pdf.cell(0, 6, f"Period: {period}", ln=1)
    pdf.ln(4)

    def row(lbl, amt):
        pdf.cell(120, 8, lbl, border=1)
        pdf.cell(60, 8, money(amt), border=1, ln=1)

    row("Base Salary", comp["base_salary"])
    row("Fuel Allowance", comp["fuel_allow"])
    row("Incentives", comp["incentives"])
    row("Reimbursement", comp["net_reimb"])
    row("Cash Received", -comp["cash_received"])
    row("Carry Forward", carry)

    pdf.ln(2)
    row("TOTAL DUE", total)

    pdf.ln(6)
    pdf.cell(0, 6, "Payments:", ln=1)
    for p in payments:
        d = pd.to_datetime(p["date"]).date() if p.get("date") else ""
        pdf.cell(0, 6, f"{d} | {money(p['amount'])} | {p.get('utr','')}", ln=1)

    return pdf.output(dest="S").encode("latin-1")


def build_driver_pdf(driver, month_lbl, period, calc):
    pdf = SalaryPDF()
    pdf.add_page()
    pdf.set_font(FONT, "", 10)

    def row(lbl, amt):
        pdf.cell(120, 8, lbl, border=1)
        pdf.cell(60, 8, money(amt), border=1, ln=1)

    pdf.cell(0, 6, f"Driver: {driver}", ln=1)
    pdf.ln(4)

    row("Base", DRV_BASE)
    row("Leave Deduction", -calc["leave_ded"])
    row("Overtime", calc["overtime_amt"])
    row("Advances", -calc["advances"])
    row("NET PAY", calc["net_pay"])

    return pdf.output(dest="S").encode("latin-1")


# ==========================================================
# UI
# ==========================================================
month_pick = st.date_input("Salary Month", date.today())
month_start, month_end = month_bounds(month_pick)
month_key = ym_key(month_start)

mode = st.radio("Mode", ["Single Employee", "All Employees"], horizontal=True)

employees = list(SALARY_MAP.keys())

if mode == "Single Employee":
    emp = st.selectbox("Employee", employees, index=employees.index(user))
    comp = calc_components(emp, month_start, month_end)

    carry = previous_pending(emp, month_key)
    total_due = comp["net_pay"] + carry

    st.metric("Net Pay", money(total_due))

    rec = load_payroll_record(emp, month_key)
    payments = rec.get("payments", [])

    if is_admin:
        st.subheader("Payments")
        for i, p in enumerate(payments):
            c1, c2, c3 = st.columns(3)
            p["date"] = c1.date_input("Date", p.get("date", date.today()), key=f"d{i}")
            p["amount"] = c2.number_input("Amount", value=to_int(p.get("amount")), key=f"a{i}")
            p["utr"] = c3.text_input("UTR", p.get("utr",""), key=f"u{i}")

        if st.button("Add payment"):
            payments.append({"date": date.today(), "amount": 0, "utr": ""})
            st.rerun()

        if st.button("Save"):
            used = allocate_to_previous(emp, month_key, sum(to_int(p["amount"]) for p in payments))
            save_payroll(emp, month_key, payments, comp, used)
            st.success("Saved")

    if st.button("Generate PDF"):
        pdf = build_employee_pdf(
            emp,
            month_start.strftime("%B %Y"),
            f"{month_start} → {month_end}",
            comp,
            carry,
            total_due,
            payments
        )
        st.download_button("Download PDF", pdf, f"Salary_{emp}_{month_key}.pdf")

else:
    for emp in employees:
        comp = calc_components(emp, month_start, month_end)
        st.markdown(f"### {emp}")
        st.write(comp)


# ==========================================================
# DRIVER SECTION
# ==========================================================
st.divider()
st.subheader("🚖 Driver Salary")

driver = st.selectbox("Driver", DRIVERS)
drv_calc = calc_driver_salary(driver, month_start, month_end)

st.metric("Driver Net Pay", money(drv_calc["net_pay"]))

if st.button("Driver PDF"):
    pdf = build_driver_pdf(
        driver,
        month_start.strftime("%B %Y"),
        f"{month_start} → {month_end}",
        drv_calc
    )
    st.download_button("Download Driver PDF", pdf, f"Driver_{driver}_{month_key}.pdf")


# ==========================================================
# ADMIN OVERVIEW
# ==========================================================
if is_admin:
    st.divider()
    st.subheader("📋 Admin – Salary Records")

    df = pd.DataFrame(list(col_payroll.find({}, {"_id": 0})))
    if not df.empty:
        st.dataframe(df, use_container_width=True)

