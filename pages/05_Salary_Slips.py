# pages/05_Salary_Slips.py
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
import os, base64

import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from fpdf import FPDF  # fpdf2

# =============================
# Page
# =============================
st.set_page_config(page_title="Salary Slips", layout="wide")
st.title("🧾 Monthly Salary Slip")

# ---- Compact UI CSS tweaks (smaller text/controls) ----
st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] { font-size: 14px !important; }
input, textarea, select { font-size: 14px !important; }
div[data-testid="stMetricValue"] { font-size: 16px !important; line-height: 1.1 !important; }
div[data-testid="stMetricLabel"] { font-size: 12px !important; }
.stDataFrame, .dataframe { font-size: 13px !important; }
.small-kv { font-size: 13px; line-height: 1.2; }
.small-kv b { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

TTL = 60  # short cache

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
        st.error("Mongo connection is not configured. Add `mongo_uri` in Secrets.")
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
col_updates  = db["package_updates"]     # incentives
col_split    = db["expense_splitwise"]   # expenses/settlements
col_payroll  = db["salary_payments"]     # salary/UTR per employee-month
col_att      = db["driver_attendance"]   # driver attendance
col_adv      = db["driver_advances"]     # driver advances
col_cars     = db["direct_car_bookings"] # direct car bookings

# =============================
# Users & login
# =============================
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
                st.warning("Using repo .streamlit/secrets.toml [dev]. Prefer Secrets in cloud.")
            return u
    except Exception:
        pass
    return {}

ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith", "Kuldeep"]))

# Base salary & fuel rules
SALARY_MAP  = {
    "Arpith": 10000,
    "Reena":  0,
    "Kuldeep": 10000,
    "Teena":  5000,
}
FUEL_MAP    = {
    "Arpith": 0,
    "Reena":  0,
    "Kuldeep": 3000,
    "Teena":  0,
}

def _login() -> Optional[str]:
    with st.sidebar:
        if st.session_state.get("user"):
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out"):
                st.experimental_set_query_params(_ts=datetime.now().timestamp())
                st.stop()


    if st.session_state.get("user"):
        return st.session_state["user"]

    users_map = load_users()
    if not users_map:
        st.error("Login not configured. Add `mongo_uri` and a [users] table in Secrets.")
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
            st.experimental_set_query_params(_ts=datetime.now().timestamp())
            st.stop()

        else:
            st.error("Invalid PIN"); st.stop()
    return None

user = _login()
if not user:
    st.stop()
is_admin = user in ADMIN_USERS

# =============================
# Helpers
# =============================
ORG_LOGO = ".streamlit/logo.png"
ORG_SIGN = ".streamlit/signature.png"

def month_bounds(d: date) -> Tuple[date, date]:
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

def _as_bytes(x) -> bytes:
    if x is None: return b""
    if isinstance(x, (bytes, bytearray)): return bytes(x)
    if isinstance(x, memoryview): return x.tobytes()
    if isinstance(x, str): return x.encode("latin-1", errors="ignore")
    try: return bytes(x)
    except Exception: return str(x).encode("latin-1", errors="ignore")

@st.cache_data(ttl=TTL, show_spinner=False)
def all_employees() -> List[str]:
    return sorted(load_users().keys())

def _ym_key(d: date) -> str:
    return d.strftime("%Y-%m")

# =============================
# Payroll helpers
# =============================
def load_payroll_record(emp: str, month_key: str) -> dict:
    return col_payroll.find_one({"employee": emp, "month": month_key}, {"_id":0}) or {}

def save_or_update_pay(
    emp: str, month_key: str, *, amount: int, paid_on: Optional[date],
    utr: str, notes: str, components: dict, paid_flag: bool
):
    payload = {
        "employee": emp,
        "month": month_key,
        "amount": int(amount),
        "paid": bool(paid_flag),
        "paid_on": datetime.combine(paid_on, datetime.min.time()) if (paid_on and paid_flag) else None,
        "utr": (utr or "").strip(),
        "notes": (notes or "").strip(),
        "components": components,
        "updated_at": datetime.utcnow(),
        "updated_by": st.session_state.get("user",""),
    }
    col_payroll.update_one({"employee": emp, "month": month_key}, {"$set": payload}, upsert=True)

def load_all_payroll_for_month(month_key: str) -> List[dict]:
    return list(col_payroll.find({"month": month_key}, {"_id":0}))

def load_all_payroll_all_months() -> pd.DataFrame:
    rows = list(col_payroll.find({}, {"_id":0}))
    if not rows:
        return pd.DataFrame(columns=["employee","month","amount","paid","components"])
    return pd.DataFrame(rows)
# ===== Carry-forward helpers =====
def previous_pending_amount(emp: str, current_month_key: str) -> int:
    cur = list(col_payroll.find(
        {"employee": emp, "month": {"$lt": current_month_key}},
        {"_id":0, "amount":1, "components":1}
    ))
    total_due = sum(_to_int((r.get("components") or {}).get("net_pay", 0)) for r in cur)
    total_paid = sum(_to_int(r.get("amount", 0)) for r in cur)
    return max(total_due - total_paid, 0)

def allocate_payment_to_previous(emp: str, current_month_key: str, amount: int) -> int:
    amt = int(amount or 0)
    if amt <= 0:
        return 0
    rows = list(col_payroll.find(
        {"employee": emp, "month": {"$lt": current_month_key}},
        {"_id":1, "amount":1, "components":1, "month":1}
    ).sort("month", 1))
    applied = 0
    for r in rows:
        due = _to_int((r.get("components") or {}).get("net_pay", 0))
        paid = _to_int(r.get("amount", 0))
        gap = due - paid
        if gap <= 0: continue
        pay = min(gap, amt - applied)
        if pay <= 0: break
        col_payroll.update_one(
            {"_id": r["_id"]},
            {"$inc": {"amount": int(pay)}, "$set": {"updated_at": datetime.utcnow(), "updated_by": st.session_state.get("user","")}}
        )
        applied += pay
        if applied >= amt: break
    return applied

# =============================
# Incentives / Expenses / Settlements
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
    rows = list(col_updates.find(q, {
        "client_mobile": 1,
        "client_name": 1,
        "final_route": 1,
        "start_date": 1,
        "booking_date": 1,
        "incentive": 1,
        "revision": 1,
        "rep_name": 1,
    }))
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    for col in ["client_mobile","client_name","final_route","start_date","booking_date","incentive","revision"]:
        if col not in df.columns:
            df[col] = None

    df["Travel date"] = pd.to_datetime(
        df["start_date"].fillna(df["booking_date"]), errors="coerce"
    ).dt.date
    df["_key"] = df[["client_mobile","Travel date","final_route"]].astype(str).agg("-".join, axis=1)

    if "revision" in df.columns:
        df = df.sort_values(["_key","revision"], ascending=[True, False]).groupby("_key", as_index=False).first()
    else:
        df = df.groupby("_key", as_index=False).first()

    return int(df["incentive"].sum())

@st.cache_data(ttl=TTL, show_spinner=False)
def splitwise_expenses(emp: str, start: date, end: date) -> pd.DataFrame:
    q = {
        "kind": "expense",
        "payer": emp,
        "date": {"$gte": datetime.combine(start, datetime.min.time()),
                 "$lte": datetime.combine(end,   datetime.max.time())}
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
        "date": {"$gte": datetime.combine(start, datetime.min.time()),
                 "$lte": datetime.combine(end,   datetime.max.time())}
    }
    return sum(_to_int(r.get("amount",0)) for r in col_split.find(q, {"amount":1}))

# =============================
# Direct Car Cash Received
# =============================
@st.cache_data(ttl=TTL, show_spinner=False)
def cash_received(emp: str, start: date, end: date) -> int:
    q = {
        "date": {"$gte": datetime.combine(start, datetime.min.time()),
                 "$lte": datetime.combine(end,   datetime.max.time())}
    }
    rows = list(col_cars.find(q, {"employees":1,"received_in":1,"amount":1}))
    total = 0
    for r in rows:
        if emp in (r.get("employees") or []):
            if r.get("received_in") == "Personal Account":
                total += _to_int(r.get("amount", 0))
    return total

def calc_components(emp: str, start: date, end: date) -> dict:
    base_salary = _to_int(SALARY_MAP.get(emp, 0))
    fuel_allow  = _to_int(FUEL_MAP.get(emp, 0))
    incentives  = incentives_for(emp, start, end)

    df_exp = splitwise_expenses(emp, start, end)
    reimb_total = int(df_exp["Amount"].sum()) if not df_exp.empty else 0
    settled_this_month = settlements_paid(emp, start, end)
    cash_recv = cash_received(emp, start, end)

    net_reimb = reimb_total - settled_this_month
    net_pay = base_salary + fuel_allow + incentives + net_reimb - cash_recv

    return {
        "base_salary": base_salary,
        "fuel_allow": fuel_allow,
        "incentives": incentives,
        "reimb_total": reimb_total,
        "settled_this_month": settled_this_month,
        "cash_received": cash_recv,
        "net_reimb": net_reimb,
        "net_pay": net_pay,
        "df_exp": df_exp,
    }

# =============================
# Driver Attendance Calc
# =============================
DRIVERS = ["Priyansh"]
DRV_BASE = 12000
DRV_LEAVE_DED = 400
DRV_OT_ADD = 300

@st.cache_data(ttl=TTL, show_spinner=False)
def load_driver_attendance(driver: str, start: date, end: date) -> pd.DataFrame:
    cur = col_att.find(
        {"driver": driver, "date": {"$gte": datetime.combine(start, datetime.min.time()),
                                    "$lte": datetime.combine(end,   datetime.max.time())}},
        {"_id":0}
    ).sort("date", 1)
    rows = []
    for r in cur:
        rows.append({
            "date": pd.to_datetime(r.get("date")).date() if r.get("date") else None,
            "status": r.get("status","Present"),
            "outstation_overnight": bool(r.get("outstation_overnight", False)),
            "overnight_client":     bool(r.get("overnight_client", False)),
            "bhasmarathi":          bool(r.get("bhasmarathi", False)),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date","status","outstation_overnight","overnight_client","bhasmarathi"])

@st.cache_data(ttl=TTL, show_spinner=False)
def load_driver_advances(driver: str, start: date, end: date) -> int:
    cur = col_adv.find(
        {"driver": driver, "date": {"$gte": datetime.combine(start, datetime.min.time()),
                                    "$lte": datetime.combine(end,   datetime.max.time())}},
        {"_id":0, "amount":1}
    )
    return sum(_to_int(r.get("amount", 0)) for r in cur)

def calc_driver_month(driver: str, start: date, end: date) -> dict:
    att = load_driver_attendance(driver, start, end)
    leave_days = 0 if att.empty else int((att["status"] == "Leave").sum())
    ot_units = 0 if att.empty else int(att["outstation_overnight"].sum()) + int(att["overnight_client"].sum()) + int(att["bhasmarathi"].sum())
    leave_ded = leave_days * DRV_LEAVE_DED
    overtime_amt = ot_units * DRV_OT_ADD
    advances = load_driver_advances(driver, start, end)
    net = (DRV_BASE - leave_ded + overtime_amt) - advances
    return {
        "leave_days": leave_days,
        "ot_units": ot_units,
        "leave_ded": leave_ded,
        "overtime_amt": overtime_amt,
        "advances": advances,
        "net_pay": net,
    }

# =============================
# PDF Helpers (header + employee slip)
# =============================
def _ascii(s: str) -> str:
    if s is None: return ""
    return (str(s)
            .replace("₹", "Rs ")
            .replace("—", "-").replace("–", "-").replace("•", "-")
            .replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
            .replace("™", "(TM)")
            )

def inr_ascii(n) -> str:
    try:
        return f"Rs {int(round(float(n))):,}"
    except Exception:
        return f"Rs {n}"

ORG = {
    "title": "TravelaajKal® – Achala Holidays Pvt. Ltd.",
    "line1": "Mangrola, Ujjain, Madhya Pradesh 456006, India",
    "line2": "Email: travelaajkal@gmail.com  |  Web: www.travelaajkal.com  |  Mob: +91-7509612798",
    "footer_rights": f"All rights reserved by TravelaajKal {datetime.now().year}-{str(datetime.now().year+1)[-2:]}"
}

class InvoiceHeaderPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_title("Salary Statement")
        self.set_author("Achala Holidays Pvt. Ltd.")

    def header(self):
        self.set_draw_color(150,150,150)
        self.rect(8, 8, 194, 281)
        if ORG_LOGO and os.path.exists(ORG_LOGO):
            try: self.image(ORG_LOGO, x=14, y=12, w=28)
            except Exception: pass
        self.set_xy(50, 12)
        self.set_font("Helvetica", "B", 14); self.cell(0, 7, _ascii(ORG["title"]), align="C", ln=1)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, _ascii(ORG["line1"]), align="C", ln=1)
        self.cell(0, 6, _ascii(ORG["line2"]), align="C", ln=1)
        self.ln(2); self.set_draw_color(0,0,0)
        self.line(12, self.get_y(), 198, self.get_y()); self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, _ascii(ORG["footer_rights"]), ln=1, align="C")
# ---- Employee PDF (2-column table) ----
def build_employee_pdf(*, emp: str, month_label: str, period_label: str, comp: dict, carry_forward: int, total_due: int) -> bytes:
    pdf = InvoiceHeaderPDF()
    pdf.add_page()

    left = 16
    th = 8
    col1_w, col2_w = 120, 66

    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(left); pdf.cell(0, 6, _ascii(f"{month_label} (Salary Statement: {period_label})"), ln=1)
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(left); pdf.cell(0, 6, _ascii(f"EMP NAME:  {emp}"), ln=1)
    pdf.ln(2)

    def header_row():
        y = pdf.get_y()
        pdf.set_font("Helvetica", "B", 10)
        pdf.rect(left, y, col1_w, th)
        pdf.rect(left + col1_w, y, col2_w, th)
        pdf.text(left + 2, y + th - 2, _ascii("Particulars"))
        pdf.text(left + col1_w + 2, y + th - 2, _ascii("Amount"))
        pdf.ln(th)
        pdf.set_font("Helvetica", "", 10)

    def row(label: str, amount):
        y = pdf.get_y()
        pdf.rect(left, y, col1_w, th)
        pdf.rect(left + col1_w, y, col2_w, th)
        pdf.text(left + 2, y + th - 2, _ascii(label))
        pdf.set_xy(left + col1_w, y); pdf.cell(col2_w - 2, th, _ascii(inr_ascii(amount)), align="R")
        pdf.ln(th)

    header_row()
    row("Base salary", comp["base_salary"])
    row("Fuel allowance", comp["fuel_allow"])
    row("Incentives", comp["incentives"])
    row("Reimbursable expenses", comp["reimb_total"])
    row("Less: Settlements (this month)", -comp["settled_this_month"])
    row("Cash received (deduction)", -comp["cash_received"])
    row("Carry forward (previous pending)", carry_forward)

    pdf.ln(2)
    y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 11)
    pdf.rect(left, y, col1_w, th)
    pdf.rect(left + col1_w, y, col2_w, th)
    pdf.text(left + 2, y + th - 2, _ascii("Total Due"))
    pdf.set_xy(left + col1_w, y); pdf.cell(col2_w - 2, th, _ascii(inr_ascii(total_due)), align="R")
    pdf.ln(th + 10)

    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _ascii("Note: This is a computer-generated statement."))

    pdf.ln(6)
    sig_w = 50
    sig_x = pdf.w - 16 - sig_w
    sig_y = pdf.get_y()
    if ORG_SIGN and os.path.exists(ORG_SIGN):
        try: pdf.image(ORG_SIGN, x=sig_x, y=sig_y, w=sig_w)
        except Exception: pass
    pdf.set_xy(sig_x, sig_y + 18)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(sig_w, 6, _ascii("Authorised Signatory"), ln=1, align="C")

    out = pdf.output(dest="S")
    return out if isinstance(out, (bytes, bytearray)) else str(out).encode("latin-1", errors="ignore")


# =============================
# UI: Month selection + modes
# =============================
emp_opts = all_employees()
if is_admin:
    mode = st.radio("View mode", ["Single employee", "All employees (overview)"], horizontal=True)
    view_emp = st.selectbox("View employee", emp_opts, index=(emp_opts.index(user) if user in emp_opts else 0)) if mode == "Single employee" else None
else:
    mode = "Single employee"
    view_emp = user

month_pick = st.date_input("Slip month", value=date.today())
month_start, month_end = month_bounds(month_pick)
month_key = _ym_key(month_start)
st.caption(f"Period: **{month_start} → {month_end}**")


# =============================
# MODE: ALL EMPLOYEES
# =============================
if mode == "All employees (overview)":
    st.subheader(f"👥 Team overview — {month_start.strftime('%B %Y')}")
    edit_states = {}
    pending_total = 0

    for emp in emp_opts:
        comp = calc_components(emp, month_start, month_end)
        payrec = load_payroll_record(emp, month_key)
        paid_flag = bool(payrec.get("paid", False))
        paid_on = pd.to_datetime(payrec.get("paid_on")).date() if payrec.get("paid_on") else date.today()
        utr = payrec.get("utr","")
        amount_paid = _to_int(payrec.get("amount", 0))

        cf = previous_pending_amount(emp, month_key)
        total_due = comp["net_pay"] + cf

        with st.container(border=True):
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.2,1,1,1,1,1.2,1.2,1.2])
            c1.markdown(f"**{emp}**")
            c2.metric("Base", money(comp["base_salary"]))
            c3.metric("Fuel", money(comp["fuel_allow"]))
            c4.metric("Incent.", money(comp["incentives"]))
            c5.metric("Reimb", money(comp["net_reimb"]))
            c6.metric("Cash recv", money(comp["cash_received"]))
            c7.metric("Net Pay", money(comp["net_pay"]))
            c8.metric("Carry Fwd", money(cf))

            paid_choice = st.selectbox(f"Paid? {emp}", ["No","Yes"], index=(1 if paid_flag else 0), key=f"paid_{emp}")
            paid_yes = (paid_choice == "Yes")

            d1, d2, d3 = st.columns([1,1,1.2])
            pay_date = d1.date_input("Paid on", value=paid_on, key=f"date_{emp}", disabled=(not paid_yes))
            pay_amt  = d2.number_input("Amt Paid", min_value=0, step=500,
                                       value=(amount_paid if paid_yes else 0),
                                       key=f"amt_{emp}", disabled=(not paid_yes))
            utr_val  = d3.text_input("UTR", value=(utr if paid_yes else ""), key=f"utr_{emp}", disabled=(not paid_yes))

            balance = total_due - (pay_amt if paid_yes else 0)
            st.caption(f"Balance after payment: {money(balance)}")
            pending_total += max(balance, 0)

            edit_states[emp] = {
                "paid": paid_yes,
                "amount": int(pay_amt if paid_yes else 0),
                "paid_on": pay_date if paid_yes else None,
                "utr": utr_val if paid_yes else "",
                "components": comp,
                "notes": f"Salary {month_key}",
                "carry_forward": cf,
            }

    st.info(f"**Total pending balance:** {money(int(pending_total))}")

    if st.button("💾 Save all changes", type="primary"):
        for emp, payload in edit_states.items():
            used_prev = allocate_payment_to_previous(emp, month_key, payload["amount"])
            remaining = max(payload["amount"] - used_prev, 0)
            save_or_update_pay(
                emp=emp,
                month_key=month_key,
                amount=remaining,
                paid_on=payload["paid_on"],
                utr=payload["utr"],
                notes=payload["notes"],
                components=payload["components"],
                paid_flag=payload["paid"],
            )
        st.success("Saved all updates.")
        st.experimental_set_query_params(_ts=datetime.now().timestamp()) st.stop()


# =============================
# MODE: SINGLE EMPLOYEE
# =============================
if mode == "Single employee" and view_emp:
    comp = calc_components(view_emp, month_start, month_end)
    cf_prev = previous_pending_amount(view_emp, month_key)
    total_due = comp["net_pay"] + cf_prev

    st.subheader(f"Salary Slip — {view_emp} ({month_start.strftime('%B %Y')})")
    st.metric("Total due", money(total_due))

    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("Base", money(comp["base_salary"]))
    k2.metric("Fuel", money(comp["fuel_allow"]))
    k3.metric("Incentives", money(comp["incentives"]))
    k4.metric("Reimb", money(comp["reimb_total"]))
    k5.metric("Settled", money(comp["settled_this_month"]))
    k6.metric("Cash recv", money(comp["cash_received"]))
    k7.metric("Carry fwd", money(cf_prev))

    if st.button("📄 Generate Salary PDF"):
        pdf_bytes = build_employee_pdf(
            emp=view_emp,
            month_label=month_start.strftime("%B-%Y"),
            period_label=f"{month_start} → {month_end}",
            comp=comp,
            carry_forward=cf_prev,
            total_due=total_due
        )
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"Salary_{view_emp}_{month_key}.pdf",
            mime="application/pdf"
        )

    if is_admin:
        existing = load_payroll_record(view_emp, month_key)
        paid_choice = st.selectbox("Paid?", ["No","Yes"], index=(1 if existing.get("paid") else 0))
        paid_yes = (paid_choice == "Yes")
        pay_amt = st.number_input("Amount paid", min_value=0, step=500,
                                  value=int(existing.get("amount",0)),
                                  disabled=(not paid_yes))
        pay_date = st.date_input("Paid on", value=pd.to_datetime(existing.get("paid_on")).date() if existing.get("paid_on") else date.today(),
                                 disabled=(not paid_yes))
        utr_val = st.text_input("UTR", value=existing.get("utr",""), disabled=(not paid_yes))
        notes = st.text_area("Notes", value=existing.get("notes",""))

        if st.button("💾 Save payment"):
            used_prev = allocate_payment_to_previous(view_emp, month_key, pay_amt)
            remaining = max(pay_amt - used_prev, 0)
            save_or_update_pay(
                emp=view_emp,
                month_key=month_key,
                amount=remaining,
                paid_on=pay_date if paid_yes else None,
                utr=utr_val,
                notes=notes,
                components=comp,
                paid_flag=paid_yes,
            )
            st.success("Saved payment.")
            st.experimental_set_query_params(_ts=datetime.now().timestamp()) st.stop()


# =============================
# DRIVER SECTION
# =============================
st.divider()
st.subheader("🚖 Driver Salary")
drv = st.selectbox("Driver", DRIVERS, index=0)
drv_month = st.date_input("Driver month", value=date.today())
drv_start, drv_end = month_bounds(drv_month)
drv_calc = calc_driver_month(drv, drv_start, drv_end)

st.metric("Net Pay", money(drv_calc["net_pay"]))
st.metric("Leaves", drv_calc["leave_days"])
st.metric("OT Units", drv_calc["ot_units"])

if st.button("📄 Driver PDF"):
    pdf_b = build_driver_pdf(driver=drv,
                             month_label=drv_start.strftime("%B-%Y"),
                             period_label=f"{drv_start} → {drv_end}",
                             calc=drv_calc)
    st.download_button(
        "⬇️ Download Driver PDF",
        data=pdf_b,
        file_name=f"Driver_{drv}_{drv_start.strftime('%Y_%m')}.pdf",
        mime="application/pdf"
    )


# =============================
# ADMIN: ALL PAYMENTS TABLE
# =============================
if is_admin:
    st.divider()
    st.subheader(f"📋 All payment records for {month_start.strftime('%B %Y')}")
    dfp = pd.DataFrame(load_all_payroll_for_month(month_key))
    if not dfp.empty:
        dfp["Paid?"] = dfp["paid"].map({True:"Yes",False:"No"})
        dfp["Paid on"] = pd.to_datetime(dfp["paid_on"]).dt.date
        st.dataframe(
            dfp[["employee","Paid?","Paid on","amount","utr","notes","updated_by"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No records yet for this month.")
