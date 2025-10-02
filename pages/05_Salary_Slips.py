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
# Driver collections
col_att      = db["driver_attendance"]
col_adv      = db["driver_advances"]

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
                st.session_state.pop("user", None)
                st.rerun()

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
            st.rerun()
        else:
            st.error("Invalid PIN"); st.stop()
    return None

user = _login()
if not user:
    st.stop()
is_admin = user in ADMIN_USERS

# Optional audit
try:
    from tak_audit import audit_pageview
    audit_pageview(st.session_state.get("user", "Unknown"), page="05_Salary_Slips")
except Exception:
    pass

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
def incentives_for(emp: str, start: date, end: date) -> int:
    q = {
        "status": "confirmed",
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
        "assigned_rep": 1  # 👈 new field from followup tracker reassignments
    }))
    if not rows:
        return 0

    df = pd.DataFrame(rows)

    # Prefer assigned_rep if present, else fall back to rep_name
    df["rep_effective"] = df["assigned_rep"].fillna(df["rep_name"])

    # Filter for the employee in question
    df = df[df["rep_effective"] == emp]

    if df.empty:
        return 0

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
# Calculators — DRIVER (attendance based)
# =============================
DRIVERS = ["Priyansh"]           # extend later
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
# PDF helpers (ASCII-safe, invoice header, table fits inside border)
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
        # border
        self.set_draw_color(150,150,150)
        self.rect(8, 8, 194, 281)
        # logo
        if ORG_LOGO and os.path.exists(ORG_LOGO):
            try: self.image(ORG_LOGO, x=14, y=12, w=28)
            except Exception: pass
        # lines
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
    # 2-column table, fits inside outer border (sum = 186)
    col1_w, col2_w = 120, 66

    # Header text
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
    row("Carry forward (previous pending)", carry_forward)

    # Total due
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

    # Signature (right)
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

# ---- Driver PDF (3-column table like attendance page) ----
def build_driver_pdf(*, driver: str, month_label: str, period_label: str, calc: dict) -> bytes:
    pdf = InvoiceHeaderPDF()
    pdf.add_page()

    left = 16
    th = 8
    # keep table within border: 88+40+58 = 186
    col1_w, col2_w, col3_w = 88, 40, 58

    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(left); pdf.cell(0, 6, _ascii(f"{month_label} (Salary Statement: {period_label})"), ln=1)
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(left); pdf.cell(0, 6, _ascii(f"EMP NAME:  {driver}"), ln=1)
    pdf.ln(2)

    # header
    y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 10)
    pdf.rect(left, y, col1_w, th)
    pdf.rect(left + col1_w, y, col2_w, th)
    pdf.rect(left + col1_w + col2_w, y, col3_w, th)
    pdf.text(left + 2, y + th - 2, _ascii("Particulars"))
    pdf.text(left + col1_w + 2, y + th - 2, _ascii("Days/Units"))
    pdf.text(left + col1_w + col2_w + 2, y + th - 2, _ascii("Amount"))
    pdf.ln(th)
    pdf.set_font("Helvetica", "", 10)

    def row(label, units, amount):
        y = pdf.get_y()
        pdf.rect(left, y, col1_w, th)
        pdf.rect(left + col1_w, y, col2_w, th)
        pdf.rect(left + col1_w + col2_w, y, col3_w, th)
        pdf.text(left + 2, y + th - 2, _ascii(label))
        pdf.set_xy(left + col1_w, y); pdf.cell(col2_w-2, th, _ascii(str(units)), align="R")
        pdf.set_xy(left + col1_w + col2_w, y); pdf.cell(col3_w-2, th, _ascii(inr_ascii(amount)), align="R")
        pdf.ln(th)

    # rows
    # total days in month is informative; not used in calc here
    # compute for label from period range
    # (end - start).days + 1 will be passed by caller if needed; show '-' for amount
    row("Salary", "-", DRV_BASE)
    row("Total Leave", calc["leave_days"], calc["leave_ded"])
    row("Over-time", calc["ot_units"], calc["overtime_amt"])
    row("Advances (deduct)", "-", calc["advances"])

    # Net
    pdf.ln(2)
    y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 11)
    pdf.rect(left, y, col1_w + col2_w, th)
    pdf.rect(left + col1_w + col2_w, y, col3_w, th)
    pdf.text(left + 2, y + th - 2, _ascii("Total Salary (Net)"))
    pdf.set_xy(left + col1_w + col2_w, y); pdf.cell(col3_w-2, th, _ascii(inr_ascii(calc["net_pay"])), align="R")
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
# Inputs (who + which month)
# =============================
emp_opts = all_employees()
if is_admin:
    mode = st.radio("View mode", ["Single employee", "All employees (overview)"], horizontal=True)
    if mode == "Single employee":
        view_emp = st.selectbox("View employee", emp_opts, index=(emp_opts.index(user) if user in emp_opts else 0))
    else:
        view_emp = None
else:
    mode = "Single employee"
    view_emp = user

month_pick = st.date_input("Slip month", value=date.today())
month_start, month_end = month_bounds(month_pick)
st.caption(f"Period: **{month_start} → {month_end}**")
month_key = _ym_key(month_start)

# =============================
# TOP: Overall pending summary (all months)
# =============================
st.divider()
st.subheader("📊 Overall pending (all months)")
df_all = load_all_payroll_all_months()
if df_all.empty:
    st.caption("No saved salary records yet.")
else:
    df_all["due"] = df_all["components"].apply(lambda c: _to_int((c or {}).get("net_pay", 0)))
    df_all["paid_amt"] = df_all["amount"].apply(_to_int)
    sel_emp = st.multiselect("Filter employees", options=emp_opts, default=emp_opts)
    df_f = df_all[df_all["employee"].isin(sel_emp)]
    agg = (
        df_f.groupby("employee", as_index=False)[["due","paid_amt"]]
        .sum()
        .assign(pending=lambda d: d["due"] - d["paid_amt"])
        .sort_values("employee")
    )
    show = agg.rename(columns={"employee":"Employee","due":"Total Due (₹)","paid_amt":"Total Paid (₹)","Pending (₹)":"Pending (₹)"})
    show["Pending (₹)"] = agg["pending"]
    show = show[["Employee","Total Due (₹)","Total Paid (₹)","Pending (₹)"]]
    for col in ["Total Due (₹)","Total Paid (₹)","Pending (₹)"]:
        show[col] = show[col].apply(lambda x: f"₹ {int(x):,}")
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()

# =============================
# MODE: ALL EMPLOYEES OVERVIEW  (COMPACT + bulk edit + save + CF)
# =============================
if mode == "All employees (overview)":
    st.subheader(f"👥 Team overview — {month_start.strftime('%B %Y')}")
    pending_total = 0
    edit_states: Dict[str, Dict] = {}

    for emp in emp_opts:
        comp = calc_components(emp, month_start, month_end)
        payrec = load_payroll_record(emp, month_key)
        paid_flag = bool(payrec.get("paid", False))
        paid_on = pd.to_datetime(payrec.get("paid_on")).date() if payrec.get("paid_on") else date.today()
        utr = payrec.get("utr","")
        amount_paid = _to_int(payrec.get("amount", 0))
        default_amt = amount_paid if amount_paid else (comp["net_pay"] if paid_flag else 0)

        cf = previous_pending_amount(emp, month_key)
        total_due = comp["net_pay"] + cf

        with st.container(border=True):
            # compact columns
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.1,0.9,0.9,0.9,0.9,1.1,1.1,1.1])
            c1.markdown(f"**{emp}**")
            c2.markdown(f'<div class="small-kv">Base<br><b>{money(comp["base_salary"])}</b></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="small-kv">Fuel<br><b>{money(comp["fuel_allow"])}</b></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="small-kv">Incent.<br><b>{money(comp["incentives"])}</b></div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="small-kv">Reimb<br><b>{money(comp["net_reimb"])}</b></div>', unsafe_allow_html=True)
            c6.markdown(f'<div class="small-kv">Net Pay<br><b>{money(comp["net_pay"])}</b></div>', unsafe_allow_html=True)
            c7.markdown(f'<div class="small-kv">Carry fwd<br><b>{money(cf)}</b></div>', unsafe_allow_html=True)

            paid_choice = c8.selectbox(f"Paid? — {emp}", ["No","Yes"], index=(1 if paid_flag else 0), key=f"paid_{emp}")
            paid_yes = (paid_choice == "Yes")

            d1, d2, d3, d4 = st.columns([1,1,1,1.1])
            with d1:
                pay_date = st.date_input("Paid on", value=(paid_on if paid_yes else date.today()),
                                         key=f"date_{emp}", disabled=(not paid_yes))
            with d2:
                pay_amt = st.number_input("Amount paid (₹)", min_value=0, step=500,
                                          value=int(default_amt if paid_yes else 0),
                                          key=f"amt_{emp}", disabled=(not paid_yes))
            with d3:
                utr_val = st.text_input("UTR / Ref", value=(utr if paid_yes else ""),
                                        key=f"utr_{emp}", placeholder="UPI/NEFT ref", disabled=(not paid_yes))
            balance = total_due - (int(pay_amt) if paid_yes else 0)
            with d4:
                st.markdown(f'<div class="small-kv">Balance<br><b>{money(balance if balance>=0 else 0)}</b></div>', unsafe_allow_html=True)

            pending_total += max(balance, 0)

            # Capture state for bulk save
            edit_states[emp] = {
                "paid": paid_yes,
                "amount": int(pay_amt if paid_yes else 0),
                "paid_on": pay_date if paid_yes else None,
                "utr": utr_val if paid_yes else "",
                "components": {
                    "base_salary": comp["base_salary"],
                    "fuel_allow": comp["fuel_allow"],
                    "incentives": comp["incentives"],
                    "net_reimb": comp["net_reimb"],
                    "net_pay": comp["net_pay"],
                    "period": {"start": str(month_start), "end": str(month_end)}
                },
                "notes": f"Salary for {month_start.strftime('%b %Y')}",
                "carry_forward": cf,
            }

    st.info(f"**Total balance (unpaid across visible rows):** {money(int(pending_total))}")

    if is_admin and st.button("💾 Save all changes", use_container_width=True, type="primary"):
        for emp, payload in edit_states.items():
            amt = payload["amount"]
            # 1) allocate to previous months first
            used_prev = allocate_payment_to_previous(emp, month_key, amt)
            # 2) remainder goes to current month record
            remaining = max(amt - used_prev, 0)
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
        st.success("Saved all payment updates with carry-forward allocation.")
        st.rerun()

    st.divider()

# =============================
# MODE: SINGLE EMPLOYEE (detailed + CF + PDF)
# =============================
if mode == "Single employee":
    comp = calc_components(view_emp, month_start, month_end)
    cf_prev = previous_pending_amount(view_emp, month_key)
    total_due = comp["net_pay"] + cf_prev

    l, r = st.columns([2,1])
    with l:
        st.markdown(f"### Salary Slip — **{view_emp}**")
        st.write({"Month": month_start.strftime("%B %Y"), "Employee": view_emp})
    with r:
        st.metric("Total due (CF + This month)", money(total_due))

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Base salary", money(comp["base_salary"]))
    k2.metric("Fuel allowance", money(comp["fuel_allow"]))
    k3.metric("Incentives (month)", money(comp["incentives"]))
    k4.metric("Reimbursable expenses", money(comp["reimb_total"]))
    k5.metric("Less: Settlements (month)", money(comp["settled_this_month"]))
    k6.metric("Carry forward", money(cf_prev))
    st.caption(f"**Net Pay (this month)** = Base + Fuel + Incentives + Net reimbursement.  **Total due** = Carry-forward + Net Pay (this month).")
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
                .rename(columns={"Amount":"Amount (₹)"}).sort_values(["Date","Category"], ascending=[True,True]),
                use_container_width=True, hide_index=True
            )
    st.divider()

    # Printable + PDF
    with st.expander("Printable slip (summary)"):
        st.markdown(f"""
**Employee:** {view_emp}  
**Month:** {month_start.strftime("%B %Y")}  

- Base salary: {money(comp['base_salary'])}  
- Fuel allowance: {money(comp['fuel_allow'])}  
- Incentives (month): {money(comp['incentives'])}  
- Reimbursable expenses (month): {money(comp['reimb_total'])}  
- Less settlements (month): {money(comp['settled_this_month'])}  
- Carry forward (previous pending): {money(cf_prev)}  
- **Total due this cycle:** {money(total_due)}  
""")

    # Generate Employee PDF
    month_label = f"{month_start.strftime('%B-%Y')}"
    period_label = f"{month_start.strftime('%d-%B-%Y')} to {month_end.strftime('%d-%B-%Y')}"
    if st.button("📄 Generate Employee Salary PDF", key="emp_pdf_btn"):
        pdf_bytes = build_employee_pdf(
            emp=view_emp,
            month_label=month_label,
            period_label=period_label,
            comp=comp,
            carry_forward=cf_prev,
            total_due=total_due
        )
        st.session_state["emp_pdf"] = _as_bytes(pdf_bytes)
        st.success("PDF ready below.")

    if "emp_pdf" in st.session_state and st.session_state["emp_pdf"]:
        b = _as_bytes(st.session_state["emp_pdf"])
        st.markdown("#### Salary slip preview")
        b64s = base64.b64encode(b).decode()
        st.components.v1.html(
            f'<iframe src="data:application/pdf;base64,{b64s}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>',
            height=620,
        )
        st.download_button(
            "⬇️ Download Employee Salary Slip (PDF)",
            data=b,
            file_name=f"Employee_Salary_{view_emp}_{month_start.strftime('%Y_%m')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="emp_pdf_dl"
        )

    # Admin payment controls with Paid? toggle
    if is_admin:
        st.divider()
        st.subheader("💵 Mark/Update payment for this month")

        existing = load_payroll_record(view_emp, month_key)
        already_paid = bool(existing.get("paid", False))
        default_paid_on = pd.to_datetime(existing.get("paid_on")).date() if existing.get("paid_on") else date.today()
        default_amt     = int(existing.get("amount", 0))
        default_utr     = existing.get("utr","")
        default_notes   = existing.get("notes","")

        paid_choice = st.selectbox("Paid?", ["No","Yes"], index=(1 if already_paid else 0))
        paid_yes = (paid_choice == "Yes")

        p1, p2, p3 = st.columns([1,1,2])
        with p1:
            pay_amount = st.number_input("Amount paid (₹)", min_value=0, step=500,
                                         value=int(default_amt if paid_yes else 0),
                                         disabled=(not paid_yes))
        with p2:
            pay_date = st.date_input("Paid on", value=(default_paid_on if paid_yes else date.today()),
                                     disabled=(not paid_yes))
        with p3:
            utr = st.text_input("UTR / Ref", value=(default_utr if paid_yes else ""),
                                placeholder="UPI/NEFT reference", disabled=(not paid_yes))

        notes = st.text_area("Notes (optional)", value=default_notes, placeholder="e.g., Salary for Aug 2025")

        balance_preview = total_due - (int(pay_amount) if paid_yes else 0)
        st.metric("Balance (after this payment)", money(balance_preview if balance_preview >= 0 else 0))

        if st.button("💾 Save payment record", type="primary"):
            used_prev = allocate_payment_to_previous(view_emp, month_key, int(pay_amount if paid_yes else 0))
            remaining = max(int(pay_amount if paid_yes else 0) - used_prev, 0)
            save_or_update_pay(
                emp=view_emp,
                month_key=month_key,
                amount=remaining,
                paid_on=(pay_date if paid_yes else None),
                utr=(utr if paid_yes else ""),
                notes=notes,
                components={
                    "base_salary": comp["base_salary"],
                    "fuel_allow": comp["fuel_allow"],
                    "incentives": comp["incentives"],
                    "net_reimb": comp["net_reimb"],
                    "net_pay": comp["net_pay"],
                    "period": {"start": str(month_start), "end": str(month_end)}
                },
                paid_flag=paid_yes,
            )
            st.success("Payment saved with carry-forward allocation.")
            st.rerun()

        cur = load_payroll_record(view_emp, month_key)
        if cur:
            st.caption("Current payment record")
            rec = {
                "Paid?": "Yes" if cur.get("paid") else "No",
                "Paid on": pd.to_datetime(cur.get("paid_on")).date() if cur.get("paid_on") else None,
                "Amount paid (₹)": _to_int(cur.get("amount",0)),
                "UTR / Ref": cur.get("utr",""),
                "Notes": cur.get("notes",""),
                "Updated by": cur.get("updated_by",""),
                "Updated at": pd.to_datetime(cur.get("updated_at")).strftime("%Y-%m-%d %H:%M") if cur.get("updated_at") else "",
            }
            st.write(rec)

# =============================
# DRIVER SALARY (attendance) — view + make payment + PDF
# =============================
st.divider()
st.subheader("🚖 Driver Salary — Attendance based")

drv = st.selectbox("Driver", DRIVERS, index=0, key="drv_pick")
drv_month_pick = st.date_input("Driver month", value=date.today(), key="drv_month_pick")
drv_start, drv_end = month_bounds(drv_month_pick)
drv_month_key = _ym_key(drv_start)
st.caption(f"Driver period: **{drv_start} → {drv_end}**")

drv_calc = calc_driver_month(drv, drv_start, drv_end)
st.columns(6)[0].metric("Leaves", drv_calc["leave_days"])
st.columns(6)[1].metric("OT units", drv_calc["ot_units"])
st.columns(6)[2].metric("Leave deduction", money(drv_calc["leave_ded"]))
st.columns(6)[3].metric("Overtime (+)", money(drv_calc["overtime_amt"]))
st.columns(6)[4].metric("Advances (−)", money(drv_calc["advances"]))
st.columns(6)[5].metric("Net Pay", money(drv_calc["net_pay"]))

# Driver PDF
drv_month_label = f"{drv_start.strftime('%B-%Y')}"
drv_period_label = f"{drv_start.strftime('%d-%B-%Y')} to {drv_end.strftime('%d-%B-%Y')}"
if st.button("📄 Generate Driver Salary PDF", key="driver_pdf_btn"):
    pdf_b = build_driver_pdf(driver=drv, month_label=drv_month_label, period_label=drv_period_label, calc=drv_calc)
    st.session_state["driver_pdf"] = _as_bytes(pdf_b)
    st.success("Driver PDF ready below.")

if "driver_pdf" in st.session_state and st.session_state["driver_pdf"]:
    b = _as_bytes(st.session_state["driver_pdf"])
    st.markdown("#### Driver slip preview")
    b64s = base64.b64encode(b).decode()
    st.components.v1.html(
        f'<iframe src="data:application/pdf;base64,{b64s}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>',
        height=620,
    )
    st.download_button(
        "⬇️ Download Driver Salary Slip (PDF)",
        data=b,
        file_name=f"Driver_Salary_{drv}_{drv_start.strftime('%Y_%m')}.pdf",
        mime="application/pdf",
        use_container_width=True,
        key="driver_pdf_dl"
    )

# Make payment to driver (uses same col_payroll and carry-forward logic)
if is_admin:
    st.markdown("### 💵 Make payment to driver")

    drv_existing = load_payroll_record(drv, drv_month_key)
    drv_already_paid = bool(drv_existing.get("paid", False))
    drv_default_paid_on = pd.to_datetime(drv_existing.get("paid_on")).date() if drv_existing.get("paid_on") else date.today()
    drv_default_amt     = int(drv_existing.get("amount", 0))
    drv_default_utr     = drv_existing.get("utr","")
    drv_default_notes   = drv_existing.get("notes","")

    drv_cf_prev = previous_pending_amount(drv, drv_month_key)
    drv_total_due = drv_calc["net_pay"] + drv_cf_prev

    e1, e2, e3, e4 = st.columns([1,1,1,1.4])
    with e1:
        drv_paid_choice = st.selectbox("Paid?", ["No","Yes"], index=(1 if drv_already_paid else 0), key="drv_paid_choice")
        drv_paid_yes = (drv_paid_choice == "Yes")
    with e2:
        drv_pay_date = st.date_input("Paid on", value=(drv_default_paid_on if drv_paid_yes else date.today()),
                                     key="drv_pay_date", disabled=(not drv_paid_yes))
    with e3:
        drv_pay_amount = st.number_input("Amount paid (₹)", min_value=0, step=500,
                                         value=int(drv_default_amt if drv_paid_yes else 0),
                                         key="drv_pay_amount", disabled=(not drv_paid_yes))
    with e4:
        drv_utr = st.text_input("UTR / Ref", value=(drv_default_utr if drv_paid_yes else ""),
                                key="drv_utr", placeholder="UPI/NEFT ref", disabled=(not drv_paid_yes))

    st.metric("Driver total due (CF + Month)", money(drv_total_due))

    drv_notes = st.text_area("Notes (optional)", value=drv_default_notes, key="drv_notes",
                             placeholder=f"Driver salary for {drv_start.strftime('%b %Y')}")

    drv_balance_preview = drv_total_due - (int(drv_pay_amount) if drv_paid_yes else 0)
    st.metric("Balance (after this payment)", money(drv_balance_preview if drv_balance_preview >= 0 else 0))

    if st.button("💾 Save driver payment", type="primary", key="drv_save_payment"):
        used_prev = allocate_payment_to_previous(drv, drv_month_key, int(drv_pay_amount if drv_paid_yes else 0))
        remaining = max(int(drv_pay_amount if drv_paid_yes else 0) - used_prev, 0)

        save_or_update_pay(
            emp=drv,
            month_key=drv_month_key,
            amount=remaining,
            paid_on=(drv_pay_date if drv_paid_yes else None),
            utr=(drv_utr if drv_paid_yes else ""),
            notes=drv_notes,
            components={
                "base_salary": DRV_BASE,
                "leave_days": drv_calc["leave_days"],
                "leave_ded": drv_calc["leave_ded"],
                "ot_units": drv_calc["ot_units"],
                "overtime_amt": drv_calc["overtime_amt"],
                "advances": drv_calc["advances"],
                "net_pay": drv_calc["net_pay"],
                "period": {"start": str(drv_start), "end": str(drv_end)}
            },
            paid_flag=drv_paid_yes,
        )
        st.success("Driver payment saved with carry-forward allocation.")
        st.rerun()

# =============================
# Admin: Month payments table (for reference)
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
                "Paid?": "Yes" if r.get("paid") else "No",
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
