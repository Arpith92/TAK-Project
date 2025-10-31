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
    """Load users->PIN map from st.secrets or local .streamlit/secrets.toml (dev)."""
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

def _normalize_user_key(x: str) -> str:
    # Allow email-like usernames or names; compare case-insensitively
    return (x or "").strip()

def _login() -> Optional[str]:
    with st.sidebar:
        if st.session_state.get("user"):
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out"):
                # Clear and rerun safely
                for k in ["user", "login_user", "login_pin"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

    if st.session_state.get("user"):
        return st.session_state["user"]

    users_map_raw = load_users()
    if not users_map_raw:
        st.error("Login not configured. Add `mongo_uri` and a [users] table in Secrets.")
        st.stop()

    # Build a normalized map but preserve display keys
    users_display = list(users_map_raw.keys())
    users_normmap = { _normalize_user_key(k).lower(): str(v) for k,v in users_map_raw.items() }

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns(2)
    with c1:
        name = st.selectbox("User", users_display, key="login_user")
    with c2:
        pin = st.text_input("PIN", type="password", key="login_pin")
    if st.button("Sign in", use_container_width=True):
        key = _normalize_user_key(name).lower()
        expected = users_normmap.get(key, None)
        ok = (expected is not None) and (str(expected).strip() == str(pin).strip())
        if ok:
            st.session_state["user"] = name  # keep original display name
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid PIN")
            st.stop()
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
# Payroll helpers (multi-payment aware)
# =============================
def load_payroll_record(emp: str, month_key: str) -> dict:
    rec = col_payroll.find_one({"employee": emp, "month": month_key}, {"_id":0}) or {}
    # Backward compatibility: promote single fields to payments[]
    if "payments" not in rec:
        amt = _to_int(rec.get("amount", 0))
        if amt:
            rec["payments"] = [{
                "date": pd.to_datetime(rec.get("paid_on")).date() if rec.get("paid_on") else None,
                "amount": amt,
                "utr": rec.get("utr","")
            }]
        else:
            rec["payments"] = []
    return rec

def save_or_update_pay_multi(
    *, emp: str, month_key: str, payments: List[dict],
    notes: str, components: dict, paid_flag: bool,
    allocated_to_previous: int = 0
):
    """
    Saves the month record with a list of payments.
    - 'amount' field stores only the portion counted against THIS month (i.e., total - allocated_to_previous).
    - 'total_paid_raw' keeps the sum of all entered payments for traceability.
    """
    total_amt = sum(_to_int(p.get("amount", 0)) for p in payments)
    amount_current = max(total_amt - _to_int(allocated_to_previous, 0), 0)

    # Normalize dates to datetimes for Mongo
    norm_payments = []
    for p in payments:
        d = p.get("date")
        if isinstance(d, date) and not isinstance(d, datetime):
            d = datetime.combine(d, datetime.min.time())
        norm_payments.append({"date": d, "amount": _to_int(p.get("amount", 0)), "utr": (p.get("utr","") or "").strip()})

    payload = {
        "employee": emp,
        "month": month_key,
        "paid": bool(paid_flag),
        "payments": norm_payments,
        "total_paid_raw": total_amt,
        "allocated_to_previous": int(allocated_to_previous or 0),
        # compatibility fields (retain for older views):
        "amount": int(amount_current),
        "paid_on": norm_payments[-1]["date"] if norm_payments else None,
        "utr": norm_payments[-1]["utr"] if norm_payments else "",
        # metadata:
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
    """
    Sum of (net_pay - amount_applied_to_that_month) for all months < current_month_key.
    'amount' field in older records contains portion applied to that month.
    """
    cur = list(col_payroll.find(
        {"employee": emp, "month": {"$lt": current_month_key}},
        {"_id":0, "amount":1, "components":1}
    ))
    total_due = sum(_to_int((r.get("components") or {}).get("net_pay", 0)) for r in cur)
    total_paid = sum(_to_int(r.get("amount", 0)) for r in cur)
    return max(total_due - total_paid, 0)

def allocate_payment_to_previous(emp: str, current_month_key: str, amount: int) -> int:
    """
    Allocate an entered payment amount to oldest pending months first.
    Returns the portion that was allocated to previous months.
    """
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
# PDF Helpers (header + employee/driver slips)
# =============================

# ✅ Unicode-capable fonts (drop these under .streamlit/)
FONT_REG = ".streamlit/DejaVuSans.ttf"
FONT_BOLD = ".streamlit/DejaVuSans-Bold.ttf"  # optional but recommended

def _ascii_fallback(text: str) -> str:
    """
    Replace common non-ASCII symbols for safe output when Unicode font is not available.
    """
    s = (text or "")
    repl = {
        "₹": "Rs ",
        "®": "(R)",
        "™": "(TM)",
        "©": "(C)",
        "–": "-",   # en dash
        "—": "-",   # em dash
        "…": "...",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "•": "*",
        "°": " deg ",
        "→": "->",
        "\u00a0": " ",  # non-breaking space
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    # Strip any other stray non-ASCII by encoding-latin1 ignore
    try:
        s.encode("latin-1")
        return s
    except Exception:
        return s.encode("latin-1", errors="ignore").decode("latin-1", errors="ignore")

def inr_fmt_val(n, unicode_ok: bool) -> str:
    try:
        s = f"{int(round(float(n))):,}"
    except Exception:
        s = str(n)
    return (f"₹ {s}" if unicode_ok else f"Rs {s}")

ORG_BASE = {
    "title": "TravelaajKal® – Achala Holidays Pvt. Ltd.",
    "line1": "Mangrola, Ujjain, Madhya Pradesh 456006, India",
    "line2": "Email: travelaajkal@gmail.com  |  Web: www.travelaajkal.com  |  Mob: +91-7509612798",
}

def _org_strings(unicode_ok: bool) -> Dict[str,str]:
    if unicode_ok:
        return {
            **ORG_BASE,
            "footer_rights": f"All rights reserved by TravelaajKal {datetime.now().year}-{str(datetime.now().year+1)[-2:]}"
        }
    # ascii-safe for all lines
    return {
        "title": _ascii_fallback(ORG_BASE["title"]),
        "line1": _ascii_fallback(ORG_BASE["line1"]),
        "line2": _ascii_fallback(ORG_BASE["line2"]),
        "footer_rights": _ascii_fallback(f"All rights reserved by TravelaajKal {datetime.now().year}-{str(datetime.now().year+1)[-2:]}"),
    }

def ensure_font(pdf: FPDF):
    """
    Try to register Unicode fonts. If unavailable, use core Helvetica and enable ASCII fallback.
    Stores:
      pdf._unicode_ok: bool
      pdf._font_regular: str
      pdf._font_bold: Optional[str]  (None if not available)
    """
    pdf._unicode_ok = False
    pdf._font_regular = "Helvetica"
    pdf._font_bold = None

    reg_ok = os.path.exists(FONT_REG)
    bold_ok = os.path.exists(FONT_BOLD)

    if reg_ok:
        try:
            pdf.add_font("DejaVu", "", FONT_REG, uni=True)
            if bold_ok:
                pdf.add_font("DejaVu", "B", FONT_BOLD, uni=True)
            # try setting to ensure usable
            pdf.set_font("DejaVu", "", 11)
            pdf._unicode_ok = True
            pdf._font_regular = "DejaVu"
            pdf._font_bold = "DejaVu" if bold_ok else None
            return
        except Exception:
            # fall through to Helvetica
            pass

    # Fallback core font (ASCII only)
    pdf.set_font("Helvetica", "", 11)
    pdf._unicode_ok = False
    pdf._font_regular = "Helvetica"
    pdf._font_bold = None

class InvoiceHeaderPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        # Use ASCII-safe title if unicode font isn't loaded yet
        # (we'll set final fonts right after)
        self.set_title(_ascii_fallback("Salary Statement"))
        self.set_author(_ascii_fallback("Achala Holidays Pvt. Ltd."))
        ensure_font(self)
        self._org = _org_strings(self._unicode_ok)

    def _set_font(self, bold=False, size=11):
        if self._unicode_ok:
            if bold and self._font_bold:
                self.set_font(self._font_bold, "B", size)
            else:
                # If bold requested but no bold TTF, use regular to avoid raising
                self.set_font(self._font_regular, "", size)
        else:
            # Core Helvetica supports "B"
            self.set_font(self._font_regular, "B" if bold else "", size)

    def _safe_text(self, s: str) -> str:
        return s if self._unicode_ok else _ascii_fallback(s)

    def header(self):
        self.set_draw_color(150,150,150)
        self.rect(8, 8, 194, 281)
        if ORG_LOGO and os.path.exists(ORG_LOGO):
            try: self.image(ORG_LOGO, x=14, y=12, w=28)
            except Exception: pass
        self.set_xy(50, 12)
        self._set_font(bold=True, size=14)
        self.cell(0, 7, self._safe_text(self._org["title"]), align="C", ln=1)
        self._set_font(bold=False, size=10)
        self.cell(0, 6, self._safe_text(self._org["line1"]), align="C", ln=1)
        self.cell(0, 6, self._safe_text(self._org["line2"]), align="C", ln=1)
        self.ln(2); self.set_draw_color(0,0,0)
        self.line(12, self.get_y(), 198, self.get_y()); self.ln(4)

    def footer(self):
        self.set_y(-15)
        self._set_font(bold=False, size=8)
        self.cell(0, 5, self._safe_text(self._org["footer_rights"]), ln=1, align="C")

def build_employee_pdf(*, emp: str, month_label: str, period_label: str,
                       comp: dict, carry_forward: int, carry_forward_label: str,
                       total_due: int, payments: List[dict]) -> bytes:
    pdf = InvoiceHeaderPDF()
    pdf.add_page()

    left = 16
    th = 8
    col1_w, col2_w = 120, 66

    def text_part(s: str) -> str:
        return s if pdf._unicode_ok else _ascii_fallback(s)

    pdf._set_font(bold=False, size=11)
    pdf.set_x(left); pdf.cell(0, 6, text_part(f"{month_label} (Salary Statement: {period_label})"), ln=1)
    pdf.ln(1)
    pdf._set_font(bold=True, size=11)
    pdf.set_x(left); pdf.cell(0, 6, text_part(f"EMP NAME:  {emp}"), ln=1)
    pdf.ln(2)

    def header_row():
        y = pdf.get_y()
        pdf._set_font(bold=True, size=10)
        pdf.rect(left, y, col1_w, th)
        pdf.rect(left + col1_w, y, col2_w, th)
        pdf.text(left + 2, y + th - 2, text_part("Particulars"))
        pdf.text(left + col1_w + 2, y + th - 2, text_part("Amount"))
        pdf.ln(th)
        pdf._set_font(bold=False, size=10)

    def row(label: str, amount):
        y = pdf.get_y()
        pdf.rect(left, y, col1_w, th)
        pdf.rect(left + col1_w, y, col2_w, th)
        pdf.text(left + 2, y + th - 2, text_part(str(label)))
        amt_str = inr_fmt_val(amount, pdf._unicode_ok)
        pdf.set_xy(left + col1_w, y); pdf.cell(col2_w - 2, th, amt_str, align="R")
        pdf.ln(th)

    header_row()
    row("Base salary", comp["base_salary"])
    row("Fuel allowance", comp["fuel_allow"])
    row("Incentives", comp["incentives"])
    row("Reimbursable expenses", comp["reimb_total"])
    row("Less: Settlements (this month)", -comp["settled_this_month"])
    row("Cash received (deduction)", -comp["cash_received"])
    row(f"Carry forward ({carry_forward_label})", carry_forward)

    pdf.ln(2)
    y = pdf.get_y()
    pdf._set_font(bold=True, size=11)
    pdf.rect(left, y, col1_w, th)
    pdf.rect(left + col1_w, y, col2_w, th)
    pdf.text(left + 2, y + th - 2, text_part("Total Due"))
    pdf.set_xy(left + col1_w, y); pdf.cell(col2_w - 2, th, inr_fmt_val(total_due, pdf._unicode_ok), align="R")
    pdf.ln(th + 6)

    # ---- Payments section
    pdf._set_font(bold=True, size=10)
    pdf.set_x(left); pdf.cell(0, 6, text_part("Payments made"), ln=1)
    pdf._set_font(bold=False, size=9)
    total_paid = 0
    if not payments:
        pdf.set_x(left); pdf.cell(0, 5, text_part("No payments recorded."), ln=1)
    else:
        for p in payments:
            d = p.get("date")
            if isinstance(d, datetime): d = d.date()
            d_str = d.strftime("%d-%b-%Y") if isinstance(d, date) else str(d or "")
            amt = _to_int(p.get("amount", 0))
            total_paid += amt
            line = f"{d_str} | Amount: {amt:,} | UTR: {p.get('utr','')}"
            pdf.set_x(left); pdf.multi_cell(0, 5, text_part(line))

    balance = total_due - total_paid
    pdf.ln(1)
    pdf._set_font(bold=True, size=10)
    pdf.set_x(left); pdf.cell(0, 6, text_part(f"Total Paid: {inr_fmt_val(total_paid, pdf._unicode_ok)}"), ln=1)
    pdf.set_x(left); pdf.cell(0, 6, text_part(f"Pending Balance: {inr_fmt_val(balance, pdf._unicode_ok)}"), ln=1)

    pdf.ln(6)
    pdf._set_font(bold=False, size=9)
    pdf.multi_cell(0, 5, text_part("Note: This is a computer-generated statement."))

    pdf.ln(6)
    sig_w = 50
    sig_x = pdf.w - 16 - sig_w
    sig_y = pdf.get_y()
    if ORG_SIGN and os.path.exists(ORG_SIGN):
        try: pdf.image(ORG_SIGN, x=sig_x, y=sig_y, w=sig_w)
        except Exception: pass
    pdf.set_xy(sig_x, sig_y + 18)
    pdf._set_font(bold=False, size=10)
    pdf.cell(sig_w, 6, text_part("Authorised Signatory"), ln=1, align="C")

    out = pdf.output(dest="S")

# Normalize to pure bytes (Streamlit's download_button needs bytes or str)
if isinstance(out, str):
    out_bytes = out.encode("latin-1", errors="ignore")
elif isinstance(out, bytearray):
    out_bytes = bytes(out)
elif isinstance(out, bytes):
    out_bytes = out
else:
    # Last resort: try bytes() constructor
    out_bytes = bytes(out)

return out_bytes


def build_driver_pdf(*, driver: str, month_label: str, period_label: str, calc: dict) -> bytes:
    pdf = InvoiceHeaderPDF()
    pdf.add_page()

    left = 16
    th = 8
    col1_w, col2_w = 120, 66

    def text_part(s: str) -> str:
        return s if pdf._unicode_ok else _ascii_fallback(s)

    pdf._set_font(bold=False, size=11)
    pdf.set_x(left); pdf.cell(0, 6, text_part(f"{month_label} (Driver Salary Statement: {period_label})"), ln=1)
    pdf.ln(1)
    pdf._set_font(bold=True, size=11)
    pdf.set_x(left); pdf.cell(0, 6, text_part(f"DRIVER NAME:  {driver}"), ln=1)
    pdf.ln(2)

    def header_row():
        y = pdf.get_y()
        pdf._set_font(bold=True, size=10)
        pdf.rect(left, y, col1_w, th)
        pdf.rect(left + col1_w, y, col2_w, th)
        pdf.text(left + 2, y + th - 2, text_part("Particulars"))
        pdf.text(left + col1_w + 2, y + th - 2, text_part("Amount"))
        pdf.ln(th)
        pdf._set_font(bold=False, size=10)

    def row(label: str, amount):
        y = pdf.get_y()
        pdf.rect(left, y, col1_w, th)
        pdf.rect(left + col1_w, y, col2_w, th)
        pdf.text(left + 2, y + th - 2, text_part(str(label)))
        amt_str = inr_fmt_val(amount, pdf._unicode_ok)
        pdf.set_xy(left + col1_w, y); pdf.cell(col2_w - 2, th, amt_str, align="R")
        pdf.ln(th)

    header_row()
    row("Base", DRV_BASE)
    row("Less: Leave deduction", -calc["leave_ded"])
    row("Add: Overtime (units)", calc["overtime_amt"])
    row("Less: Advances", -calc["advances"])

    pdf.ln(2)
    y = pdf.get_y()
    pdf._set_font(bold=True, size=11)
    pdf.rect(left, y, col1_w, th)
    pdf.rect(left + col1_w, y, col2_w, th)
    pdf.text(left + 2, y + th - 2, text_part("Net Pay"))
    pdf.set_xy(left + col1_w, y); pdf.cell(col2_w - 2, th, inr_fmt_val(calc["net_pay"], pdf._unicode_ok), align="R")
    pdf.ln(th + 10)

    pdf._set_font(bold=False, size=9)
    pdf.multi_cell(0, 5, text_part("Note: This is a computer-generated statement."))

    pdf.ln(6)
    sig_w = 50
    sig_x = pdf.w - 16 - sig_w
    sig_y = pdf.get_y()
    if ORG_SIGN and os.path.exists(ORG_SIGN):
        try: pdf.image(ORG_SIGN, x=sig_x, y=sig_y, w=sig_w)
        except Exception: pass
    pdf.set_xy(sig_x, sig_y + 18)
    pdf._set_font(bold=False, size=10)
    pdf.cell(sig_w, 6, text_part("Authorised Signatory"), ln=1, align="C")

    out = pdf.output(dest="S")

# Normalize to pure bytes (Streamlit's download_button needs bytes or str)
if isinstance(out, str):
    out_bytes = out.encode("latin-1", errors="ignore")
elif isinstance(out, bytearray):
    out_bytes = bytes(out)
elif isinstance(out, bytes):
    out_bytes = out
else:
    # Last resort: try bytes() constructor
    out_bytes = bytes(out)

return out_bytes


# =============================
# UI: Month selection + modes
# =============================
@st.cache_data(ttl=TTL, show_spinner=False)
def _employees_cached() -> List[str]:
    return all_employees()

emp_opts = _employees_cached()
if is_admin:
    mode = st.radio("View mode", ["Single employee", "All employees (overview)"], horizontal=True)
    view_emp = st.selectbox("View employee", emp_opts, index=(emp_opts.index(user) if user in emp_opts else 0)) if mode == "Single employee" else None
else:
    mode = "Single employee"
    view_emp = user

month_pick = st.date_input("Slip month", value=date.today())
month_start, month_end = month_bounds(month_pick)
month_key = _ym_key(month_start)
prev_month_label = (month_start - timedelta(days=1)).strftime("%B %Y")
st.caption(f"Period: **{month_start} → {month_end}**")

# =============================
# MODE: ALL EMPLOYEES
# =============================
if mode == "All employees (overview)":
    st.subheader(f"👥 Team overview — {month_start.strftime('%B %Y')}")
    pending_total = 0

    for emp in emp_opts:
        comp = calc_components(emp, month_start, month_end)
        payrec = load_payroll_record(emp, month_key)

        cf = previous_pending_amount(emp, month_key)  # auto carry-forward from older months
        total_due = comp["net_pay"] + cf

        # Initialize session-based payment list
        session_key = f"payments_{emp}_{month_key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = payrec.get("payments", []) or [
                {"date": date.today(), "amount": 0, "utr": ""}
            ]
        payments_list = st.session_state[session_key]

        with st.container(border=True):
            # Summary metrics header
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.2,1,1,1,1,1.2,1.2,1.2])
            c1.markdown(f"**{emp}**")
            c2.metric("Base", money(comp["base_salary"]))
            c3.metric("Fuel", money(comp["fuel_allow"]))
            c4.metric("Incent.", money(comp["incentives"]))
            c5.metric("Reimb", money(comp["net_reimb"]))
            c6.metric("Cash recv", money(comp["cash_received"]))
            c7.metric("Net Pay", money(comp["net_pay"]))
            c8.metric(f"Carry Fwd ({prev_month_label})", money(cf))

            st.write("#### Payment Entries")

            # --- Display existing rows ---
            to_delete = []
            for i, p in enumerate(payments_list):
                d1, d2, d3, d4 = st.columns([1,1,1,0.3])
                pay_date = d1.date_input(
                    "Paid on",
                    value=pd.to_datetime(p.get("date", date.today())).date(),
                    key=f"date_{emp}_{i}",
                )
                pay_amt = d2.number_input(
                    "Amt Paid",
                    min_value=0,
                    step=500,
                    value=_to_int(p.get("amount", 0)),
                    key=f"amt_{emp}_{i}",
                )
                utr_val = d3.text_input(
                    "UTR",
                    value=p.get("utr", ""),
                    key=f"utr_{emp}_{i}",
                )
                if d4.button("❌", key=f"del_{emp}_{i}"):
                    to_delete.append(i)

                payments_list[i] = {"date": pay_date, "amount": pay_amt, "utr": utr_val}

            # Remove deleted rows
            for i in sorted(to_delete, reverse=True):
                del payments_list[i]
            st.session_state[session_key] = payments_list

            # Add new row
            if st.button(f"➕ Add payment row {emp}", key=f"add_{emp}"):
                payments_list.append({"date": date.today(), "amount": 0, "utr": ""})
                st.session_state[session_key] = payments_list
                st.rerun()

            # Totals
            total_paid = sum(_to_int(p.get("amount", 0)) for p in payments_list)
            balance = total_due - total_paid
            pending_total += max(balance, 0)
            st.caption(
                f"Total Paid: {money(total_paid)} | Balance after payment: {money(balance)}"
            )

            # Save
            if st.button(f"💾 Save {emp}", key=f"save_{emp}"):
                used_prev = allocate_payment_to_previous(emp, month_key, total_paid)
                save_or_update_pay_multi(
                    emp=emp,
                    month_key=month_key,
                    payments=payments_list,
                    notes=f"Salary {month_key}",
                    components=comp,
                    paid_flag=(total_paid > 0),
                    allocated_to_previous=used_prev,
                )
                st.success(f"Saved {emp} payments.")
                st.rerun()

    st.info(f"**Total pending balance (after entered payments):** {money(int(pending_total))}")

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
    k7.metric(f"Carry fwd ({prev_month_label})", money(cf_prev))

    # Editable multi-payment block (admin only)
    existing = load_payroll_record(view_emp, month_key)

    if is_admin:
        st.write("#### Payment Entries")

        session_key_single = f"payments_single_{view_emp}_{month_key}"
        if session_key_single not in st.session_state:
            st.session_state[session_key_single] = existing.get("payments", []) or [
                {"date": date.today(), "amount": 0, "utr": ""}
            ]
        payments_single = st.session_state[session_key_single]

        # Render rows
        to_delete = []
        for i, p in enumerate(payments_single):
            d1, d2, d3, d4 = st.columns([1,1,1,0.3])
            pay_date = d1.date_input(
                "Paid on",
                value=pd.to_datetime(p.get("date", date.today())).date(),
                key=f"single_date_{i}"
            )
            pay_amt  = d2.number_input(
                "Amt Paid",
                min_value=0, step=500,
                value=_to_int(p.get("amount", 0)),
                key=f"single_amt_{i}"
            )
            utr_val  = d3.text_input("UTR", value=p.get("utr",""), key=f"single_utr_{i}")
            if d4.button("❌", key=f"single_del_{i}"):
                to_delete.append(i)
            payments_single[i] = {"date": pay_date, "amount": pay_amt, "utr": utr_val}

        for i in sorted(to_delete, reverse=True):
            del payments_single[i]
        st.session_state[session_key_single] = payments_single

        if st.button("➕ Add payment row", key="single_add_row"):
            payments_single.append({"date": date.today(), "amount": 0, "utr": ""})
            st.session_state[session_key_single] = payments_single
            st.rerun()

        total_paid_single = sum(_to_int(p.get("amount", 0)) for p in payments_single)
        balance_single = total_due - total_paid_single
        st.caption(f"Total Paid: {money(total_paid_single)}  |  Balance after payment: {money(balance_single)}")

        if st.button("💾 Save payment(s)", key="single_save"):
            used_prev = allocate_payment_to_previous(view_emp, month_key, total_paid_single)
            save_or_update_pay_multi(
                emp=view_emp,
                month_key=month_key,
                payments=payments_single,
                notes=existing.get("notes","") or f"Salary {month_key}",
                components=comp,
                paid_flag=(total_paid_single > 0),
                allocated_to_previous=used_prev
            )
            st.success("Saved payments.")
            st.rerun()

    # PDF download
    if st.button("📄 Generate Salary PDF"):
        # reload (after any edits) to ensure we pass current payments to the PDF
        cur = load_payroll_record(view_emp, month_key)
        pdf_bytes = build_employee_pdf(
            emp=view_emp,
            month_label=month_start.strftime("%B-%Y"),
            period_label=f"{month_start} → {month_end}",
            comp=comp,
            carry_forward=cf_prev,
            carry_forward_label=prev_month_label,
            total_due=total_due,
            payments=cur.get("payments", [])
        )
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"Salary_{view_emp}_{month_key}.pdf",
            mime="application/pdf"
        )

# =============================
# DRIVER SECTION
# =============================
st.divider()
st.subheader("🚖 Driver Salary")
drv = st.selectbox("Driver", DRIVERS, index=0)
drv_month = st.date_input("Driver month", value=date.today())
drv_start, drv_end = month_bounds(drv_month)
drv_calc = calc_driver_month(drv, drv_start, drv_end)

c1, c2, c3 = st.columns(3)
c1.metric("Net Pay", money(drv_calc["net_pay"]))
c2.metric("Leaves", drv_calc["leave_days"])
c3.metric("OT Units", drv_calc["ot_units"])

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
        # Expand payments into a readable summary column
        def summarize_payments(row):
            ps = row.get("payments", []) or []
            parts = []
            for p in ps:
                d = p.get("date")
                try:
                    if isinstance(d, str):
                        d = pd.to_datetime(d, errors="coerce")
                    if isinstance(d, (pd.Timestamp, datetime)):
                        d = d.date()
                except Exception:
                    pass
                d_str = d.strftime("%d-%b") if isinstance(d, date) else ""
                parts.append(f"{d_str}:{_to_int(p.get('amount',0)):,} ({p.get('utr','')})")
            return " | ".join(parts)

        dfp["Payments"] = dfp.apply(summarize_payments, axis=1)
        dfp["Paid?"] = dfp["paid"].map({True: "Yes", False: "No"})
        dfp["Paid on"] = pd.to_datetime(dfp.get("paid_on"), errors="coerce").dt.date

        # Fill missing optional fields so KeyError never occurs
        for col in ["amount", "allocated_to_previous", "total_paid_raw", "notes", "updated_by"]:
            if col not in dfp.columns:
                dfp[col] = None

        shown = dfp[[
            "employee", "Paid?", "Paid on",
            "amount", "allocated_to_previous", "total_paid_raw",
            "Payments", "notes", "updated_by"
        ]].rename(columns={
            "amount": "Amount (this month)",
            "allocated_to_previous": "Allocated to previous",
            "total_paid_raw": "Total paid (all rows)"
        })

        st.dataframe(shown, use_container_width=True, hide_index=True)
    else:
        st.info("No records yet for this month.")
