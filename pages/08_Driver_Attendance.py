# pages/08_Driver_Attendance.py
from __future__ import annotations

# ==============================
# Imports & setup
# ==============================
import os, base64
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional, Tuple, Dict

import pandas as pd
import streamlit as st
from pymongo import MongoClient
from bson import ObjectId
from fpdf import FPDF  # fpdf2

# Page
st.set_page_config(page_title="Driver Attendance & Salary", layout="wide")
st.title("🚖 Driver Attendance & Salary")

TTL = 90  # short cache keeps UI snappy

# --- Freeze headers for dataframes/editors ---
st.markdown("""
<style>
/* Make headers sticky inside data editor / dataframe scroll areas */
div[data-testid="stDataEditor"] thead tr th,
div[data-testid="stDataFrame"] thead tr th {
  position: sticky; top: 0; z-index: 2; background: var(--background-color,#fff);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Admin gate
# ==============================
def require_admin() -> bool:
    ADMIN_PASS_DEFAULT = "Arpith&92"
    ADMIN_PASS = str(st.secrets.get("admin_pass", ADMIN_PASS_DEFAULT))
    with st.sidebar:
        st.markdown("### Admin access")
        p = st.text_input("Enter admin password", type="password", placeholder="enter pass", key="adm_pw")
    ok = ((p or "").strip() == ADMIN_PASS.strip())
    st.session_state["is_admin"] = ok
    st.session_state["user"] = "Admin" if ok else st.session_state.get("user", "Driver")
    return ok

is_admin = require_admin()

# ==============================
# Mongo
# ==============================
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
        st.error("Mongo connection is not configured. Add `mongo_uri` in Secrets.")
        st.stop()
    cli = MongoClient(uri, appName="TAK_DriverAttendance", serverSelectionTimeoutMS=8000, connectTimeoutMS=8000, tz_aware=True)
    cli.admin.command("ping")
    return cli

DB = _get_client()["TAK_DB"]
col_att = DB["driver_attendance"]
col_adv = DB["driver_advances"]
col_updates = DB["package_updates"]
col_itins   = DB["itineraries"]

# optional audit
try:
    from tak_audit import audit_pageview
    audit_pageview(st.session_state.get("user", "Unknown"), page="08_Driver_Attendance")
except Exception:
    pass

# ==============================
# Constants / helpers
# ==============================
DRIVERS = ["Priyansh"]  # extend later
CARS = ["TAK Sedan", "TAK Ertiga"]
BASE_SALARY = 12000
LEAVE_DEDUCT = 400
OVERTIME_ADD = 300

ORG_LOGO = ".streamlit/logo.png"
ORG_SIGN = ".streamlit/signature.png"

def _to_int(x, default=0) -> int:
    try:
        if x is None: return default
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def _d(x) -> Optional[date]:
    try:
        if x is None or pd.isna(x): return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

def month_bounds(d: date) -> Tuple[date, date]:
    first = d.replace(day=1)
    next_first = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    last = next_first - timedelta(days=1)
    return first, last

def inr(n: int) -> str:
    return f"₹ {int(n):,}"

def _as_bytes(x) -> bytes:
    if x is None:
        return b""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, memoryview):
        return x.tobytes()
    if isinstance(x, str):
        return x.encode("latin-1", errors="ignore")
    try:
        return bytes(x)
    except Exception:
        return str(x).encode("latin-1", errors="ignore")

# ==============================
# Confirmed customers (ACH/Name) for linking
# ==============================
@st.cache_data(ttl=TTL, show_spinner=False)
def load_confirmed_customers() -> pd.DataFrame:
    ups = list(col_updates.find({"status": "confirmed"}, {"_id":0, "itinerary_id":1}))
    if not ups:
        return pd.DataFrame(columns=["itinerary_id","ach_id","client_name","client_mobile"])
    iids = [str(u.get("itinerary_id") or "") for u in ups if u.get("itinerary_id")]
    it_rows = []
    if iids:
        obj_ids = [ObjectId(i) for i in iids if len(i) == 24]
        if obj_ids:
            for r in col_itins.find({"_id": {"$in": obj_ids}}, {"ach_id":1,"client_name":1,"client_mobile":1}):
                it_rows.append({
                    "itinerary_id": str(r["_id"]),
                    "ach_id": r.get("ach_id",""),
                    "client_name": r.get("client_name",""),
                    "client_mobile": r.get("client_mobile","")
                })
        str_ids = [i for i in iids if len(i) != 24]
        if str_ids:
            for r in col_itins.find({"itinerary_id": {"$in": str_ids}},
                                    {"_id":1,"itinerary_id":1,"ach_id":1,"client_name":1,"client_mobile":1}):
                it_rows.append({
                    "itinerary_id": str(r.get("itinerary_id") or r.get("_id")),
                    "ach_id": r.get("ach_id",""),
                    "client_name": r.get("client_name",""),
                    "client_mobile": r.get("client_mobile","")
                })
    if not it_rows:
        return pd.DataFrame(columns=["itinerary_id","ach_id","client_name","client_mobile"])
    df = pd.DataFrame(it_rows).drop_duplicates(subset=["itinerary_id"]).reset_index(drop=True)
    for c in ("ach_id","client_name","client_mobile","itinerary_id"):
        df[c] = df[c].fillna("").astype(str)
    return df

def customer_pick_options() -> tuple[list[str], Dict[str, Dict[str, str]]]:
    """
    Build dropdown options and a reverse map:
    label = "Name — Mobile | ACH | ItineraryID"
    """
    confirmed = load_confirmed_customers()
    options = ["", "➕ Custom / Other"]
    rev_map: Dict[str, Dict[str, str]] = {}
    if confirmed.empty:
        return options, rev_map
    tmp = confirmed.copy()
    tmp["label"] = (
        tmp["client_name"].str.strip() + " — " + tmp["client_mobile"].str.strip()
        + " | " + tmp["ach_id"].fillna("").astype(str).str.strip()
        + " | " + tmp["itinerary_id"].astype(str).str.strip()
    )
    for _, r in tmp.iterrows():
        lbl = r["label"]
        options.append(lbl)
        rev_map[lbl] = {
            "cust_name": r.get("client_name",""),
            "cust_ach_id": r.get("ach_id",""),
            "cust_itinerary_id": r.get("itinerary_id",""),
        }
    return options, rev_map

# ==============================
# Cached loaders
# ==============================
@st.cache_data(ttl=TTL, show_spinner=False)
def load_attendance(driver: str, start: date, end: date) -> pd.DataFrame:
    cur = col_att.find(
        {"driver": driver, "date": {"$gte": datetime.combine(start, dtime.min),
                                    "$lte": datetime.combine(end,   dtime.max)}},
        {"_id":0}
    ).sort("date", 1)
    rows = []
    for r in cur:
        rows.append({
            "date": _d(r.get("date")),
            "driver": r.get("driver",""),
            "car": r.get("car",""),
            "in_time": r.get("in_time",""),
            "out_time": r.get("out_time",""),
            "status": r.get("status","Present"),
            "outstation_overnight": bool(r.get("outstation_overnight", False)),
            "overnight_client": bool(r.get("overnight_client", False)),
            "overnight_client_name": r.get("overnight_client_name",""),
            "bhasmarathi": bool(r.get("bhasmarathi", False)),
            "bhas_client_name": r.get("bhas_client_name",""),
            "notes": r.get("notes",""),
            "cust_itinerary_id": r.get("cust_itinerary_id",""),
            "cust_ach_id": r.get("cust_ach_id",""),
            "cust_name": r.get("cust_name",""),
            "cust_is_custom": bool(r.get("cust_is_custom", False)),
            "billable_salary": _to_int(r.get("billable_salary", 0)),
            "billable_ot_units": _to_int(r.get("billable_ot_units", 0)),
            "billable_ot_amount": _to_int(r.get("billable_ot_amount", 0)),
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "date","driver","car","in_time","out_time","status",
        "outstation_overnight","overnight_client","overnight_client_name",
        "bhasmarathi","bhas_client_name","notes",
        "cust_itinerary_id","cust_ach_id","cust_name","cust_is_custom",
        "billable_salary","billable_ot_units","billable_ot_amount"
    ])
    if not df.empty:
        df["billable_ot_amount"] = df["billable_ot_units"].fillna(0).astype(int) * OVERTIME_ADD
    return df

@st.cache_data(ttl=TTL, show_spinner=False)
def load_advances(driver: str, start: date, end: date) -> pd.DataFrame:
    cur = col_adv.find(
        {"driver": driver, "date": {"$gte": datetime.combine(start, dtime.min),
                                    "$lte": datetime.combine(end,   dtime.max)}},
        {"_id":0}
    ).sort("date", 1)
    rows = [{"date": _d(r.get("date")), "driver": r.get("driver",""),
             "amount": _to_int(r.get("amount",0)), "note": r.get("note","")} for r in cur]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date","driver","amount","note"])

# ==============================
# DB upserts
# ==============================
def upsert_attendance(
    *, driver: str, day: date, car: str,
    in_time: str, out_time: str, status: str,
    outstation_overnight: bool, overnight_client: bool, overnight_client_name: str,
    bhasmarathi: bool, bhas_client_name: str, notes: str,
    cust_itinerary_id: str, cust_ach_id: str, cust_name: str, cust_is_custom: bool,
    billable_salary: int, billable_ot_units: int
):
    key_dt = datetime.combine(day, dtime.min)
    billable_ot_amount = int(billable_ot_units) * OVERTIME_ADD
    payload = {
        "driver": driver,
        "date": key_dt,
        "car": car,
        "in_time": in_time,
        "out_time": out_time,
        "status": status,
        "outstation_overnight": bool(outstation_overnight),
        "overnight_client": bool(overnight_client),
        "overnight_client_name": overnight_client_name or "",
        "bhasmarathi": bool(bhasmarathi),
        "bhas_client_name": bhas_client_name or "",
        "notes": notes or "",
        "cust_itinerary_id": str(cust_itinerary_id or ""),
        "cust_ach_id": cust_ach_id or "",
        "cust_name": cust_name or "",
        "cust_is_custom": bool(cust_is_custom),
        "billable_salary": int(billable_salary or 0),
        "billable_ot_units": int(billable_ot_units or 0),
        "billable_ot_amount": int(billable_ot_amount),
        "updated_at": datetime.utcnow(),
    }
    col_att.update_one({"driver": driver, "date": key_dt}, {"$set": payload}, upsert=True)

def add_advance(*, driver: str, day: date, amount: int, note: str):
    col_adv.insert_one({
        "driver": driver,
        "date": datetime.combine(day, dtime.min),
        "amount": int(amount),
        "note": note or "",
        "created_at": datetime.utcnow(),
    })

def bulk_upsert_range(
    *, driver: str, start: date, end: date,
    status: str, mark_outstation: bool, mark_overnight_client: bool, mark_bhas: bool
):
    cur = start
    while cur <= end:
        upsert_attendance(
            driver=driver, day=cur, car="", in_time="", out_time="", status=status,
            outstation_overnight=mark_outstation,
            overnight_client=mark_overnight_client,
            overnight_client_name="",
            bhasmarathi=mark_bhas, bhas_client_name="",
            notes="(bulk update)",
            cust_itinerary_id="", cust_ach_id="", cust_name="", cust_is_custom=False,
            billable_salary=0, billable_ot_units=0
        )
        cur += timedelta(days=1)

# ==============================
# Salary calculator
# ==============================
def calc_salary(df_att: pd.DataFrame, df_adv: pd.DataFrame, month_start: date, month_end: date) -> dict:
    days_in_month = (month_end - month_start).days + 1
    leave_days = 0 if df_att.empty else int((df_att["status"] == "Leave").sum())
    ot_units = 0
    if not df_att.empty:
        ot_units += int(df_att["outstation_overnight"].fillna(False).sum())
        ot_units += int(df_att["overnight_client"].fillna(False).sum())
        ot_units += int(df_att["bhasmarathi"].fillna(False).sum())

    leave_ded = leave_days * LEAVE_DEDUCT
    overtime_amt = ot_units * OVERTIME_ADD
    advances = 0 if df_adv.empty else int(df_adv["amount"].sum())

    gross = BASE_SALARY - leave_ded + overtime_amt
    net = gross - advances

    return {
        "days_in_month": days_in_month,
        "leave_days": leave_days,
        "ot_units": ot_units,
        "leave_ded": leave_ded,
        "overtime_amt": overtime_amt,
        "advances": advances,
        "gross": gross,
        "net": net,
    }

# ==============================
# PDF Slip (ASCII-safe)
# ==============================
def _ascii_downgrade(s: str) -> str:
    if s is None:
        return ""
    return (str(s)
        .replace("₹", "Rs ")
        .replace("—", "-").replace("–", "-").replace("•", "-")
        .replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        .replace("™", "(TM)")
    )

ORG = {
    "title": "TravelaajKal® – Achala Holidays Pvt. Ltd.",
    "line1": "Mangrola, Ujjain, Madhya Pradesh 456006, India",
    "line2": "Email: travelaajkal@gmail.com  |  Web: www.travelaajkal.com  |  Mob: +91-7509612798",
    "footer_rights": f"All rights reserved by TravelaajKal {datetime.now().year}-{str(datetime.now().year+1)[-2:]}"
}

class SalaryPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_title("Driver Salary Statement")
        self.set_author("Achala Holidays Pvt. Ltd.")
        self.set_creator("TravelaajKal – Streamlit")
        self.set_subject("Salary Statement")

    def _txt(self, s: str) -> str:
        return _ascii_downgrade(s)

    def header(self):
        self.set_draw_color(150,150,150)
        self.rect(8, 8, 194, 281)
        if ORG_LOGO and os.path.exists(ORG_LOGO):
            try:
                self.image(ORG_LOGO, x=14, y=12, w=28)
            except Exception:
                pass
        self.set_xy(50, 12)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 7, self._txt(ORG["title"]), align="C", ln=1)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, self._txt(ORG["line1"]), align="C", ln=1)
        self.cell(0, 6, self._txt(ORG["line2"]), align="C", ln=1)
        self.ln(2)
        self.set_draw_color(0,0,0)
        self.line(12, self.get_y(), 198, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, self._txt(ORG["footer_rights"]), ln=1, align="C")

def inr_ascii(n) -> str:
    try:
        return f"Rs {int(round(float(n))):,}"
    except Exception:
        return f"Rs {n}"

def build_salary_pdf(*, emp_name: str, month_label: str, period_label: str, calc: dict) -> bytes:
    BASE_SALARY_LINE = 12000
    days_in_month = int(calc.get("days_in_month", 0))
    leave_days    = int(calc.get("leave_days", 0))
    leave_ded     = int(calc.get("leave_ded", 0) or calc.get("leave_deduction", 0))
    ot_units      = int(calc.get("ot_units", 0))
    ot_amount     = int(calc.get("overtime_amt", 0) or calc.get("ot_amount", 0))
    advances      = int(calc.get("advances", 0))
    net           = int(calc.get("net", 0))

    pdf = SalaryPDF()
    pdf.add_page()

    left = 16
    th = 8
    col1_w, col2_w, col3_w = 88, 40, 58

    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(left)
    pdf.cell(0, 6, pdf._txt(f"{month_label} (Salary Statement: {period_label})"), ln=1)
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(left)
    pdf.cell(0, 6, pdf._txt(f"EMP NAME:  {emp_name}"), ln=1)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    y = pdf.get_y()
    pdf.rect(left, y, col1_w, th)
    pdf.rect(left + col1_w, y, col2_w, th)
    pdf.rect(left + col1_w + col2_w, y, col3_w, th)
    pdf.text(left + 2, y + th - 2, pdf._txt("Particulars"))
    pdf.text(left + col1_w + 2, y + th - 2, pdf._txt("Days/Units"))
    pdf.text(left + col1_w + col2_w + 2, y + th - 2, pdf._txt("Amount"))
    pdf.ln(th)

    pdf.set_font("Helvetica", "", 10)

    def row(label: str, units, amount):
        y = pdf.get_y()
        pdf.rect(left, y, col1_w, th)
        pdf.rect(left + col1_w, y, col2_w, th)
        pdf.rect(left + col1_w + col2_w, y, col3_w, th)
        pdf.text(left + 2, y + th - 2, pdf._txt(label))
        pdf.set_xy(left + col1_w, y); pdf.cell(col2_w - 2, th, pdf._txt(str(units)), align="R")
        pdf.set_xy(left + col1_w + col2_w, y); pdf.cell(col3_w - 2, th, pdf._txt(inr_ascii(amount)), align="R")
        pdf.ln(th)

    row("Total Days in Month", days_in_month, 0)
    row("Salary", "-", BASE_SALARY_LINE)
    row("Total Leave", leave_days, leave_ded)
    row("Over-time", ot_units, ot_amount)
    row("Advances (deduct)", "-", advances)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    y = pdf.get_y()
    pdf.rect(left, y, col1_w + col2_w, th)
    pdf.rect(left + col1_w + col2_w, y, col3_w, th)
    pdf.text(left + 2, y + th - 2, pdf._txt("Total Salary (Net)"))
    pdf.set_xy(left + col1_w + col2_w, y)
    pdf.cell(col3_w - 2, th, pdf._txt(inr_ascii(net)), align="R")
    pdf.ln(th + 10)

    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf._txt("Note: This is a computer-generated statement."))

    pdf.ln(6)
    sig_w = 50
    sig_x = pdf.w - 16 - sig_w
    sig_y = pdf.get_y()
    if ORG_SIGN and os.path.exists(ORG_SIGN):
        try:
            pdf.image(ORG_SIGN, x=sig_x, y=sig_y, w=sig_w)
        except Exception:
            pass
    pdf.set_xy(sig_x, sig_y + 18)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(sig_w, 6, pdf._txt("Authorised Signatory"), ln=1, align="C")

    out = pdf.output(dest="S")
    return out if isinstance(out, (bytes, bytearray)) else str(out).encode("latin-1", errors="ignore")

# ==============================
# Shared: editor + saver
# ==============================
def render_editable_table(df: pd.DataFrame, *, key: str) -> pd.DataFrame:
    """Return edited DataFrame using st.data_editor with proper widgets, including customer dropdown."""
    if df.empty:
        st.info("No entries found for the selected period.")
        return df

    df = df.copy()
    df["billable_ot_amount"] = df["billable_ot_units"].fillna(0).astype(int) * OVERTIME_ADD

    # customer dropdown options (confirmed only)
    cust_opts, _ = customer_pick_options()

    cols = [
        "date","driver","car","in_time","out_time","status",
        "outstation_overnight","overnight_client","overnight_client_name",
        "bhasmarathi","bhas_client_name","notes",
        # customer helper + details
        "customer_pick", "cust_name","cust_ach_id","cust_itinerary_id","cust_is_custom",
        # allocation
        "billable_salary","billable_ot_units","billable_ot_amount",
    ]

    # Ensure required cols exist
    for c in cols:
        if c not in df.columns:
            if c in ["outstation_overnight","overnight_client","bhasmarathi","cust_is_custom"]:
                df[c] = False
            elif c == "billable_ot_units":
                df[c] = 0
            elif c == "billable_ot_amount":
                df[c] = 0
            else:
                df[c] = ""

    # Derive a default 'customer_pick' label if data matches a confirmed customer
    if "customer_pick" not in df.columns:
        df["customer_pick"] = ""
    # (We keep it blank; users can choose an option. Matching back requires mobile which we don't store.)

    edited = st.data_editor(
        df[cols].sort_values("date"),
        hide_index=True,
        use_container_width=True,
        height=520,  # internal scroll -> sticky header visible
        key=key,
        column_config={
            "date": st.column_config.DateColumn("date", help="Attendance date", disabled=True),
            "driver": st.column_config.TextColumn("driver", disabled=True),
            "car": st.column_config.SelectboxColumn("car", options=[""] + CARS),
            "in_time": st.column_config.TextColumn("in_time", help="e.g., 08:00"),
            "out_time": st.column_config.TextColumn("out_time", help="e.g., 20:30"),
            "status": st.column_config.SelectboxColumn("status", options=["Present","Leave"]),
            "outstation_overnight": st.column_config.CheckboxColumn("outstation_overnight"),
            "overnight_client": st.column_config.CheckboxColumn("overnight_client"),
            "overnight_client_name": st.column_config.TextColumn("overnight_client_name"),
            "bhasmarathi": st.column_config.CheckboxColumn("bhasmarathi"),
            "bhas_client_name": st.column_config.TextColumn("bhas_client_name"),
            "notes": st.column_config.TextColumn("notes", width="large"),

            # Customer selection & details
            "customer_pick": st.column_config.SelectboxColumn(
                "customer (confirmed)",
                options=cust_opts,
                help="Pick a confirmed customer+package (Name — Mobile | ACH | IID) or choose '➕ Custom / Other' and fill 'cust_name'."
            ),
            "cust_name": st.column_config.TextColumn("cust_name"),
            "cust_ach_id": st.column_config.TextColumn("cust_ach_id"),
            "cust_itinerary_id": st.column_config.TextColumn("cust_itinerary_id"),
            "cust_is_custom": st.column_config.CheckboxColumn("cust_is_custom"),

            "billable_salary": st.column_config.NumberColumn("billable_salary", min_value=0, step=100),
            "billable_ot_units": st.column_config.NumberColumn("billable_ot_units", min_value=0, step=1),
            "billable_ot_amount": st.column_config.NumberColumn("billable_ot_amount", help="auto = units × 300", disabled=True),
        }
    )
    return edited

def save_table_changes(_: pd.DataFrame, edited_df: pd.DataFrame) -> int:
    """Persist all rows from edited_df by upserting each row (keyed on driver+date)."""
    if edited_df.empty:
        return 0
    _, cust_map = customer_pick_options()

    cnt = 0
    for _, r in edited_df.iterrows():
        day = _d(r["date"])
        if not day:
            continue

        # Resolve customer from dropdown, if selected
        cust_is_custom = bool(r.get("cust_is_custom", False))
        cust_name = str(r.get("cust_name",""))
        cust_ach_id = str(r.get("cust_ach_id",""))
        cust_itinerary_id = str(r.get("cust_itinerary_id",""))
        pick = str(r.get("customer_pick","")).strip()

        if pick and not pick.startswith("➕") and pick in cust_map:
            # Overwrite from confirmed pick
            ref = cust_map[pick]
            cust_name = ref.get("cust_name","")
            cust_ach_id = ref.get("cust_ach_id","")
            cust_itinerary_id = ref.get("cust_itinerary_id","")
            cust_is_custom = False
        # else keep whatever is typed, incl. custom

        upsert_attendance(
            driver = str(r.get("driver","")),
            day    = day,
            car    = str(r.get("car","")),
            in_time= str(r.get("in_time","")),
            out_time=str(r.get("out_time","")),
            status = str(r.get("status","Present")),
            outstation_overnight = bool(r.get("outstation_overnight", False)),
            overnight_client      = bool(r.get("overnight_client", False)),
            overnight_client_name = str(r.get("overnight_client_name","")),
            bhasmarathi           = bool(r.get("bhasmarathi", False)),
            bhas_client_name      = str(r.get("bhas_client_name","")),
            notes                 = str(r.get("notes","")),
            cust_itinerary_id     = cust_itinerary_id,
            cust_ach_id           = cust_ach_id,
            cust_name             = cust_name,
            cust_is_custom        = cust_is_custom,
            billable_salary       = _to_int(r.get("billable_salary",0)),
            billable_ot_units     = _to_int(r.get("billable_ot_units",0)),
        )
        cnt += 1
    load_attendance.clear()
    return cnt

# ==============================
# UI – Tabs: Driver / Admin
# ==============================
tab_driver, tab_admin = st.tabs(["Driver Entry & My Salary", "Admin Panel"]) if is_admin else (st.container(), None)

# ---------------- DRIVER VIEW ----------------
with tab_driver:
    st.subheader("Driver – Daily Entry")
    driver = st.selectbox("Driver", DRIVERS, index=0, key="drv_driver")
    day = st.date_input("Date", value=date.today(), key="drv_date")
    c1, c2, c3 = st.columns(3)
    with c1:
        car = st.selectbox("Car", [""] + CARS, index=0, key="drv_car")
    with c2:
        in_time = st.text_input("In time (e.g., 08:00)", key="drv_in")
    with c3:
        out_time = st.text_input("Out time (e.g., 20:30)", key="drv_out")

    c4, c5 = st.columns(2)
    with c4:
        status = st.selectbox("Status", ["Present", "Leave"], index=0, key="drv_status")
    with c5:
        outstation_overnight = st.checkbox("Overnight – Outstation stay", value=False, key="drv_outst")

    c6, c7 = st.columns(2)
    with c6:
        overnight_client = st.checkbox("Overnight – Client pickup/drop (post-midnight)", value=False, key="drv_ovtcli")
        overnight_client_name = st.text_input("Client (for overnight-client)", value="", key="drv_ovtcli_name", disabled=(not overnight_client))
    with c7:
        bhasmarathi = st.checkbox("Bhasmarathi duty", value=False, key="drv_bhas")
        bhas_client = st.text_input("Bhasmarathi client", value="", key="drv_bhas_name", disabled=(not bhasmarathi))

    # --- Customer link & cost allocation (single-entry form)
    st.markdown("### Link to Customer (optional) & Cost Allocation")
    confirmed = load_confirmed_customers()

    cust_itinerary_id = ""
    cust_ach_id = ""
    cust_name = ""
    cust_is_custom = False

    if confirmed.empty:
        st.caption("No confirmed packages found.")
        first_pick = st.selectbox("Select customer", ["", "➕ Add new / other"], index=0, key="drv_cust_first")
        if first_pick == "➕ Add new / other":
            cust_is_custom = True
            cust_name = st.text_input("Enter customer name (free text)", key="drv_cust_custom")
    else:
        confirmed["key"] = (
            confirmed["client_name"].str.strip().str.lower() + " | " +
            confirmed["client_mobile"].str.strip()
        )
        unique_clients = confirmed.drop_duplicates(subset=["key"]).copy()
        unique_clients["label"] = (
            unique_clients["client_name"].str.strip() + " — " +
            unique_clients["client_mobile"].str.strip()
        )
        unique_opts = ["", "➕ Add new / other"] + unique_clients["label"].tolist()
        first_pick = st.selectbox("Search / pick client (unique)", unique_opts, index=0, key="drv_cust_first")

        if first_pick == "➕ Add new / other":
            cust_is_custom = True
            cust_name = st.text_input("Enter customer name (free text)", key="drv_cust_custom")
        elif first_pick:
            name_part, mobile_part = [p.strip() for p in first_pick.split(" — ", 1)]
            client_key = name_part.lower() + " | " + mobile_part
            all_for_client = confirmed[confirmed["key"] == client_key].copy()
            cust_name = name_part
            cust_ach_id = (all_for_client["ach_id"].iloc[0] if not all_for_client.empty else "")
            all_for_client["iid_label"] = all_for_client["ach_id"].fillna("") + " | " + all_for_client["itinerary_id"]
            iid_options = all_for_client["iid_label"].tolist()
            sel_iid_lbl = st.selectbox("Select package (ACH | IID)", iid_options, index=0, key="drv_cust_iid_pick")
            try:
                cust_ach_id, cust_itinerary_id = [p.strip() for p in sel_iid_lbl.split("|", 1)]
            except Exception:
                cust_itinerary_id = all_for_client["itinerary_id"].iloc[0]

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        billable_salary = st.number_input("Billable salary to this customer (₹)", min_value=0, step=100, value=0, key="drv_bill_sal")
    with ac2:
        billable_ot_units = st.number_input("Billable OT units (× ₹300)", min_value=0, step=1, value=0, key="drv_bill_ot")
    with ac3:
        st.metric("Billable OT amount", inr(int(billable_ot_units) * OVERTIME_ADD))

    notes = st.text_area("Notes (optional)", key="drv_notes")

    if st.button("💾 Save today’s entry", key="drv_save"):
        upsert_attendance(
            driver=driver, day=day, car=car, in_time=in_time, out_time=out_time, status=status,
            outstation_overnight=outstation_overnight,
            overnight_client=overnight_client, overnight_client_name=overnight_client_name,
            bhasmarathi=bhasmarathi, bhas_client_name=bhas_client, notes=notes,
            cust_itinerary_id=cust_itinerary_id, cust_ach_id=cust_ach_id,
            cust_name=cust_name, cust_is_custom=cust_is_custom,
            billable_salary=int(billable_salary), billable_ot_units=int(billable_ot_units)
        )
        load_attendance.clear()
        st.success("Saved.")
        st.rerun()

    st.divider()
    st.subheader("My Salary (this month)")
    mstart, mend = month_bounds(date.today())
    df_att = load_attendance(driver, mstart, mend)
    df_adv = load_advances(driver, mstart, mend)
    calc = calc_salary(df_att, df_adv, mstart, mend)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Days in month", calc["days_in_month"])
    k2.metric("Leave days", calc["leave_days"])
    k3.metric("OT units", calc["ot_units"])
    k4.metric("Base", inr(BASE_SALARY))
    k5.metric("Deductions", inr(calc["leave_ded"] + calc["advances"]))
    k6.metric("Net Pay", inr(calc["net"]))

    with st.expander("Edit my entries (this month) — customer details included", expanded=True):
        edited_df = render_editable_table(df_att, key="drv_editor")
        c1, c2 = st.columns([1,3])
        if c1.button("✅ Apply changes", key="drv_apply_changes"):
            saved = save_table_changes(df_att, edited_df)
            if saved:
                st.success(f"Updated {saved} row(s).")
                load_attendance.clear()
                st.rerun()
            else:
                st.info("No changes to save.")

    with st.expander("Advances (this month)"):
        st.dataframe(df_adv.sort_values("date"), use_container_width=True, hide_index=True, height=320)

# ---------------- ADMIN VIEW ----------------
if is_admin and tab_admin is not None:
    with tab_admin:
        st.subheader("Admin — Bulk Update & Salary")

        a1, a2 = st.columns(2)
        with a1:
            admin_driver = st.selectbox("Driver", DRIVERS, index=0, key="adm_driver_select")
        with a2:
            ref_month = st.date_input("Salary month anchor", value=date.today(), key="adm_ref_month")
        mstart, mend = month_bounds(ref_month)
        st.caption(f"Period: **{mstart} → {mend}**")

        st.markdown("#### Bulk mark days")
        b1, b2 = st.columns(2)
        with b1:
            bfrom = st.date_input("From date", value=mstart, key="adm_bulk_from")
        with b2:
            bto = st.date_input("To date", value=mend, key="adm_bulk_to")
            if bto < bfrom: bto = bfrom

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            status_sel = st.selectbox("Status", ["Present","Leave"], index=0, key="adm_status_sel")
        with c2:
            mark_outstation = st.checkbox("Overnight – Outstation", value=False, key="adm_mark_outstation")
        with c3:
            mark_overnight_client = st.checkbox("Overnight – Client", value=False, key="adm_mark_ovt_client")
        with c4:
            mark_bhas = st.checkbox("Bhasmarathi", value=False, key="adm_mark_bhas")

        if st.button("🚀 Apply bulk", key="adm_apply_bulk_btn"):
            bulk_upsert_range(
                driver=admin_driver, start=bfrom, end=bto,
                status=status_sel, mark_outstation=mark_outstation,
                mark_overnight_client=mark_overnight_client, mark_bhas=mark_bhas
            )
            load_attendance.clear()
            st.success("Bulk update done.")

        st.divider()
        st.markdown("#### Add Advance (deduct from salary)")
        ad1, ad2, ad3 = st.columns(3)
        with ad1:
            adv_date = st.date_input("Advance date", value=mstart, key="adm_adv_date")
        with ad2:
            adv_amt = st.number_input("Amount (₹)", min_value=0, step=100, value=0, key="adm_adv_amt")
        with ad3:
            adv_note = st.text_input("Note", "", key="adm_adv_note")
        if st.button("➕ Add advance", key="adm_add_adv_btn"):
            if adv_amt > 0:
                add_advance(driver=admin_driver, day=adv_date, amount=int(adv_amt), note=adv_note)
                load_advances.clear()
                st.success("Advance added.")
            else:
                st.warning("Enter amount > 0")

        st.divider()
        st.subheader("Review, Edit & Generate Salary Slip")
        df_att_m = load_attendance(admin_driver, mstart, mend)
        df_adv_m = load_advances(admin_driver, mstart, mend)

        with st.expander("Attendance (month) — editable with customer dropdown", expanded=True):
            edited_admin_df = render_editable_table(df_att_m, key="adm_editor")
            c1, c2 = st.columns([1,3])
            if c1.button("✅ Apply changes (admin)", key="adm_apply_changes"):
                saved = save_table_changes(df_att_m, edited_admin_df)
                if saved:
                    st.success(f"Updated {saved} row(s).")
                    load_attendance.clear()
                    st.rerun()
                else:
                    st.info("No changes to save.")

        with st.expander("Advances (month)", expanded=False):
            st.dataframe(df_adv_m.sort_values("date"), use_container_width=True, hide_index=True, height=320)

        # Recompute after potential edits
        df_att_m = load_attendance(admin_driver, mstart, mend)
        df_adv_m = load_advances(admin_driver, mstart, mend)
        calc_m = calc_salary(df_att_m, df_adv_m, mstart, mend)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Leaves", calc_m["leave_days"])
        k2.metric("OT units", calc_m["ot_units"])
        k3.metric("Advances", inr(calc_m["advances"]))
        k4.metric("Net Pay", inr(calc_m["net"]))

        st.markdown("#### Export customer-wise allocations (for expense posting)")
        alloc = df_att_m.copy()
        if not alloc.empty:
            alloc["billable_ot_amount"] = alloc["billable_ot_units"].apply(lambda x: int(x or 0) * OVERTIME_ADD)
            cols = [
                "date","driver","cust_ach_id","cust_name","cust_itinerary_id",
                "billable_salary","billable_ot_units","billable_ot_amount",
                "notes"
            ]
            for c in cols:
                if c not in alloc.columns: alloc[c] = ""
            csv_bytes = alloc[cols].sort_values(["date","cust_name"]).to_csv(index=False).encode("utf-8")
        else:
            csv_bytes = pd.DataFrame(columns=[
                "date","driver","cust_ach_id","cust_name","cust_itinerary_id",
                "billable_salary","billable_ot_units","billable_ot_amount","notes"
            ]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download allocations CSV (month)",
            data=csv_bytes,
            file_name=f"driver_customer_allocations_{admin_driver}_{mstart}_to_{mend}.csv",
            mime="text/csv",
            use_container_width=True,
            key="adm_alloc_csv_dl"
        )

        # PDF
        month_label = f"{mstart.strftime('%B-%Y')}"
        period_label = f"{mstart.strftime('%d-%B-%Y')} to {mend.strftime('%d-%B-%Y')}"
        if st.button("📄 Generate Salary PDF", key="adm_pdf_btn"):
            pdf_bytes = build_salary_pdf(
                emp_name=admin_driver, month_label=month_label,
                period_label=period_label, calc=calc_m
            )
            st.session_state["drv_pdf"] = _as_bytes(pdf_bytes)
            st.success("PDF ready below.")

        # Preview + Download
        if "drv_pdf" in st.session_state and st.session_state["drv_pdf"]:
            slip_b = _as_bytes(st.session_state["drv_pdf"])
            st.markdown("#### Salary slip preview")
            b64s = base64.b64encode(slip_b).decode()
            st.components.v1.html(
                f'<iframe src="data:application/pdf;base64,{b64s}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>',
                height=620,
            )
            st.download_button(
                "⬇️ Download Salary Slip (PDF)",
                data=slip_b,
                file_name=f"Salary_{admin_driver}_{mstart.strftime('%Y_%m')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="adm_pdf_dl_btn"
            )
