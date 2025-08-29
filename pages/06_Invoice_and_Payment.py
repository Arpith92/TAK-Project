# pages/06_Invoice_and_Payment.py
from __future__ import annotations

import os, base64
from datetime import datetime, date
from typing import Optional, Dict, Tuple, List

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from fpdf import FPDF

# ================= Page config =================
st.set_page_config(page_title="Invoice & Payment Slip", layout="wide")
st.title("🧾 Invoice & Payment Slip (Confirmed packages only)")

TTL = 90  # small cache, keeps the page snappy

# ================= Admin gate ==================
def require_admin():
    ADMIN_PASS_DEFAULT = "Arpith&92"
    ADMIN_PASS = str(st.secrets.get("admin_pass", ADMIN_PASS_DEFAULT))
    with st.sidebar:
        st.markdown("### Admin access")
        p = st.text_input("Enter admin password", type="password", placeholder="enter pass")
    if (p or "").strip() != ADMIN_PASS.strip():
        st.stop()
    st.session_state["user"] = "Admin"
    st.session_state["is_admin"] = True

require_admin()

# ================= Mongo =======================
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
        appName="TAK_InvoiceSlip",
        serverSelectionTimeoutMS=6000,
        connectTimeoutMS=6000,
        retryWrites=True,
        tz_aware=True,
    )
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        st.error(f"Could not connect to MongoDB. Details: {e}")
        st.stop()
    return client

from tak_audit import audit_pageview
audit_pageview(st.session_state.get("user", "Unknown"), page="06_Invoice_and_Payment")

db = _get_client()["TAK_DB"]
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_expenses    = db["expenses"]

# ================= Org / assets =================
ORG = {
    "title": "TravelaajKal™ – Achala Holidays Pvt. Ltd",
    "line1": "Mangrola, Ujjain, Madhya Pradesh 456006, India",
    "line2": "Email: travelaajkal@gmail.com  |  Web: www.travelaajkal.com | Mob: +91-7509612798" ,
    
    "footer_rights": "All the rights reserved by TravelAajkal 2025-26",
}
# Put your files in .streamlit/
ORG_LOGO = ".streamlit/logo.png"
ORG_SIGN = ".streamlit/signature.png"   # update to your file name if different

# ================= Helpers =====================
def _to_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        return int(float(str(x).replace(",", "")))
    except Exception:
        return default

def _safe_date(x) -> Optional[date]:
    try:
        if x is None or pd.isna(x):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

def _str(x) -> str:
    return "" if x is None else str(x)

def _fmt_money(n: int) -> str:
    return f"Rs {int(n):,}"  # ASCII (no unicode rupee for PDF safety)

def _nights_days(start: Optional[date], end: Optional[date]) -> str:
    if not start or not end:
        return ""
    try:
        days = (end - start).days + 1
        if days < 1:
            days = 1
        nights = max(days - 1, 0)
        return f"{days} days {nights} nights"
    except Exception:
        return ""

def _final_cost(iid: str) -> Dict[str, int]:
    """
    Final cost preference:
      1) expenses.final_package_cost
      2) expenses.base_package_cost - expenses.discount
      3) itinerary.package_cost - itinerary.discount
    """
    exp = col_expenses.find_one(
        {"itinerary_id": str(iid)},
        {"final_package_cost":1,"base_package_cost":1,"discount":1,"package_cost":1}
    ) or {}

    if "final_package_cost" in exp:
        return {
            "base": _to_int(exp.get("base_package_cost", exp.get("package_cost", 0))),
            "discount": _to_int(exp.get("discount", 0)),
            "final": _to_int(exp.get("final_package_cost", 0))
        }

    base = _to_int(exp.get("base_package_cost", exp.get("package_cost", 0)))
    disc = _to_int(exp.get("discount", 0))
    if base or disc:
        return {"base": base, "discount": disc, "final": max(0, base - disc)}

    it = col_itineraries.find_one({"_id": ObjectId(iid)}, {"package_cost":1,"discount":1}) or {}
    base2 = _to_int(it.get("package_cost", 0))
    disc2 = _to_int(it.get("discount", 0))
    return {"base": base2, "discount": disc2, "final": max(0, base2 - disc2)}

@st.cache_data(ttl=TTL, show_spinner=False)
def fetch_confirmed() -> pd.DataFrame:
    # itineraries (essentials)
    its = list(col_itineraries.find({}, {
        "_id":1, "ach_id":1, "client_name":1, "client_mobile":1,
        "final_route":1, "total_pax":1, "start_date":1, "end_date":1
    }))
    if not its:
        return pd.DataFrame()
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        r["start_date"] = _safe_date(r.get("start_date"))
        r["end_date"]   = _safe_date(r.get("end_date"))
        r.pop("_id", None)
    df_i = pd.DataFrame(its)

    # confirmed updates
    ups = list(col_updates.find({"status":"confirmed"}, {
        "_id":0, "itinerary_id":1, "status":1, "booking_date":1,
        "advance_amount":1, "rep_name":1, "incentive":1
    }))
    if not ups:
        return pd.DataFrame()
    for u in ups:
        u["itinerary_id"] = str(u.get("itinerary_id"))
        u["booking_date"]   = _safe_date(u.get("booking_date"))
        u["advance_amount"] = _to_int(u.get("advance_amount", 0))
    df_u = pd.DataFrame(ups)

    df = df_i.merge(df_u, on="itinerary_id", how="inner")

    # enrich final cost snapshot
    bases, discs, finals = [], [], []
    for iid in df["itinerary_id"]:
        c = _final_cost(iid)
        bases.append(c["base"]); discs.append(c["discount"]); finals.append(c["final"])
    df["base_amount"] = bases
    df["discount"]    = discs
    df["final_cost"]  = finals

    return df

def _update_advance_and_booking(iid: str, adv: int, bdate: Optional[date]) -> None:
    payload = {
        "itinerary_id": str(iid),
        "status": "confirmed",   # keep it confirmed
        "advance_amount": int(adv),
        "updated_at": datetime.utcnow(),
    }
    if bdate:
        payload["booking_date"] = datetime.combine(bdate, datetime.min.time())
    col_updates.update_one({"itinerary_id": str(iid)}, {"$set": payload}, upsert=True)

# ================= FPDF (ASCII-safe) =============
def _latin(s: str) -> str:
    if s is None:
        s = ""
    s = (str(s)
         .replace("₹", "Rs ")
         .replace("—", "-").replace("–", "-").replace("•", "-")
         .replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
         .replace("™", "(TM)"))
    try:
        return s.encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return s

class PDF(FPDF):
    def header(self):
        # Outer border rectangle
        self.set_draw_color(150,150,150)
        self.rect(8, 8, 194, 281)

        # Logo (enlarged)
        x0 = 14
        try:
            if ORG_LOGO and os.path.exists(ORG_LOGO):
                self.image(ORG_LOGO, x=x0, y=12, w=34)  # enlarged logo
                x0 = 52
        except Exception:
            x0 = 14

        # Company title & address
        self.set_xy(x0, 12)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, _latin(ORG["title"]), ln=1, align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, _latin(ORG["line1"]), ln=1, align="C")
        self.cell(0, 6, _latin(ORG["line2"]), ln=1, align="C")
        self.ln(2)
        self.set_draw_color(0,0,0)
        self.line(12, self.get_y(), 198, self.get_y())
        self.ln(4)

    def footer(self):
        # rights line
        self.set_y(-22)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, _latin(ORG["footer_rights"]), ln=1, align="C")

def _pdf_bytes(pdf: FPDF) -> bytes:
    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytes, bytearray)) else str(out).encode("latin-1", errors="ignore")

# ------------- invoice builder -------------
def build_invoice_pdf(row: dict, subject: str) -> bytes:
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    left = 16
    pdf.set_x(left)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, _latin("INVOICE"), ln=1)

    pdf.set_x(left)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, _latin(f"Invoice Date: {datetime.now().strftime('%Y-%m-%d')}"), ln=1)
    pdf.set_x(left); pdf.cell(0, 6, _latin(f"ACH ID: {_str(row.get('ach_id'))}"), ln=1)
    pdf.ln(2)

    # Customer block (name line only on top; not repeated below)
    pdf.set_x(left); pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, _latin("Bill To:"), ln=1)
    pdf.set_x(left); pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, _latin(f"Customer Name: {_str(row.get('client_name'))}"), ln=1)
    pdf.set_x(left); pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, _latin(f"Mobile: {_str(row.get('client_mobile'))}"), ln=1)
    travel = f"{_str(row.get('start_date'))} to {_str(row.get('end_date'))}"
    pdf.set_x(left); pdf.cell(0, 6, _latin(f"Travel: {travel}"), ln=1)
    pdf.set_x(left); pdf.cell(0, 6, _latin(f"Route: {_str(row.get('final_route'))}"), ln=1)
    pdf.ln(2)

    # Subject (no customer name)
    pdf.set_x(left); pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, _latin(f"Subject: {subject}"))
    pdf.ln(1)

    # Line item
    days_nights = _nights_days(row.get("start_date"), row.get("end_date"))
    base  = int(row.get("base_amount", 0))
    disc  = int(row.get("discount", 0))
    final = int(row.get("final_cost", 0))
    desc = f"{days_nights} {_str(row.get('final_route'))} travel package for {_str(row.get('total_pax'))} pax"

    # Table
    col1_w, col2_w = 130, 52
    th = 8

    # header row
    pdf.set_font("Helvetica", "B", 10)
    y = pdf.get_y()
    pdf.rect(left, y, col1_w, th);  pdf.rect(left+col1_w, y, col2_w, th)
    pdf.text(left+2,  y+th-2, _latin("Description"))
    pdf.text(left+col1_w+2, y+th-2, _latin("Amount"))
    pdf.ln(th)

    # item
    pdf.set_font("Helvetica", "", 10)
    y = pdf.get_y()
    pdf.rect(left, y, col1_w, th); pdf.rect(left+col1_w, y, col2_w, th)
    pdf.text(left+2, y+th-2, _latin(desc))
    pdf.set_xy(left+col1_w, y)
    pdf.cell(col2_w-2, th, _latin(_fmt_money(base)), align="R")
    pdf.ln(th)

    # discount (if any)
    if disc > 0:
        y = pdf.get_y()
        pdf.rect(left, y, col1_w, th); pdf.rect(left+col1_w, y, col2_w, th)
        pdf.text(left+2, y+th-2, _latin("Less: Discount"))
        pdf.set_xy(left+col1_w, y)
        pdf.cell(col2_w-2, th, _latin(f"- {_fmt_money(disc)}"), align="R")
        pdf.ln(th)

    # total
    y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 10)
    pdf.rect(left, y, col1_w, th); pdf.rect(left+col1_w, y, col2_w, th)
    pdf.text(left+2, y+th-2, _latin("Total Payable"))
    pdf.set_xy(left+col1_w, y)
    pdf.cell(col2_w-2, th, _latin(_fmt_money(final)), align="R")
    pdf.ln(th + 4)

    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin("Note: This invoice is generated for your confirmed booking. Please retain for your records."))

    # signature block (enlarged)
    pdf.ln(8)
    sig_y = pdf.get_y()
    if ORG_SIGN and os.path.exists(ORG_SIGN):
        try:
            pdf.image(ORG_SIGN, x=left+110, y=sig_y-4, w=55)  # enlarged signature
        except Exception:
            pass
    pdf.set_y(sig_y + 20)
    pdf.set_x(left+110)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(80, 6, _latin("Authorised Signatory"), ln=1, align="C")
    return _pdf_bytes(pdf)

# ------------- payment slip builder -------------
def build_payment_slip_pdf(row: dict, payment_date: Optional[date]) -> bytes:
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    left = 16
    pdf.set_x(left)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, _latin("PAYMENT SLIP"), ln=1)

    pdf.set_x(left); pdf.set_font("Helvetica", "", 11)
    slip_date = payment_date or row.get("booking_date")
    slip_date_str = _str(slip_date)
    pdf.cell(0, 6, _latin(f"Slip Date: {slip_date_str}"), ln=1)
    pdf.set_x(left); pdf.cell(0, 6, _latin(f"ACH ID: {_str(row.get('ach_id'))}"), ln=1)
    pdf.ln(2)

    # Customer block (name on top only)
    pdf.set_x(left); pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 6, _latin("Customer:"), ln=1)
    pdf.set_x(left); pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, _latin(f"Customer Name: {_str(row.get('client_name'))}"), ln=1)
    pdf.set_x(left); pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, _latin(f"Mobile: {_str(row.get('client_mobile'))}"), ln=1)
    travel = f"{_str(row.get('start_date'))} to {_str(row.get('end_date'))}"
    pdf.set_x(left); pdf.cell(0, 6, _latin(f"Travel: {travel}  |  Route: {_str(row.get('final_route'))}"), ln=1)
    pdf.ln(2)

    advance = int(row.get("advance_amount", 0))
    final   = int(row.get("final_cost", 0))
    bal     = max(final - advance, 0)

    # table
    col1_w, col2_w, th = 130, 52, 8
    pdf.set_font("Helvetica", "B", 10)
    y = pdf.get_y()
    pdf.rect(left, y, col1_w, th);  pdf.rect(left+col1_w, y, col2_w, th)
    pdf.text(left+2,  y+th-2, _latin("Item"))
    pdf.text(left+col1_w+2, y+th-2, _latin("Amount"))
    pdf.ln(th)

    pdf.set_font("Helvetica", "", 10)
    # row 1
    y = pdf.get_y()
    pdf.rect(left, y, col1_w, th); pdf.rect(left+col1_w, y, col2_w, th)
    pdf.text(left+2, y+th-2, _latin("Amount Paid (Advance)"))
    pdf.set_xy(left+col1_w, y); pdf.cell(col2_w-2, th, _latin(_fmt_money(advance)), align="R")
    pdf.ln(th)
    # row 2
    y = pdf.get_y()
    pdf.rect(left, y, col1_w, th); pdf.rect(left+col1_w, y, col2_w, th)
    pdf.text(left+2, y+th-2, _latin("Total Package Value"))
    pdf.set_xy(left+col1_w, y); pdf.cell(col2_w-2, th, _latin(_fmt_money(final)), align="R")
    pdf.ln(th)
    # row 3
    y = pdf.get_y()
    pdf.rect(left, y, col1_w, th); pdf.rect(left+col1_w, y, col2_w, th)
    pdf.text(left+2, y+th-2, _latin("Balance Due"))
    pdf.set_xy(left+col1_w, y); pdf.cell(col2_w-2, th, _latin(_fmt_money(bal)), align="R")
    pdf.ln(th+4)

    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin(f"Payment received on: {slip_date_str}. This is a computer generated receipt."))

    # signature block (enlarged)
    pdf.ln(8)
    sig_y = pdf.get_y()
    if ORG_SIGN and os.path.exists(ORG_SIGN):
        try:
            pdf.image(ORG_SIGN, x=left+110, y=sig_y-4, w=55)
        except Exception:
            pass
    pdf.set_y(sig_y + 20)
    pdf.set_x(left+110)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(80, 6, _latin("Authorised Signatory"), ln=1, align="C")
    return _pdf_bytes(pdf)

# ================= UI ===========================
df = fetch_confirmed()
if df.empty:
    st.info("No confirmed packages found.")
    st.stop()

search = st.text_input("🔎 Search (name / mobile / ACH ID / route)")
view = df.copy()
if (search or "").strip():
    s = search.lower().strip()
    view = view[
        view["client_name"].astype(str).str.lower().str.contains(s) |
        view["client_mobile"].astype(str).str.lower().str.contains(s) |
        view["ach_id"].astype(str).str.lower().str.contains(s) |
        view["final_route"].astype(str).str.lower().str.contains(s)
    ]

left, right = st.columns([2, 1])
with left:
    show_cols = ["ach_id","client_name","client_mobile","final_route","total_pax",
                 "start_date","end_date","booking_date","final_cost","advance_amount"]
    grid = view[show_cols].rename(columns={
        "ach_id":"ACH ID","client_name":"Customer","client_mobile":"Mobile",
        "final_route":"Route","total_pax":"Pax","start_date":"Start",
        "end_date":"End","booking_date":"Booked on","final_cost":"Final Cost",
        "advance_amount":"Advance Paid"
    }).sort_values(["Booked on","Start","Customer"], na_position="last")
    st.dataframe(grid, use_container_width=True, hide_index=True)

with right:
    st.markdown("**Select a confirmed package**")
    opts = (view["ach_id"].fillna("") + " | " +
            view["client_name"].fillna("") + " | " +
            view["booking_date"].astype(str).fillna("") + " | " +
            view["itinerary_id"])
    choice = st.selectbox("Choose", opts.tolist())
    chosen_id = choice.split(" | ")[-1] if choice else None

st.divider()
if not chosen_id:
    st.stop()

row = df[df["itinerary_id"] == chosen_id].iloc[0].to_dict()

# --- Small admin utility: keep DB in sync before we make a slip ---
st.subheader("🛠️ Update booking/advance (optional, saves to DB)")
ub1, ub2, ub3 = st.columns([1,1,1])
with ub1:
    booked_on = row.get("booking_date") or date.today()
    if not isinstance(booked_on, date):
        try:
            booked_on = pd.to_datetime(booked_on).date()
        except Exception:
            booked_on = date.today()
    edit_bdate = st.date_input("Booking date", value=booked_on)
with ub2:
    edit_adv = st.number_input("Advance amount (₹)", min_value=0, step=500, value=int(row.get("advance_amount", 0)))
with ub3:
    st.caption(" ")
    if st.button("💾 Save to DB"):
        _update_advance_and_booking(chosen_id, int(edit_adv), edit_bdate)
        # refresh cache + row snapshot
        fetch_confirmed.clear()
        df = fetch_confirmed()
        row = df[df["itinerary_id"] == chosen_id].iloc[0].to_dict()
        st.success("Saved. Refreshed values will reflect in Payment Slip.")

st.markdown("---")

# Subject line WITHOUT customer name
dn = _nights_days(row.get("start_date"), row.get("end_date"))
default_subject = f"{dn} {_latin(_str(row.get('final_route')))} travel package"

st.subheader("🧾 Invoice")
subject = st.text_input("Subject line for invoice", value=default_subject)

c1, c2 = st.columns([1,1])
with c1:
    if st.button("Generate Invoice PDF"):
        inv_bytes = build_invoice_pdf(row, subject=_latin(subject))
        st.session_state["inv_pdf"] = inv_bytes

with c2:
    pay_date_default = row.get("booking_date") or date.today()
    if not isinstance(pay_date_default, date):
        try:
            pay_date_default = pd.to_datetime(pay_date_default).date()
        except Exception:
            pay_date_default = date.today()
    pay_date = st.date_input("Payment made date (for slip)", value=pay_date_default)
    if st.button("Generate Payment Slip PDF"):
        slip_bytes = build_payment_slip_pdf(row, payment_date=pay_date)
        st.session_state["slip_pdf"] = slip_bytes

st.markdown("---")

# ===== Preview + Download (Invoice) =====
if "inv_pdf" in st.session_state:
    st.markdown("#### Invoice preview")
    b64 = base64.b64encode(st.session_state["inv_pdf"]).decode()
    st.components.v1.html(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>',
        height=620,
    )
    fname = f"Invoice_{_latin(_str(row.get('ach_id') or row.get('client_name') or 'TAK'))}.pdf"
    st.download_button(
        "⬇️ Download Invoice (PDF)",
        data=st.session_state["inv_pdf"],
        file_name=fname,
        mime="application/pdf",
        use_container_width=True,
    )

# ===== Preview + Download (Payment Slip) =====
if "slip_pdf" in st.session_state:
    st.markdown("#### Payment slip preview")
    b64s = base64.b64encode(st.session_state["slip_pdf"]).decode()
    st.components.v1.html(
        f'<iframe src="data:application/pdf;base64,{b64s}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>',
        height=620,
    )
    fname2 = f"PaymentSlip_{_latin(_str(row.get('ach_id') or row.get('client_name') or 'TAK'))}.pdf"
    st.download_button(
        "⬇️ Download Payment Slip (PDF)",
        data=st.session_state["slip_pdf"],
        file_name=fname2,
        mime="application/pdf",
        use_container_width=True,
    )
