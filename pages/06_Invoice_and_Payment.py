# pages/06_Invoice_and_Payment.py
from __future__ import annotations

import io
import os
import base64
from datetime import datetime, date
from typing import Optional, Dict

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# ========= Page config =========
st.set_page_config(page_title="Invoice & Payment Slip", layout="wide")
st.title("🧾 Invoice & Payment Slip (Confirmed packages only)")

# ========= Admin-only gate (same style as other admin pages) =========
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

# ========= Mongo boot =========
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
    uri = _find_uri() or st.secrets["mongo_uri"]
    client = MongoClient(uri, appName="TAK_InvoiceSlip", serverSelectionTimeoutMS=6000, tz_aware=True)
    client.admin.command("ping")
    return client

db = _get_client()["TAK_DB"]
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_expenses    = db["expenses"]

# ========= Helpers =========
def _to_int(x, default=0):
    try:
        if x is None: return default
        return int(float(str(x).replace(",", "")))
    except Exception:
        return default

def _safe_date(x) -> Optional[date]:
    try:
        if pd.isna(x): return None
        d = pd.to_datetime(x)
        return d.date()
    except Exception:
        return None

def _str(x):
    return "" if x is None else str(x)

def _fmt_money(n: int) -> str:
    # ASCII-friendly (avoid ₹ to keep core fonts happy)
    return f"Rs {int(n):,}"

def _final_cost(iid: str) -> Dict[str, int]:
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

@st.cache_data(ttl=120, show_spinner=False)
def fetch_confirmed() -> pd.DataFrame:
    its = list(col_itineraries.find({}, {
        "_id":1, "ach_id":1, "client_name":1, "client_mobile":1,
        "final_route":1, "total_pax":1, "start_date":1, "end_date":1
    }))
    if not its: return pd.DataFrame()
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        r["start_date"] = _safe_date(r.get("start_date"))
        r["end_date"]   = _safe_date(r.get("end_date"))
        r.pop("_id", None)
    df_i = pd.DataFrame(its)

    ups = list(col_updates.find({"status":"confirmed"}, {
        "_id":0, "itinerary_id":1, "status":1, "booking_date":1,
        "advance_amount":1, "rep_name":1, "incentive":1
    }))
    if not ups:
        return pd.DataFrame(columns=[
            "ach_id","client_name","client_mobile","final_route","total_pax",
            "start_date","end_date","booking_date","advance_amount","rep_name","incentive","itinerary_id"
        ])
    for u in ups:
        u["booking_date"]   = _safe_date(u.get("booking_date"))
        u["advance_amount"] = _to_int(u.get("advance_amount", 0))
    df_u = pd.DataFrame(ups)

    df = df_i.merge(df_u, on="itinerary_id", how="inner")
    # attach cost fields
    bases, discs, finals = [], [], []
    for iid in df["itinerary_id"]:
        c = _final_cost(iid)
        bases.append(c["base"]); discs.append(c["discount"]); finals.append(c["final"])
    df["base_amount"] = bases
    df["discount"]    = discs
    df["final_cost"]  = finals
    return df

# ========= PDF builders (fpdf2; sanitize text to Latin-1) =========
from fpdf import FPDF

def _sanitize_text(s: str) -> str:
    """
    Convert to Latin-1 safe text for FPDF core fonts.
    - Replace common Unicode punctuation (₹, —, –, “”, ‘’, •) with ASCII equivalents.
    - Then encode/decode with latin-1 and replace unencodable chars by '?'.
    """
    if s is None:
        s = ""
    # Common replacements
    s = (str(s)
         .replace("₹", "Rs ")
         .replace("—", "-")
         .replace("–", "-")
         .replace("•", "-")
         .replace("“", '"').replace("”", '"')
         .replace("‘", "'").replace("’", "'"))
    try:
        return s.encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return s

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, _sanitize_text("TRAVEL & KAILASH - TAX INVOICE / RECEIPT"), ln=1, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, _sanitize_text("Address: Your Office Address, City | Phone: +91-XXXXXXXXXX | Email: info@example.com"), ln=1, align="C")
        self.ln(2)
        self.set_draw_color(0,0,0)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, _sanitize_text(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"), align="R")

def _pdf_bytes(pdf: FPDF) -> bytes:
    """
    fpdf2 behavior differs by version:
      - Some return bytes for output(dest="S")
      - Older return a str (latin-1)
    Handle both safely.
    """
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    # else we assume str
    return str(out).encode("latin-1", errors="ignore")

def build_invoice_pdf(row: dict, subject: str) -> bytes:
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)

    # Invoice title & meta
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize_text("INVOICE"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _sanitize_text(f"Invoice Date: {datetime.now().strftime('%Y-%m-%d')}"), ln=1)
    pdf.cell(0, 6, _sanitize_text(f"ACH ID: {_str(row.get('ach_id'))}"), ln=1)
    pdf.ln(2)

    # Bill To
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, _sanitize_text("Bill To:"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _sanitize_text(f"Client: {_str(row.get('client_name'))}"), ln=1)
    pdf.cell(0, 6, _sanitize_text(f"Mobile: {_str(row.get('client_mobile'))}"), ln=1)
    travel = f"{_str(row.get('start_date'))} to {_str(row.get('end_date'))}"
    pdf.cell(0, 6, _sanitize_text(f"Travel: {travel}"), ln=1)
    pdf.cell(0, 6, _sanitize_text(f"Route: {_str(row.get('final_route'))}"), ln=1)
    pdf.ln(2)

    # Subject
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, _sanitize_text(f"Subject: {subject}"))
    pdf.ln(2)

    # Line items
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(120, 7, _sanitize_text("Description"), border=1)
    pdf.cell(60, 7, _sanitize_text("Amount"), border=1, ln=1, align="R")
    pdf.set_font("Helvetica", "", 10)

    base = int(row.get("base_amount", 0))
    disc = int(row.get("discount", 0))
    final = int(row.get("final_cost", 0))

    desc = f"Travel Package - {_str(row.get('final_route'))} ({_str(row.get('total_pax'))} pax)"
    pdf.cell(120, 7, _sanitize_text(desc), border=1)
    pdf.cell(60, 7, _sanitize_text(_fmt_money(base)), border=1, ln=1, align="R")

    if disc > 0:
        pdf.cell(120, 7, _sanitize_text("Less: Discount"), border=1)
        pdf.cell(60, 7, _sanitize_text(f"- {_fmt_money(disc)}"), border=1, ln=1, align="R")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(120, 7, _sanitize_text("Total Payable"), border=1)
    pdf.cell(60, 7, _sanitize_text(_fmt_money(final)), border=1, ln=1, align="R")

    pdf.ln(6)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _sanitize_text("Note: This invoice is generated for your confirmed booking. Please retain for your records."))

    return _pdf_bytes(pdf)

def build_payment_slip_pdf(row: dict, payment_date: Optional[date]) -> bytes:
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize_text("PAYMENT SLIP"), ln=1)
    pdf.set_font("Helvetica", "", 10)

    slip_date = payment_date or row.get("booking_date")
    slip_date_str = _str(slip_date)

    pdf.cell(0, 6, _sanitize_text(f"Slip Date: {slip_date_str}"), ln=1)
    pdf.cell(0, 6, _sanitize_text(f"ACH ID: {_str(row.get('ach_id'))}"), ln=1)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, _sanitize_text("Client:"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _sanitize_text(f"{_str(row.get('client_name'))}  |  Mobile: {_str(row.get('client_mobile'))}"), ln=1)
    travel = f"{_str(row.get('start_date'))} to {_str(row.get('end_date'))}"
    pdf.cell(0, 6, _sanitize_text(f"Travel: {travel}  |  Route: {_str(row.get('final_route'))}"), ln=1)
    pdf.ln(2)

    advance = int(row.get("advance_amount", 0))
    final   = int(row.get("final_cost", 0))

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(120, 7, _sanitize_text("Item"), border=1)
    pdf.cell(60, 7, _sanitize_text("Amount"), border=1, ln=1, align="R")
    pdf.set_font("Helvetica", "", 10)

    pdf.cell(120, 7, _sanitize_text("Amount Paid (Advance)"), border=1)
    pdf.cell(60, 7, _sanitize_text(_fmt_money(advance)), border=1, ln=1, align="R")

    pdf.cell(120, 7, _sanitize_text("Total Package Value"), border=1)
    pdf.cell(60, 7, _sanitize_text(_fmt_money(final)), border=1, ln=1, align="R")

    bal = max(final - advance, 0)
    pdf.cell(120, 7, _sanitize_text("Balance Due"), border=1)
    pdf.cell(60, 7, _sanitize_text(_fmt_money(bal)), border=1, ln=1, align="R")

    pdf.ln(6)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _sanitize_text(f"Payment received on: {slip_date_str}. This is a computer generated receipt."))

    return _pdf_bytes(pdf)

# ========= UI =========
df = fetch_confirmed()
if df.empty:
    st.info("No confirmed packages found.")
    st.stop()

search = st.text_input("🔎 Search (name / mobile / ACH ID / route)")
view = df.copy()
if search.strip():
    s = search.lower()
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
        "ach_id":"ACH ID","client_name":"Client","client_mobile":"Mobile",
        "final_route":"Route","total_pax":"Pax","start_date":"Start",
        "end_date":"End","booking_date":"Booked on","final_cost":"Final Cost",
        "advance_amount":"Advance Paid"
    }).sort_values(["Booked on","Start","Client"], na_position="last")
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

st.subheader("🧾 Invoice")
default_subject = f"Travel Package - {_sanitize_text(_str(row.get('final_route')))} for {_sanitize_text(_str(row.get('client_name')))}"
subject = st.text_input("Subject line for invoice", value=default_subject)

c1, c2 = st.columns([1,1])
with c1:
    if st.button("Generate Invoice PDF"):
        inv_bytes = build_invoice_pdf(row, subject=_sanitize_text(subject))
        st.session_state["inv_pdf"] = inv_bytes
with c2:
    pay_date_default = row.get("booking_date") or date.today()
    # ensure it's a date object
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
    st.components.v1.html(f"""
        <iframe src="data:application/pdf;base64,{b64}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>
    """, height=620, scrolling=True)
    st.download_button(
        "⬇️ Download Invoice (PDF)",
        data=st.session_state["inv_pdf"],
        file_name=f"Invoice_{_sanitize_text(_str(row.get('ach_id')))}.pdf",
        mime="application/pdf"
    )

# ===== Preview + Download (Payment Slip) =====
if "slip_pdf" in st.session_state:
    st.markdown("#### Payment slip preview")
    b64s = base64.b64encode(st.session_state["slip_pdf"]).decode()
    st.components.v1.html(f"""
        <iframe src="data:application/pdf;base64,{b64s}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>
    """, height=620, scrolling=True)
    st.download_button(
        "⬇️ Download Payment Slip (PDF)",
        data=st.session_state["slip_pdf"],
        file_name=f"PaymentSlip_{_sanitize_text(_str(row.get('ach_id')))}.pdf",
        mime="application/pdf"
    )
