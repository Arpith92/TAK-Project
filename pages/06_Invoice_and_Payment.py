# pages/06_Invoice_and_Payment.py
from __future__ import annotations

import io
import os
import base64
from datetime import datetime, date, timedelta
from typing import Optional, Dict

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# ================= Page config =================
st.set_page_config(page_title="Invoice & Payment Slip", layout="wide")
st.title("🧾 Invoice & Payment Slip (Confirmed packages only)")

# ================= Admin gate =================
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

# ================= Brand / Template =================
# Place your assets in these paths. Change if needed.
BRAND = {
    "company": "Achala Holidays Pvt Limited",
    "address_lines": [
        "Mangrola",
        "Ujjain, Madhya Pradesh 456006, India"
    ],
    "phone": "+91-XXXXXXXXXX",
    "email": "travelaajkal@gmail.com",
    "website": "www.travelaajkal.com",
    "logo_path": "assets/logo.png",          # <- put your logo here
    "signature_path": "assets/signature.png",# <- put your sign image here
    "unicode_ttf": "assets/DejaVuSans.ttf",  # <- optional font to render ₹ and long dashes
    "notes": "Thanks for your business.",
}

# ================= Mongo boot =================
CAND_KEYS = ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]

def _find_uri() -> Optional[str]:
    for k in CAND_KEYS:
        try: v = st.secrets.get(k)
        except Exception: v = None
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

# ================= Helpers =================
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

def _str(x): return "" if x is None else str(x)

# Money formatting: if Unicode font is available we will print ₹, else "Rs"
def _rupee_symbol():
    try:
        if os.path.exists(BRAND["unicode_ttf"]):
            return "₹"
    except Exception:
        pass
    return "Rs"

def _fmt_money(n: int) -> str:
    s = f"{int(n):,}"
    sym = _rupee_symbol()
    if sym == "₹":
        return f"{sym}{s}"
    return f"{sym} {s}"

# Words (best-effort without adding packages)
def _amount_in_words(n: int) -> str:
    try:
        # try num2words if present
        from num2words import num2words
        words = num2words(n, lang="en_IN").replace(",", "").title()
        return f"Indian Rupee {words} Only"
    except Exception:
        return f"Indian Rupee {n:,} Only"

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

    bases, discs, finals = [], [], []
    for iid in df["itinerary_id"]:
        c = _final_cost(iid)
        bases.append(c["base"]); discs.append(c["discount"]); finals.append(c["final"])
    df["base_amount"] = bases
    df["discount"]    = discs
    df["final_cost"]  = finals
    return df

# ================= PDF (fpdf2) =================
from fpdf import FPDF

def _font_setup(pdf: FPDF):
    """Load Unicode TTF if available to allow ₹ and fancy punctuation."""
    ttf = BRAND["unicode_ttf"]
    if os.path.exists(ttf):
        try:
            pdf.add_font("DejaVu", "", ttf, uni=True)
            pdf.add_font("DejaVu", "B", ttf, uni=True)
            pdf.set_font("DejaVu", "", 11)
            return ("DejaVu",)
        except Exception:
            pass
    pdf.set_font("Helvetica", "", 11)
    return ("Helvetica",)

def _pdf_bytes(pdf: FPDF) -> bytes:
    out = pdf.output(dest="S")
    return out if isinstance(out, (bytes, bytearray)) else str(out).encode("latin-1", "ignore")

def _header_block(pdf: FPDF, title: str):
    # Logo
    if os.path.exists(BRAND["logo_path"]):
        try:
            pdf.image(BRAND["logo_path"], x=10, y=10, w=36)
        except Exception:
            pass

    # Company name & contacts
    pdf.set_xy(50, 10)
    pdf.set_font(pdf.font_family, "B", 16)
    pdf.cell(0, 8, BRAND["company"], ln=1)
    pdf.set_x(50)
    pdf.set_font(pdf.font_family, "", 9)
    line = " | ".join([", ".join(BRAND["address_lines"]), f"Phone: {BRAND['phone']}", f"Email: {BRAND['email']}"])
    pdf.multi_cell(0, 5, line)
    if BRAND.get("website"):
        pdf.set_x(50)
        pdf.cell(0, 5, f"Website: {BRAND['website']}", ln=1)
    pdf.ln(2)
    pdf.set_draw_color(0,0,0)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Big title
    pdf.set_font(pdf.font_family, "B", 13)
    pdf.cell(0, 8, title, ln=1)

def _two_col_label(pdf: FPDF, label: str, value: str, w_label=32, h=6):
    pdf.set_font(pdf.font_family, "B", 10); pdf.cell(w_label, h, f"{label}:", ln=0)
    pdf.set_font(pdf.font_family, "", 10);   pdf.cell(0, h, value, ln=1)

def build_invoice_pdf(row: dict, subject: str) -> bytes:
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    _font_setup(pdf)

    _header_block(pdf, "TAX INVOICE")

    # Meta box (like Zoho top-right table)
    top_y = pdf.get_y()
    pdf.set_font(pdf.font_family, "", 10)
    inv_no = _str(row.get("ach_id")) or f"INV-{datetime.now():%Y%m%d}"
    inv_date = date.today()
    due_date = inv_date  # Due on Receipt

    # Left: Bill To
    left_x = 10
    pdf.set_xy(left_x, top_y + 2)
    pdf.set_font(pdf.font_family, "B", 11); pdf.cell(0, 6, "Bill To:", ln=1)
    pdf.set_font(pdf.font_family, "", 10)
    _two_col_label(pdf, "Client", _str(row.get("client_name")))
    _two_col_label(pdf, "Mobile", _str(row.get("client_mobile")))
    travel = f"{_str(row.get('start_date'))} to {_str(row.get('end_date'))}"
    _two_col_label(pdf, "Travel", travel)
    _two_col_label(pdf, "Route", _str(row.get("final_route")))
    pdf.ln(2)

    # Right: Invoice meta table
    right_x = 120
    right_y = top_y + 2
    pdf.set_xy(right_x, right_y)
    pdf.set_font(pdf.font_family, "B", 10)
    pdf.cell(80, 7, "Invoice Details", border=1, ln=1, align="C")
    pdf.set_font(pdf.font_family, "", 10)
    def rrow(lbl, val):
        pdf.set_x(right_x)
        pdf.cell(35, 7, lbl, border="L")
        pdf.cell(45, 7, val, border="R", ln=1, align="R")
    rrow("Invoice #", inv_no)
    rrow("Invoice Date", inv_date.strftime("%d/%m/%Y"))
    rrow("Terms", "Due on Receipt")
    rrow("Due Date", due_date.strftime("%d/%m/%Y"))
    pdf.set_x(right_x); pdf.cell(80, 0, "", border="T", ln=1)
    pdf.ln(3)

    # Subject
    pdf.set_font(pdf.font_family, "B", 11)
    pdf.cell(0, 7, "Subject :", ln=1)
    pdf.set_font(pdf.font_family, "", 11)
    pdf.multi_cell(0, 7, subject)
    pdf.ln(1)

    # Items grid (Description | Qty | Rate | Amount)
    base = int(row.get("base_amount", 0))
    disc = int(row.get("discount", 0))
    final = int(row.get("final_cost", 0))
    qty = 1

    # Header
    pdf.set_font(pdf.font_family, "B", 10)
    x0 = 10
    col_w = [100, 20, 35, 35]
    pdf.set_x(x0)
    pdf.cell(col_w[0], 8, "Item & Description", border=1)
    pdf.cell(col_w[1], 8, "Qty", border=1, align="R")
    pdf.cell(col_w[2], 8, "Rate", border=1, align="R")
    pdf.cell(col_w[3], 8, "Amount", border=1, ln=1, align="R")

    # Row 1
    pdf.set_font(pdf.font_family, "", 10)
    desc = f"Travel Package - {_str(row.get('final_route'))} ({_str(row.get('total_pax'))} pax)"
    pdf.set_x(x0)
    pdf.cell(col_w[0], 8, desc, border=1)
    pdf.cell(col_w[1], 8, f"{qty:.2f}", border=1, align="R")
    pdf.cell(col_w[2], 8, _fmt_money(base), border=1, align="R")
    pdf.cell(col_w[3], 8, _fmt_money(base), border=1, ln=1, align="R")

    # Totals box (Sub Total, Discount, Total, Payment Made, Balance Due)
    pdf.ln(1)
    box_w = col_w[1] + col_w[2] + col_w[3]
    totals_x = x0 + col_w[0]
    def trow(label, value, bold=False):
        pdf.set_x(totals_x)
        pdf.set_font(pdf.font_family, "B" if bold else "", 10)
        pdf.cell(col_w[1] + col_w[2], 8, label, border=1)
        pdf.cell(col_w[3], 8, value, border=1, ln=1, align="R")

    trow("Sub Total", _fmt_money(base))
    if disc > 0:
        trow("Discount (-)", _fmt_money(disc))
    trow("Total", _fmt_money(final), bold=True)
    advance = int(row.get("advance_amount", 0))
    if advance > 0:
        trow("Payment Made (-)", _fmt_money(advance))
    balance = max(final - advance, 0)
    trow("Balance Due", _fmt_money(balance), bold=True)

    # Total in words + Notes + Signature block
    pdf.ln(2)
    pdf.set_font(pdf.font_family, "B", 10)
    pdf.cell(0, 6, "Total In Words", ln=1)
    pdf.set_font(pdf.font_family, "", 10)
    pdf.multi_cell(0, 6, _amount_in_words(final))
    pdf.ln(1)

    pdf.set_font(pdf.font_family, "B", 10)
    pdf.cell(0, 6, "Notes", ln=1)
    pdf.set_font(pdf.font_family, "", 10)
    pdf.multi_cell(0, 6, BRAND["notes"])
    pdf.ln(6)

    # Signature lane (right aligned)
    sign_y_top = pdf.get_y()
    if os.path.exists(BRAND["signature_path"]):
        try:
            pdf.image(BRAND["signature_path"], x=130, y=sign_y_top, w=50)
            pdf.set_y(sign_y_top + 22)
        except Exception:
            pass
    pdf.set_x(120)
    pdf.cell(70, 0, "", border="T", ln=1)
    pdf.set_x(120)
    pdf.set_font(pdf.font_family, "", 10)
    pdf.cell(70, 6, "Authorized Signature", ln=1, align="C")

    return _pdf_bytes(pdf)

def build_payment_slip_pdf(row: dict, payment_date: Optional[date]) -> bytes:
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    _font_setup(pdf)

    _header_block(pdf, "PAYMENT RECEIPT")

    inv_no = _str(row.get("ach_id")) or f"INV-{datetime.now():%Y%m%d}"
    slip_date = payment_date or row.get("booking_date") or date.today()
    if not isinstance(slip_date, date):
        slip_date = _safe_date(slip_date) or date.today()

    # Bill To + meta
    pdf.set_font(pdf.font_family, "B", 11); pdf.cell(0, 6, "Bill To:", ln=1)
    pdf.set_font(pdf.font_family, "", 10)
    _two_col_label(pdf, "Client", _str(row.get("client_name")))
    _two_col_label(pdf, "Mobile", _str(row.get("client_mobile")))
    travel = f"{_str(row.get('start_date'))} to {_str(row.get('end_date'))}"
    _two_col_label(pdf, "Travel", travel)
    _two_col_label(pdf, "Route", _str(row.get("final_route")))
    pdf.ln(2)

    pdf.set_font(pdf.font_family, "B", 10)
    pdf.cell(0, 6, "Receipt Details", ln=1)
    pdf.set_font(pdf.font_family, "", 10)
    _two_col_label(pdf, "Invoice #", inv_no)
    _two_col_label(pdf, "Slip Date", slip_date.strftime("%d/%m/%Y"))
    pdf.ln(2)

    # Amount table: Paid, Total, Balance
    base = int(row.get("base_amount", 0))
    disc = int(row.get("discount", 0))
    final = int(row.get("final_cost", 0))
    advance = int(row.get("advance_amount", 0))
    balance = max(final - advance, 0)

    x0 = 10
    col_w = [100, 55, 35]
    pdf.set_font(pdf.font_family, "B", 10)
    pdf.set_x(x0)
    pdf.cell(col_w[0], 8, "Item", border=1)
    pdf.cell(col_w[1], 8, "Description", border=1)
    pdf.cell(col_w[2], 8, "Amount", border=1, ln=1, align="R")

    pdf.set_font(pdf.font_family, "", 10)
    pdf.set_x(x0)
    pdf.cell(col_w[0], 8, "Amount Paid", border=1)
    pdf.cell(col_w[1], 8, f"Advance paid on {slip_date.strftime('%d/%m/%Y')}", border=1)
    pdf.cell(col_w[2], 8, _fmt_money(advance), border=1, ln=1, align="R")

    pdf.set_x(x0)
    pdf.cell(col_w[0], 8, "Total Package Value", border=1)
    pdf.cell(col_w[1], 8, "After discount" if disc>0 else "Package price", border=1)
    pdf.cell(col_w[2], 8, _fmt_money(final), border=1, ln=1, align="R")

    pdf.set_x(x0)
    pdf.cell(col_w[0]+col_w[1], 8, "Balance Due", border=1)
    pdf.cell(col_w[2], 8, _fmt_money(balance), border=1, ln=1, align="R")

    pdf.ln(8)
    # Signature lane
    sign_y_top = pdf.get_y()
    if os.path.exists(BRAND["signature_path"]):
        try:
            pdf.image(BRAND["signature_path"], x=130, y=sign_y_top, w=50)
            pdf.set_y(sign_y_top + 22)
        except Exception:
            pass
    pdf.set_x(120)
    pdf.cell(70, 0, "", border="T", ln=1)
    pdf.set_x(120)
    pdf.set_font(pdf.font_family, "", 10)
    pdf.cell(70, 6, "Authorized Signature", ln=1, align="C")

    return _pdf_bytes(pdf)

# ================= UI =================
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
default_subject = f"2days {_str(row.get('final_route'))} tour package"  # match subject style
subject = st.text_input("Subject line for invoice", value=default_subject)

c1, c2 = st.columns([1,1])
with c1:
    if st.button("Generate Invoice PDF"):
        inv_bytes = build_invoice_pdf(row, subject=subject)
        st.session_state["inv_pdf"] = inv_bytes
with c2:
    pay_date_default = row.get("booking_date") or date.today()
    if not isinstance(pay_date_default, date):
        try: pay_date_default = pd.to_datetime(pay_date_default).date()
        except Exception: pay_date_default = date.today()
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
        height=620, scrolling=True
    )
    st.download_button(
        "⬇️ Download Invoice (PDF)",
        data=st.session_state["inv_pdf"],
        file_name=f"Invoice_{_str(row.get('ach_id'))}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# ===== Preview + Download (Payment Slip) =====
if "slip_pdf" in st.session_state:
    st.markdown("#### Payment slip preview")
    b64s = base64.b64encode(st.session_state["slip_pdf"]).decode()
    st.components.v1.html(
        f'<iframe src="data:application/pdf;base64,{b64s}" width="100%" height="600" style="border:1px solid #ddd;"></iframe>',
        height=620, scrolling=True
    )
    st.download_button(
        "⬇️ Download Payment Slip (PDF)",
        data=st.session_state["slip_pdf"],
        file_name=f"PaymentSlip_{_str(row.get('ach_id'))}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
