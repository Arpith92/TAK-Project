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

# ================= Page config =================
st.set_page_config(page_title="Invoice & Payment Slip", layout="wide")
st.title("🧾 Invoice & Payment Slip (Confirmed packages only)")

# ================= Admin-only gate =================
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

# ================= Mongo boot =================
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

# ================= Helpers =================
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
SIGN_PATH = os.path.join(ASSETS_DIR, "signature.png")
FONT_REG  = os.path.join(ASSETS_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(ASSETS_DIR, "DejaVuSans-Bold.ttf")

def _asset_exists(p: str) -> bool:
    try:
        return os.path.exists(p) and os.path.getsize(p) > 0
    except Exception:
        return False

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

def _fmt_money_inr(n: int, symbol=True) -> str:
    s = f"{int(n):,}"
    return (f"₹{s}" if symbol else s) if _asset_exists(FONT_REG) else f"Rs {s}"

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

# --------- "Amount in words" (Indian format up to 999,99,99,999) ----------
ONES = ["", "One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten",
        "Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"]
TENS = ["","", "Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"]

def _two_digit_words(n):
    if n < 20: return ONES[n]
    return TENS[n//10] + ("" if n%10==0 else f" {ONES[n%10]}")

def inr_words(n: int) -> str:
    if n == 0: return "Zero"
    parts = []
    crores = n // 10_000_000; n %= 10_000_000
    lakhs  = n // 100_000;    n %= 100_000
    thous  = n // 1000;       n %= 1000
    hund   = n // 100;        n %= 100
    tens   = n

    if crores: parts.append(f"{_two_digit_words(crores)} Crore")
    if lakhs:  parts.append(f"{_two_digit_words(lakhs)} Lakh")
    if thous:  parts.append(f"{_two_digit_words(thous)} Thousand")
    if hund:   parts.append(f"{ONES[hund]} Hundred")
    if tens:
        if parts: parts.append("and " + _two_digit_words(tens))
        else: parts.append(_two_digit_words(tens))
    return " ".join(p for p in parts if p).strip()

# ================= PDF builders (FPDF) =================
from fpdf import FPDF

def _has_unicode_font() -> bool:
    return _asset_exists(FONT_REG) and _asset_exists(FONT_BOLD)

def _apply_font(pdf: FPDF, bold=False, size=10):
    if _has_unicode_font():
        # registered names
        pdf.set_font("DejaVu", "B" if bold else "", size)
    else:
        pdf.set_font("Helvetica", "B" if bold else "", size)

def _register_fonts(pdf: FPDF):
    if _has_unicode_font():
        try:
            pdf.add_font("DejaVu", "", FONT_REG, uni=True)
        except Exception:
            pass
        try:
            pdf.add_font("DejaVu", "B", FONT_BOLD, uni=True)
        except Exception:
            pass

def _pdf_bytes(pdf: FPDF) -> bytes:
    out = pdf.output(dest="S")
    return out if isinstance(out, (bytes, bytearray)) else str(out).encode("latin-1", "ignore")

class TemplatePDF(FPDF):
    def header(self):
        # Top strip
        if _asset_exists(LOGO_PATH):
            try:
                self.image(LOGO_PATH, x=10, y=10, w=28)
            except Exception:
                pass
        _apply_font(self, bold=True, size=14)
        self.cell(0, 8, "TAX INVOICE", ln=1, align="R")
        # Company block
        _apply_font(self, bold=True, size=11)
        self.set_xy(10, 16)
        self.cell(0, 6, "Achala Holidays Pvt Limited")
        self.ln(5)
        _apply_font(self, size=9)
        self.set_x(10)
        self.multi_cell(90, 5, "Mangrola\nUjjain Madhya Pradesh 456006\nIndia\ntravelaajkal@gmail.com\nwww.travelaajkal.com")
        # Divider
        self.set_draw_color(200,200,200)
        self.set_line_width(0.2)
        self.line(10, 45, 200, 45)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        _apply_font(self, size=8)
        self.cell(0, 8, f"Powered by TAK • Generated {datetime.now().strftime('%d/%m/%Y %H:%M')}", align="R")

def _field(pdf, label, value, w_label=26, w_value=58, h=6):
    _apply_font(pdf, bold=True, size=9)
    pdf.cell(w_label, h, label, border=0)
    _apply_font(pdf, size=9)
    pdf.cell(w_value, h, str(value), border=0)

def build_invoice_pdf(row: dict, subject: str) -> bytes:
    pdf = TemplatePDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    _register_fonts(pdf)
    pdf.add_page()

    # Right meta panel (Invoice #, Date, Terms, Due Date)
    x0, y0 = 120, 12
    pdf.set_xy(x0, y0)
    _apply_font(pdf, bold=True, size=11); pdf.cell(0, 6, f"# : {_str(row.get('ach_id'))}", ln=1)
    pdf.set_x(x0); _field(pdf, "Invoice Date :", datetime.now().strftime("%d/%m/%Y")); pdf.ln(6)
    pdf.set_x(x0); _field(pdf, "Terms :", "Due on Receipt"); pdf.ln(6)
    pdf.set_x(x0); _field(pdf, "Due Date :", datetime.now().strftime("%d/%m/%Y")); pdf.ln(8)

    # Bill To
    _apply_font(pdf, bold=True, size=11)
    pdf.set_xy(10, 48)
    pdf.cell(0, 6, "Bill To", ln=1)
    _apply_font(pdf, size=10)
    pdf.cell(0, 6, _str(row.get("client_name")), ln=1)
    pdf.ln(2)

    # Subject
    _apply_font(pdf, bold=True, size=10)
    pdf.cell(0, 6, "Subject :", ln=1)
    _apply_font(pdf, size=10)
    pdf.multi_cell(0, 6, subject)
    pdf.ln(2)

    # Items table header
    _apply_font(pdf, bold=True, size=10)
    pdf.set_fill_color(245,245,245)
    pdf.cell(8,  8, "#", 1, 0, "C", True)
    pdf.cell(102, 8, "Item & Description", 1, 0, "L", True)
    pdf.cell(20, 8, "Qty", 1, 0, "C", True)
    pdf.cell(30, 8, "Rate", 1, 0, "R", True)
    pdf.cell(30, 8, "Amount", 1, 1, "R", True)

    # Single line item derived from package
    _apply_font(pdf, size=10)
    desc = f"1. Sedan Car  2. Hotel Room at 3 Star Hotel\nRoute: {_str(row.get('final_route'))} • {_str(row.get('total_pax'))} pax • Travel: {_str(row.get('start_date'))} to {_str(row.get('end_date'))}"
    base = int(row.get("base_amount", 0))
    disc = int(row.get("discount", 0))
    final = int(row.get("final_cost", 0))

    y_before = pdf.get_y()
    pdf.cell(8,  10, "1", 1, 0, "C")
    # description cell (multi-line): draw a cell and write inside
    x_desc = pdf.get_x(); y_desc = pdf.get_y()
    pdf.multi_cell(102, 10, desc, border=1)
    # go back to the right cells same row height
    row_h = max(10, pdf.get_y() - y_before)
    pdf.set_xy(x_desc+102, y_before)
    pdf.cell(20, row_h, "1.00", 1, 0, "C")
    pdf.cell(30, row_h, _fmt_money_inr(base, symbol=False), 1, 0, "R")
    pdf.cell(30, row_h, _fmt_money_inr(base, symbol=False), 1, 1, "R")

    # Totals block (right)
    pdf.ln(2)
    x_tot = 120
    pdf.set_xy(x_tot, pdf.get_y())

    def _tot_row(label, val, bold=False):
        _apply_font(pdf, bold=bold, size=10)
        pdf.cell(40, 8, label, 1, 0, "R")
        pdf.cell(30, 8, _fmt_money_inr(val), 1, 1, "R")

    _tot_row("Sub Total", base)
    _tot_row("Discount (-)", -disc)
    _tot_row("Total", final, bold=True)
    _tot_row("Payment Made (-)", -int(row.get("advance_amount", 0)))
    bal = max(final - int(row.get("advance_amount", 0)), 0)
    _tot_row("Balance Due", bal, bold=True)

    # Amount in words
    pdf.ln(2)
    _apply_font(pdf, bold=True, size=10)
    pdf.cell(0, 6, "Total In Words", ln=1)
    _apply_font(pdf, size=10)
    pdf.multi_cell(0, 6, f"Indian Rupee {inr_words(final)} Only")

    # Notes
    pdf.ln(2)
    _apply_font(pdf, bold=True, size=10); pdf.cell(0,6,"Notes", ln=1)
    _apply_font(pdf, size=9)
    pdf.multi_cell(0,5,"Thanks for your business.")

    # Signature
    pdf.ln(6)
    if _asset_exists(SIGN_PATH):
        try:
            y_sig = pdf.get_y()
            pdf.image(SIGN_PATH, x=150, y=y_sig, w=35)
            pdf.set_y(y_sig + 22)
        except Exception:
            pass
    _apply_font(pdf, size=10)
    pdf.cell(0,6,"Authorized Signature", ln=1, align="R")

    return _pdf_bytes(pdf)

def build_payment_slip_pdf(row: dict, payment_date: Optional[date]) -> bytes:
    pdf = TemplatePDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    _register_fonts(pdf)
    pdf.add_page()

    # Header right label
    pdf.set_xy(120, 12)
    _apply_font(pdf, bold=True, size=11); pdf.cell(0, 6, "PAYMENT SLIP", ln=1)

    # Client block
    _apply_font(pdf, bold=True, size=11)
    pdf.set_xy(10, 48); pdf.cell(0, 6, "Client", ln=1)
    _apply_font(pdf, size=10)
    pdf.cell(0, 6, f"{_str(row.get('client_name'))}", ln=1)
    pdf.cell(0, 6, f"Mobile: {_str(row.get('client_mobile'))}", ln=1)
    pdf.cell(0, 6, f"Route: {_str(row.get('final_route'))}", ln=1)
    pdf.cell(0, 6, f"Travel: {_str(row.get('start_date'))} to {_str(row.get('end_date'))}", ln=1)
    pdf.ln(2)

    # Slip meta
    slip_date = payment_date or row.get("booking_date")
    slip_date_str = _str(slip_date)
    _apply_font(pdf, size=10)
    _field(pdf, "Slip Date :", slip_date_str); pdf.ln(6)
    _field(pdf, "ACH ID :", _str(row.get("ach_id"))); pdf.ln(8)

    # Table
    _apply_font(pdf, bold=True, size=10)
    pdf.set_fill_color(245,245,245)
    pdf.cell(120, 8, "Item", 1, 0, "L", True)
    pdf.cell(60, 8, "Amount", 1, 1, "R", True)
    _apply_font(pdf, size=10)

    adv = int(row.get("advance_amount", 0))
    final = int(row.get("final_cost", 0))
    bal  = max(final - adv, 0)

    pdf.cell(120, 8, "Amount Paid (Advance)", 1)
    pdf.cell(60, 8, _fmt_money_inr(adv), 1, 1, "R")

    pdf.cell(120, 8, "Total Package Value", 1)
    pdf.cell(60, 8, _fmt_money_inr(final), 1, 1, "R")

    _apply_font(pdf, bold=True, size=10)
    pdf.cell(120, 8, "Balance Due", 1)
    pdf.cell(60, 8, _fmt_money_inr(bal), 1, 1, "R")

    # Note + signature
    pdf.ln(6)
    _apply_font(pdf, size=9)
    pdf.multi_cell(0, 5, f"Payment received on: {slip_date_str}. This is a computer generated receipt.")
    pdf.ln(6)
    if _asset_exists(SIGN_PATH):
        try:
            y_sig = pdf.get_y()
            pdf.image(SIGN_PATH, x=150, y=y_sig, w=35)
            pdf.set_y(y_sig + 22)
        except Exception:
            pass
    _apply_font(pdf, size=10)
    pdf.cell(0,6,"Authorized Signature", ln=1, align="R")

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
    choice = st.selectbox("Choose", opts.tolist() if not opts.empty else [])
    chosen_id = choice.split(" | ")[-1] if choice else None

st.divider()
if not chosen_id:
    st.stop()

row = df[df["itinerary_id"] == chosen_id].iloc[0].to_dict()

st.subheader("🧾 Invoice")
default_subject = f"2days {_str(row.get('final_route'))} tour package"
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
