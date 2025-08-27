# App.py
from __future__ import annotations

# ----------------- Compatibility safety -----------------
try:
    import streamlit as st, rich
    from packaging.version import Version
    import sys, subprocess
    if Version(st.__version__) < Version("1.42.0") and Version(rich.__version__) >= Version("14.0.0"):
        subprocess.run([sys.executable, "-m", "pip", "install", "rich==13.9.4"], check=True)
        st.warning("Adjusted rich to 13.9.4 for compatibility. Rerunning…")
        st.experimental_rerun()
except Exception:
    import streamlit as st  # ensure st is available

# ----------------- Imports -----------------
import io, math, locale, datetime, os
from collections.abc import Mapping
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from pymongo import MongoClient

IST = ZoneInfo("Asia/Kolkata")

# ----------------- App config -----------------
st.set_page_config(page_title="TAK – Itinerary Generator", layout="wide")
st.title("🧭 TAK Project – Itinerary Generator")

# ----------------- Masters URLs -----------------
CODE_FILE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Code.xlsx"
BHASMARATHI_TYPE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Bhasmarathi_Type.xlsx"
STAY_CITY_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Stay_City.xlsx"

# ================= LOGIN (PIN) =================
def _load_users() -> dict:
    """Load [users] from Secrets or local .streamlit/secrets.toml."""
    try:
        raw = st.secrets.get("users", {})
        if isinstance(raw, Mapping): return dict(raw)
        if isinstance(raw, dict):    return raw
    except Exception:
        pass
    try:
        try:
            import tomllib
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, Mapping): return dict(u)
        if isinstance(u, dict):    return u
    except Exception:
        pass
    return {}

def _login() -> str | None:
    with st.sidebar:
        if st.session_state.get("user"):
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out"):
                st.session_state.pop("user", None)
                st.rerun()

    if st.session_state.get("user"):
        return st.session_state["user"]

    users_map = _load_users()
    if not users_map:
        with st.sidebar:
            st.caption("Secrets debug")
            try:
                st.write("Visible secret keys:", list(st.secrets.keys()))
                if "users" in st.secrets:
                    st.write("type(users):", type(st.secrets["users"]).__name__)
            except Exception:
                st.write("secrets unavailable")
        st.error("Login not configured. Add a **[users]** section in Secrets with PINs.")
        st.stop()

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns([1, 1])
    with c1:
        name = st.selectbox("User", list(users_map.keys()), key="login_user")
    with c2:
        pin = st.text_input("PIN", type="password", key="login_pin")

    if st.button("Sign in"):
        if str(users_map.get(name, "")).strip() == str(pin).strip():
            st.session_state["user"] = name
            try:
                audit_login(name)  # IST audit
            except Exception:
                pass
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid PIN"); st.stop()
    return None

# ================= Mongo =======================
def _find_uri() -> str | None:
    for k in ("mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"):
        try:
            v = st.secrets.get(k)
        except Exception:
            v = None
        if v: return v
    for k in ("mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"):
        v = os.getenv(k)
        if v: return v
    return None

@st.cache_resource
def mongo_client():
    uri = _find_uri()
    if not uri:
        st.error("Mongo URI not configured. Add `mongo_uri` in Secrets.")
        st.stop()
    client = MongoClient(uri, appName="TAK_App", maxPoolSize=100, serverSelectionTimeoutMS=5000, tz_aware=True)
    client.admin.command("ping")
    return client

@st.cache_resource
def get_collections():
    client = mongo_client()
    db = client["TAK_DB"]
    return {
        "itineraries": db["itineraries"],
        "updates": db["package_updates"],
        "expenses": db["expenses"],
        "followups": db["followups"],
        "splitwise": db["expense_splitwise"],
        "vendor_expenses": db["vendor_expenses"],
        "audit_logins": db["audit_logins"],
    }

cols = get_collections()
col_it = cols["itineraries"]
col_audit = cols["audit_logins"]
col_it_prod = cols["itineraries"]  # for referral source search

# ================= Audit helper =================
def audit_login(user: str):
    """Write a login audit row with IST timestamp. Safe on Cloud."""
    now_utc = datetime.datetime.utcnow()
    try:
        doc = {
            "user": str(user),
            "ts_utc": now_utc,
            "ts_ist": now_utc.replace(tzinfo=datetime.timezone.utc).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "page": "App.py",
        }
        col_audit.insert_one(doc)
    except Exception:
        pass

# ensure a login exists if user already in session (first entry to page)
user = _login()
if not user:
    st.stop()

# ================= Caching helpers =================
@st.cache_data(ttl=900)
def read_excel_from_url(url, sheet_name=None):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name)

# ================= Load static masters (cached) =================
try:
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
    bhasmarathi_type_df = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet_name="Bhasmarathi_Type")
except Exception as e:
    st.error(f"Failed to load master sheets: {e}")
    st.stop()

# ================= Small utils =================
def is_valid_mobile(num: str) -> bool:
    digits = "".join(ch for ch in str(num or "") if ch.isdigit())
    return len(digits) == 10

def in_locale(n: int) -> str:
    try:
        locale.setlocale(locale.LC_ALL, 'en_IN')
        return locale.format_string("%d", int(n), grouping=True)
    except Exception:
        return f"{int(n):,}"

def ceil_to_999(n: float) -> int:
    return (math.ceil(n/1000)*1000 - 1) if n > 0 else 0

# ================= Input mode toggle =================
st.markdown("### 1) Provide Input")
mode = st.radio(
    "Input Mode",
    ["Form Table (No Excel)", "Excel Upload"],
    horizontal=True,
    index=0,
    help="Use an editable table instead of uploading Excel. Old Excel flow is still available."
)

# ===== Common header inputs (both modes) =====
c0, c1, c2, c3 = st.columns([1.4, 1, 1, 1])
with c0:
    sheet = st.text_input("Client Name*", placeholder="e.g., Mayur Gupta / Family", key="client_name_form")
with c1:
    client_mobile_raw = st.text_input("Client mobile (10 digits)*", help="Used to uniquely identify the itinerary.")
with c2:
    rep = st.selectbox("Representative*", ["-- Select --", "Arpith", "Reena", "Kuldeep", "Teena"])
with c3:
    default_pax = st.number_input("Total Pax (default)", min_value=1, value=2, step=1)

# ----- Referred By (from confirmed clients in PROD) -----
def _load_client_refs() -> list[dict]:
    try:
        cur = col_it_prod.aggregate([
            {"$group": {"_id": {"name":"$client_name","mobile":"$client_mobile"}}},
            {"$project": {"_id":0, "name":"$_id.name", "mobile":"$_id.mobile"}}
        ])
        items = []
        for x in cur:
            n = (x.get("name") or "").strip()
            m = (x.get("mobile") or "").strip()
            label = f"{n} — {m}" if n and m else n or m
            if label: items.append({"label": label, "name": n, "mobile": m})
        items = sorted({i["label"]: i for i in items}.values(), key=lambda k: k["label"].lower())
        return items
    except Exception:
        return []

ref_items = _load_client_refs()
ref_labels = ["-- None --"] + [i["label"] for i in ref_items]
referred_sel = st.selectbox("Referred By (select confirmed client; gives 10% discount)", ref_labels, index=0)
referred_by = None if referred_sel == "-- None --" else referred_sel
has_ref = referred_by is not None
discount_pct = 10 if has_ref else 0

# ====== Bhasmarathi (outside the table) ======
bcol1, bcol2, bcol3, bcol4 = st.columns([0.9, 1, 1, 1.2])
with bcol1:
    bhas_required = st.selectbox("Bhasmarathi required?", ["No", "Yes"], index=0)
with bcol2:
    bhas_options = ["V-BH", "P-BH", "BH"]
    bhas_type = st.selectbox("Bhasmarathi Type", bhas_options, index=0, disabled=(bhas_required=="No"))
with bcol3:
    bhas_pkg_cost = st.number_input("Bhasmarathi Package Cost (₹)", min_value=0, value=0, step=100, disabled=(bhas_required=="No"))
with bcol4:
    bhas_actual_cost = st.number_input("Bhasmarathi Actual Cost (₹)", min_value=0, value=0, step=100, disabled=(bhas_required=="No"))

# ====== Cost header (top) ======
hc1, hc2, hc3 = st.columns([1,1,1])
with hc1:
    header_package_cost = st.number_input("Quoted Package Cost (₹)", min_value=0, value=0, step=500)
with hc2:
    header_actual_cost = st.number_input("Estimated Actual Cost (₹)", min_value=0, value=0, step=500, help="Optional override for actuals.")
with hc3:
    st.caption("If **Referred By** is chosen → auto 10% discount shown & saved.")

# ====== MODE A: FORM TABLE (No Excel) ======
if mode == "Form Table (No Excel)":
    h1, h2, h3 = st.columns([1, 1, 1])
    with h1:
        start_date_input = st.date_input("Start date", value=datetime.date.today())
    with h2:
        days = st.number_input("No. of days", min_value=1, value=3, step=1)
    with h3:
        st.caption("Use + Add row at bottom of the table if needed.")

    # Build persistent table only once
    if "form_rows" not in st.session_state:
        dates = [start_date_input + datetime.timedelta(days=i) for i in range(days)]
        st.session_state.form_rows = pd.DataFrame({
            "Date": dates,
            "Time": [""] * days,
            "Code": [""] * days,
            "Car Type": [""] * days,
            "Hotel Type": [""] * days,
            "Stay City": [""] * days,
            "Room Type": [""] * days,
            "Pkg Cost (item)": [0.0] * days,
            "Actual Cost (item)": [0.0] * days,
            "Total Pax": [default_pax] * days,
        })

    def _reset_rows():
        dates = [start_date_input + datetime.timedelta(days=i) for i in range(days)]
        st.session_state.form_rows = pd.DataFrame({
            "Date": dates,
            "Time": [""] * days,
            "Code": [""] * days,
            "Car Type": [""] * days,
            "Hotel Type": [""] * days,
            "Stay City": [""] * days,
            "Room Type": [""] * days,
            "Pkg Cost (item)": [0.0] * days,
            "Actual Cost (item)": [0.0] * days,
            "Total Pax": [default_pax] * days,
        })
    st.button("↻ Reset rows to new dates/days", on_click=_reset_rows)

    # Dropdown options
    code_options = code_df["Code"].dropna().astype(str).unique().tolist() if not code_df.empty else []
    stay_city_options = sorted(stay_city_df["Stay City"].dropna().astype(str).unique().tolist()) if "Stay City" in stay_city_df.columns else []
    # Car options with AC/Non-AC
    base_cars = ["Sedan","Ertiga","Innova","Tempo Traveller"]
    car_options = []
    for c in base_cars:
        car_options += [f"AC {c}", f"Non AC {c}"]

    col_cfg = {
        "Date": st.column_config.DateColumn("Date"),
        "Time": st.column_config.TextColumn("Time"),
        "Code": st.column_config.SelectboxColumn("Code", options=code_options, help="Searchable"),
        "Car Type": st.column_config.SelectboxColumn("Car Type", options=car_options, help="Search with 1 letter"),
        "Hotel Type": st.column_config.TextColumn("Hotel Type"),
        "Stay City": st.column_config.SelectboxColumn("Stay City", options=stay_city_options),
        "Room Type": st.column_config.TextColumn("Room Type"),
        "Pkg Cost (item)": st.column_config.NumberColumn("Pkg Cost (item)", min_value=0.0, step=100.0),
        "Actual Cost (item)": st.column_config.NumberColumn("Actual Cost (item)", min_value=0.0, step=100.0),
        "Total Pax": st.column_config.NumberColumn("Total Pax", min_value=1, step=1),
    }

    st.markdown("### Fill Line Items")
    edited_df = st.data_editor(
        st.session_state.form_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config=col_cfg,
        hide_index=True
    )
    st.session_state.form_rows = edited_df.copy()
    df = edited_df.copy()

# ====== MODE B: EXCEL UPLOAD (Original flow) ======
else:
    uploaded = st.file_uploader("Choose file (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.info("Upload the Excel to proceed.")
        st.stop()
    try:
        xls = pd.ExcelFile(uploaded)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()

    # pick sheet as client name if not typed
    if not sheet:
        sheet = st.selectbox("Select client (sheet name)", xls.sheet_names, index=0, key="sheet_pick")
    # keep rep selection consistent
    if rep == "-- Select --":
        rep = st.selectbox("Representative*", ["-- Select --", "Arpith", "Reena", "Kuldeep", "Teena"], key="rep_excel")

    if not (sheet and rep != "-- Select --"):
        st.stop()

    try:
        df = xls.parse(sheet)
    except Exception as e:
        st.error(f"Error reading sheet: {e}")
        st.stop()

    # Ensure columns present (compat with old structure)
    for col in ["Date","Time","Code","Car Type","Hotel Type","Stay City","Room Type","Pkg Cost (item)","Actual Cost (item)","Total Pax"]:
        if col not in df.columns:
            # Map old cost columns if present
            if col == "Pkg Cost (item)" and "Car Cost" in df.columns or "Hotel Cost" in df.columns or "Bhasmarathi Cost" in df.columns:
                df["Pkg Cost (item)"] = pd.to_numeric(df.get("Car Cost",0), errors="coerce").fillna(0) \
                                        + pd.to_numeric(df.get("Hotel Cost",0), errors="coerce").fillna(0)
            elif col == "Actual Cost (item)":
                df[col] = 0.0
            elif col == "Total Pax":
                df[col] = df.get("Total Pax", default_pax)
            else:
                df[col] = ""

# ---------- Common compute from df (both modes) ----------
required_cols = ["Date", "Total Pax"]
if any(c not in df.columns for c in required_cols):
    st.error(f"Required columns missing: {', '.join([c for c in required_cols if c not in df.columns])}")
    st.stop()

dates = pd.to_datetime(df["Date"], errors="coerce")
if dates.isna().all():
    st.error("No valid dates found.")
    st.stop()

start_date = dates.min().date()
end_date = dates.max().date()
total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
total_nights = max(total_days - 1, 0)
total_pax = int(pd.to_numeric(df["Total Pax"].iloc[0], errors="coerce") or default_pax)

# ================= Build itinerary lines from Code master =================
def _code_to_desc(code: str) -> str:
    if not code: return "No code provided"
    m = code_df.loc(code_df["Code"].astype(str) == str(code), "Particulars") if hasattr(code_df, "loc") else None
    m = code_df.loc[code_df["Code"].astype(str) == str(code), "Particulars"]
    return (m.iloc[0] if not m.empty else f"No description found for code {code}")

def _code_to_route(code: str) -> str | None:
    if not code: return None
    m = code_df.loc[code_df["Code"].astype(str) == str(code), "Route"]
    return (str(m.iloc[0]) if not m.empty else None)

itinerary = []
for _, r in df.iterrows():
    code = str(r.get("Code","") or "")
    desc = _code_to_desc(code) if not code_df.empty else ("No code provided" if not code else f"No description found for code {code}")
    itinerary.append({
        "Date": r.get("Date", ""),
        "Time": r.get("Time", ""),
        "Description": str(desc)
    })

# ================= Derive route =================
route_parts = []
if "Code" in df.columns and "Route" in code_df.columns:
    for c in df["Code"].astype(str):
        rt = _code_to_route(c)
        if rt: route_parts.append(rt)
route_raw = "-".join(route_parts).replace(" -", "-").replace("- ", "-")
route_list = [x for x in route_raw.split("-") if x]
final_route = "-".join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i-1]])

# ================= Package/Actual from items + bhas =================
sum_pkg_items = float(pd.to_numeric(df.get("Pkg Cost (item)"), errors="coerce").fillna(0).sum())
sum_actual_items = float(pd.to_numeric(df.get("Actual Cost (item)"), errors="coerce").fillna(0).sum())
actual_total_from_items = sum_actual_items + (bhas_actual_cost if bhas_required=="Yes" else 0)
package_total_from_items = sum_pkg_items + (bhas_pkg_cost if bhas_required=="Yes" else 0)

actual_total = header_actual_cost if header_actual_cost > 0 else actual_total_from_items
package_total = header_package_cost if header_package_cost > 0 else package_total_from_items
package_total = ceil_to_999(package_total)

package_after_discount = int(round(package_total * (1 - (discount_pct/100.0)))) if has_ref else package_total
profit = int(package_total - actual_total)

# ================= Type strings =================
car_types = "-".join(pd.Series(df.get("Car Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")
hotel_types = "-".join(pd.Series(df.get("Hotel Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")

# Bhas desc (if selected)
bhas_desc_str = ""
if bhas_required == "Yes":
    mm = bhasmarathi_type_df.loc[bhasmarathi_type_df["Bhasmarathi Type"].astype(str) == str(bhas_type), "Description"]
    if not mm.empty: bhas_desc_str = str(mm.iloc[0])

# ================= Header badges =================
badge_color = "#16a34a" if profit >= 4000 else "#dc2626"
hint = "" if profit >= 4000 else " • Keep profit margin ≥ ₹4,000"

st.markdown(
    f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; margin-bottom:8px;">
      <div style="padding:8px 12px; border-radius:8px; background:#0ea5e9; color:white;">
        Package Cost: <b>₹{in_locale(package_total)}</b>
      </div>
      {('<div style="padding:8px 12px; border-radius:8px; background:#7c3aed; color:white;">After Referral (10%): <b>₹'+in_locale(package_after_discount)+'</b></div>') if has_ref else ''}
      <div style="padding:8px 12px; border-radius:8px; background:#475569; color:white;">
        Actual Cost: <b>₹{in_locale(actual_total)}</b>
      </div>
      <div style="padding:8px 12px; border-radius:8px; background:%s; color:white;">
        Profit: <b>₹%s</b>%s
      </div>
    </div>
    """ % (badge_color, in_locale(profit), hint),
    unsafe_allow_html=True
)

# ================= Text build =================
night_txt = "Night" if total_nights == 1 else "Nights"
person_txt = "Person" if total_pax == 1 else "Persons"

if not sheet:
    st.warning("Enter **Client Name**.")
    st.stop()
if not client_mobile_raw or not is_valid_mobile(client_mobile_raw):
    st.error("Enter a valid **10-digit** mobile. Package will not be created without it.")
    st.stop()
if rep == "-- Select --":
    st.warning("Please select **Representative**.")
    st.stop()

client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())

greet = f"Greetings from TravelAajkal,\n\n*Client Name: {sheet}*\n\n"
plan = f"*Plan:- {total_days}Days and {total_nights}{night_txt} {final_route} for {total_pax} {person_txt}*"

# group by day
grouped = {}
for e in itinerary:
    dstr = pd.to_datetime(e["Date"]).strftime("%d-%b-%Y") if pd.notna(e["Date"]) and str(e["Date"]) else "N/A"
    time_part = f"{e.get('Time','')}: " if str(e.get('Time','')).strip() else ""
    grouped.setdefault(dstr, []).append(f"{time_part}{e['Description']}")

itinerary_text = greet + plan + "\n\n*Itinerary:*\n"
for i, (d, evs) in enumerate(grouped.items(), 1):
    itinerary_text += f"\n*Day{i}:{d}*\n" + "\n".join(evs) + "\n"

details_bits = [x for x in [car_types or None, hotel_types or None, bhas_desc_str or None] if x]
details_line = "(" + ",".join(details_bits) + ")" if details_bits else ""

pkg_cost_fmt = in_locale(package_total)
itinerary_text += f"\n*Package cost: ₹{pkg_cost_fmt}/-*\n"
if has_ref:
    itinerary_text += f"*Package cost (after referral 10%): ₹{in_locale(package_after_discount)}/-*\n"
itinerary_text += f"{details_line}"

# Inclusions (short & dynamic)
inc = []
if car_types:
    inc += [
        f"Entire travel as per itinerary by {car_types}.",
        "Toll, parking, and driver bata are included.",
        "Airport/ Railway station pickup and drop."
    ]
if bhas_desc_str:
    inc += [
        f"{bhas_desc_str} for {total_pax} {person_txt}.",
        "Bhasm-Aarti pickup and drop."
    ]
if "Stay City" in df.columns and "Room Type" in df.columns and not stay_city_df.empty:
    stay_series = df["Stay City"].astype(str).fillna("")
    city_nights = stay_series[stay_series != ""].value_counts().to_dict()
    used = 0
    for stay_city, nn in city_nights.items():
        if used >= total_nights: break
        match = stay_city_df[stay_city_df["Stay City"].astype(str) == str(stay_city)]
        if not match.empty:
            city_name = match["City"].iloc[0]
            rt = df.loc[df["Stay City"].astype(str) == str(stay_city), "Room Type"].dropna().astype(str).unique()
            inc.append(f"{min(nn, total_nights-used)}Night stay in {city_name} with {'/'.join(rt) or 'room'} in {hotel_types or 'hotel'}.")
            used += nn
if hotel_types:
    inc += [
        "*Standard check-in at 12:00 PM and check-out at 09:00 AM.*",
        "Early check-in and late check-out are subject to room availability."
    ]
inclusions = "*Inclusions:-*\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(inc)]) if inc else ""

# Exclusions & notes (from original)
exclusions = "*Exclusions:-*\n" + "\n".join([
    "1. Any meals/beverages not specified (breakfast/lunch/dinner/snacks/personal drinks).",
    "2. Entry fees for attractions/temples unless included.",
    "3. Travel insurance.",
    "4. Personal shopping/tips.",
    "5. Early check-in/late check-out if rooms unavailable.",
    "6. Natural events/roadblocks/personal itinerary changes.",
    "7. Extra sightseeing not listed."
])
notes = "\n*Important Notes:-*\n" + "\n".join([
    "1. Any attractions not in itinerary will be chargeable.",
    "2. Visits subject to traffic/temple rules; closures are beyond control & non-refundable.",
    "3. Bhasm-Aarti: we provide tickets; arrival/seating beyond our control; cost at actuals; subject to availability & cancellations by temple.",
    "4. Hotel entry as per rules; valid ID required; only married couples allowed.",
    "5. >9 yrs considered adult; <9 yrs share bed; extra bed chargeable."
])
cxl = """
*Cancellation Policy:-*
1. 30+ days → 20% of advance deducted.
2. 15–29 days → 50% of advance deducted.
3. <15 days → No refund on advance.
4. No refund for no-shows/early departures.
5. One-time reschedule allowed ≥15 days prior, subject to availability.
"""
pay = """*Payment Terms:-*
50% advance and remaining 50% after arrival at Ujjain.
"""
acct = """For booking confirmation, please make the advance payment to the company's current account provided below.

*Company Account details:-*
Account Name: ACHALA HOLIDAYS PVT LTD
Bank: Axis Bank
Account No: 923020071937652
IFSC Code: UTIB0000329
MICR Code: 452211003
Branch Address: Ground Floor, 77, Dewas Road, Ujjain, Madhya Pradesh 456010

Regards,
Team TravelAajKal™️
Reg. Achala Holidays Pvt Limited
Visit :- www.travelaajkal.com
Follow us :- https://www.instagram.com/travelaaj_kal/

*Great news! ACHALA HOLIDAYS PVT LTD is now a DPIIT-recognized Startup by the Government of India.*
*Thank you for your support as we continue to redefine travel.*
*Travel Aaj aur Kal with us!*

TravelAajKal® is a registered trademark of Achala Holidays Pvt Ltd.
"""

final_output = (
    itinerary_text
    + ("\n\n" + inclusions if inclusions else "")
    + "\n\n" + exclusions + "\n\n" + notes + "\n\n" + cxl + "\n\n" + pay + "\n\n" + acct
)

# ================= Auto-save / Upsert (PROD) =================
key_filter = {"client_mobile": client_mobile, "start_date": str(start_date)}
record = {
    "client_name": sheet,
    "client_mobile": client_mobile,
    "representative": rep,
    "upload_date": datetime.datetime.utcnow(),
    "start_date": str(start_date),
    "end_date": str(end_date),
    "total_days": int(total_days),
    "total_pax": int(total_pax),
    "final_route": final_route,
    "car_types": car_types,
    "hotel_types": hotel_types,
    "bhasmarathi_required": (bhas_required=="Yes"),
    "bhasmarathi_type": bhas_type if bhas_required=="Yes" else None,
    "bhasmarathi_pkg_cost": int(bhas_pkg_cost) if bhas_required=="Yes" else 0,
    "bhasmarathi_actual_cost": int(bhas_actual_cost) if bhas_required=="Yes" else 0,
    "sum_pkg_items": int(sum_pkg_items),
    "sum_actual_items": int(sum_actual_items),
    "header_package_cost": int(header_package_cost),
    "header_actual_cost": int(header_actual_cost),
    "package_total": int(package_total),
    "package_after_referral": int(package_after_discount) if has_ref else int(package_total),
    "actual_total": int(actual_total),
    "profit": int(profit),
    "referred_by": referred_by,
    "referral_discount_pct": discount_pct,
    "bhasmarathi_types": bhas_desc_str,  # keep the field name used earlier for compatibility
    "itinerary_text": final_output
}

# avoid multiple banners on rerun
saved_key = f"{client_mobile}|{start_date}"
already = st.session_state.get("_last_saved_key") == saved_key

try:
    res = col_it.update_one(key_filter, {"$set": record}, upsert=True)
    st.session_state["_last_saved_key"] = saved_key
    if not already:
        if res.upserted_id:
            st.success(f"✅ Saved new itinerary (ID: {res.upserted_id}).")
        else:
            st.info("🔁 Updated existing itinerary for this mobile + start date.")
except Exception as e:
    st.error(f"Could not save itinerary: {e}")

# ================= Preview & Download =================
st.markdown("### 2) Preview & Share")
c1, c2 = st.columns([1, 1])
with c1:
    st.text_area("Preview (copy from here)", final_output, height=420)
with c2:
    st.download_button(
        label="⬇️ Download itinerary as .txt",
        data=final_output,
        file_name=f"itinerary_{sheet}_{start_date}.txt",
        mime="text/plain",
        use_container_width=True
    )
