# pages/01_Test – Itinerary Form (Writes to DB).py
from __future__ import annotations

import io, math, locale, datetime, os
from collections.abc import Mapping
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import streamlit as st
from pymongo import MongoClient

IST = ZoneInfo("Asia/Kolkata")

st.set_page_config(page_title="TAK – Test Itinerary (DB Write)", layout="wide")
st.title("🧪 TEST – Itinerary Form (Writes to DB)")

# ------------- Masters URLs (same as main app) -------------
CODE_FILE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Code.xlsx"
BHASMARATHI_TYPE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Bhasmarathi_Type.xlsx"
STAY_CITY_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Stay_City.xlsx"

# ================= LOGIN (PIN) =================
def _load_users() -> dict:
    try:
        raw = st.secrets.get("users", {})
        if isinstance(raw, Mapping): return dict(raw)
        if isinstance(raw, dict):    return raw
    except Exception:
        pass
    try:
        try:
            import tomllib  # py311+
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
        st.error("Login not configured. Add a **[users]** section in Secrets with PINs.")
        st.stop()

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns([1, 1])
    with c1:
        name = st.selectbox("User", list(users_map.keys()), key="login_user_test")
    with c2:
        pin = st.text_input("PIN", type="password", key="login_pin_test")

    if st.button("Sign in", key="signin_test"):
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
    client = MongoClient(uri, appName="TAK_App_Test", maxPoolSize=100, serverSelectionTimeoutMS=5000, tz_aware=True)
    client.admin.command("ping")
    return client

@st.cache_resource
def get_collections():
    client = mongo_client()
    db = client["TAK_DB"]
    return {
        # separate test collection
        "itineraries_test": db["itineraries_test"],
    }

col_test = get_collections()["itineraries_test"]

# ================= Helpers ====================
@st.cache_data(ttl=900)
def read_excel_from_url(url, sheet_name=None):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name)

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
    import math
    return (math.ceil(n/1000)*1000 - 1) if n > 0 else 0

# ================= Load Masters =================
try:
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
    bhasmarathi_type_df = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet_name="Bhasmarathi_Type")
except Exception as e:
    st.error(f"Failed to load master sheets: {e}")
    st.stop()

st.info("This is a TEST page. Data writes to **TAK_DB.itineraries_test** ✅")

# ================= Header inputs =================
c0, c1, c2, c3 = st.columns([1.3, 1, 1, 1])
with c0:
    client_name = st.text_input("Client Name", placeholder="e.g., Test Client / Family", key="client_name_form_test")
with c1:
    client_mobile_raw = st.text_input("Client mobile (10 digits)", help="Used as the unique key.", key="mob_test")
with c2:
    rep = st.selectbox("Representative", ["-- Select --", "Arpith", "Reena", "Kuldeep", "Teena"], key="rep_test")
with c3:
    default_pax = st.number_input("Total Pax (default)", min_value=1, value=2, step=1, key="pax_test")

h1, h2 = st.columns([1, 1])
with h1:
    start_date_input = st.date_input("Start date", value=datetime.date.today(), key="start_date_test")
with h2:
    days = st.number_input("No. of days", min_value=1, value=3, step=1, key="days_test")

# ================= Editable Table =================
if "form_rows_test" not in st.session_state:
    dates = [start_date_input + datetime.timedelta(days=i) for i in range(days)]
    st.session_state.form_rows_test = pd.DataFrame({
        "Date": dates,
        "Time": [""] * days,
        "Code": [""] * days,
        "Car Type": ["" for _ in range(days)],
        "Hotel Type": ["" for _ in range(days)],
        "Bhasmarathi Type": ["" for _ in range(days)],
        "Stay City": ["" for _ in range(days)],
        "Room Type": ["" for _ in range(days)],
        "Car Cost": [0.0 for _ in range(days)],
        "Hotel Cost": [0.0 for _ in range(days)],
        "Bhasmarathi Cost": [0.0 for _ in range(days)],
        "Total Pax": [default_pax for _ in range(days)],
    })

regen = st.button("↻ Regenerate rows (keeps existing where possible)", key="regen_test")
if regen:
    dates = [start_date_input + datetime.timedelta(days=i) for i in range(days)]
    df_new = pd.DataFrame({
        "Date": dates,
        "Time": [""] * days,
        "Code": [""] * days,
        "Car Type": ["" for _ in range(days)],
        "Hotel Type": ["" for _ in range(days)],
        "Bhasmarathi Type": ["" for _ in range(days)],
        "Stay City": ["" for _ in range(days)],
        "Room Type": ["" for _ in range(days)],
        "Car Cost": [0.0 for _ in range(days)],
        "Hotel Cost": [0.0 for _ in range(days)],
        "Bhasmarathi Cost": [0.0 for _ in range(days)],
        "Total Pax": [default_pax for _ in range(days)],
    })
    try:
        st.session_state.form_rows_test = df_new.combine_first(st.session_state.form_rows_test).iloc[:days].reset_index(drop=True)
    except Exception:
        st.session_state.form_rows_test = df_new

code_options = code_df["Code"].dropna().astype(str).unique().tolist() if not code_df.empty else []

col_cfg = {
    "Date": st.column_config.DateColumn("Date"),
    "Time": st.column_config.TextColumn("Time"),
    "Code": st.column_config.SelectboxColumn("Code", options=code_options),
    "Car Type": st.column_config.TextColumn("Car Type"),
    "Hotel Type": st.column_config.TextColumn("Hotel Type"),
    "Bhasmarathi Type": st.column_config.TextColumn("Bhasmarathi Type"),
    "Stay City": st.column_config.TextColumn("Stay City"),
    "Room Type": st.column_config.TextColumn("Room Type"),
    "Car Cost": st.column_config.NumberColumn("Car Cost", min_value=0.0),
    "Hotel Cost": st.column_config.NumberColumn("Hotel Cost", min_value=0.0),
    "Bhasmarathi Cost": st.column_config.NumberColumn("Bhasmarathi Cost", min_value=0.0),
    "Total Pax": st.column_config.NumberColumn("Total Pax", min_value=1, step=1),
}

st.caption("Fill one row per date. Use **+ Add row** below if needed.")
edited_df = st.data_editor(
    st.session_state.form_rows_test,
    num_rows="dynamic",
    use_container_width=True,
    column_config=col_cfg,
    hide_index=True,
    key="editor_test"
)
st.session_state.form_rows_test = edited_df.copy()

# ================= Validation =================
if not client_name:
    st.warning("Enter **Client Name**.")
    st.stop()
if not client_mobile_raw or not is_valid_mobile(client_mobile_raw):
    st.warning("Enter a valid **Client mobile (10 digits)**.")
    st.stop()
if rep == "-- Select --":
    st.warning("Please select **Representative**.")
    st.stop()

# ================= Compute =====================
df = edited_df.copy()

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
total_pax = int(pd.to_numeric(df["Total Pax"].iloc[0], errors="coerce") or 0)

# Ensure optional cols exist
for col in ["Time","Code","Car Type","Hotel Type","Bhasmarathi Type","Stay City","Room Type","Car Cost","Hotel Cost","Bhasmarathi Cost"]:
    if col not in df.columns:
        df[col] = "" if "Cost" not in col else 0.0

# Route and descriptions
itinerary = []
code_set = set(code_df["Code"].astype(str)) if not code_df.empty else set()
for _, r in df.iterrows():
    code = str(r.get("Code", "") or "")
    if code and code in code_set:
        desc = code_df.loc[code_df["Code"].astype(str) == code, "Particulars"].iloc[0]
    else:
        desc = f"No description found for code {code}" if code else "No code provided"
    itinerary.append({
        "Date": r.get("Date", ""),
        "Time": r.get("Time", ""),
        "Description": str(desc)
    })

route_parts = []
if "Code" in df.columns and "Route" in code_df.columns:
    for c in df["Code"].astype(str):
        m = code_df.loc[code_df["Code"].astype(str) == c, "Route"]
        if not m.empty:
            route_parts.append(str(m.iloc[0]))
route_raw = "-".join(route_parts).replace(" -", "-").replace("- ", "-")
route_list = [x for x in route_raw.split("-") if x]
final_route = "-".join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i-1]])

# Package cost
for req in ["Car Cost", "Hotel Cost", "Bhasmarathi Cost"]:
    if req not in df.columns:
        df[req] = 0
tcar = float(pd.to_numeric(df["Car Cost"], errors="coerce").fillna(0).sum())
thot = float(pd.to_numeric(df["Hotel Cost"], errors="coerce").fillna(0).sum())
tbha = float(pd.to_numeric(df["Bhasmarathi Cost"], errors="coerce").fillna(0).sum())
total_package_cost = ceil_to_999(tcar + thot + tbha)
pkg_cost_fmt = in_locale(total_package_cost)

# Type strings
car_types = "-".join(pd.Series(df.get("Car Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")
hotel_types = "-".join(pd.Series(df.get("Hotel Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")
bhas_types = pd.Series(df.get("Bhasmarathi Type", [])).dropna().astype(str).replace("","").unique().tolist()
bhas_descs = []
if not bhasmarathi_type_df.empty:
    for b in bhas_types:
        m = bhasmarathi_type_df.loc[bhasmarathi_type_df["Bhasmarathi Type"].astype(str) == str(b), "Description"]
        if not m.empty:
            bhas_descs.append(str(m.iloc[0]))
bhas_desc_str = "-".join([x for x in bhas_descs if x])

# Build text
night_txt = "Night" if total_nights == 1 else "Nights"
person_txt = "Person" if total_pax == 1 else "Persons"
client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())

greet = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
plan = f"*Plan:- {total_days}Days and {total_nights}{night_txt} {final_route} for {total_pax} {person_txt}*"

grouped = {}
for e in itinerary:
    dstr = pd.to_datetime(e["Date"]).strftime("%d-%b-%Y") if pd.notna(e["Date"]) and str(e["Date"]) else "N/A"
    time_part = f"{e.get('Time','')}: " if str(e.get('Time','')).strip() else ""
    grouped.setdefault(dstr, []).append(f"{time_part}{e['Description']}")

itinerary_text = greet + plan + "\n\n*Itinerary:*\n"
for i, (d, evs) in enumerate(grouped.items(), 1):
    itinerary_text += f"\n*Day{i}:{d}*\n" + "\n".join(evs) + "\n"

details_line = "(" + ",".join([x for x in [car_types, hotel_types, bhas_desc_str] if x]) + ")"
itinerary_text += f"\n*Package cost: {pkg_cost_fmt}/-*\n{details_line}"

# Minimal inclusions/exclusions block (short for test)
inclusions = "*Inclusions:-*\n" + "\n".join([
    f"1. Travel by {car_types or 'vehicle'} as per itinerary.",
    "2. Toll, parking, driver bata.",
    "3. Pickup and drop as planned."
])
exclusions = "*Exclusions:-*\n" + "\n".join([
    "1. Meals not specified.",
    "2. Entry fees unless included.",
    "3. Personal expenses."
])

final_output = itinerary_text + "\n\n" + inclusions + "\n\n" + exclusions

# ================= DB WRITE (TEST COLLECTION) =================
st.markdown("### 2) Generate & Save (DB: itineraries_test)")
if st.button("✅ Generate & Write to DB (TEST)", use_container_width=True):
    key_filter = {"client_mobile": client_mobile, "start_date": str(start_date), "is_test": True}
    record = {
        "is_test": True,
        "client_name": client_name,
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
        "bhasmarathi_types": bhas_desc_str,
        "package_cost": int(total_package_cost),
        "itinerary_text": final_output
    }
    try:
        res = col_test.update_one(key_filter, {"$set": record}, upsert=True)
        if res.upserted_id:
            st.success(f"✅ Wrote to DB (TEST) – New ID: {res.upserted_id}")
        else:
            st.info("🔁 Wrote to DB (TEST) – Updated existing document for this mobile + start date.")
    except Exception as e:
        st.error(f"❌ DB write failed: {e}")

# ================= Preview & Download =================
st.markdown("### 3) Preview & Download")
c1, c2 = st.columns([1, 1])
with c1:
    st.text_area("Preview (copy from here)", final_output, height=420)
with c2:
    st.download_button(
        label="⬇️ Download itinerary as .txt",
        data=final_output,
        file_name=f"TEST_itinerary_{client_name}_{start_date}.txt",
        mime="text/plain",
        use_container_width=True
    )

st.caption("Note: This page is for testing only. Writes to **itineraries_test** collection.")
