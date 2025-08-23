# App.py
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
    pass

# ----------------- Imports -----------------
import io, math, locale, datetime, os
import pandas as pd
import requests
from pymongo import MongoClient
from bson import ObjectId

# ----------------- App config -----------------
st.set_page_config(page_title="TAK – Itinerary Generator", layout="wide")
st.title("🧭 TAK Project – Itinerary Generator")

# ----------------- Constants -----------------
CODE_FILE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Code.xlsx"
BHASMARATHI_TYPE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Bhasmarathi_Type.xlsx"
STAY_CITY_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Stay_City.xlsx"

# ----------------- Caching helpers -----------------
@st.cache_data(ttl=900)
def read_excel_from_url(url, sheet_name=None):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name)

@st.cache_resource
def mongo_client():
    # prefer secrets, then env
    uri = (
        st.secrets.get("mongo_uri")
        or os.getenv("mongo_uri")
        or os.getenv("MONGO_URI")
        or os.getenv("mongodb_uri")
        or os.getenv("MONGODB_URI")
    )
    if not uri:
        st.error("Mongo URI not configured. Add `mongo_uri` in Secrets.")
        st.stop()
    return MongoClient(uri, appName="TAK_App", maxPoolSize=100, serverSelectionTimeoutMS=5000)

@st.cache_resource
def get_collections():
    client = mongo_client()
    client.admin.command("ping")
    db = client["TAK_DB"]
    return {
        "itineraries": db["itineraries"],
        "updates": db["package_updates"],
        "expenses": db["expenses"],
        "followups": db["followups"],
        "splitwise": db["expense_splitwise"],
        "vendor_expenses": db["vendor_expenses"],
    }

cols = get_collections()
col_it = cols["itineraries"]

# ----------------- Small utils -----------------
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

# ----------------- Load static masters (cached) -----------------
try:
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
    bhasmarathi_type_df = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet_name="Bhasmarathi_Type")
except Exception as e:
    st.error(f"Failed to load master sheets: {e}")
    st.stop()

# ----------------- Upload zone -----------------
with st.container():
    st.markdown("### 1) Upload your date-wise Excel")
    uploaded = st.file_uploader("Choose file (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.info("Upload the Excel to proceed.")
        st.stop()

try:
    xls = pd.ExcelFile(uploaded)
except Exception as e:
    st.error(f"Error reading uploaded file: {e}")
    st.stop()

sheet = st.selectbox("Select client (sheet name)", xls.sheet_names, index=0)
client_mobile_raw = st.text_input("Client mobile (10 digits)", help="Used to uniquely identify the itinerary.")
rep = st.selectbox("Representative", ["-- Select --", "Arpith", "Reena", "Kuldeep", "Teena"])

if not (sheet and client_mobile_raw and rep != "-- Select --"):
    st.stop()

if not is_valid_mobile(client_mobile_raw):
    st.error("Enter a valid 10-digit mobile.")
    st.stop()

client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())

# ----------------- Parse selected sheet -----------------
try:
    df = xls.parse(sheet)
except Exception as e:
    st.error(f"Error reading sheet: {e}")
    st.stop()

if "Date" not in df.columns:
    st.error("Required column 'Date' missing.")
    st.stop()

dates = pd.to_datetime(df["Date"], errors="coerce")
if dates.isna().all():
    st.error("No valid dates found.")
    st.stop()

start_date = dates.min().date()
end_date = dates.max().date()
total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
total_nights = max(total_days - 1, 0)

if "Total Pax" not in df.columns:
    st.error("Required column 'Total Pax' missing.")
    st.stop()

total_pax = int(pd.to_numeric(df["Total Pax"].iloc[0], errors="coerce") or 0)

# ----------------- Build itinerary lines from Code master -----------------
itinerary = []
for _, r in df.iterrows():
    code = r.get("Code", None)
    if code_df is not None and code in set(code_df["Code"]):
        desc = code_df.loc[code_df["Code"] == code, "Particulars"].iloc[0]
    else:
        desc = f"No description found for code {code}" if code else "No code provided"
    itinerary.append({
        "Date": r.get("Date", ""),
        "Time": r.get("Time", ""),
        "Description": str(desc)
    })

# ----------------- Derive route -----------------
route_parts = []
if "Code" in df.columns and "Route" in code_df.columns:
    for c in df["Code"]:
        m = code_df.loc[code_df["Code"] == c, "Route"]
        if not m.empty:
            route_parts.append(str(m.iloc[0]))
route_raw = "-".join(route_parts).replace(" -", "-").replace("- ", "-")
route_list = [x for x in route_raw.split("-") if x]
final_route = "-".join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i-1]])

# ----------------- Package cost (from sheet columns if present) -----------------
for req in ["Car Cost", "Hotel Cost", "Bhasmarathi Cost"]:
    if req not in df.columns:
        df[req] = 0
tcar = float(pd.to_numeric(df["Car Cost"], errors="coerce").fillna(0).sum())
thot = float(pd.to_numeric(df["Hotel Cost"], errors="coerce").fillna(0).sum())
tbha = float(pd.to_numeric(df["Bhasmarathi Cost"], errors="coerce").fillna(0).sum())
total_package_cost = ceil_to_999(tcar + thot + tbha)
pkg_cost_fmt = in_locale(total_package_cost)

# ----------------- Type strings -----------------
car_types = "-".join(pd.Series(df.get("Car Type", [])).dropna().astype(str).unique().tolist())
hotel_types = "-".join(pd.Series(df.get("Hotel Type", [])).dropna().astype(str).unique().tolist())
bhas_types = pd.Series(df.get("Bhasmarathi Type", [])).dropna().astype(str).unique().tolist()
bhas_descs = []
if not bhasmarathi_type_df.empty:
    for b in bhas_types:
        m = bhasmarathi_type_df.loc[bhasmarathi_type_df["Bhasmarathi Type"] == b, "Description"]
        if not m.empty:
            bhas_descs.append(str(m.iloc[0]))
bhas_desc_str = "-".join(bhas_descs)

# ----------------- Text build -----------------
night_txt = "Night" if total_nights == 1 else "Nights"
person_txt = "Person" if total_pax == 1 else "Persons"

greet = f"Greetings from TravelAajkal,\n\n*Client Name: {sheet}*\n\n"
plan = f"*Plan:- {total_days}Days and {total_nights}{night_txt} {final_route} for {total_pax} {person_txt}*"

# group by day
grouped = {}
for e in itinerary:
    dstr = pd.to_datetime(e["Date"]).strftime("%d-%b-%Y") if pd.notna(e["Date"]) and str(e["Date"]) else "N/A"
    grouped.setdefault(dstr, []).append(f"{e.get('Time','')}: {e['Description']}")

itinerary_text = greet + plan + "\n\n*Itinerary:*\n"
for i, (d, evs) in enumerate(grouped.items(), 1):
    itinerary_text += f"\n*Day{i}:{d}*\n" + "\n".join(evs) + "\n"

details_line = "(" + ",".join([x for x in [car_types, hotel_types, bhas_desc_str] if x]) + ")"
itinerary_text += f"\n*Package cost: {pkg_cost_fmt}/-*\n{details_line}"

# Inclusions (light, same as your logic but compact)
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
    # simple city-night roll-up
    stay_series = df["Stay City"].astype(str).fillna("")
    city_nights = stay_series[stay_series != ""].value_counts().to_dict()
    used = 0
    for stay_city, nn in city_nights.items():
        if used >= total_nights: break
        match = stay_city_df[stay_city_df["Stay City"] == stay_city]
        if not match.empty:
            city_name = match["City"].iloc[0]
            rt = df.loc[df["Stay City"] == stay_city, "Room Type"].dropna().astype(str).unique()
            inc.append(f"{min(nn, total_nights-used)}Night stay in {city_name} with {'/'.join(rt) or 'room'} in {hotel_types or 'hotel'}.")
            used += nn
if hotel_types:
    inc += [
        "*Standard check-in at 12:00 PM and check-out at 09:00 AM.*",
        "Early check-in and late check-out are subject to room availability."
    ]
inclusions = "*Inclusions:-*\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(inc)])

# Exclusions & notes (unchanged spirit, compact)
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
    itinerary_text + "\n\n" + inclusions + "\n\n" + exclusions + "\n\n" + notes + "\n\n" + cxl + "\n\n" + pay + "\n\n" + acct
)

# ----------------- Dedupe logic: 1 upload -> 1 entry -----------------
# key = (client_mobile, start_date)
key_filter = {"client_mobile": client_mobile, "start_date": str(start_date)}
existing = col_it.find_one(key_filter, {"_id": 1})

# build record
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
    "bhasmarathi_types": bhas_desc_str,
    "package_cost": total_package_cost,  # store numeral
    "itinerary_text": final_output
}

# Save/update gated by a button (prevent reruns creating duplicates)
st.markdown("### 2) Save & share")
c1, c2 = st.columns([1, 1])
with c1:
    st.text_area("Preview (copy from here)", final_output, height=420)
with c2:
    if st.button("💾 Save to MongoDB (create/update)"):
        if existing:
            col_it.update_one({"_id": existing["_id"]}, {"$set": record})
            st.success("✅ Updated existing itinerary (same mobile + start date).")
            iid = str(existing["_id"])
        else:
            res = col_it.insert_one(record)
            st.success("✅ Saved new itinerary.")
            iid = str(res.inserted_id)
        st.session_state["last_iid"] = iid

    if st.session_state.get("last_iid"):
        st.info(f"Saved as ID: {st.session_state['last_iid']}")

st.download_button(
    label="⬇️ Download itinerary as .txt",
    data=final_output,
    file_name=f"itinerary_{sheet}_{start_date}.txt",
    mime="text/plain",
    use_container_width=True
)
