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
        st.error("Login not configured. Add a **[users]** section in Secrets with PINs.")
        st.stop()

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns(2)
    with c1: name = st.selectbox("User", list(users_map.keys()), key="login_user")
    with c2: pin  = st.text_input("PIN", type="password", key="login_pin")
    if st.button("Sign in"):
        if str(users_map.get(name, "")).strip() == str(pin).strip():
            st.session_state["user"] = name
            try: audit_login(name)
            except Exception: pass
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid PIN"); st.stop()
    return None

# ================= Mongo =======================
def _find_uri() -> str | None:
    for k in ("mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"):
        try: v = st.secrets.get(k)
        except Exception: v = None
        if v: return v
    for k in ("mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"):
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
    db = mongo_client()["TAK_DB"]
    return {
        "itineraries": db["itineraries"],
        "audit_logins": db["audit_logins"],
    }

cols = get_collections()
col_it = cols["itineraries"]
col_audit = cols["audit_logins"]

def audit_login(user: str):
    now_utc = datetime.datetime.utcnow()
    try:
        col_audit.insert_one({
            "user": str(user),
            "ts_utc": now_utc,
            "ts_ist": now_utc.replace(tzinfo=datetime.timezone.utc).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "page": "App.py",
        })
    except Exception:
        pass

user = _login()
if not user:
    st.stop()

# ================= Helpers =================
@st.cache_data(ttl=900)
def read_excel_from_url(url, sheet_name=None):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name)

try:
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
    bhas_df = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet_name="Bhasmarathi_Type")
except Exception as e:
    st.error(f"Failed to load master sheets: {e}")
    st.stop()

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

# ================= UI: shared header =================
st.markdown("### 1) Provide Input")
mode = st.radio("Input Mode", ["Form Table (No Excel)", "Excel Upload"], horizontal=True, index=0)

c0, c1, c2, c3 = st.columns([1.4, 1, 1, 1])
with c0: sheet = st.text_input("Client Name*", placeholder="e.g., Mayur Gupta / Family")
with c1: client_mobile_raw = st.text_input("Client mobile (10 digits)*")
with c2: rep = st.selectbox("Representative*", ["-- Select --","Arpith","Reena","Kuldeep","Teena"])
with c3: default_pax = st.number_input("Total Pax (default)", min_value=1, value=2, step=1)

# Referral (from confirmed clients)
def _load_client_refs() -> list[str]:
    try:
        cur = col_it.aggregate([
            {"$group": {"_id": {"name": "$client_name", "mobile": "$client_mobile"}}},
            {"$project": {"_id": 0, "name": "$_id.name", "mobile": "$_id.mobile"}}
        ])
        labels = []
        for x in cur:
            n = (x.get("name") or "").strip()
            m = (x.get("mobile") or "").strip()
            label = f"{n} — {m}" if n and m else n or m
            if label: labels.append(label)
        labels = sorted(set(labels), key=lambda s: s.lower())
        return labels
    except Exception:
        return []

ref_labels = ["-- None --"] + _load_client_refs()
referred_sel = st.selectbox("Referred By (10% discount applies)", ref_labels, index=0)
has_ref = referred_sel != "-- None --"
discount_pct = 10 if has_ref else 0

# ====== Masters for dropdowns ======
stay_city_options = sorted(stay_city_df["Stay City"].dropna().astype(str).unique().tolist()) if "Stay City" in stay_city_df.columns else []
code_options = code_df["Code"].dropna().astype(str).unique().tolist() if not code_df.empty else []

base_cars = ["Sedan","Ertiga","Innova","Tempo Traveller"]
car_options = [f"{ac} {c}" for c in base_cars for ac in ("AC","Non AC")]

hotel_options = [
    "AC Standard AC",
    "Non-AC Standard AC",
    "3Star AC Hotel room",
    "4Star AC Hotel room",
    "5Star AC Hotel room",
]

room_options = [f"{occ} occupancy {i} room" for occ in ["Double","Triple","Quad","Quint"] for i in range(1,5)]

bhas_type_options = ["— None —","V-BH","P-BH","BH"]  # first option disables BH calc

# ================= MODE A: Form Table =================
if mode == "Form Table (No Excel)":
    h1, h2 = st.columns(2)
    with h1:
        start_date_input = st.date_input("Start date", value=datetime.date.today())
    with h2:
        days = st.number_input("No. of days", min_value=1, value=3, step=1)

    # Create once and never rebuild unless Reset is clicked
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
            # --- finance columns exactly like your sheet ---
            "Hotel Cost": [0.0] * days,
            "Bhasmarathi Type": ["— None —"] * days,
            "Bhasmarathi Unit Cost": [0.0] * days,
            "Total Pax": [default_pax] * days,
            "Actual-Car Cost": [0.0] * days,
            "Actual-Hotel Cost": [0.0] * days,
            "Actual-Bhasmarathi Cost": [0.0] * days,
            "Package Cost": [0.0] * days,
            # Calculated columns (shown, but recalculated each run)
            "Bhasmarathi Cost": [0.0] * days,
            "Actual Cost": [0.0] * days,
            "Profit": [0.0] * days,
        })

    def _reset_rows():
        dates = [start_date_input + datetime.timedelta(days=i) for i in range(days)]
        df_new = st.session_state.form_rows.copy()
        needed = len(dates)
        if len(df_new) < needed:
            extra = needed - len(df_new)
            add = pd.DataFrame({c: [df_new[c].iloc[0] if len(df_new) else 0] * extra for c in df_new.columns})
            add["Date"] = [dates[len(df_new) + i] for i in range(extra)]
            df_new = pd.concat([df_new, add], ignore_index=True)
        df_new.loc[:needed-1, "Date"] = dates
        df_new = df_new.iloc[:needed].reset_index(drop=True)
        st.session_state.form_rows = df_new

    st.button("↻ Reset rows for new dates/days (optional)", on_click=_reset_rows)

    # Configure editor (searchable dropdowns)
    col_cfg = {
        "Date": st.column_config.DateColumn("Date"),
        "Time": st.column_config.TextColumn("Time"),
        "Code": st.column_config.SelectboxColumn("Code", options=code_options, help="Searchable"),
        "Car Type": st.column_config.SelectboxColumn("Car Type", options=car_options),
        "Hotel Type": st.column_config.SelectboxColumn("Hotel Type", options=hotel_options),
        "Stay City": st.column_config.SelectboxColumn("Stay City", options=stay_city_options),
        "Room Type": st.column_config.SelectboxColumn("Room Type", options=room_options),
        "Hotel Cost": st.column_config.NumberColumn("Hotel Cost", min_value=0.0, step=100.0),
        "Bhasmarathi Type": st.column_config.SelectboxColumn("Bhasmarathi Type", options=bhas_type_options),
        "Bhasmarathi Unit Cost": st.column_config.NumberColumn("Bhasmarathi Unit Cost", min_value=0.0, step=100.0),
        "Total Pax": st.column_config.NumberColumn("Total Pax", min_value=1, step=1),
        "Actual-Car Cost": st.column_config.NumberColumn("Actual-Car Cost", min_value=0.0, step=100.0),
        "Actual-Hotel Cost": st.column_config.NumberColumn("Actual-Hotel Cost", min_value=0.0, step=100.0),
        "Actual-Bhasmarathi Cost": st.column_config.NumberColumn("Actual-Bhasmarathi Cost", min_value=0.0, step=100.0),
        "Package Cost": st.column_config.NumberColumn("Package Cost", min_value=0.0, step=100.0),
        # Calculated columns (read-only appearance)
        "Bhasmarathi Cost": st.column_config.NumberColumn("Bhasmarathi Cost", disabled=True),
        "Actual Cost": st.column_config.NumberColumn("Actual Cost", disabled=True),
        "Profit": st.column_config.NumberColumn("Profit", disabled=True),
    }

    edited_df = st.data_editor(
        st.session_state.form_rows,
        num_rows="dynamic",         # you can add rows any time
        use_container_width=True,
        column_config=col_cfg,
        hide_index=True,
        key="editor_main"
    )

    # ---- Recalculate dependent columns (no data loss) ----
    df = edited_df.copy()

    # safe numeric helpers
    def _num(s): 
        try:
            return float(s)
        except Exception:
            return 0.0

    # compute per row
    bh_cost = []
    act_cost = []
    profit = []
    for i, r in df.iterrows():
        pax = int(_num(r.get("Total Pax")))
        unit = _num(r.get("Bhasmarathi Unit Cost"))
        bh_type = str(r.get("Bhasmarathi Type") or "— None —").strip()
        bh_total = unit * pax if bh_type != "— None —" else 0.0

        ac_car = _num(r.get("Actual-Car Cost"))
        ac_hot = _num(r.get("Actual-Hotel Cost"))
        ac_bh  = _num(r.get("Actual-Bhasmarathi Cost"))
        actual = ac_car + ac_hot + ac_bh

        pkg = _num(r.get("Package Cost"))

        bh_cost.append(bh_total)
        act_cost.append(actual)
        profit.append(pkg - actual)

    df["Bhasmarathi Cost"] = bh_cost
    df["Actual Cost"] = act_cost
    df["Profit"] = profit

    # persist back
    st.session_state.form_rows = df.copy()

# ================= MODE B: Excel Upload =================
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
    if not sheet:
        sheet = st.selectbox("Select client (sheet name)", xls.sheet_names, index=0)
    if rep == "-- Select --":
        rep = st.selectbox("Representative*", ["-- Select --","Arpith","Reena","Kuldeep","Teena"])

    if not (sheet and rep != "-- Select --"):
        st.stop()

    try:
        df = xls.parse(sheet)
    except Exception as e:
        st.error(f"Error reading sheet: {e}")
        st.stop()

    # add any missing columns used by our flow so you can edit them
    wanted = ["Date","Time","Code","Car Type","Hotel Type","Stay City","Room Type",
              "Hotel Cost","Bhasmarathi Type","Bhasmarathi Unit Cost","Total Pax",
              "Actual-Car Cost","Actual-Hotel Cost","Actual-Bhasmarathi Cost","Package Cost",
              "Bhasmarathi Cost","Actual Cost","Profit"]
    for c in wanted:
        if c not in df.columns:
            if c in ("Bhasmarathi Type",):
                df[c] = "— None —"
            elif c in ("Date",):
                df[c] = pd.NaT
            elif c in ("Time","Code","Car Type","Hotel Type","Stay City","Room Type"):
                df[c] = ""
            else:
                df[c] = 0.0

    col_cfg = {
        "Date": st.column_config.DateColumn("Date"),
        "Time": st.column_config.TextColumn("Time"),
        "Code": st.column_config.SelectboxColumn("Code", options=code_options),
        "Car Type": st.column_config.SelectboxColumn("Car Type", options=car_options),
        "Hotel Type": st.column_config.SelectboxColumn("Hotel Type", options=hotel_options),
        "Stay City": st.column_config.SelectboxColumn("Stay City", options=stay_city_options),
        "Room Type": st.column_config.SelectboxColumn("Room Type", options=room_options),
        "Hotel Cost": st.column_config.NumberColumn("Hotel Cost", min_value=0.0, step=100.0),
        "Bhasmarathi Type": st.column_config.SelectboxColumn("Bhasmarathi Type", options=bhas_type_options),
        "Bhasmarathi Unit Cost": st.column_config.NumberColumn("Bhasmarathi Unit Cost", min_value=0.0, step=100.0),
        "Total Pax": st.column_config.NumberColumn("Total Pax", min_value=1, step=1),
        "Actual-Car Cost": st.column_config.NumberColumn("Actual-Car Cost", min_value=0.0, step=100.0),
        "Actual-Hotel Cost": st.column_config.NumberColumn("Actual-Hotel Cost", min_value=0.0, step=100.0),
        "Actual-Bhasmarathi Cost": st.column_config.NumberColumn("Actual-Bhasmarathi Cost", min_value=0.0, step=100.0),
        "Package Cost": st.column_config.NumberColumn("Package Cost", min_value=0.0, step=100.0),
        "Bhasmarathi Cost": st.column_config.NumberColumn("Bhasmarathi Cost", disabled=True),
        "Actual Cost": st.column_config.NumberColumn("Actual Cost", disabled=True),
        "Profit": st.column_config.NumberColumn("Profit", disabled=True),
    }

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config=col_cfg,
        hide_index=True,
        key="editor_excel"
    )

    # recompute calculated columns
    def _num(x):
        try: return float(x)
        except Exception: return 0.0

    bh_cost, act_cost, profit = [], [], []
    for _, r in edited_df.iterrows():
        pax = int(_num(r.get("Total Pax")))
        unit = _num(r.get("Bhasmarathi Unit Cost"))
        bh_type = str(r.get("Bhasmarathi Type") or "— None —").strip()
        bh_total = unit * pax if bh_type != "— None —" else 0.0
        ac_car = _num(r.get("Actual-Car Cost"))
        ac_hot = _num(r.get("Actual-Hotel Cost"))
        ac_bh  = _num(r.get("Actual-Bhasmarathi Cost"))
        actual = ac_car + ac_hot + ac_bh
        pkg = _num(r.get("Package Cost"))
        bh_cost.append(bh_total); act_cost.append(actual); profit.append(pkg - actual)

    edited_df["Bhasmarathi Cost"] = bh_cost
    edited_df["Actual Cost"] = act_cost
    edited_df["Profit"] = profit
    df = edited_df.copy()

# ================= Validations =================
if not sheet:
    st.warning("Enter **Client Name**."); st.stop()
if not is_valid_mobile(client_mobile_raw):
    st.error("Enter a valid **10-digit** mobile. Package will not be created without it."); st.stop()
if rep == "-- Select --":
    st.warning("Select **Representative**."); st.stop()

client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())

# ================= Safe Code helpers =================
def _code_to_desc(code) -> str:
    if code is None: return "No code provided"
    s = str(code).strip()
    if s == "" or s.lower() in ("none","nan"): return "No code provided"
    try:
        m = code_df.loc[code_df["Code"].astype(str) == s, "Particulars"]
        return str(m.iloc[0]) if not m.empty else f"No description found for code {s}"
    except Exception:
        return f"No description found for code {s}"

def _code_to_route(code) -> str | None:
    if code is None: return None
    s = str(code).strip()
    if s == "" or s.lower() in ("none","nan"): return None
    try:
        m = code_df.loc[code_df["Code"].astype(str) == s, "Route"]
        return str(m.iloc[0]) if not m.empty else None
    except Exception:
        return None

# ================= Totals / badges =================
sum_pkg = float(pd.to_numeric(df.get("Package Cost"), errors="coerce").fillna(0).sum())
sum_actual = float(pd.to_numeric(df.get("Actual Cost"), errors="coerce").fillna(0).sum())
total_package = ceil_to_999(sum_pkg)
total_actual = sum_actual
profit_total = int(total_package - total_actual)
after_ref = int(round(total_package * (1 - discount_pct/100.0))) if has_ref else total_package

badge_color = "#16a34a" if profit_total >= 4000 else "#dc2626"
hint = "" if profit_total >= 4000 else " • Keep profit margin ≥ ₹4,000"

st.markdown(
    f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 4px 0;">
      <div style="padding:8px 12px; border-radius:8px; background:#0ea5e9; color:white;">
        Package Total: <b>₹{in_locale(total_package)}</b>
      </div>
      {('<div style="padding:8px 12px; border-radius:8px; background:#7c3aed; color:white;">After Referral (10%): <b>₹'+in_locale(after_ref)+'</b></div>') if has_ref else ''}
      <div style="padding:8px 12px; border-radius:8px; background:#475569; color:white;">
        Actual Total: <b>₹{in_locale(total_actual)}</b>
      </div>
      <div style="padding:8px 12px; border-radius:8px; background:%s; color:white;">
        Profit: <b>₹%s</b>%s
      </div>
    </div>
    """ % (badge_color, in_locale(profit_total), hint),
    unsafe_allow_html=True
)

# ================= Build itinerary text =================
dates = pd.to_datetime(df["Date"], errors="coerce")
if dates.isna().all():
    st.error("No valid dates found."); st.stop()

start_date = dates.min().date()
end_date = dates.max().date()
total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
total_nights = max(total_days - 1, 0)

# itinerary items from codes
items = []
for _, r in df.iterrows():
    desc = _code_to_desc(r.get("Code", ""))
    items.append({"Date": r.get("Date",""), "Time": r.get("Time",""), "Description": desc})

# route from codes
route_parts = []
for c in df["Code"]:
    rt = _code_to_route(c)
    if rt: route_parts.append(rt)
route_raw = "-".join(route_parts).replace(" -","-").replace("- ","-")
route_list = [x for x in route_raw.split("-") if x]
final_route = "-".join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i-1]])

# type strings
car_types = "-".join(pd.Series(df.get("Car Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")
hotel_types = "-".join(pd.Series(df.get("Hotel Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")

# BH description (aggregate: if any row has BH type, use first description)
bhas_desc_str = ""
try:
    first_bh = next((t for t in df["Bhasmarathi Type"].astype(str) if t not in ("— None —","", "None", "nan")), None)
except Exception:
    first_bh = None
if first_bh:
    m = bhas_df.loc[bhas_df["Bhasmarathi Type"].astype(str) == str(first_bh), "Description"]
    if not m.empty: bhas_desc_str = str(m.iloc[0])

# text
total_pax_any = int(pd.to_numeric(df["Total Pax"].dropna().iloc[0], errors="coerce")) if not df["Total Pax"].dropna().empty else default_pax
night_txt = "Night" if total_nights == 1 else "Nights"
person_txt = "Person" if total_pax_any == 1 else "Persons"

greet = f"Greetings from TravelAajkal,\n\n*Client Name: {sheet}*\n\n"
plan  = f"*Plan:- {total_days}Days and {total_nights}{night_txt} {final_route} for {total_pax_any} {person_txt}*"

grouped = {}
for e in items:
    dstr = pd.to_datetime(e["Date"]).strftime("%d-%b-%Y") if pd.notna(e["Date"]) and str(e["Date"]) else "N/A"
    tp = f"{e.get('Time','')}: " if str(e.get('Time','')).strip() else ""
    grouped.setdefault(dstr, []).append(f"{tp}{e['Description']}")

itinerary_text = greet + plan + "\n\n*Itinerary:*\n"
for i,(d,evs) in enumerate(grouped.items(),1):
    itinerary_text += f"\n*Day{i}:{d}*\n" + "\n".join(evs) + "\n"

details_bits = [x for x in [car_types or None, hotel_types or None, bhas_desc_str or None] if x]
details_line = "(" + ",".join(details_bits) + ")" if details_bits else ""

itinerary_text += f"\n*Package cost: ₹{in_locale(total_package)}/-*\n"
if has_ref:
    itinerary_text += f"*Package cost (after referral 10%): ₹{in_locale(after_ref)}/-*\n"
itinerary_text += f"{details_line}"

# inclusions
inc = []
if car_types:
    inc += ["Entire travel as per itinerary by " + car_types + ".", "Toll, parking, and driver bata included.", "Pickup & drop included."]
if bhas_desc_str:
    inc += [f"{bhas_desc_str} for {total_pax_any} {person_txt}.", "Bhasm-Aarti pickup & drop."]
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
    inc += ["*Standard check-in 12:00 PM, check-out 09:00 AM.*", "Early check-in/late check-out subject to availability."]
if inc:
    itinerary_text += "\n\n*Inclusions:-*\n" + "\n".join([f"{i+1}. {x}" for i,x in enumerate(inc)])

# exclusions, notes, cxl, pay, acct (unchanged)
exclusions = "*Exclusions:-*\n" + "\n".join([
    "1. Any meals/beverages not specified.",
    "2. Entry fees unless included.",
    "3. Travel insurance.",
    "4. Personal shopping/tips.",
    "5. Early check-in/late check-out if rooms unavailable.",
    "6. Natural events/roadblocks/personal itinerary changes.",
    "7. Extra sightseeing not listed."
])
notes = "\n*Important Notes:-*\n" + "\n".join([
    "1. Any attractions not in itinerary will be chargeable.",
    "2. Visits subject to traffic/temple rules; closures beyond control & non-refundable.",
    "3. Bhasm-Aarti: tickets by us; arrival/seating beyond our control; at actuals; subject to availability/cancellations.",
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

final_output = itinerary_text + "\n\n" + exclusions + "\n\n" + notes + "\n\n" + cxl + "\n\n" + pay + "\n\n" + acct

# ================= Save to DB =================
key_filter = {"client_mobile": client_mobile, "start_date": str(start_date)}
record = {
    "client_name": sheet,
    "client_mobile": client_mobile,
    "representative": rep,
    "upload_date": datetime.datetime.utcnow(),
    "start_date": str(start_date),
    "end_date": str(end_date),
    "total_days": int(total_days),
    "final_route": final_route,
    "car_types": "-".join(pd.Series(df.get("Car Type", [])).dropna().astype(str).unique().tolist()),
    "hotel_types": "-".join(pd.Series(df.get("Hotel Type", [])).dropna().astype(str).unique().tolist()),
    "package_total": int(total_package),
    "package_after_referral": int(after_ref),
    "actual_total": int(total_actual),
    "profit_total": int(profit_total),
    "referred_by": referred_sel if has_ref else None,
    "referral_discount_pct": discount_pct,
    "rows": df.to_dict(orient="records"),
    "itinerary_text": final_output
}
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
c1, c2 = st.columns(2)
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
