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
import io, math, datetime, os, re
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

def audit_login(user: str):
    now_utc = datetime.datetime.utcnow()
    try:
        cols["audit_logins"].insert_one({
            "user": str(user),
            "ts_utc": now_utc,
            "ts_ist": now_utc.replace(tzinfo=datetime.timezone.utc).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "page": "App.py",
        })
    except Exception:
        pass

def _login() -> str | None:
    with st.sidebar:
        if st.session_state.get("user"):
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out"):
                st.session_state.pop("user", None)
                st.rerun()

    if st.session_state.get("user"): return st.session_state["user"]

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

# ---------- Sidebar: User Trail ----------
with st.sidebar.expander("👤 User trail (last 25)"):
    try:
        logs = list(cols["audit_logins"].find({}, {"_id":0}).sort([("ts_utc", -1)]).limit(25))
        if logs:
            for l in logs:
                st.caption(f"{l.get('ts_ist','')} — {l.get('user','')} ({l.get('page','')})")
        else:
            st.caption("No logins yet.")
    except Exception:
        st.caption("Trail unavailable.")

# ================= Caching helpers =================
@st.cache_data(ttl=900)
def read_excel_from_url(url, sheet_name=None):
    r = requests.get(url, timeout=20); r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name)

# ================= Load static masters (cached) =================
try:
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df      = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
    bhas_df      = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet_name="Bhasmarathi_Type")
except Exception as e:
    st.error(f"Failed to load master sheets: {e}"); st.stop()

# ================= Small utils =================
def is_valid_mobile(num: str) -> bool:
    digits = "".join(ch for ch in str(num or "") if ch.isdigit())
    return len(digits) == 10

def in_locale(n: int) -> str:
    return f"{int(n):,}"

def ceil_to_999(n: float) -> int:
    return (math.ceil(n/1000)*1000 - 1) if n > 0 else 0

# ---------- ensure login ----------
user = _login()
if not user: st.stop()

# ============================
#        RETRIEVE OLD PACKAGE
# ============================
with st.expander("🔎 Retrieve old package"):
    q = st.text_input("Search by client name or mobile", placeholder="e.g., Akshay or 8962235121")
    if "loaded_doc" not in st.session_state: st.session_state.loaded_doc = None
    matches = []
    if q.strip():
        try:
            rx = re.escape(q.strip())
            matches = list(col_it.find(
                {"$or":[{"client_name":{"$regex":rx,"$options":"i"}},{"client_mobile":{"$regex":rx}}]},
                {"itinerary_text":0}
            ).sort([("upload_date",-1)]).limit(10))
        except Exception:
            matches = []
    if matches:
        labels = [f"{m.get('client_name','?')} — {m.get('client_mobile','?')} • start: {m.get('start_date','?')} • rev:{m.get('revision_num',0)} • id:{str(m.get('_id',''))[:8]}" for m in matches]
        pick = st.selectbox("Select a previous package", labels, index=0)
        if st.button("Load this package"):
            doc = matches[labels.index(pick)]
            # store base doc in session to mark revisions later
            st.session_state.loaded_doc = doc
            # ---- prefill header fields
            st.session_state.prefill = {
                "client_name": doc.get("client_name",""),
                "client_mobile_raw": doc.get("client_mobile",""),
                "rep": doc.get("representative","-- Select --"),
                "total_pax": int(doc.get("total_pax",1) or 1),
                "start_date": datetime.date.fromisoformat(doc.get("start_date")),
                "days": int(doc.get("total_days", len(doc.get("rows",[])) or 1)),
                "referred_sel": doc.get("referred_by","-- None --") or "-- None --",
                # Bhasmarathi
                "bhas_required": "Yes" if doc.get("bhasmarathi_required") else "No",
                "bhas_type": doc.get("bhasmarathi_type","V-BH"),
                "bhas_persons": int(doc.get("bhasmarathi_persons",0) or 0),
                "bhas_unit_pkg": int(doc.get("bhasmarathi_unit_pkg",0) or 0),
                "bhas_unit_actual": int(doc.get("bhasmarathi_unit_actual",0) or 0),
                # rows
                "rows": doc.get("rows",[])
            }
            st.success("Previous package loaded into the form below. You can make changes and save as a revision.")
            st.experimental_rerun()
    else:
        if q.strip():
            st.info("No matches found.")

# ============================
#        FORM UI
# ============================

st.markdown("### 1) Provide Input")

# Prefill values if a package was loaded
pf = st.session_state.get("prefill", {}) if "prefill" in st.session_state else {}

c0, c1, c2, c3 = st.columns([1.6, 1, 1, 1])
with c0: client_name = st.text_input("Client Name*", value=pf.get("client_name",""))
with c1: client_mobile_raw = st.text_input("Client mobile (10 digits)*", value=pf.get("client_mobile_raw",""))
with c2: rep = st.selectbox("Representative*", ["-- Select --","Arpith","Reena","Kuldeep","Teena"], index= ["-- Select --","Arpith","Reena","Kuldeep","Teena"].index(pf.get("rep","-- Select --")) if pf else 0)
with c3: total_pax = st.number_input("Total Pax*", min_value=1, value=pf.get("total_pax",2), step=1)

# Referral
def _load_client_refs() -> list[str]:
    try:
        cur = cols["itineraries"].aggregate([
            {"$group": {"_id": {"name": "$client_name", "mobile": "$client_mobile"}}},
            {"$project": {"_id": 0, "name": "$_id.name", "mobile": "$_id.mobile"}}
        ])
        labels = []
        for x in cur:
            n = (x.get("name") or "").strip(); m = (x.get("mobile") or "").strip()
            if n or m: labels.append(f"{n} — {m}" if n and m else n or m)
        return sorted(set(labels), key=lambda s: s.lower())
    except Exception:
        return []
ref_labels = ["-- None --"] + _load_client_refs()
referred_sel = st.selectbox("Referred By (applies 10% discount)", ref_labels, index= ref_labels.index(pf.get("referred_sel","-- None --")) if pf else 0)
has_ref = referred_sel != "-- None --"; discount_pct = 10 if has_ref else 0

# Dates / rows
h1, h2 = st.columns(2)
with h1: start_date = st.date_input("Start date", value=pf.get("start_date", datetime.date.today()))
with h2: days = st.number_input("No. of days", min_value=1, value=pf.get("days",2), step=1)

# Dropdown options
stay_city_options = sorted(stay_city_df["Stay City"].dropna().astype(str).unique().tolist()) if "Stay City" in stay_city_df.columns else []
code_options = code_df["Code"].dropna().astype(str).unique().tolist() if not code_df.empty else []
base_cars = ["Sedan","Ertiga","Innova","Tempo Traveller"]
car_options = [f"{ac} {c}" for c in base_cars for ac in ("AC","Non AC")]
hotel_options = ["AC Standard AC","Non-AC Standard AC","3Star AC Hotel room","4Star AC Hotel room","5Star AC Hotel room"]
room_options = [f"{occ} occupancy {i} room" for occ in ["Double","Triple","Quad","Quint"] for i in range(1,5)]

def _time_list(step_minutes=15):
    base = datetime.datetime(2000,1,1,0,0)
    return [(base + datetime.timedelta(minutes=i)).time().strftime("%I:%M %p") for i in range(0,24*60,step_minutes)]
time_options = _time_list(15)

# Bhasmarathi
bhc1, bhc2, bhc3 = st.columns(3)
with bhc1:
    bhas_required = st.selectbox("Bhasmarathi required?", ["No","Yes"], index= 1 if pf.get("bhas_required","No")=="Yes" else 0)
with bhc2:
    bhas_type = st.selectbox("Bhasmarathi Type", ["V-BH","P-BH","BH"], index= ["V-BH","P-BH","BH"].index(pf.get("bhas_type","V-BH")))
with bhc3:
    bhas_persons = st.number_input("Persons for Bhasmarathi", min_value=0, value=pf.get("bhas_persons",0), step=1, disabled=(bhas_required=="No"))
bhc4, bhc5 = st.columns(2)
with bhc4:
    bhas_unit_pkg = st.number_input("Bhasmarathi unit cost (Package)", min_value=0, value=pf.get("bhas_unit_pkg",0), step=100, disabled=(bhas_required=="No"))
with bhc5:
    bhas_unit_actual = st.number_input("Bhasmarathi unit cost (Actual)", min_value=0, value=pf.get("bhas_unit_actual",0), step=100, disabled=(bhas_required=="No"))

# ===== Stable rows (with optional prefill of 'rows' from loaded package) =====
def _blank_df(n_rows: int, start: datetime.date) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": [start + datetime.timedelta(days=i) for i in range(n_rows)],
        "Time": ["" for _ in range(n_rows)],
        "Code": ["" for _ in range(n_rows)],
        "Car Type": ["" for _ in range(n_rows)],
        "Hotel Type": ["" for _ in range(n_rows)],
        "Stay City": ["" for _ in range(n_rows)],
        "Room Type": ["" for _ in range(n_rows)],
        "Pkg-Car Cost": [0.0 for _ in range(n_rows)],
        "Pkg-Hotel Cost": [0.0 for _ in range(n_rows)],
        "Act-Car Cost": [0.0 for _ in range(n_rows)],
        "Act-Hotel Cost": [0.0 for _ in range(n_rows)],
    })

def _prefill_rows(rows: list, n_rows: int, start: datetime.date) -> pd.DataFrame:
    if not rows: return _blank_df(n_rows, start)
    df = pd.DataFrame(rows)
    # normalize column names
    for col in ["Pkg-Car Cost","Pkg-Hotel Cost","Act-Car Cost","Act-Hotel Cost"]:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["Time","Code","Car Type","Hotel Type","Stay City","Room Type"]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].astype(str)
    # fix date
    if "Date" in df.columns:
        try: df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        except Exception: df["Date"] = [start + datetime.timedelta(days=i) for i in range(len(df))]
    # pad/trim to n_rows
    if len(df) < n_rows:
        add = _blank_df(n_rows - len(df), start + datetime.timedelta(days=len(df)))
        df = pd.concat([df, add], ignore_index=True)
    elif len(df) > n_rows:
        df = df.iloc[:n_rows].reset_index(drop=True)
    # recompute date sequence
    df["Date"] = [start + datetime.timedelta(days=i) for i in range(n_rows)]
    return df

def _ensure_rows(days: int, start: datetime.date):
    if "form_rows" not in st.session_state:
        # prefill from loaded doc if available
        if pf.get("rows"):
            st.session_state.form_rows = _prefill_rows(pf["rows"], days, start)
        else:
            st.session_state.form_rows = _blank_df(days, start)
        st.session_state._days = days
    else:
        df = st.session_state.form_rows
        prev_days = st.session_state.get("_days", len(df))
        if days > prev_days:
            add = _blank_df(days - prev_days, start + datetime.timedelta(days=prev_days))
            st.session_state.form_rows = pd.concat([df, add], ignore_index=True)
            st.session_state._days = days
        elif days < prev_days:
            st.session_state.form_rows = df.iloc[:days].reset_index(drop=True)
            st.session_state._days = days
    # keep dates in sync
    st.session_state.form_rows.loc[:, "Date"] = [start + datetime.timedelta(days=i) for i in range(days)]

_ensure_rows(days, start_date)

EDITABLE_COLS = ["Time","Code","Car Type","Hotel Type","Stay City","Room Type","Pkg-Car Cost","Pkg-Hotel Cost","Act-Car Cost","Act-Hotel Cost"]
col_cfg = {
    "Date": st.column_config.DateColumn("Date", disabled=True),
    "Time": st.column_config.SelectboxColumn("Time", options=time_options),
    "Code": st.column_config.SelectboxColumn("Code", options=code_options),
    "Car Type": st.column_config.SelectboxColumn("Car Type", options=car_options),
    "Hotel Type": st.column_config.SelectboxColumn("Hotel Type", options=hotel_options),
    "Stay City": st.column_config.SelectboxColumn("Stay City", options=stay_city_options),
    "Room Type": st.column_config.SelectboxColumn("Room Type", options=room_options),
    "Pkg-Car Cost": st.column_config.NumberColumn("Pkg-Car Cost", min_value=0.0, step=100.0),
    "Pkg-Hotel Cost": st.column_config.NumberColumn("Pkg-Hotel Cost", min_value=0.0, step=100.0),
    "Act-Car Cost": st.column_config.NumberColumn("Act-Car Cost", min_value=0.0, step=100.0),
    "Act-Hotel Cost": st.column_config.NumberColumn("Act-Hotel Cost", min_value=0.0, step=100.0),
}
st.markdown("### Fill Line Items")
edited_df = st.data_editor(st.session_state.form_rows, num_rows="fixed", use_container_width=True, column_config=col_cfg, hide_index=True, key="editor_main")
base = st.session_state.form_rows.copy()
for col in EDITABLE_COLS:
    if col in edited_df.columns: base[col] = edited_df[col]
st.session_state.form_rows = base
base = st.session_state.form_rows

# ---- Validations
if not client_name: st.warning("Enter **Client Name**."); st.stop()
if not is_valid_mobile(client_mobile_raw): st.error("Enter a valid **10-digit** mobile."); st.stop()
if rep == "-- Select --": st.warning("Select **Representative**."); st.stop()
client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())

# ---- Code helpers
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

# ---- Totals
pkg_car   = pd.to_numeric(base.get("Pkg-Car Cost", 0), errors="coerce").fillna(0).sum()
pkg_hotel = pd.to_numeric(base.get("Pkg-Hotel Cost", 0), errors="coerce").fillna(0).sum()
act_car   = pd.to_numeric(base.get("Act-Car Cost", 0), errors="coerce").fillna(0).sum()
act_hotel = pd.to_numeric(base.get("Act-Hotel Cost", 0), errors="coerce").fillna(0).sum()

bhas_pkg_total    = (bhas_unit_pkg * bhas_persons) if bhas_required=="Yes" else 0
bhas_actual_total = (bhas_unit_actual * bhas_persons) if bhas_required=="Yes" else 0

package_cost_rows = float(pkg_car + pkg_hotel)
actual_cost_rows  = float(act_car + act_hotel)
total_package = ceil_to_999(package_cost_rows + bhas_pkg_total)
total_actual  = actual_cost_rows + bhas_actual_total
profit_total  = int(total_package - total_actual)
after_ref     = int(round(total_package * 0.9)) if referred_sel != "-- None --" else total_package

badge_color = "#16a34a" if profit_total >= 4000 else "#dc2626"
hint = "" if profit_total >= 4000 else " • Keep profit margin ≥ ₹4,000"
ref_html = f'<div style="padding:8px 12px; border-radius:8px; background:#7c3aed; color:white;">After Referral (10%): <b>₹{in_locale(after_ref)}</b></div>' if referred_sel != "-- None --" else ""
totals_html = (
    '<div style="display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 4px 0;">'
    f'<div style="padding:8px 12px; border-radius:8px; background:#0ea5e9; color:white;">Package Cost: <b>₹{in_locale(total_package)}</b></div>'
    f'{ref_html}'
    f'<div style="padding:8px 12px; border-radius:8px; background:#475569; color:white;">Actual Cost: <b>₹{in_locale(total_actual)}</b></div>'
    f'<div style="padding:8px 12px; border-radius:8px; background:{badge_color}; color:white;">Profit: <b>₹{in_locale(profit_total)}</b>{hint}</div>'
    '</div>'
)
st.markdown(totals_html, unsafe_allow_html=True)

# ---- Build itinerary text
base["Date"] = [start_date + datetime.timedelta(days=i) for i in range(days)]
dates_series   = pd.to_datetime(base["Date"], errors="coerce")
start_date_calc = dates_series.min().date()
end_date_calc   = dates_series.max().date()
total_days_calc = (pd.to_datetime(end_date_calc) - pd.to_datetime(start_date_calc)).days + 1
total_nights    = max(total_days_calc - 1, 0)

items = [{"Date": r["Date"], "Time": r.get("Time",""), "Code": r.get("Code","")} for _, r in base.iterrows()]
route_parts = []
for r in base["Code"]:
    rt = _code_to_route(r)
    if rt: route_parts.append(rt)
route_raw  = "-".join(route_parts).replace(" -","-").replace("- ","-")
route_list = [x for x in route_raw.split("-") if x]
final_route = "-".join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i-1]])

car_types   = "-".join(pd.Series(base.get("Car Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")
hotel_types = "-".join(pd.Series(base.get("Hotel Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")

bhas_desc_str = ""
if bhas_required == "Yes":
    mm = bhas_df.loc[bhas_df["Bhasmarathi Type"].astype(str) == str(bhas_type), "Description"]
    if not mm.empty: bhas_desc_str = str(mm.iloc[0])

night_txt  = "Night" if total_nights == 1 else "Nights"
person_txt = "Person" if total_pax == 1 else "Persons"

greet = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
plan  = f"*Plan:- {total_days_calc}Days and {total_nights}{night_txt} {final_route} for {total_pax} {person_txt}*"

grouped = {}
for it in items:
    dstr = pd.to_datetime(it["Date"]).strftime("%d-%b-%Y") if pd.notna(it["Date"]) and str(it["Date"]) else "N/A"
    tp = f"{it.get('Time','')}: " if str(it.get('Time','')).strip() else ""
    grouped.setdefault(dstr, []).append(f"{tp}{_code_to_desc(it['Code'])}")

itinerary_text = greet + plan + "\n\n*Itinerary:*\n"
for i,(d,evs) in enumerate(grouped.items(),1):
    itinerary_text += f"\n*Day{i}:{d}*\n" + "\n".join(evs) + "\n"

details_bits = [x for x in [car_types or None, hotel_types or None, bhas_desc_str or None] if x]
details_line = "(" + ",".join(details_bits) + ")" if details_bits else ""

itinerary_text += f"\n*Package cost: ₹{in_locale(total_package)}/-*\n"
if referred_sel != "-- None --":
    itinerary_text += f"*Package cost (after referral 10%): ₹{in_locale(after_ref)}/-*\n"
itinerary_text += f"{details_line}"

# Exclusions / Notes / Policy / Payment / Account (same as earlier, trimmed for brevity)
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
    "3. Bhasm-Aarti: tickets at actuals; subject to availability/cancellations.",
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
acct = """*Company Account details:-*
Account Name: ACHALA HOLIDAYS PVT LTD
Bank: Axis Bank
Account No: 923020071937652
IFSC Code: UTIB0000329
MICR Code: 452211003
Branch: Ground Floor, 77, Dewas Road, Ujjain, MP 456010

Regards,
Team TravelAajKal™️ • Reg. Achala Holidays Pvt Ltd
Visit: www.travelaajkal.com • IG: @travelaaj_kal
DPIIT-recognized Startup • TravelAajKal® is a registered trademark.
"""

final_output = itinerary_text + "\n\n" + exclusions + "\n\n" + notes + "\n\n" + cxl + "\n\n" + pay + "\n\n" + acct

# ================= Serialize rows for Mongo (no datetime objects)
rows_serialized = base.copy()
if "Date" in rows_serialized.columns:
    try:
        rows_serialized["Date"] = pd.to_datetime(rows_serialized["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        rows_serialized["Date"] = rows_serialized["Date"].fillna("")
    except Exception:
        rows_serialized["Date"] = rows_serialized["Date"].astype(str)

# ================= Auto-save / Upsert (with revision support) =================
# Detect if we loaded a previous doc
loaded = st.session_state.get("loaded_doc")
loaded_id_str = str(loaded.get("_id")) if loaded else None

# Minimal revision notes
rev_notes = []
if loaded:
    try:
        if loaded.get("package_total") != int(total_package): rev_notes.append("package_total changed")
        if loaded.get("actual_total")  != int(total_actual):  rev_notes.append("actual_total changed")
        if loaded.get("final_route","") != final_route:       rev_notes.append("route changed")
        if loaded.get("start_date","") != str(start_date_calc) or loaded.get("end_date","") != str(end_date_calc):
            rev_notes.append("dates changed")
    except Exception:
        pass

key_filter = {"client_mobile": client_mobile, "start_date": str(start_date_calc)}
record = {
    "client_name": client_name,
    "client_mobile": client_mobile,
    "representative": rep,
    "upload_date": datetime.datetime.utcnow(),
    "start_date": str(start_date_calc),
    "end_date": str(end_date_calc),
    "total_days": int(total_days_calc),
    "total_pax": int(total_pax),
    "final_route": final_route,
    "car_types": car_types,
    "hotel_types": hotel_types,
    # Bhasmarathi
    "bhasmarathi_required": (bhas_required=="Yes"),
    "bhasmarathi_type": bhas_type if bhas_required=="Yes" else None,
    "bhasmarathi_persons": int(bhas_persons) if bhas_required=="Yes" else 0,
    "bhasmarathi_unit_pkg": int(bhas_unit_pkg) if bhas_required=="Yes" else 0,
    "bhasmarathi_unit_actual": int(bhas_unit_actual) if bhas_required=="Yes" else 0,
    "bhasmarathi_pkg_total": int(bhas_pkg_total),
    "bhasmarathi_actual_total": int(bhas_actual_total),
    # totals
    "package_total": int(total_package),
    "package_after_referral": int(after_ref),
    "actual_total": int(total_actual),
    "profit_total": int(profit_total),
    "referred_by": referred_sel if referred_sel != "-- None --" else None,
    "referral_discount_pct": 10 if referred_sel != "-- None --" else 0,
    # rows
    "rows": rows_serialized.to_dict(orient="records"),
    # legacy/back-compat
    "package_cost": int(total_package),
    "bhasmarathi_types": bhas_desc_str,
    # text
    "itinerary_text": final_output,
    # revision metadata (when applicable)
    "is_revision": bool(loaded),
    "revision_of": loaded_id_str if loaded else None,
    "revision_num": (int(loaded.get("revision_num",0)) + 1) if loaded else 0,
    "revision_notes": ", ".join(rev_notes) if rev_notes else None,
}

saved_key = f"{client_mobile}|{start_date_calc}"
already = st.session_state.get("_last_saved_key") == saved_key
try:
    res = col_it.update_one(key_filter, {"$set": record}, upsert=True)
    st.session_state["_last_saved_key"] = saved_key
    if not already:
        if res.upserted_id:
            st.success(f"✅ Saved new itinerary (ID: {res.upserted_id}).")
        else:
            if loaded:
                st.info("🔁 Updated as a revision of the previously loaded package.")
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
        file_name=f"itinerary_{client_name}_{start_date_calc}.txt",
        mime="text/plain",
        use_container_width=True
    )
