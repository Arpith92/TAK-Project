# app.py
from __future__ import annotations

# ----------------- Compatibility safety -----------------
try:
    import streamlit as st, rich  # noqa
    from packaging.version import Version  # noqa
    import sys, subprocess  # noqa
    if Version(st.__version__) < Version("1.42.0") and Version(rich.__version__) >= Version("14.0.0"):
        subprocess.run([sys.executable, "-m", "pip", "install", "rich==13.9.4"], check=True)
        st.warning("Adjusted rich to 13.9.4 for compatibility. Rerunning…")
        st.rerun()
except Exception:
    import streamlit as st  # ensure st is available

# ----------------- Imports -----------------
import io, math, datetime as dt, os, re
from collections.abc import Mapping
from datetime import timedelta, date as _date
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from pymongo import MongoClient
from bson import ObjectId  # ← needed for the new section

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
    # 1) Try Streamlit secrets
    try:
        raw = st.secrets.get("users", {})
        if isinstance(raw, Mapping):
            return dict(raw)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    # 2) Fallback to local .streamlit/secrets.toml (dev)
    try:
        try:
            import tomllib  # py311+
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, Mapping):
            return dict(u)
        if isinstance(u, dict):
            return u
    except Exception:
        pass
    return {}

def audit_login(user: str):
    now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    try:
        cols["audit_logins"].insert_one({
            "user": str(user),
            "ts_utc": now_utc,
            "ts_ist": now_utc.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S %Z"),
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

        if st.session_state.get("user"):
            return st.session_state["user"]

        users_map = _load_users()
        if not users_map:
            st.error("Login not configured. Add a **[users]** section in Secrets with PINs.")
            st.stop()

        st.markdown("### 🔐 Login")
        c1, c2 = st.columns(2)
        with c1:
            name = st.selectbox("User", list(users_map.keys()), key="login_user")
        with c2:
            pin = st.text_input("PIN", type="password", key="login_pin")

        if st.button("Sign in"):
            if str(users_map.get(name, "")).strip() == str(pin).strip():
                st.session_state["user"] = name
                try:
                    audit_login(name)
                except Exception:
                    pass
                st.success(f"Welcome, {name}!")
                st.rerun()
            else:
                st.error("Invalid PIN")
                st.stop()
    return None

# ================= Mongo =======================
def _find_uri() -> str | None:
    for k in ("mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"):
        try:
            v = st.secrets.get(k)
        except Exception:
            v = None
        if v:
            return v
    for k in ("mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"):
        v = os.getenv(k)
        if v:
            return v
    return None

@st.cache_resource
def mongo_client():
    uri = _find_uri()
    if not uri:
        st.error("Mongo URI not configured. Add mongo_uri in Secrets.")
        st.stop()
    client = MongoClient(uri, appName="TAK_App", maxPoolSize=100, serverSelectionTimeoutMS=5000, tz_aware=True)
    client.admin.command("ping")
    return client

@st.cache_resource
def get_collections():
    """
    Always return a dict with the required keys.
    """
    db = mongo_client()["TAK_DB"]
    return {
        "itineraries":     db["itineraries"],
        "audit_logins":    db["audit_logins"],
        "package_updates": db["package_updates"],
        "followups":       db["followups"],
        "expenses":        db["expenses"],
    }

cols = get_collections()
_db = mongo_client()["TAK_DB"]  # fallback handle

def _col(cols_dict: dict, key: str):
    val = cols_dict.get(key, None)
    return val if val is not None else _db[key]

col_it        = _col(cols, "itineraries")
col_updates   = _col(cols, "package_updates")
col_followups = _col(cols, "followups")
col_expenses  = _col(cols, "expenses")

for _k in ("itineraries", "audit_logins", "package_updates", "followups", "expenses"):
    if _k not in cols or cols[_k] is None:
        cols[_k] = _db[_k]

from datetime import datetime as _dt

def _to_int(x, default=0):
    try:
        if x is None:
            return default
        return int(float(str(x).replace(",", "")))
    except Exception:
        return default

def upsert_update_from_rep(itinerary_id: str, representative: str, actor_user: str):
    """
    Ensure `package_updates` exists and is assigned to `representative`.
    - If no doc exists: create with status='followup'
    - If exists and not confirmed/cancelled: keep status, but set assigned_to=<representative>
    """
    itinerary_id = str(itinerary_id or "")
    representative = (representative or "").strip()
    if not itinerary_id or not representative:
        return

    pre = col_updates.find_one({"itinerary_id": itinerary_id}, {"_id": 1, "assigned_to": 1, "status": 1})
    if not pre:
        col_updates.update_one(
            {"itinerary_id": itinerary_id},
            {"$set": {
                "status": "followup",
                "assigned_to": representative,
                "updated_at": _dt.utcnow(),
            }},
            upsert=True,
        )
        # first assignment trail
        col_followups.insert_one({
            "itinerary_id": itinerary_id,
            "created_at": _dt.utcnow(),
            "created_by": actor_user,
            "status": "followup",
            "comment": f"Auto-assigned from representative {representative}",
            "credited_to": representative,
            "next_followup_on": None,
        })
    else:
        if pre.get("status") not in ("confirmed", "cancelled") and pre.get("assigned_to") != representative:
            col_updates.update_one(
                {"itinerary_id": itinerary_id},
                {"$set": {"assigned_to": representative, "updated_at": _dt.utcnow()}}
            )

def sync_initial_cost_to_expenses_and_updates(itinerary_id: str, total_package: int):
    """
    Mirror initial cost so other pages immediately see amounts.
    Writes both to `expenses` and `package_updates` without changing status.
    """
    final_cost = _to_int(total_package, 0)
    base_amt   = final_cost
    disc_amt   = 0

    # expenses
    col_expenses.update_one(
        {"itinerary_id": str(itinerary_id)},
        {"$set": {
            "itinerary_id": str(itinerary_id),
            "base_package_cost": int(base_amt),
            "discount": int(disc_amt),
            "final_package_cost": int(final_cost),
            "package_cost": int(final_cost),
            "saved_at": _dt.utcnow(),
        }},
        upsert=True,
    )

    # package_updates (do not clobber status; just mirror cost fields)
    col_updates.update_one(
        {"itinerary_id": str(itinerary_id)},
        {"$set": {
            "package_cost": int(final_cost),
            "final_package_cost": int(final_cost),
            "base_package_cost": int(base_amt),
            "discount": int(disc_amt),
            "updated_at": _dt.utcnow(),
        }},
        upsert=True,
    )

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

# ---------- Daily counter ----------
def _today_created_count() -> int:
    now_ist = dt.datetime.now(IST)
    start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ist = start_ist + dt.timedelta(days=1)
    start_utc = start_ist.astimezone(dt.timezone.utc)
    end_utc = end_ist.astimezone(dt.timezone.utc)
    try:
        return col_it.count_documents({"upload_date": {"$gte": start_utc, "$lt": end_utc}})
    except Exception:
        return 0

st.info(f"📦 Packages created today: **{_today_created_count()}** (resets at 23:59 IST)")

# ================= Caching helpers =================
@st.cache_data(ttl=900)
def read_excel_from_url(url, sheet_name=None):
    r = requests.get(url, timeout=20); r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name)

# ================= Load static masters (cached) =================
try:
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
    bhas_df = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet_name="Bhasmarathi_Type")
except Exception as e:
    st.error(f"Failed to load master sheets: {e}")
    st.stop()

# ================= Small utils =================
def is_valid_mobile(num: str) -> bool:
    digits = "".join(ch for ch in str(num or "") if ch.isdigit())
    return len(digits) == 10

def in_locale(n: int) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

def ceil_to_999(n: float) -> int:
    return (math.ceil(n/1000)*1000 - 1) if n > 0 else 0

def _code_to_desc(code) -> str:
    if code is None:
        return "No code provided"
    s = str(code).strip()
    if s == "" or s.lower() in ("none","nan"):
        return "No code provided"
    try:
        m = code_df.loc[code_df["Code"].astype(str) == s, "Particulars"]
        return str(m.iloc[0]) if not m.empty else f"No description found for code {s}"
    except Exception:
        return f"No description found for code {s}"

def _code_to_route(code) -> str | None:
    if code is None:
        return None
    s = str(code).strip()
    if s == "" or s.lower() in ("none","nan"):
        return None
    try:
        m = code_df.loc[code_df["Code"].astype(str) == s, "Route"]
        return str(m.iloc[0]) if not m.empty else None
    except Exception:
        return None

# ---------- editor helpers ----------
TARGET_COLS = ["Date","Time","Code","Car Type","Hotel Type","Stay City","Room Type",
               "Pkg-Car Cost","Pkg-Hotel Cost","Act-Car Cost","Act-Hotel Cost"]
_EDITOR_KEY = "editor_widget"
_MODEL_KEY = "editor_model"
_SEARCH_DOC_KEY = "search_loaded_doc"

def _blank_df(n_rows: int, start: _date) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": [start + timedelta(days=i) for i in range(n_rows)],
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
    })[TARGET_COLS]

def _normalize_to_model(obj) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame(columns=TARGET_COLS)
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            df = pd.DataFrame(obj["data"])
        else:
            try:
                df = pd.DataFrame.from_dict(obj)
            except Exception:
                return pd.DataFrame(columns=TARGET_COLS)
    else:
        return pd.DataFrame(columns=TARGET_COLS)

    for c in TARGET_COLS:
        if c not in df.columns:
            df[c] = 0.0 if "Cost" in c else ""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    for c in ["Pkg-Car Cost","Pkg-Hotel Cost","Act-Car Cost","Act-Hotel Cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df[TARGET_COLS].copy()

def _seed_editor_model(n_rows: int, start: _date):
    st.session_state.setdefault(_MODEL_KEY, _blank_df(n_rows, start))

def _apply_dates_days(n_rows: int, start: _date):
    df = st.session_state.get(_MODEL_KEY)
    if df is None:
        _seed_editor_model(n_rows, start)
        return
    cur = len(df)
    if cur < n_rows:
        add = _blank_df(n_rows - cur, start + timedelta(days=cur))
        df = pd.concat([df, add], ignore_index=True)
    elif cur > n_rows:
        df = df.iloc[:n_rows].reset_index(drop=True)
    df.loc[:, "Date"] = [start + timedelta(days=i) for i in range(n_rows)]
    st.session_state[_MODEL_KEY] = df

def _editor_sync():
    """Robust sync for st.data_editor – persists edits on first attempt."""
    raw = st.session_state.get(_EDITOR_KEY, None)
    if raw is None:
        return

    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        for c in TARGET_COLS:
            if c not in df.columns:
                df[c] = 0.0 if "Cost" in c else ""
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        for c in ["Pkg-Car Cost","Pkg-Hotel Cost","Act-Car Cost","Act-Hotel Cost"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df[TARGET_COLS].copy()

    # ✅ always coerce and store immediately regardless of payload shape
    if isinstance(raw, pd.DataFrame):
        st.session_state[_MODEL_KEY] = _coerce(raw.copy())
        return

    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        st.session_state[_MODEL_KEY] = _coerce(pd.DataFrame(raw["data"]))
        return

    if isinstance(raw, dict) and ("edited_rows" in raw or "added_rows" in raw or "deleted_rows" in raw):
        base = st.session_state.get(_MODEL_KEY, pd.DataFrame(columns=TARGET_COLS))
        df = base.copy()
        edited = raw.get("edited_rows", {}) or {}
        for idx_str, changes in edited.items():
            try:
                idx = int(idx_str)
            except Exception:
                idx = idx_str
            if idx in df.index:
                for col, val in (changes or {}).items():
                    if col in df.columns:
                        df.at[idx, col] = val
        st.session_state[_MODEL_KEY] = _coerce(df)
        return

    if _MODEL_KEY not in st.session_state:
        st.session_state[_MODEL_KEY] = _coerce(pd.DataFrame(columns=TARGET_COLS))

# ---------- dropdown options ----------
stay_city_options = sorted(stay_city_df["Stay City"].dropna().astype(str).unique().tolist()) if "Stay City" in stay_city_df.columns else []
code_options = code_df["Code"].dropna().astype(str).unique().tolist() if not code_df.empty else []
base_cars = ["Sedan","Ertiga","Innova","Tempo Traveller"]
car_options = [f"{ac} {c}" for c in base_cars for ac in ("AC","Non AC")]
hotel_options = ["Homestay","AC Standard AC","Non-AC Standard AC","3Star AC Hotel room","3Star AC Hotel room with Breakfast","4Star AC Hotel room","4Star AC Hotel room with Breakfast","5Star AC Hotel room","5Star AC Hotel room with Breakfast"]
room_options = [f"{occ} occupancy {i} room" for occ in ["Double","Triple","Quad","Quint"] for i in range(1,5)]

def _time_list(step_minutes=15):
    base = dt.datetime(2000,1,1,0,0)
    return [(base + dt.timedelta(minutes=i)).time().strftime("%I:%M %p") for i in range(0,24*60,step_minutes)]

time_options = _time_list(15)

# ---------- login ----------
user = _login()
if not user:
    st.stop()

# ---------- Mode ----------
st.subheader("Mode")
mode = st.radio(
    "Choose what you want to do",
    ["Create new itinerary", "Search itinerary", "Upload itinerary"],  # ← added third option
    horizontal=True,
    label_visibility="collapsed"
)

# ---------- referral helpers ----------
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
            if n or m:
                labels.append(f"{n} — {m}" if n and m else n or m)
        return sorted(set(labels), key=lambda s: s.lower())
    except Exception:
        return []

def _client_suggestions(prefix: str) -> list[str]:
    if not prefix:
        return []
    try:
        rx = f"^{re.escape(prefix)}"
        cur = col_it.aggregate([
            {"$match": {"$or": [
                {"client_name": {"$regex": rx, "$options": "i"}},
                {"client_mobile": {"$regex": rx}}
            ]}},
            {"$group": {"_id": {"n": "$client_name", "m": "$client_mobile"}}},
            {"$project": {"_id": 0, "name": "$_id.n", "mobile": "$_id.m"}},
            {"$limit": 50}
        ])
        res = []
        for x in cur:
            n = (x.get("name") or "").strip()
            m = (x.get("mobile") or "").strip()
            if n or m:
                res.append(f"{n} — {m}" if n and m else n or m)
        return sorted(set(res), key=lambda s: s.lower())
    except Exception:
        return []

# =========================================================
# CREATE NEW
# =========================================================
if mode == "Create new itinerary":
    # defaults
    st.session_state.setdefault("k_client_name", "")
    st.session_state.setdefault("k_mobile", "")
    st.session_state.setdefault("k_rep", "-- Select --")
    st.session_state.setdefault("k_total_pax", 2)
    st.session_state.setdefault("k_start", dt.date.today())
    st.session_state.setdefault("k_days", 2)
    st.session_state.setdefault("k_ref_sel", "-- None --")
    st.session_state.setdefault("k_bhas_req", "No")
    st.session_state.setdefault("k_bhas_type", "V-BH")
    st.session_state.setdefault("k_bhas_pax", 0)
    st.session_state.setdefault("k_bhas_pkg", 0)
    st.session_state.setdefault("k_bhas_act", 0)

    st.markdown("### Create new itinerary")
    c0, c1, c2, c3 = st.columns([1.6, 1, 1, 1])
    with c0:
        client_name = st.text_input("Client Name*", key="k_client_name")
    with c1:
        client_mobile_raw = st.text_input("Client mobile (10 digits)*", key="k_mobile")
    with c2:
        rep = st.selectbox("Representative*", ["-- Select --","Arpith","Reena","Kuldeep","Teena"], key="k_rep")
    with c3:
        total_pax = st.number_input("Total Pax*", min_value=1, step=1, key="k_total_pax")

    ref_labels = ["-- None --"] + _load_client_refs()
    referred_sel = st.selectbox("Referred By (applies 10% discount)", ref_labels, key="k_ref_sel")
    has_ref = referred_sel != "-- None --"
    discount_pct = 10 if has_ref else 0

    h1, h2 = st.columns(2)
    with h1:
        start_date = st.date_input("Start date", key="k_start")
    with h2:
        days = st.number_input("No. of days", min_value=1, step=1, key="k_days")

    if st.button("📅 Apply dates & days to table"):
        _apply_dates_days(int(days), start_date)
        st.success("Dates & rows applied. You can edit the table now.")

    if _MODEL_KEY not in st.session_state:
        _seed_editor_model(int(days), start_date)

    # Bhas outside table
    bhc1, bhc2, bhc3 = st.columns(3)
    with bhc1:
        bhas_required = st.selectbox("Bhasmarathi required?", ["No","Yes"], key="k_bhas_req")
    with bhc2:
        bhas_type = st.selectbox("Bhasmarathi Type", ["V-BH","P-BH","BH"], key="k_bhas_type")
    with bhc3:
        bhas_persons = st.number_input("Persons for Bhasmarathi", min_value=0, step=1, key="k_bhas_pax",
                                       disabled=(st.session_state["k_bhas_req"]=="No"))
    bhc4, bhc5 = st.columns(2)
    with bhc4:
        bhas_unit_pkg = st.number_input("Bhasmarathi unit cost (Package)", min_value=0, step=100, key="k_bhas_pkg",
                                        disabled=(st.session_state["k_bhas_req"]=="No"))
    with bhc5:
        bhas_unit_actual = st.number_input("Bhasmarathi unit cost (Actual)", min_value=0, step=100, key="k_bhas_act",
                                           disabled=(st.session_state["k_bhas_req"]=="No"))

    # ---------- Line items table ----------
    st.markdown("### Fill line items")
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
    st.data_editor(
        st.session_state[_MODEL_KEY],
        key=_EDITOR_KEY,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config=col_cfg,
        on_change=_editor_sync,
    )
    _editor_sync()

    # --- Live totals preview (before Generate) ---
    df_prev = st.session_state.get(_MODEL_KEY, pd.DataFrame(columns=TARGET_COLS)).copy()
    pkg_car = pd.to_numeric(df_prev.get("Pkg-Car Cost", 0), errors="coerce").fillna(0).sum()
    pkg_hotel = pd.to_numeric(df_prev.get("Pkg-Hotel Cost", 0), errors="coerce").fillna(0).sum()
    act_car = pd.to_numeric(df_prev.get("Act-Car Cost", 0), errors="coerce").fillna(0).sum()
    act_hotel = pd.to_numeric(df_prev.get("Act-Hotel Cost", 0), errors="coerce").fillna(0).sum()

    bhas_required = st.session_state.get("k_bhas_req", "No")
    bhas_persons = int(st.session_state.get("k_bhas_pax", 0) or 0)
    bhas_unit_pkg = int(st.session_state.get("k_bhas_pkg", 0) or 0)
    bhas_unit_actual = int(st.session_state.get("k_bhas_act", 0) or 0)
    has_ref = st.session_state.get("k_ref_sel", "-- None --") != "-- None --"

    bhas_pkg_total = (bhas_unit_pkg * bhas_persons) if bhas_required == "Yes" else 0
    bhas_actual_total = (bhas_unit_actual * bhas_persons) if bhas_required == "Yes" else 0

    package_cost_rows = float(pkg_car + pkg_hotel)
    actual_cost_rows = float(act_car + act_hotel)

    preview_package = ceil_to_999(package_cost_rows + bhas_pkg_total)
    preview_actual = actual_cost_rows + bhas_actual_total
    preview_profit = int(preview_package - preview_actual)
    preview_after_ref = int(round(preview_package * 0.9)) if has_ref else preview_package

    badge_color = "#16a34a" if preview_profit >= 4000 else "#dc2626"
    hint = ""if preview_profit >= 4000 else " • Keep profit margin ≥ ₹4,000"

    ref_html = (
        f'<div style="padding:8px 12px; border-radius:8px; background:#7c3aed; color:white;">'
        f'After Referral (10%): <b>₹{in_locale(preview_after_ref)}</b></div>'
    ) if has_ref else ""

    totals_html = (
        '<div style="display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 4px 0;">'
        f'<div style="padding:8px 12px; border-radius:8px; background:#0ea5e9; color:white;">'
        f'Package Cost: <b>₹{in_locale(preview_package)}</b></div>'
        f'{ref_html}'
        f'<div style="padding:8px 12px; border-radius:8px; background:#475569; color:white;">'
        f'Actual Cost: <b>₹{in_locale(preview_actual)}</b></div>'
        f'<div style="padding:8px 12px; border-radius:8px; background:{badge_color}; color:white;">'
        f'Profit: <b>₹{in_locale(preview_profit)}</b>{hint}</div>'
        '</div>'
    )
    st.markdown(totals_html, unsafe_allow_html=True)
    st.markdown("")

    # ---------- Generate & save ----------
    if st.button("✅ Generate itinerary & save (rev 1 for new / next rev for existing)", use_container_width=True):
        if not client_name:
            st.error("Enter **Client Name**.")
            st.stop()
        if not is_valid_mobile(client_mobile_raw):
            st.error("Enter a valid **10-digit** mobile.")
            st.stop()
        if rep == "-- Select --":
            st.error("Select **Representative**.")
            st.stop()

        client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())
        base = st.session_state[_MODEL_KEY].copy()
        if base.empty:
            st.error("No rows to save. Apply dates & days and fill the table.")
            st.stop()

        base["Date"] = [start_date + dt.timedelta(days=i) for i in range(len(base))]

        # totals
        pkg_car = pd.to_numeric(base.get("Pkg-Car Cost", 0), errors="coerce").fillna(0).sum()
        pkg_hotel = pd.to_numeric(base.get("Pkg-Hotel Cost", 0), errors="coerce").fillna(0).sum()
        act_car = pd.to_numeric(base.get("Act-Car Cost", 0), errors="coerce").fillna(0).sum()
        act_hotel = pd.to_numeric(base.get("Act-Hotel Cost", 0), errors="coerce").fillna(0).sum()

        bhas_pkg_total = (bhas_unit_pkg * bhas_persons) if bhas_required=="Yes" else 0
        bhas_actual_total = (bhas_unit_actual * bhas_persons) if bhas_required=="Yes" else 0

        total_package = ceil_to_999(float(pkg_car + pkg_hotel) + bhas_pkg_total)
        total_actual = float(act_car + act_hotel) + bhas_actual_total
        profit_total = int(total_package - total_actual)
        after_ref = int(round(total_package * 0.9)) if has_ref else total_package

        # route
        route_parts = []
        for r in base["Code"]:
            rt = _code_to_route(r)
            if rt:
                route_parts.append(rt)
        route_raw = "-".join(route_parts).replace(" -","-").replace("- ","-")
        route_list = [x for x in route_raw.split("-") if x]
        final_route = "-".join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i-1]])

        car_types = "-".join(pd.Series(base.get("Car Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")
        hotel_types = "-".join(pd.Series(base.get("Hotel Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")

        # Bhas desc
        bhas_desc_str = ""
        if bhas_required == "Yes":
            mm = bhas_df.loc[bhas_df["Bhasmarathi Type"].astype(str) == str(st.session_state.get("k_bhas_type","V-BH")), "Description"]
            if not mm.empty:
                bhas_desc_str = str(mm.iloc[0])

        # meta
        total_days_calc = base.shape[0]
        total_nights = max(total_days_calc - 1, 0)
        night_txt = "Night" if total_nights == 1 else "Nights"
        person_txt = "Person" if total_pax == 1 else "Persons"

        # day-wise items
        items = [{"Date": r["Date"], "Time": r.get("Time",""), "Code": r.get("Code","")} for _, r in base.iterrows()]
        grouped = {}
        for it in items:
            dstr = pd.to_datetime(it["Date"]).strftime("%d-%b-%Y") if (pd.notna(it["Date"]) and str(it["Date"])) else "N/A"
            tp = f"{it.get('Time','')}: " if str(it.get('Time','')).strip() else ""
            grouped.setdefault(dstr, []).append(f"{tp}{_code_to_desc(it['Code'])}")

        greet = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
        plan = f"*Plan:- {total_days_calc}Days and {total_nights}{night_txt} {final_route} for {total_pax} {person_txt}*"
        itinerary_text = greet + plan + "\n\n*Itinerary:*\n"
        for i,(d,evs) in enumerate(grouped.items(),1):
            itinerary_text += f"\n*Day{i}:{d}*\n" + "\n".join(evs) + "\n"

        # dynamic inclusions
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
        if "Stay City" in base.columns and "Room Type" in base.columns and not stay_city_df.empty:
            stay_series = base["Stay City"].astype(str).fillna("")
            city_nights = stay_series[stay_series != ""].value_counts().to_dict()
            used = 0
            for stay_code, nn in city_nights.items():
                if used >= total_nights:
                    break
                match = stay_city_df[stay_city_df["Stay City"] == stay_code]
                if not match.empty:
                    city_name = str(match["City"].iloc[0])
                    rt = base.loc[base["Stay City"] == stay_code, "Room Type"].dropna().astype(str).unique()
                    inc.append(f"{min(nn, total_nights-used)}Night stay in {city_name} with {'/'.join(rt) or 'room'} in {hotel_types or 'hotel'}.")
                    used += nn
        if hotel_types:
            inc += [
                "*Standard check-in at 12:00 PM and check-out at 09:00 AM.*",
                "Early check-in and late check-out are subject to room availability."
            ]

        inclusions = "*Inclusions:-*\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(inc)]) if inc else "*Inclusions:-*\n1. As per itinerary."

        details_bits = [x for x in [car_types or None, hotel_types or None, bhas_desc_str or None] if x]
        details_line = "(" + ",".join(details_bits) + ")" if details_bits else ""

        itinerary_text += f"\n*Package cost: ₹{in_locale(total_package)}/-*\n"
        if has_ref:
            itinerary_text += f"*Package cost (after referral 10%): ₹{in_locale(after_ref)}/-*\n"
        itinerary_text += f"{details_line}"

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

        cxl = (
            "*Cancellation Policy:-*\n"
            "1. 30+ days → 20% of advance deducted.\n"
            "2. 15–29 days → 50% of advance deducted.\n"
            "3. <15 days → No refund on advance.\n"
            "4. No refund for no-shows/early departures.\n"
            "5. One-time reschedule allowed ≥15 days prior, subject to availability.\n"
        )

        pay = "*Payment Terms:-*\n50% advance and remaining 50% after arrival at Ujjain.\n"

        acct = (
            "For booking confirmation, please make the advance payment to the company's current account provided below.\n\n"
            "*Company Account details:-*\n"
            "Account Name: ACHALA HOLIDAYS PVT LTD\n"
            "Bank: Axis Bank\n"
            "Account No: 923020071937652\n"
            "IFSC Code: UTIB0000329\n"
            "MICR Code: 452211003\n"
            "Branch: Ground Floor, 77, Dewas Road, Ujjain, MP 456010\n\n"
            "Regards,\n"
            "Team TravelAajKal™️ • Reg. Achala Holidays Pvt Ltd\n"
            "Visit: www.travelaajkal.com • IG: @travelaaj_kal\n"
            "DPIIT-recognized Startup • TravelAajKal® is a registered trademark.\n"
        )

        final_output = itinerary_text + "\n\n" + inclusions + "\n\n" + exclusions + "\n" + notes + "\n\n" + cxl + "\n" + pay + "\n" + acct

        rows_serialized = base.copy()
        rows_serialized["Date"] = pd.to_datetime(rows_serialized["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        start_key = str(start_date)

        try:
            mx = -1
            for ddoc in col_it.find({"client_mobile": client_mobile, "start_date": start_key}, {"revision_num":1}):
                mx = max(mx, int(ddoc.get("revision_num", 0) or 0))
            next_rev = 1 if mx < 1 else (mx + 1)
        except Exception:
            next_rev = 1

        now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        record = {
            "client_name": client_name,
            "client_mobile": client_mobile,
            "representative": rep,
            "upload_date": now_utc,
            "start_date": start_key,
            "end_date": str(start_date + dt.timedelta(days=len(base)-1)),
            "total_days": int(base.shape[0]),
            "total_pax": int(total_pax),
            "final_route": final_route,
            "car_types": car_types,
            "hotel_types": hotel_types,
            "bhasmarathi_required": (bhas_required=="Yes"),
            "bhasmarathi_type": st.session_state.get("k_bhas_type") if bhas_required=="Yes" else None,
            "bhasmarathi_persons": int(bhas_persons) if bhas_required=="Yes" else 0,
            "bhasmarathi_unit_pkg": int(bhas_unit_pkg) if bhas_required=="Yes" else 0,
            "bhasmarathi_unit_actual": int(bhas_unit_actual) if bhas_required=="Yes" else 0,
            "bhasmarathi_pkg_total": int(bhas_pkg_total),
            "bhasmarathi_actual_total": int(bhas_actual_total),
            "package_total": int(total_package),
            "package_after_referral": int(after_ref),
            "actual_total": int(total_actual),
            "profit_total": int(profit_total),
            "referred_by": referred_sel if has_ref else None,
            "referral_discount_pct": discount_pct,
            "rows": rows_serialized.to_dict(orient="records"),
            "package_cost": int(total_package),
            "itinerary_text": final_output,
            "revision_num": int(next_rev),
            "is_revision": True if next_rev > 1 else False,
            "revision_notes": "initial" if next_rev == 1 else "auto: new version",
        }

        try:
            res = col_it.insert_one(record)   # INSERT
            new_id = str(res.inserted_id)

            # (optional) stash for navigation or follow-up pages
            st.session_state["last_saved_itinerary_id"] = new_id

            # ⬇️ Immediately reflect assignment & cost for visibility in other pages
            upsert_update_from_rep(new_id, rep, actor_user=user)
            sync_initial_cost_to_expenses_and_updates(new_id, total_package)

            st.success(f"✅ Saved package for **{client_name}** ({client_mobile}) • **rev {next_rev}**")
            st.session_state["last_preview_text"] = final_output
            st.session_state["last_generated_meta"] = {"client": client_name, "mobile": client_mobile, "rev": next_rev}

        except Exception as e:
            st.error(f"Could not save itinerary: {e}")
            st.stop()

    st.divider()
    st.caption("Tip: Click “Apply dates & days” before editing the table. Your first edits will persist.")

# =========================================================
# SEARCH / LOAD
# =========================================================
elif mode == "Search itinerary":
    st.markdown("### Search itinerary")
    q = st.text_input(
        "Type client name or mobile (1 character shows suggestions)",
        key="search_q", placeholder="e.g., Gaurav or 9576226271"
    )
    suggestions = _client_suggestions(q.strip()) if q.strip() else []

    picked_client_name, picked_client_mobile = "", ""
    sel_client = st.selectbox("Suggestions", ["--"] + suggestions, index=0, key="sel_client")

    if sel_client != "--":
        parts = [p.strip() for p in sel_client.split("—", 1)]
        if len(parts) == 2:
            picked_client_name, picked_client_mobile = parts[0], parts[1]
        elif parts and parts[0].isdigit():
            picked_client_mobile = parts[0]
        else:
            picked_client_name = parts[0]

    loaded_doc = st.session_state.get(_SEARCH_DOC_KEY)

    # ------- Selection of package & revision -------
    if picked_client_mobile:
        rx_name = f"^{re.escape(picked_client_name)}$" if picked_client_name else ".*"
        docs = list(
            col_it.find(
                {"client_mobile": picked_client_mobile, "client_name": {"$regex": rx_name, "$options": "i"}},
                {}
            ).sort([("start_date", -1), ("revision_num", -1), ("upload_date", -1)])
        )
        if not docs:
            st.info("No itineraries found for this client yet.")
        else:
            from collections import defaultdict
            by_start = defaultdict(list)
            for d in docs:
                by_start[str(d.get("start_date", ""))].append(d)

            start_dates = sorted(by_start.keys(), reverse=True)
            sel_start = st.selectbox("Select package start date", start_dates, key=f"start_{picked_client_mobile}")

            revs = sorted(by_start.get(sel_start, []), key=lambda x: int(x.get("revision_num", 0) or 0), reverse=True)
            rev_labels = [
                f"rev:{int(d.get('revision_num',0) or 0)} • uploaded:{str(d.get('upload_date',''))[:10]}"
                for d in revs
            ]
            sel_rev_idx = st.selectbox(
                "Select revision",
                list(range(len(revs))),
                format_func=lambda i: rev_labels[i],
                key=f"rev_{picked_client_mobile}_{sel_start}"
            )

            if st.button("📂 Load this revision",
                         key=f"load_{picked_client_mobile}_{sel_start}_{sel_rev_idx}",
                         use_container_width=False):
                chosen = revs[sel_rev_idx]
                st.session_state[_SEARCH_DOC_KEY] = chosen
                loaded_key = (
                    f"{chosen.get('client_mobile','')}:"
                    f"{str(chosen.get('start_date',''))}:"
                    f"rev{int(chosen.get('revision_num',0) or 0)}"
                )
                rows_df = pd.DataFrame(chosen.get("rows") or [])
                st.session_state[_MODEL_KEY] = _normalize_to_model(rows_df)
                st.session_state.pop(_EDITOR_KEY, None)  # force redraw
                st.session_state["_loaded_key"] = loaded_key
                st.rerun()

    # -------- Hydrate editor once selection is set --------
    loaded_doc = st.session_state.get(_SEARCH_DOC_KEY)
    if loaded_doc:
        current_key = (
            f"{loaded_doc.get('client_mobile','')}:"
            f"{str(loaded_doc.get('start_date',''))}:"
            f"rev{int(loaded_doc.get('revision_num',0) or 0)}"
        )
        if st.session_state.get("_loaded_key") != current_key:
            rows_df = pd.DataFrame(loaded_doc.get("rows") or [])
            st.session_state[_MODEL_KEY] = _normalize_to_model(rows_df)
            st.session_state.pop(_EDITOR_KEY, None)  # pick up new data
            st.session_state["_loaded_key"] = current_key

        # Pre-fill top fields from loaded_doc
        client_name = loaded_doc.get("client_name", "")
        client_mobile = loaded_doc.get("client_mobile", "")
        rep = loaded_doc.get("representative", "-- Select --") or "-- Select --"
        total_pax = int(loaded_doc.get("total_pax", 1) or 1)
        start_date = pd.to_datetime(loaded_doc.get("start_date")).date() if loaded_doc.get("start_date") else dt.date.today()
        days = int(loaded_doc.get("total_days", len(st.session_state.get(_MODEL_KEY, []))) or 1)

        c0, c1, c2, c3 = st.columns([1.6,1,1,1])
        with c0:
            st.text_input("Client Name*", value=client_name, key="s_client_name")
        with c1:
            st.text_input("Client mobile (10 digits)*", value=client_mobile, key="s_mobile", disabled=True)
        with c2:
            reps = ["-- Select --","Arpith","Reena","Kuldeep","Teena"]
            st.selectbox("Representative*", reps,
                         index=max(0, reps.index(rep) if rep in reps else 0), key="s_rep")
        with c3:
            st.number_input("Total Pax*", min_value=1, step=1, value=total_pax, key="s_pax")

        # Bhas from doc
        bhas_required = "Yes" if loaded_doc.get("bhasmarathi_required") else "No"
        bhas_type = loaded_doc.get("bhasmarathi_type","V-BH") or "V-BH"
        bhas_persons = int(loaded_doc.get("bhasmarathi_persons",0) or 0)
        bhas_unit_pkg = int(loaded_doc.get("bhasmarathi_unit_pkg",0) or 0)
        bhas_unit_actual = int(loaded_doc.get("bhasmarathi_unit_actual",0) or 0)

        bhc1, bhc2, bhc3 = st.columns(3)
        with bhc1:
            st.selectbox("Bhasmarathi required?", ["No","Yes"], index=(1 if bhas_required=="Yes" else 0), key="s_breq")
        with bhc2:
            st.selectbox("Bhasmarathi Type", ["V-BH","P-BH","BH"],
                         index=max(0, ["V-BH","P-BH","BH"].index(bhas_type) if bhas_type in ["V-BH","P-BH","BH"] else 0),
                         key="s_btype")
        with bhc3:
            st.number_input("Persons for Bhasmarathi", min_value=0, step=1, value=bhas_persons, key="s_bpax")
        bhc4, bhc5 = st.columns(2)
        with bhc4:
            st.number_input("Bhasmarathi unit cost (Package)", min_value=0, step=100, value=bhas_unit_pkg, key="s_bpkg")
        with bhc5:
            st.number_input("Bhasmarathi unit cost (Actual)", min_value=0, step=100, value=bhas_unit_actual, key="s_bact")

        # Initialize editor model if absent
        if _MODEL_KEY not in st.session_state:
            st.session_state[_MODEL_KEY] = (
                pd.DataFrame(loaded_doc.get("rows") or [])[TARGET_COLS]
                if loaded_doc.get("rows") else _blank_df(days, start_date)
            )

        st.markdown("### Fill line items")
        st.data_editor(
            st.session_state[_MODEL_KEY],
            key=_EDITOR_KEY,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
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
            },
            on_change=_editor_sync,
        )
        _editor_sync()

        if st.button("🔁 Update itinerary (save as next revision)", use_container_width=True):
            client_name_new = st.session_state.get("s_client_name", client_name)
            rep_new = st.session_state.get("s_rep", rep)
            total_pax_new = int(st.session_state.get("s_pax", total_pax) or 1)

            bhas_req_new = "Yes" if st.session_state.get("s_breq", "No") == "Yes" else "No"
            bhas_type_new = st.session_state.get("s_btype", "V-BH")
            bhas_pax_new = int(st.session_state.get("s_bpax", 0) or 0)
            bhas_unit_pkg_new = int(st.session_state.get("s_bpkg", 0) or 0)
            bhas_unit_act_new = int(st.session_state.get("s_bact", 0) or 0)

            base = _normalize_to_model(st.session_state[_MODEL_KEY]).copy()
            base["Date"] = [start_date + dt.timedelta(days=i) for i in range(len(base))]

            pkg_car = pd.to_numeric(base.get("Pkg-Car Cost", 0), errors="coerce").fillna(0).sum()
            pkg_hotel = pd.to_numeric(base.get("Pkg-Hotel Cost", 0), errors="coerce").fillna(0).sum()
            act_car = pd.to_numeric(base.get("Act-Car Cost", 0), errors="coerce").fillna(0).sum()
            act_hotel = pd.to_numeric(base.get("Act-Hotel Cost", 0), errors="coerce").fillna(0).sum()

            bhas_pkg_total = (bhas_unit_pkg_new * bhas_pax_new) if bhas_req_new == "Yes" else 0
            bhas_act_total = (bhas_unit_act_new * bhas_pax_new) if bhas_req_new == "Yes" else 0

            total_package = ceil_to_999(float(pkg_car + pkg_hotel) + bhas_pkg_total)
            total_actual = float(act_car + act_hotel) + bhas_act_total
            profit_total = int(total_package - total_actual)

            route_parts = []
            for r in base["Code"]:
                rt = _code_to_route(r)
                if rt:
                    route_parts.append(rt)
            route_raw = "-".join(route_parts).replace(" -","-").replace("- ","-")
            route_list = [x for x in route_raw.split("-") if x]
            final_route = "-".join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i-1]])

            car_types = "-".join(pd.Series(base.get("Car Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")
            hotel_types = "-".join(pd.Series(base.get("Hotel Type", [])).dropna().astype(str).replace("","").unique().tolist()).strip("-")

            total_days = len(base)
            total_nights = max(total_days - 1, 0)
            night_txt = "Night" if total_nights == 1 else "Nights"
            person_txt = "Person" if total_pax_new == 1 else "Persons"

            bhas_desc_str = ""
            if bhas_req_new == "Yes":
                mm = bhas_df.loc[bhas_df["Bhasmarathi Type"].astype(str) == str(bhas_type_new), "Description"]
                if not mm.empty:
                    bhas_desc_str = str(mm.iloc[0])

            items = [{"Date": r["Date"], "Time": r.get("Time",""), "Code": r.get("Code","")} for _, r in base.iterrows()]
            grouped = {}
            for it in items:
                dstr = pd.to_datetime(it["Date"]).strftime("%d-%b-%Y") if (pd.notna(it["Date"]) and str(it["Date"])) else "N/A"
                tp = f"{it.get('Time','')}: " if str(it.get('Time','')).strip() else ""
                grouped.setdefault(dstr, []).append(f"{tp}{_code_to_desc(it['Code'])}")

            greet = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name_new}*\n\n"
            plan = f"*Plan:- {total_days}Days and {total_nights}{night_txt} {final_route} for {total_pax_new} {person_txt}*"
            itinerary_text = greet + plan + "\n\n*Itinerary:*\n"
            for i,(d,evs) in enumerate(grouped.items(),1):
                itinerary_text += f"\n*Day{i}:{d}*\n" + "\n".join(evs) + "\n"

            inc = []
            if car_types:
                inc += [
                    f"Entire travel as per itinerary by {car_types}.",
                    "Toll, parking, and driver bata are included.",
                    "Airport/ Railway station pickup and drop."
                ]
            if bhas_desc_str:
                inc += [
                    f"{bhas_desc_str} for {total_pax_new} {person_txt}.",
                    "Bhasm-Aarti pickup and drop."
                ]
            if "Stay City" in base.columns and "Room Type" in base.columns and not stay_city_df.empty:
                stay_series = base["Stay City"].astype(str).fillna("")
                city_nights = stay_series[stay_series != ""].value_counts().to_dict()
                used = 0
                for stay_code, nn in city_nights.items():
                    if used >= total_nights:
                        break
                    match = stay_city_df[stay_city_df["Stay City"] == stay_code]
                    if not match.empty:
                        city_name = str(match["City"].iloc[0])
                        rt = base.loc[base["Stay City"] == stay_code, "Room Type"].dropna().astype(str).unique()
                        inc.append(f"{min(nn, total_nights-used)}Night stay in {city_name} with {'/'.join(rt) or 'room'} in {hotel_types or 'hotel'}.")
                        used += nn
            if hotel_types:
                inc += [
                    "*Standard check-in at 12:00 PM and check-out at 09:00 AM.*",
                    "Early check-in and late check-out are subject to room availability."
                ]

            inclusions_block = "*Inclusions:-*\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(inc)]) if inc else "*Inclusions:-*\n1. As per itinerary."
            details_bits = [x for x in [car_types or None, hotel_types or None, bhas_desc_str or None] if x]
            details_line = "(" + ",".join(details_bits) + ")" if details_bits else ""

            itinerary_text += f"\n*Package cost: ₹{in_locale(total_package)}/-*\n" + details_line

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

            cxl = (
                "*Cancellation Policy:-*\n"
                "1. 30+ days → 20% of advance deducted.\n"
                "2. 15–29 days → 50% of advance deducted.\n"
                "3. <15 days → No refund on advance.\n"
                "4. No refund for no-shows/early departures.\n"
                "5. One-time reschedule allowed ≥15 days prior, subject to availability.\n"
            )

            pay = "*Payment Terms:-*\n50% advance and remaining 50% after arrival at Ujjain.\n"

            acct = (
                "For booking confirmation, please make the advance payment to the company's current account provided below.\n\n"
                "*Company Account details:-*\n"
                "Account Name: ACHALA HOLIDAYS PVT LTD\n"
                "Bank: Axis Bank\n"
                "Account No: 923020071937652\n"
                "IFSC Code: UTIB0000329\n"
                "MICR Code: 452211003\n"
                "Branch: Ground Floor, 77, Dewas Road, Ujjain, MP 456010\n\n"
                "Regards,\n"
                "Team TravelAajKal™️ • Reg. Achala Holidays Pvt Ltd\n"
                "Visit: www.travelaajkal.com • IG: @travelaaj_kal\n"
                "DPIIT-recognized Startup • TravelAajKal® is a registered trademark.\n"
            )

            final_output = itinerary_text + "\n\n" + inclusions_block + "\n\n" + exclusions + "\n\n" + notes + "\n" + cxl + "\n\n" + pay + "\n\n" + acct

            # Next revision #
            mx = -1
            for ddoc in col_it.find({"client_mobile": client_mobile, "start_date": str(start_date)}, {"revision_num":1}):
                mx = max(mx, int(ddoc.get("revision_num", 0) or 0))
            next_rev = 1 if mx < 1 else (mx + 1)

            rows_serialized = base.copy()
            rows_serialized["Date"] = pd.to_datetime(rows_serialized["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
            now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
            record = {
                "client_name": client_name_new,
                "client_mobile": client_mobile,
                "representative": rep_new,
                "upload_date": now_utc,
                "start_date": str(start_date),
                "end_date": str(start_date + dt.timedelta(days=len(base)-1)),
                "total_days": int(len(base)),
                "total_pax": int(total_pax_new),
                "final_route": final_route,
                "car_types": car_types,
                "hotel_types": hotel_types,
                "bhasmarathi_required": (bhas_req_new == "Yes"),
                "bhasmarathi_type": bhas_type_new if bhas_req_new == "Yes" else None,
                "bhasmarathi_persons": int(bhas_pax_new) if bhas_req_new == "Yes" else 0,
                "bhasmarathi_unit_pkg": int(bhas_unit_pkg_new) if bhas_req_new == "Yes" else 0,
                "bhasmarathi_unit_actual": int(bhas_unit_act_new) if bhas_req_new == "Yes" else 0,
                "bhasmarathi_pkg_total": int(bhas_pkg_total),
                "bhasmarathi_actual_total": int(bhas_act_total),
                "package_total": int(total_package),
                "package_after_referral": int(total_package),  # (no referral toggle in Search-edit path)
                "actual_total": int(total_actual),
                "profit_total": int(profit_total),
                "rows": rows_serialized.to_dict(orient="records"),
                "package_cost": int(total_package),
                "itinerary_text": final_output,
                "revision_num": int(next_rev),
                "is_revision": True,
                "revision_notes": "edit from search",
            }

            try:
                res = col_it.insert_one(record)
                inserted_id = str(res.inserted_id)

                # Keep assignment aligned to representative edits (if any)
                upsert_update_from_rep(inserted_id, rep_new, actor_user=user)
                # Mirror latest quoted package total to expenses/updates
                sync_initial_cost_to_expenses_and_updates(inserted_id, total_package)

                st.success(f"✅ Updated & saved as **rev {next_rev}**")
                st.session_state["last_preview_text"] = final_output
                st.session_state["last_generated_meta"] = {"client": client_name_new, "mobile": client_mobile, "rev": next_rev}
            except Exception as e:
                st.error(f"Could not save itinerary: {e}")
                st.stop()

# =========================================================
# UPLOAD / QUICK UPDATE (new third section)
# =========================================================
elif mode == "Upload itinerary":
    st.markdown("### Upload / Update an existing itinerary")
    st.caption("Pick a package, edit dates/rep/final amount, and either confirm now or set to follow-up. "
               "Optionally upload a fresh itinerary .txt. All changes reflect in DB and dashboards.")

    # --- quick search + picker (latest first) ---
    qtext = st.text_input("Quick search (name / mobile / ACH)", "")
    try:
        docs = list(
            col_it.find(
                {},
                {
                    "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
                    "start_date": 1, "end_date": 1, "representative": 1,
                    "upload_date": 1, "revision_num": 1, "package_total": 1
                }
            ).sort([("upload_date", -1), ("revision_num", -1)]).limit(600)
        )
    except Exception as e:
        st.error(f"Could not load packages: {e}")
        docs = []

    def _mk_label(d):
        ach = str(d.get("ach_id","") or "—")
        nm  = str(d.get("client_name","") or "")
        mb  = str(d.get("client_mobile","") or "")
        stt = str(d.get("start_date","") or "—")
        rev = int(d.get("revision_num",1) or 1)
        return f"{ach} | {nm} | {mb} | start:{stt} | rev:{rev} | {str(d['_id'])}"

    options = [(_mk_label(d), str(d["_id"])) for d in docs]
    if qtext.strip():
        s = qtext.strip().lower()
        options = [(lbl, iid) for (lbl, iid) in options if s in lbl.lower()]

    if not options:
        st.info("No matching packages.")
        st.stop()

    sel_lbl = st.selectbox("Pick a package", [o[0] for o in options], key="upd_pick")
    chosen_id = next((iid for lbl, iid in options if lbl == sel_lbl), None)
    if not chosen_id:
        st.stop()

    # --- load current docs ---
    it_doc = col_it.find_one({"_id": ObjectId(chosen_id)}) or {}
    up_doc = col_updates.find_one({"itinerary_id": str(chosen_id)}) or {}
    ex_doc = col_expenses.find_one({"itinerary_id": str(chosen_id)}) or {}

    def _norm_date_local(x):
        try:
            if not x: return None
            return pd.to_datetime(x).date()
        except Exception:
            return None

    # Prefills
    ach_id  = str(it_doc.get("ach_id","") or "")
    nm      = str(it_doc.get("client_name","") or "")
    mb      = str(it_doc.get("client_mobile","") or "")
    rep_cur = str(it_doc.get("representative","") or "")
    start_d = _norm_date_local(it_doc.get("start_date")) or dt.date.today()
    end_d   = _norm_date_local(it_doc.get("end_date")) or start_d
    upl_d   = _norm_date_local(it_doc.get("upload_date")) or dt.date.today()

    final_default = int(
        ex_doc.get("final_package_cost", ex_doc.get("package_cost", it_doc.get("package_total", 0))) or 0
    )

    st.markdown("**Current selection**")
    cc = st.columns(4)
    with cc[0]: st.write({"ACH": ach_id})
    with cc[1]: st.write({"Client": nm})
    with cc[2]: st.write({"Mobile": mb})
    with cc[3]: st.write({"Current rep": rep_cur})

    st.markdown("### Edit fields")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        rep_new = st.selectbox(
            "Representative",
            ["Arpith","Reena","Kuldeep","Teena"],
            index=(["Arpith","Reena","Kuldeep","Teena"].index(rep_cur) if rep_cur in ["Arpith","Reena","Kuldeep","Teena"] else 0)
        )
    with c2:
        start_new = st.date_input("Travel start date", value=start_d)
    with c3:
        end_new = st.date_input("Travel end date", value=end_d)
    with c4:
        upload_new = st.date_input("Upload date (optional)", value=upl_d)

    c5,c6 = st.columns(2)
    with c5:
        final_amt = st.number_input("Final package amount (₹)", min_value=0, step=500, value=int(final_default or 0))
    with c6:
        next_status = st.radio("Next step", ["Confirm now", "Follow-up"], horizontal=True)

    it_text_file = st.file_uploader("(Optional) Upload itinerary .txt to store in DB", type=["txt"])
    note = st.text_area("Comment / internal note (optional)", "")

    # Extra inputs when confirming
    book_date = None; adv_amt = 0; utr = ""
    if next_status == "Confirm now":
        k1,k2,k3 = st.columns(3)
        with k1: book_date = st.date_input("Booking date", value=dt.date.today())
        with k2: adv_amt   = st.number_input("Advance (₹)", min_value=0, step=500)
        with k3: utr       = st.text_input("UTR / Payment reference", placeholder="UPI/NEFT ref")

    if st.button("💾 Save changes", use_container_width=True):
        try:
            # 1) Update core itinerary fields
            it_set = {
                "representative": rep_new,
                "start_date": str(start_new),
                "end_date": str(end_new),
            }
            if upload_new:
                it_set["upload_date"] = dt.datetime(
                    upload_new.year, upload_new.month, upload_new.day, 12, 0, 0, tzinfo=dt.timezone.utc
                )
            if final_amt and final_amt > 0:
                it_set["package_total"] = int(final_amt)
                it_set["package_cost"] = int(final_amt)
                # keep referral-adjusted field in sync if applicable
                ref_pct = int(it_doc.get("referral_discount_pct", 0) or 0)
                if ref_pct > 0:
                    it_set["package_after_referral"] = max(int(final_amt) - int(round(final_amt * ref_pct / 100.0)), 0)

            if it_text_file is not None:
                try:
                    it_text = it_text_file.read().decode("utf-8", errors="ignore")
                    it_set["itinerary_text"] = it_text
                except Exception:
                    st.warning("Could not read uploaded text; skipping itinerary_text update.")

            col_it.update_one({"_id": ObjectId(chosen_id)}, {"$set": it_set})

            # 2) Mirror amount to Expenses + Package Updates
            if final_amt and final_amt > 0:
                sync_initial_cost_to_expenses_and_updates(chosen_id, int(final_amt))

            # 3) Status path
            if next_status == "Follow-up":
                col_updates.update_one(
                    {"itinerary_id": str(chosen_id)},
                    {"$set": {
                        "status": "followup",
                        "assigned_to": rep_new,
                        "updated_at": dt.datetime.utcnow(),
                        "package_cost": int(final_amt or 0),
                        "final_package_cost": int(final_amt or 0),
                        "base_package_cost": int(final_amt or 0),
                        "discount": 0,
                    }},
                    upsert=True
                )
                col_followups.insert_one({
                    "itinerary_id": str(chosen_id),
                    "created_at": dt.datetime.utcnow(),
                    "created_by": user,
                    "status": "followup",
                    "comment": note or "Updated via Upload itinerary; set to follow-up",
                    "next_followup_on": None,
                })
                upsert_update_from_rep(str(chosen_id), rep_new, actor_user=user)

            else:
                # Confirm now
                if not book_date:
                    st.error("Please choose a booking date.")
                    st.stop()
                if adv_amt <= 0:
                    st.error("Advance must be > 0 to confirm.")
                    st.stop()
                if not (utr or "").strip():
                    st.error("UTR / payment reference is required.")
                    st.stop()

                bdt = dt.datetime.combine(book_date, dt.time(0,0))
                if final_amt and final_amt > 0:
                    sync_initial_cost_to_expenses_and_updates(chosen_id, int(final_amt))

                col_updates.update_one(
                    {"itinerary_id": str(chosen_id)},
                    {"$set": {
                        "status": "confirmed",
                        "booking_date": bdt,
                        "advance_amount": int(adv_amt),
                        "utr": str(utr).strip(),
                        "rep_name": rep_new,
                        "assigned_to": None,
                        "updated_at": dt.datetime.utcnow(),
                        "package_cost": int(final_amt or 0),
                        "final_package_cost": int(final_amt or 0),
                        "base_package_cost": int(final_amt or 0),
                        "discount": 0,
                    }},
                    upsert=True
                )
                col_followups.insert_one({
                    "itinerary_id": str(chosen_id),
                    "created_at": dt.datetime.utcnow(),
                    "created_by": user,
                    "status": "confirmed",
                    "comment": note or "Confirmed via Upload itinerary",
                    "next_followup_on": None,
                })

            st.success("✅ Saved. Changes will appear with other packages on all pages.")
        except Exception as e:
            st.error(f"Update failed: {e}")

# ============= Shared: show preview & download if available =============
if "last_preview_text" in st.session_state:
    text = str(st.session_state.get("last_preview_text", ""))
    meta = st.session_state.get("last_generated_meta", {}) or {}
    client = str(meta.get("client", "") or "")
    mobile = str(meta.get("mobile", "") or "")
    rev = str(meta.get("rev", "") or "")

    st.success(f"Generated for {client} ({mobile}) • rev {rev}")
    st.markdown("### Preview of generated itinerary")
    st.text_area("Copy from here", value=text, height=420, key="itinerary_preview")

    import re as _re
    def _slug(s: str) -> str:
        s = s.strip().replace(" ", "_")
        return _re.sub(r"[^A-Za-z0-9._-]+", "", s)

    today_str = str(dt.date.today())
    fname = f"itinerary_{_slug(client)}_{_slug(mobile)}_{today_str}.txt"
    st.download_button(
        label="⬇️ Download itinerary as .txt",
        data=text,
        file_name=fname,
        mime="text/plain",
        use_container_width=True,
    )
