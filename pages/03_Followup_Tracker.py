# pages/03_Followup_Tracker.py
from __future__ import annotations

from datetime import datetime, date, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import os
import io
import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# =========================
# App config
# =========================
st.set_page_config(page_title="Follow-up Tracker", layout="wide")
IST = ZoneInfo("Asia/Kolkata")

# --- Theme toggle (Dark / Normal) ---
with st.sidebar:
    dark_mode = st.toggle("🌙 Dark mode", value=False, help="Switch between dark and normal theme")
# Robust CSS for dark mode across widgets, tables, inputs & metrics
if dark_mode:
    st.markdown("""
        <style>
        :root{
          --bg:#0e1116; --fg:#e5e7eb; --muted:#a3a3a3; --card:#141821; --card2:#111520;
          --border:#2a2f3a; --accent:#22d3ee; --btn:#1f2937; --btnh:#374151;
        }
        html, body, [data-testid="stAppViewContainer"]{ background:var(--bg)!important; color:var(--fg)!important; }
        [data-testid="stHeader"]{ background:var(--bg)!important; }
        .stMarkdown, .stText, .stDataFrame, .stMetric, .stCaption, .st-emotion-cache { color:var(--fg)!important; }
        /* inputs & select */
        input, textarea, select, .stTextInput>div>div>input, .stNumberInput input{
           background:var(--card)!important; color:var(--fg)!important; border:1px solid var(--border)!important;
        }
        label, .stSelectbox label, .stCheckbox>label{ color:var(--fg)!important; }
        /* buttons */
        .stButton>button { background:var(--btn)!important; color:var(--fg)!important; 
           border-radius:10px; border:1px solid var(--border)!important; }
        .stButton>button:hover{ background:var(--btnh)!important; }
        /* dataframes */
        [data-testid="stDataFrame"] div, [data-testid="stDataFrame"] th{
            color:var(--fg)!important; background:var(--card2)!important; border-color:var(--border)!important;
        }
        /* expanders/cards */
        .st-af, .st-bc, .st-bb{ background:var(--card)!important; color:var(--fg)!important; }
        /* metrics */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"]{ color:var(--fg)!important; }
        a{ color:var(--accent)!important; }
        </style>
    """, unsafe_allow_html=True)

st.title("📞 Follow-up Tracker")

# =========================
# Incentive policy constants
# =========================
INCENTIVE_START_DATE: date = date(2025, 8, 1)  # Incentives only for bookings on/after 2025-08-01

def _eligible_for_incentive(booking_dt: Optional[datetime]) -> bool:
    """Return True only if booking_dt is present and on/after INCENTIVE_START_DATE."""
    if not booking_dt:
        return False
    try:
        return booking_dt.date() >= INCENTIVE_START_DATE
    except Exception:
        return False

# =========================
# Mongo helpers
# =========================
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
    uri = _find_uri()
    if not uri:
        st.error("Mongo URI missing. Add `mongo_uri` in Secrets.")
        st.stop()
    client = MongoClient(
        uri,
        appName="TAK_FollowupTracker",
        maxPoolSize=100,
        serverSelectionTimeoutMS=6000,
        connectTimeoutMS=6000,
        retryWrites=True,
        tz_aware=True,
    )
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        st.error(f"Could not connect MongoDB: {e}")
        st.stop()
    return client

@st.cache_resource
def get_db():
    return _get_client()["TAK_DB"]

db = get_db()
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_followups   = db["followups"]
col_expenses    = db["expenses"]

# =========================
# Users + login  (robust)
# =========================
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    # local fallback
    try:
        try:
            import tomllib
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, dict) and u:
            with st.sidebar:
                st.warning("Using users from repo .streamlit/secrets.toml (dev fallback).")
            return u
    except Exception:
        pass
    # env fallback (optional)
    try:
        import json
        raw = os.getenv("USERS_JSON")
        if raw:
            u = json.loads(raw)
            if isinstance(u, dict) and u:
                with st.sidebar:
                    st.warning("Using users from USERS_JSON env var.")
                return u
    except Exception:
        pass
    return {}

def _login() -> Optional[str]:
    with st.sidebar:
        if st.session_state.get("user"):
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out"):
                st.session_state.pop("user", None)
                st.rerun()

    if st.session_state.get("user"):
        return st.session_state["user"]

    users_map = load_users()
    if not users_map:
        st.error(
            "Login not configured.\n\n"
            "Add a `[users]` table in **Manage app → Secrets** or in `.streamlit/secrets.toml`.\n\n"
            "[users]\nArpith=\"Arpith&92\"\nReena=\"Reena&90\"\nTeena=\"Teena@123\"\nKuldeep=\"Kuldeep&96\""
        )
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
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid PIN"); st.stop()
    return None

user = _login()
if not user:
    st.stop()

USERS_MAP = load_users()
ALL_USERS = list(USERS_MAP.keys())
is_admin   = (str(user).strip().lower() == "arpith") or (user == "Kuldeep")
is_manager = (str(user).strip() == "Kuldeep")
can_reassign = is_admin or is_manager

# Audit (safe import)
try:
    from tak_audit import audit_pageview
    audit_pageview(st.session_state.get("user", "Unknown"), page="03_Followup_Tracker")
except Exception:
    pass

# =========================
# Utils
# =========================
def _to_int(x, default=0):
    try:
        if x is None: return default
        return int(float(str(x).replace(",", "")))
    except Exception:
        return default

def _clean_dt(x: object) -> Optional[datetime]:
    if x is None: return None
    try:
        ts = pd.to_datetime(x)
        if isinstance(ts, pd.Timestamp): ts = ts.to_pydatetime()
        return ts
    except Exception:
        return None

def _today_utc() -> date:
    return datetime.utcnow().date()

def month_bounds(d: date) -> Tuple[date, date]:
    first = d.replace(day=1)
    last = (first + pd.offsets.MonthEnd(1)).date()
    return first, last

def _fmt_ist(dt: datetime | None) -> str:
    if not dt: return ""
    try:
        return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M UTC")

def _oid_time(iid: str) -> Optional[datetime]:
    try:
        return ObjectId(str(iid)).generation_time
    except Exception:
        return None

def _get_itinerary(iid: str, projection: Optional[dict] = None) -> dict:
    projection = projection or {}
    doc = None
    try:
        doc = col_itineraries.find_one({"_id": ObjectId(iid)}, projection)
    except Exception:
        doc = col_itineraries.find_one({"itinerary_id": str(iid)}, projection)
    return doc or {}

# =========================
# Final cost logic + SYNC
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def _final_cost_map(ids: List[str]) -> Dict[str, int]:
    if not ids:
        return {}
    cur = list(col_expenses.find(
        {"itinerary_id": {"$in": [str(x) for x in ids]}},
        {"_id": 0, "itinerary_id": 1, "final_package_cost": 1, "base_package_cost": 1, "discount": 1, "package_cost": 1}
    ))
    out: Dict[str, int] = {}
    for d in cur:
        iid = str(d.get("itinerary_id"))
        f = _to_int(d.get("final_package_cost", 0))
        if f > 0:
            out[iid] = f
        else:
            base = _to_int(d.get("base_package_cost", d.get("package_cost", 0)))
            disc = _to_int(d.get("discount", 0))
            out[iid] = max(0, base - disc)
    return out

def _final_cost_for(iid: str) -> int:
    m = _final_cost_map([iid])
    return m.get(str(iid), 0)

def _compute_incentive(final_amt: int) -> int:
    if 5000 < final_amt < 20000: return 250
    if final_amt >= 20000: return 500
    return 0

def _sync_cost_to_updates(iid: str, final_cost: int, base: int, disc: int) -> None:
    col_updates.update_one(
        {"itinerary_id": str(iid)},
        {"$set": {
            "package_cost": int(final_cost),
            "final_package_cost": int(final_cost),
            "base_package_cost": int(base),
            "discount": int(disc),
            "updated_at": datetime.utcnow(),
        }},
        upsert=True,
    )

# =========================
# Assignment heal from itinerary.representative
# =========================
def _ensure_assignment_from_rep(iid: str, actor_user: str):
    it = _get_itinerary(iid, {"representative": 1, "client_name":1, "client_mobile":1, "ach_id":1})
    rep = (it.get("representative") or "").strip()
    if not rep:
        return
    upd = col_updates.find_one({"itinerary_id": str(iid)}, {"_id":1, "assigned_to":1, "status":1})
    if not upd:
        col_updates.update_one(
            {"itinerary_id": str(iid)},
            {"$set": {"status": "followup", "assigned_to": rep, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        col_followups.insert_one({
            "itinerary_id": str(iid),
            "created_at": datetime.utcnow(),
            "created_by": actor_user,
            "status": "followup",
            "comment": f"Auto-assigned from itinerary representative {rep}",
            "credited_to": rep,
            "client_name": it.get("client_name",""),
            "client_mobile": it.get("client_mobile",""),
            "ach_id": it.get("ach_id",""),
        })
    else:
        if upd.get("assigned_to") != rep and upd.get("status") != "confirmed":
            col_updates.update_one(
                {"itinerary_id": str(iid)},
                {"$set": {"assigned_to": rep, "updated_at": datetime.utcnow()}}
            )

# ======== Auto-confirm other packages (same mobile) ========
def _auto_confirm_other_packages(current_iid: str, credit_user: str, booking_date: Optional[date], actor_user: str):
    base_doc = _get_itinerary(current_iid, {"client_mobile":1, "client_name":1, "ach_id":1})
    mobile = str(base_doc.get("client_mobile","")).strip()
    if not mobile:
        return

    others = list(col_itineraries.find(
        {"client_mobile": mobile},
        {"_id":1, "ach_id":1, "client_name":1, "client_mobile":1}
    ))
    other_ids = [str(o["_id"]) for o in others if str(o["_id"]) != str(current_iid)]
    if not other_ids:
        return

    existing_upds = list(col_updates.find({"itinerary_id": {"$in": other_ids}}, {"itinerary_id":1, "status":1}))
    status_map = {str(u.get("itinerary_id")): u.get("status") for u in existing_upds}

    for oid in other_ids:
        if status_map.get(oid) in ("confirmed", "cancelled"):
            continue

        fc = _final_cost_for(oid)
        exp_doc = col_expenses.find_one({"itinerary_id": str(oid)},
                                        {"base_package_cost":1,"discount":1,"final_package_cost":1,"package_cost":1}) or {}
        base_amt = _to_int(exp_doc.get("base_package_cost", 0))
        disc_amt = _to_int(exp_doc.get("discount", 0))
        if fc <= 0:
            fc = max(0, base_amt - disc_amt)

        base_it = _get_itinerary(oid, {"client_name":1,"client_mobile":1,"ach_id":1})
        col_followups.insert_one({
            "itinerary_id": str(oid),
            "created_at": datetime.utcnow(),
            "created_by": actor_user,
            "status": "confirmed",
            "comment": f"Auto-confirmed due to confirmation of another package for same mobile {mobile}.",
            "next_followup_on": None,
            "cancellation_reason": "",
            "credited_to": credit_user,
            "client_name": base_it.get("client_name",""),
            "client_mobile": base_it.get("client_mobile",""),
            "ach_id": base_it.get("ach_id",""),
        })

        bdt = datetime.combine(booking_date, dtime.min) if booking_date else None
        inc_val = _compute_incentive(fc) if _eligible_for_incentive(bdt) else 0

        upd = {
            "status": "confirmed",
            "booking_date": bdt,
            "advance_amount": 0,
            "incentive": int(inc_val),
            "rep_name": credit_user,
            "assigned_to": None,
            "package_cost": int(fc),
            "final_package_cost": int(fc),
            "base_package_cost": int(base_amt),
            "discount": int(disc_amt),
            "updated_at": datetime.utcnow(),
        }
        col_updates.update_one({"itinerary_id": str(oid)}, {"$set": upd}, upsert=True)

# =========================
# Cached fetchers (existing)
# =========================
@st.cache_data(ttl=45, show_spinner=False)
def fetch_assigned_followups_raw(assigned_to: Optional[str] = None) -> pd.DataFrame:
    q = {"status": "followup"}
    if assigned_to is not None:
        q["assigned_to"] = assigned_to
    ups = list(col_updates.find(q, {"_id":0, "itinerary_id":1, "assigned_to":1, "status":1}))
    if not ups:
        return pd.DataFrame(columns=["itinerary_id","assigned_to","status"])
    df_u = pd.DataFrame(ups)
    df_u["itinerary_id"] = df_u["itinerary_id"].astype(str)

    its = list(col_itineraries.find({}, {
        "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
        "start_date": 1, "end_date": 1, "final_route": 1, "total_pax": 1, "representative": 1
    }))
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        r["_created_utc"] = _oid_time(r["itinerary_id"])
        for k in ("start_date","end_date"):
            try:
                r[k] = pd.to_datetime(r.get(k)).date()
            except Exception:
                r[k] = None
    df_i = pd.DataFrame(its).drop(columns=["_id"]) if its else pd.DataFrame()
    return df_u.merge(df_i, on="itinerary_id", how="left")

def _dedupe_latest_by_mobile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df2 = df.copy()
    df2["client_mobile"] = df2["client_mobile"].astype(str).fillna("")
    df2["_created_utc"] = df2["_created_utc"].apply(lambda x: x or datetime.min)
    df2.sort_values(["client_mobile", "_created_utc"], ascending=[True, False], inplace=True)
    latest = df2.groupby("client_mobile", as_index=False).first()
    return latest

@st.cache_data(ttl=45, show_spinner=False)
def fetch_latest_followup_log_map(itinerary_ids: List[str]) -> Dict[str, dict]:
    if not itinerary_ids:
        return {}
    cur = col_followups.find({"itinerary_id": {"$in": itinerary_ids}}, {"_id":0})
    latest: Dict[str, dict] = {}
    for d in cur:
        iid = str(d.get("itinerary_id"))
        ts = _clean_dt(d.get("created_at")) or datetime.min
        if iid not in latest or ts > latest[iid].get("_ts", datetime.min):
            d["_ts"] = ts
            latest[iid] = d
    return latest

# =========================
# NEW: Unified updates+itineraries with client key
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_updates_joined() -> pd.DataFrame:
    ups = list(col_updates.find({}, {"_id":0}))
    df_u = pd.DataFrame(ups) if ups else pd.DataFrame()
    its = list(col_itineraries.find({}, {
        "_id":1,"ach_id":1,"client_name":1,"client_mobile":1,"final_route":1,
        "start_date":1,"end_date":1
    }))
    df_i = pd.DataFrame([{
        "itinerary_id": str(r["_id"]),
        "ach_id": r.get("ach_id",""),
        "client_name": r.get("client_name",""),
        "client_mobile": r.get("client_mobile",""),
        "final_route": r.get("final_route",""),
        "start_date": pd.to_datetime(r.get("start_date")).date() if r.get("start_date") else None,
        "end_date": pd.to_datetime(r.get("end_date")).date() if r.get("end_date") else None,
        "_created_utc": ObjectId(str(r["_id"])).generation_time if ObjectId.is_valid(str(r["_id"])) else None
    } for r in its])
    if df_u.empty and df_i.empty:
        return pd.DataFrame()
    df = df_i.merge(df_u, on="itinerary_id", how="left")
    # client key
    def _ck(row):
        mob = str(row.get("client_mobile") or "").strip()
        if mob: return f"M:{mob}"
        ach = str(row.get("ach_id") or "").strip()
        nam = str(row.get("client_name") or "").strip()
        return f"A:{ach}|N:{nam}"
    df["_client_key"] = df.apply(_ck, axis=1)
    _bk = pd.to_datetime(df.get("booking_date"), errors="coerce", utc=True)
    _cr = pd.to_datetime(df.get("_created_utc"), errors="coerce", utc=True)
    df["_booking"] = _bk.dt.tz_convert(None)
    df["_created"] = _cr.dt.tz_convert(None)
    return df

def latest_per_client(df: pd.DataFrame, user_filter: Optional[str]=None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if user_filter:
        df = df[(df["assigned_to"] == user_filter) | (df["rep_name"] == user_filter)]
    # Sort by booking desc, then created desc, keep first per client
    df = df.sort_values(["_client_key","_booking","_created"], ascending=[True, False, False])
    latest = df.groupby("_client_key", as_index=False).first()
    return latest

def count_unique_statuses(df_latest: pd.DataFrame) -> Dict[str,int]:
    if df_latest.empty:
        return {"confirmed":0,"pending":0,"under_discussion":0,"followup":0,"cancelled":0}
    s = df_latest["status"].fillna("")
    return {
        "confirmed": int((s=="confirmed").sum()),
        "pending": int((s=="pending").sum()),
        "under_discussion": int((s=="under_discussion").sum()),
        "followup": int((s=="followup").sum()),
        "cancelled": int((s=="cancelled").sum()),
    }

def count_confirmed_unique_range(user_filter: Optional[str], start_d: date, end_d: date) -> int:
    df = fetch_updates_joined()
    df_l = latest_per_client(df, user_filter)
    if df_l.empty:
        return 0
    mask = (df_l["status"]=="confirmed") & df_l["_booking"].notna()
    mask &= (df_l["_booking"] >= pd.Timestamp(datetime.combine(start_d, dtime.min))) & \
            (df_l["_booking"] <= pd.Timestamp(datetime.combine(end_d, dtime.max)))
    return int(mask.sum())

# =========================
# Incentives fetchers (month-wise & per-customer)
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_confirmed_incentives(user_filter: Optional[str], start_d: date, end_d: date) -> int:
    """Sum incentives only for bookings within [start_d, end_d] and on/after INCENTIVE_START_DATE."""
    start_window = max(start_d, INCENTIVE_START_DATE)
    if start_window > end_d:
        return 0
    q = {
        "status": "confirmed",
        "booking_date": {"$gte": datetime.combine(start_window, dtime.min),
                         "$lte": datetime.combine(end_d, dtime.max)}
    }
    if user_filter:
        q["rep_name"] = user_filter
    cur = col_updates.find(q, {"_id":0, "incentive":1})
    return sum(_to_int(d.get("incentive", 0)) for d in cur)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_user_months_with_totals(rep_name: str) -> pd.DataFrame:
    q = {
        "status": "confirmed",
        "rep_name": rep_name,
        "booking_date": {"$gte": datetime.combine(INCENTIVE_START_DATE, dtime.min)}
    }
    cur = list(col_updates.find(q, {"_id":0, "booking_date":1, "incentive":1}))
    if not cur:
        return pd.DataFrame(columns=["Month", "Total Incentive (₹)"])
    df = pd.DataFrame(cur)
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df["Month"] = df["booking_date"].dt.strftime("%Y-%m")
    df["incentive"] = df["incentive"].apply(_to_int)
    out = df.groupby("Month")["incentive"].sum().reset_index().rename(columns={"incentive": "Total Incentive (₹)"})
    out.sort_values("Month", inplace=True)
    return out

@st.cache_data(ttl=60, show_spinner=False)
def fetch_user_customer_incentives_for_month(rep_name: str, month_start: date, month_end: date) -> pd.DataFrame:
    start_window = max(month_start, INCENTIVE_START_DATE)
    if start_window > month_end:
        return pd.DataFrame(columns=[
            "ACH ID","Client","Mobile","Route","Booking date","Final package (₹)","Incentive (₹)"
        ])

    q = {
        "status": "confirmed",
        "rep_name": rep_name,
        "booking_date": {"$gte": datetime.combine(start_window, dtime.min),
                         "$lte": datetime.combine(month_end, dtime.max)}
    }
    rows = list(col_updates.find(q, {"_id":0, "itinerary_id":1, "booking_date":1, "incentive":1, "final_package_cost":1}))
    if not rows:
        return pd.DataFrame(columns=[
            "ACH ID","Client","Mobile","Route","Booking date","Final package (₹)","Incentive (₹)"
        ])
    df_u = pd.DataFrame(rows)
    df_u["itinerary_id"] = df_u["itinerary_id"].astype(str)

    its = list(col_itineraries.find(
        {"_id": {"$in": [ObjectId(x) for x in df_u["itinerary_id"].unique() if ObjectId.is_valid(x)]}},
        {"_id":1, "ach_id":1, "client_name":1, "client_mobile":1, "final_route":1}
    ))
    df_i = pd.DataFrame([{
        "itinerary_id": str(i["_id"]),
        "ACH ID": i.get("ach_id",""),
        "Client": i.get("client_name",""),
        "Mobile": i.get("client_mobile",""),
        "Route": i.get("final_route",""),
    } for i in its])

    df_u["Booking date"] = pd.to_datetime(df_u["booking_date"]).dt.date
    df_u["Incentive (₹)"] = df_u["incentive"].apply(_to_int)
    df_u["Final package (₹)"] = df_u["final_package_cost"].apply(_to_int)
    view = df_u.merge(df_i, on="itinerary_id", how="left")[
        ["ACH ID","Client","Mobile","Route","Booking date","Final package (₹)","Incentive (₹)","itinerary_id"]
    ].sort_values(["Booking date","Client"])
    return view

# =========================
# Reassign + updaters + booking-date editor
# =========================
def _latest_next_followup_date(iid: str) -> Optional[datetime]:
    d = col_followups.find_one({"itinerary_id": str(iid)}, sort=[("created_at", -1)], projection={"next_followup_on": 1})
    return d.get("next_followup_on") if d else None

def reassign_followup(iid: str, from_user: str, to_user: str) -> None:
    next_dt = _latest_next_followup_date(iid)
    base = _get_itinerary(iid, {"client_name":1,"client_mobile":1,"ach_id":1})
    col_followups.insert_one({
        "itinerary_id": str(iid),
        "created_at": datetime.utcnow(),
        "created_by": from_user,
        "status": "followup",
        "comment": f"Reassigned from {from_user} to {to_user}",
        "next_followup_on": next_dt,
        "client_name": base.get("client_name",""),
        "client_mobile": base.get("client_mobile",""),
        "ach_id": base.get("ach_id",""),
    })
    col_updates.update_one(
        {"itinerary_id": str(iid)},
        {"$set": {"status": "followup", "assigned_to": to_user, "updated_at": datetime.utcnow()}},
        upsert=True
    )

def upsert_update_status(
    iid: str, status: str, actor_user: str, credit_user: str,
    next_followup_on: Optional[date], booking_date: Optional[date],
    comment: str, cancellation_reason: Optional[str], advance_amount: Optional[int],
) -> None:
    base = _get_itinerary(iid, {"client_name":1,"client_mobile":1,"ach_id":1})
    col_followups.insert_one({
        "itinerary_id": str(iid),
        "created_at": datetime.utcnow(),
        "created_by": actor_user,
        "status": status,
        "comment": str(comment or ""),
        "next_followup_on": (datetime.combine(next_followup_on, dtime.min) if next_followup_on else None),
        "cancellation_reason": (str(cancellation_reason or "") if status == "cancelled" else ""),
        "credited_to": credit_user,
        "client_name": base.get("client_name",""),
        "client_mobile": base.get("client_mobile",""),
        "ach_id": base.get("ach_id",""),
    })

    final_status = status if status in ("followup","cancelled") else "confirmed"
    upd = {"itinerary_id": str(iid), "status": final_status, "updated_at": datetime.utcnow()}

    if final_status == "followup":
        upd["assigned_to"] = credit_user

    elif final_status == "confirmed":
        bdt = datetime.combine(booking_date, dtime.min) if booking_date else None
        if bdt: upd["booking_date"] = bdt
        if advance_amount is not None: upd["advance_amount"] = int(advance_amount)

        fc = _final_cost_for(iid)
        exp_doc = col_expenses.find_one({"itinerary_id": str(iid)},
                                        {"base_package_cost":1,"discount":1,"final_package_cost":1,"package_cost":1}) or {}
        base_amt = _to_int(exp_doc.get("base_package_cost", 0))
        disc_amt = _to_int(exp_doc.get("discount", 0))
        if fc <= 0:
            fc = max(0, base_amt - disc_amt)

        inc_val = _compute_incentive(fc) if _eligible_for_incentive(bdt) else 0

        upd.update({
            "incentive": int(inc_val),
            "rep_name": credit_user,
            "assigned_to": None,
            "package_cost": int(fc),
            "final_package_cost": int(fc),
            "base_package_cost": int(base_amt),
            "discount": int(disc_amt),
        })

        _auto_confirm_other_packages(current_iid=str(iid),
                                     credit_user=credit_user,
                                     booking_date=(bdt.date() if bdt else None),
                                     actor_user=actor_user)

    elif final_status == "cancelled":
        upd["cancellation_reason"] = str(cancellation_reason or "")
        upd["assigned_to"] = None

    col_updates.update_one({"itinerary_id": str(iid)}, {"$set": upd}, upsert=True)

def save_final_package_cost(iid: str, base_amount: int, discount: int, actor_user: str, credit_user: Optional[str]=None) -> None:
    base = _to_int(base_amount); disc = _to_int(discount)
    final_cost = max(0, base - disc)
    col_expenses.update_one(
        {"itinerary_id": str(iid)},
        {"$set": {
            "itinerary_id": str(iid),
            "base_package_cost": int(base),
            "discount": int(disc),
            "final_package_cost": int(final_cost),
            "package_cost": int(final_cost),
            "saved_at": datetime.utcnow(),
        }},
        upsert=True
    )
    _sync_cost_to_updates(iid=str(iid), final_cost=final_cost, base=base, disc=disc)

    upd = col_updates.find_one({"itinerary_id": str(iid)}, {"status":1, "rep_name":1, "booking_date":1})
    if upd and upd.get("status") == "confirmed":
        bdt = _clean_dt(upd.get("booking_date"))
        inc = _compute_incentive(int(final_cost)) if _eligible_for_incentive(bdt) else 0
        rep = credit_user or upd.get("rep_name") or actor_user
        col_updates.update_one(
            {"itinerary_id": str(iid)},
            {"$set": {"incentive": int(inc), "rep_name": rep, "updated_at": datetime.utcnow()}}
        )

def batch_update_booking_dates(rows: List[dict], actor_user: str) -> int:
    """
    rows: list of {itinerary_id, booking_date, final_package_cost(optional)} from editable grid.
    Recompute incentives per policy after date change.
    """
    updated = 0
    for r in rows:
        iid = str(r.get("itinerary_id","")).strip()
        if not iid: continue
        bdt = _clean_dt(r.get("booking_date"))
        # pull final cost (from updates or expenses)
        upd = col_updates.find_one({"itinerary_id": iid}, {"final_package_cost":1, "rep_name":1, "status":1})
        if not upd or upd.get("status") != "confirmed":
            continue
        fc = _to_int((r.get("final_package_cost") if r.get("final_package_cost") is not None else upd.get("final_package_cost", 0)))
        if fc <= 0:
            fc = _final_cost_for(iid)
        inc = _compute_incentive(fc) if _eligible_for_incentive(bdt) else 0
        col_updates.update_one(
            {"itinerary_id": iid},
            {"$set": {"booking_date": bdt, "incentive": int(inc), "updated_at": datetime.utcnow()}}
        )
        updated += 1
    return updated

# =============================================================================
# Sidebar performance controls
# =============================================================================
with st.sidebar:
    st.markdown("---")
    defer_loads = st.toggle("⚡ Defer heavy loads", value=False,
                            help="Skip loading big tables until you press Refresh or after a Save.")
    refresh_now = st.button("🔄 Refresh data now", use_container_width=True)
if refresh_now:
    st.session_state["force_refresh"] = True

def _defer_guard(msg: str) -> bool:
    if defer_loads and not st.session_state.get("force_refresh", False):
        st.info(f"Deferred: {msg}\n\nClick **🔄 Refresh data now** (sidebar) to load.")
        return True
    return False

def _clear_force_refresh():
    if st.session_state.get("force_refresh"):
        st.session_state.pop("force_refresh", None)

# =============================================================================
# UI
# =============================================================================
with st.sidebar:
    if is_admin:
        view_user = st.selectbox("Filter follow-ups by user", ["All users"] + ALL_USERS, index=0)
        user_filter = None if view_user == "All users" else view_user
        st.caption("Admin mode: view all or filter by user. Record updates on behalf of a user.")
    else:
        view_user = user
        user_filter = user
        st.caption("User mode: viewing your assigned follow-ups.")

# Admin/Manager quick overview
if is_admin or is_manager:
    if not _defer_guard("Follow-ups by assignee"):
        # Use the existing cached fetcher defined above
        df_over = fetch_assigned_followups_raw(assigned_to=None)

        if not df_over.empty:
            # Build a unique-by-client key and count latest per client per assignee
            df_over["_ck"] = df_over["client_mobile"].fillna("").astype(str)
            df_over["_created_utc"] = pd.to_datetime(df_over.get("_created_utc"))
            df_over = df_over.sort_values(
                ["assigned_to", "_ck", "_created_utc"],
                ascending=[True, True, False]
            )
            # keep only latest row per client key per assignee
            uniq = df_over.groupby(["assigned_to", "_ck"], as_index=False).first()
            agg = (
                uniq.groupby("assigned_to")["itinerary_id"]
                .nunique()
                .rename("Follow-ups")
                .reset_index()
                .sort_values("Follow-ups", ascending=False)
            )

            st.markdown("### 👥 Follow-ups by assignee")
            st.dataframe(
                agg.rename(columns={"assigned_to": "User"}),
                use_container_width=True,
                hide_index=True
            )

    # Admin tools
    with st.expander("🧰 Admin tools"):
        if st.button("Resync assignments from representatives (recent 500)"):
            try:
                recent_its = list(
                    col_itineraries.find({}, {"_id": 1}).sort([("_id", -1)]).limit(500)
                )
                fixed = 0
                for r in recent_its:
                    iid = str(r["_id"])
                    _ensure_assignment_from_rep(iid, actor_user=user)
                    fixed += 1
                fetch_assigned_followups_raw.clear()
                st.success(f"Resynced assignment on ~{fixed} itineraries.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to resync: {e}")

    st.divider()


tabs = st.tabs(["🗂️ Follow-ups", "📘 All packages", "💰 Incentives"])

# =========================
# TAB 1: Follow-ups (dedup by mobile)
# =========================
with tabs[0]:
    if _defer_guard("Follow-ups list"):
        _clear_force_refresh()
        st.stop()

    df_raw = fetch_assigned_followups_raw(user_filter)
    df_assigned = _dedupe_latest_by_mobile(df_raw)

    if not df_assigned.empty:
        q = st.text_input("🔎 Search (name / mobile / ACH / route)", "")
        if q.strip():
            s = q.strip().lower()
            df_assigned = df_assigned[
                df_assigned["client_name"].astype(str).str.lower().str.contains(s) |
                df_assigned["client_mobile"].astype(str).str.lower().str.contains(s) |
                df_assigned["ach_id"].astype(str).str.lower().str.contains(s) |
                df_assigned["final_route"].astype(str).str.lower().str.contains(s)
            ]

    if df_assigned.empty:
        st.info("No follow-ups found for the selected filter.")
        _clear_force_refresh()
        st.stop()

    itinerary_ids = df_assigned["itinerary_id"].astype(str).tolist()
    latest_map = fetch_latest_followup_log_map(itinerary_ids)
    fc_map = _final_cost_map(itinerary_ids)

    df_assigned["next_followup_on"] = df_assigned["itinerary_id"].map(
        lambda x: (latest_map.get(str(x), {}) or {}).get("next_followup_on")
    ).apply(lambda x: pd.to_datetime(x).date() if pd.notna(x) else None)
    df_assigned["last_comment"] = df_assigned["itinerary_id"].map(
        lambda x: (latest_map.get(str(x), {}) or {}).get("comment", "")
    )
    df_assigned["final_cost"] = df_assigned["itinerary_id"].map(lambda x: fc_map.get(str(x), 0))

    # KPIs (unique-based)
    today = _today_utc()
    tmr = today + timedelta(days=1)
    in7 = today + timedelta(days=7)
    total_pkgs = len(df_assigned)
    due_today = int((df_assigned["next_followup_on"] == today).sum())
    due_tomorrow = int((df_assigned["next_followup_on"] == tmr).sum())
    due_week = int(((df_assigned["next_followup_on"] >= today) & (df_assigned["next_followup_on"] <= in7)).sum())

    first_this, last_this = month_bounds(today)
    first_last, last_last = month_bounds(first_this - timedelta(days=1))

    # Unique-confirmed counts (booking-date based)
    confirmed_this_month  = count_confirmed_unique_range(user_filter, first_this, last_this)
    confirmed_all_time    = count_confirmed_unique_range(user_filter, date(1970,1,1), date(2999,12,31))

    # Incentive sums (policy already enforced inside)
    this_month_incentive = fetch_confirmed_incentives(user_filter, first_this, last_this)
    last_month_incentive  = fetch_confirmed_incentives(user_filter, first_last, last_last)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Follow-ups (unique mobiles)", total_pkgs)
    c2.metric("Due today", due_today)
    c3.metric("Due tomorrow", due_tomorrow)
    c4.metric("Due in next 7 days", due_week)
    label_inc = "My incentive" if user_filter else "Incentive (all users)"
    c5.metric(label_inc, f"₹ {this_month_incentive:,}", help=f"Last month: ₹ {last_month_incentive:,}")
    c6, c7 = st.columns(2)
    c6.metric("Confirmed this month (unique)", confirmed_this_month)
    c7.metric("Confirmed (all time, unique)", confirmed_all_time)

    st.divider()
    st.subheader("Follow-ups list (unique by mobile)")

    table = df_assigned[[
        "ach_id","client_name","client_mobile","start_date","end_date",
        "final_route","assigned_to","next_followup_on","last_comment","final_cost","itinerary_id"
    ]].copy().sort_values(["next_followup_on","start_date"], na_position="last")
    table.rename(columns={
        "ach_id":"ACH ID","client_name":"Client","client_mobile":"Mobile",
        "start_date":"Start","end_date":"End","final_route":"Route","assigned_to":"Assigned to",
        "next_followup_on":"Next follow-up","last_comment":"Last comment","final_cost":"Final package cost (₹)"
    }, inplace=True)

    left, right = st.columns([2,1])
    with left:
        st.dataframe(table.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)
        if st.button("⬇️ Export current list (CSV)"):
            out = io.StringIO()
            table.drop(columns=["itinerary_id"]).to_csv(out, index=False)
            st.download_button("Download CSV", data=out.getvalue().encode("utf-8"),
                               file_name=f"followups_{today}.csv", mime="text/csv",
                               use_container_width=True)

    with right:
        options = (table["ACH ID"].fillna("").astype(str) + " | " +
                   table["Client"].fillna("") + " | " +
                   table["Mobile"].fillna("") + " | " +
                   table["itinerary_id"])
        sel = st.selectbox("Open client", options.tolist())
        chosen_id = sel.split(" | ")[-1] if sel else None

    if not chosen_id:
        _clear_force_refresh()
        st.stop()

    # Auto-heal assignment mismatch from itinerary.representative
    try:
        _ensure_assignment_from_rep(chosen_id, actor_user=user)
        st.caption("✅ Assignment checked against itinerary representative.")
    except Exception:
        pass

    st.divider()
    st.subheader("Details & Update")

    it_doc = _get_itinerary(chosen_id, {"ach_id":1,"client_name":1,"client_mobile":1,"final_route":1,"total_pax":1,
                                        "start_date":1,"end_date":1,"representative":1,"itinerary_text":1}) or {}
    upd_doc = col_updates.find_one({"itinerary_id": str(chosen_id)}, {"_id":0}) or {}

    created_dt_utc = _oid_time(chosen_id)
    created_ist_str = _fmt_ist(created_dt_utc)
    created_utc_str = created_dt_utc.strftime("%Y-%m-%d %H:%M %Z") if created_dt_utc else ""

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**Client & Package**")
        st.write({
            "ACH ID": it_doc.get("ach_id",""),
            "Client": it_doc.get("client_name",""),
            "Mobile": it_doc.get("client_mobile",""),
            "Route": it_doc.get("final_route",""),
            "Pax": it_doc.get("total_pax",""),
            "Travel": f"{it_doc.get('start_date','')} → {it_doc.get('end_date','')}",
            "Representative": it_doc.get("representative",""),
            "Created (IST)": created_ist_str,
            "Created (UTC)": created_utc_str,
        })
    with dc2:
        st.markdown("**Current Status**")
        st.write({
            "Status": upd_doc.get("status",""),
            "Assigned To": upd_doc.get("assigned_to",""),
            "Booking date": upd_doc.get("booking_date",""),
            "Advance (₹)": upd_doc.get("advance_amount",0),
            "Incentive (₹)": upd_doc.get("incentive",0),
            "Rep (credited to)": upd_doc.get("rep_name",""),
            "Final package cost (₹)": _final_cost_for(chosen_id),
        })

    # Reassign (single)
    if can_reassign:
        st.markdown("### Reassign this follow-up")
        current_assignee = (upd_doc or {}).get("assigned_to", "")
        candidates = [u for u in ALL_USERS if u != current_assignee] or ALL_USERS
        to_user = st.selectbox("Move to user", candidates, key="reassign_to")
        if st.button("➡️ Reassign now"):
            try:
                reassign_followup(chosen_id, from_user=user, to_user=to_user)
                st.success(f"Moved to {to_user}.")
                fetch_assigned_followups_raw.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Could not reassign: {e}")

    # Final package cost
    st.markdown("### Final package cost")
    exp_doc = col_expenses.find_one(
        {"itinerary_id": str(chosen_id)},
        {"base_package_cost":1, "discount":1, "final_package_cost":1, "package_cost":1}
    ) or {}
    base_default = _to_int(exp_doc.get("base_package_cost", 0)) or _to_int(it_doc.get("package_cost", 0))
    disc_default = _to_int(exp_doc.get("discount", 0))

    if is_admin or is_manager:
        credit_for_cost = st.selectbox("Credit incentive to (on confirm)", ["(keep existing)"] + ALL_USERS)
        credit_user_cost = None if credit_for_cost == "(keep existing)" else credit_for_cost
    else:
        credit_user_cost = None

    c1c, c2c, c3c = st.columns(3)
    with c1c:
        base_amount = st.number_input("Quoted/Initial amount (₹)", min_value=0, step=500, value=int(base_default))
    with c2c:
        discount = st.number_input("Discount (₹)", min_value=0, step=500, value=int(disc_default))
    with c3c:
        st.metric("Final cost (auto)", f"₹ {max(0, int(base_amount) - int(discount)):,}")

    if st.button("💾 Save final package cost"):
        try:
            save_final_package_cost(chosen_id, int(base_amount), int(discount),
                                    actor_user=user, credit_user=credit_user_cost)
            st.success("Final package cost saved. Synced to updates; incentive updated if already confirmed.")
            _final_cost_map.clear()
            fetch_assigned_followups_raw.clear()
            st.session_state["force_refresh"] = True
            st.rerun()
        except Exception as e:
            st.error(f"Could not save package cost: {e}")

    with st.expander("Show full itinerary text"):
        st.text_area("Itinerary shared with client",
                     value=it_doc.get("itinerary_text",""), height=260, disabled=True)

    st.markdown("### Follow-up trail")
    trail = list(col_followups.find({"itinerary_id": str(chosen_id)}).sort("created_at", -1))
    if trail:
        df_trail = pd.DataFrame([{
            "When (UTC)": t.get("created_at"),
            "When (IST)": _fmt_ist(_clean_dt(t.get("created_at"))),
            "By": t.get("created_by"),
            "Credited to": t.get("credited_to", ""),
            "Status": t.get("status"),
            "Next follow-up": (pd.to_datetime(t.get("next_followup_on")).date() if t.get("next_followup_on") else None),
            "Comment": t.get("comment",""),
            "Cancel reason": t.get("cancellation_reason",""),
        } for t in trail])
        st.dataframe(df_trail, use_container_width=True, hide_index=True)
        if st.button("⬇️ Export trail (CSV)"):
            out2 = io.StringIO()
            df_trail.to_csv(out2, index=False)
            st.download_button("Download CSV", data=out2.getvalue().encode("utf-8"),
                               file_name=f"followup_trail_{chosen_id}.csv", mime="text/csv",
                               use_container_width=True)
    else:
        st.caption("No follow-up logs yet for this client.")

    st.markdown("---")
    st.markdown("### Add follow-up update")

    if is_admin and ALL_USERS:
        default_idx = ALL_USERS.index(upd_doc.get("assigned_to")) if upd_doc.get("assigned_to") in ALL_USERS else 0
        record_as = st.selectbox("Record this update on behalf of", ALL_USERS, index=default_idx)
    else:
        record_as = user

    with st.form("followup_form"):
        status_choice = st.selectbox("Status", ["followup required", "confirmed", "cancelled"])
        comment = st.text_area("Comment", placeholder="Write your update…")
        next_date = None; cancel_reason = None; booking_date = None; advance_amt = None
        if status_choice == "followup required":
            next_date = st.date_input("Next follow-up on")
        elif status_choice == "confirmed":
            booking_date = st.date_input("Booking date")
            advance_amt = st.number_input("Advance amount (₹) — optional", min_value=0, step=500, value=0)
        elif status_choice == "cancelled":
            cancel_reason = st.text_input("Reason for cancellation", placeholder="Required")
        submitted = st.form_submit_button("💾 Save update")

    if submitted:
        if status_choice == "followup required" and not next_date:
            st.error("Please choose the next follow-up date."); st.stop()
        if status_choice == "cancelled" and not (cancel_reason or "").strip():
            st.error("Please provide a reason for cancellation."); st.stop()
        if status_choice == "confirmed" and not booking_date:
            st.error("Please choose the booking date."); st.stop()

        upsert_update_status(
            iid=chosen_id,
            status=("followup" if status_choice == "followup required" else status_choice),
            actor_user=user,
            credit_user=record_as,
            next_followup_on=next_date,
            booking_date=booking_date,
            comment=comment,
            cancellation_reason=cancel_reason,
            advance_amount=int(advance_amt) if advance_amt is not None else None
        )
        st.success("Update saved.")
        fetch_assigned_followups_raw.clear()
        fetch_latest_followup_log_map.clear()
        st.session_state["force_refresh"] = True
        st.rerun()

    _clear_force_refresh()

# =========================
# TAB 2: All packages (overview, unique client last revision)
# =========================
with tabs[1]:
    if is_admin:
        filter_user_all = st.selectbox("Filter packages by user (assigned/credited)", ["All users"] + ALL_USERS, index=0)
        fu = None if filter_user_all == "All users" else filter_user_all
    else:
        st.caption("Showing packages related to you (assigned to you or credited to you).")
        fu = user

    if _defer_guard("All packages table"):
        _clear_force_refresh()
    else:
        df_join = fetch_updates_joined()
        df_latest = latest_per_client(df_join, fu)
        if df_latest.empty:
            st.info("No packages to display for the selected filter.")
        else:
            # KPIs (unique-based)
            counts = count_unique_statuses(df_latest)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("✅ Confirmed (unique)", counts["confirmed"])
            m2.metric("🟡 Pending (unique)", counts["pending"])
            m3.metric("🟠 Under discussion (unique)", counts["under_discussion"])
            m4.metric("🔵 Follow-up (unique)", counts["followup"])
            m5.metric("🔴 Cancelled (unique)", counts["cancelled"])

            # Attach final cost
            id_list = df_latest["itinerary_id"].astype(str).tolist()
            fc_map_all = _final_cost_map(id_list)
            df_latest["final_cost"] = df_latest["itinerary_id"].map(lambda x: fc_map_all.get(str(x), 0))

            q2 = st.text_input("Search in packages (name / mobile / ACH / route)", "")
            if q2.strip():
                s2 = q2.strip().lower()
                df_latest = df_latest[
                    df_latest["client_name"].astype(str).str.lower().str.contains(s2) |
                    df_latest["client_mobile"].astype(str).str.lower().str.contains(s2) |
                    df_latest["ach_id"].astype(str).str.lower().str.contains(s2) |
                    df_latest["final_route"].astype(str).str.lower().str.contains(s2)
                ]

            view = df_latest[[
                "ach_id","client_name","client_mobile","final_route","start_date","end_date","status",
                "assigned_to","rep_name","_booking","advance_amount","incentive","final_cost","itinerary_id"
            ]].copy()
            view.rename(columns={
                "ach_id":"ACH ID","client_name":"Client","client_mobile":"Mobile","final_route":"Route",
                "start_date":"Start","end_date":"End","assigned_to":"Assigned to","rep_name":"Rep (credited)",
                "_booking":"Booking date","final_cost":"Final package cost (₹)"
            }, inplace=True)

            st.dataframe(
                view.sort_values(["Booking date","Start","Client"], na_position="last").drop(columns=["itinerary_id"]),
                use_container_width=True, hide_index=True
            )

            # Bulk reassign (Admin/Manager)
            if is_admin or is_manager:
                st.markdown("### 🔁 Bulk reassign (unique list)")
                options_all = (
                    view["ACH ID"].fillna("").astype(str) + " | " +
                    view["Client"].fillna("") + " | " +
                    view["Mobile"].fillna("") + " | " +
                    view["itinerary_id"]
                ).tolist()
                sel_multi = st.multiselect("Select one or more packages to reassign", options_all,
                                           help="Type to search by ACH, client or mobile")
                target_user = st.selectbox("Reassign selected to", ALL_USERS, index=0)
                if st.button("➡️ Reassign selected", type="primary", use_container_width=True, disabled=(not sel_multi)):
                    try:
                        moved = 0
                        for row in sel_multi:
                            iid = row.split(" | ")[-1]
                            reassign_followup(iid, from_user=user, to_user=target_user)
                            moved += 1
                        fetch_updates_joined.clear()
                        fetch_assigned_followups_raw.clear()
                        st.success(f"Reassigned {moved} package(s) to {target_user}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not reassign: {e}")

            if st.button("⬇️ Export table (CSV)"):
                out3 = io.StringIO()
                view.drop(columns=["itinerary_id"]).to_csv(out3, index=False)
                st.download_button("Download CSV", data=out3.getvalue().encode("utf-8"),
                                   file_name=f"packages_{_today_utc()}.csv", mime="text/csv",
                                   use_container_width=True)
        _clear_force_refresh()

# =========================
# TAB 3: 💰 Incentives (+ Admin booking-date editor)
# =========================
with tabs[2]:
    st.markdown("#### View incentives (booking-date based, policy from **01-Aug-2025**)")

    # Admin can pick user; others fixed to self
    if is_admin:
        rep_for_view = st.selectbox("Select user", ALL_USERS, index=ALL_USERS.index(user) if user in ALL_USERS else 0)
    else:
        rep_for_view = user
        st.caption(f"Showing incentives for **{rep_for_view}**")

    if _defer_guard("Incentives totals"):
        _clear_force_refresh()
    else:
        month_totals = fetch_user_months_with_totals(rep_for_view)
        if month_totals.empty:
            st.info("No incentives yet (policy applies from Aug-2025).")
        else:
            st.markdown("**Month-wise totals**")
            st.dataframe(month_totals, use_container_width=True, hide_index=True)

            months = month_totals["Month"].tolist()
            default_month = months[-1] if months else datetime.utcnow().strftime("%Y-%m")
            chosen_month = st.selectbox("Select month", months, index=months.index(default_month))

            yr, mo = map(int, chosen_month.split("-"))
            month_start = date(yr, mo, 1)
            month_end = (pd.Timestamp(month_start) + pd.offsets.MonthEnd(1)).date()

            details = fetch_user_customer_incentives_for_month(rep_for_view, month_start, month_end)
            st.markdown(f"**Customer-wise incentives for {chosen_month}**")
            if details.empty:
                st.info("No incentives for the selected month.")
            else:
                agg = details.groupby(["Client","Mobile"], as_index=False)["Incentive (₹)"].sum().sort_values("Incentive (₹)", ascending=False)
                c1, c2 = st.columns([1,1])
                with c1:
                    st.markdown("**Customer totals**")
                    st.dataframe(agg, use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Package-wise details**")
                    st.dataframe(details.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)

                # --- Admin booking date editor ---
                if is_admin:
                    st.markdown("### ✏️ Admin: Edit booking dates for this month")
                    # editable grid with current dates; allow change and then save
                    edit_df = details[["itinerary_id","ACH ID","Client","Mobile","Route","Booking date","Final package (₹)","Incentive (₹)"]].copy()
                    edit_df.rename(columns={"Booking date":"booking_date","Final package (₹)":"final_package_cost"}, inplace=True)
                    edited = st.data_editor(
                        edit_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "booking_date": st.column_config.DateColumn("Booking date", format="YYYY-MM-DD"),
                            "final_package_cost": st.column_config.NumberColumn("Final package (₹)", step=500, min_value=0),
                        },
                        disabled=["itinerary_id","ACH ID","Client","Mobile","Route","Incentive (₹)"]
                    )
                    if st.button("💾 Save booking date changes"):
                        try:
                            rows = edited.to_dict(orient="records")
                            updated = batch_update_booking_dates(rows, actor_user=user)
                            # clear caches & refresh
                            fetch_user_months_with_totals.clear()
                            fetch_user_customer_incentives_for_month.clear()
                            fetch_updates_joined.clear()
                            st.success(f"Updated {updated} record(s). Incentives recalculated.")
                            st.session_state["force_refresh"] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update dates: {e}")

                # Exports
                if st.button("⬇️ Export month customer totals (CSV)"):
                    buf = io.StringIO(); agg.to_csv(buf, index=False)
                    st.download_button("Download CSV (totals)", data=buf.getvalue().encode("utf-8"),
                                       file_name=f"incentives_{rep_for_view}_{chosen_month}_totals.csv",
                                       mime="text/csv", use_container_width=True)
                if st.button("⬇️ Export month package details (CSV)"):
                    buf2 = io.StringIO(); details.drop(columns=["itinerary_id"]).to_csv(buf2, index=False)
                    st.download_button("Download CSV (details)", data=buf2.getvalue().encode("utf-8"),
                                       file_name=f"incentives_{rep_for_view}_{chosen_month}_details.csv",
                                       mime="text/csv", use_container_width=True)
        _clear_force_refresh()
