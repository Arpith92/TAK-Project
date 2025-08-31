# pages/03_Followup_Tracker.py
from __future__ import annotations

from datetime import datetime, date, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import os
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
        .stMarkdown, .stText, .stDataFrame, .stMetric, .stCaption { color:var(--fg)!important; }
        input, textarea, select, .stTextInput>div>div>input, .stNumberInput input{
           background:var(--card)!important; color:var(--fg)!important; border:1px solid var(--border)!important;
        }
        label, .stSelectbox label, .stCheckbox>label{ color:var(--fg)!important; }
        .stButton>button { background:var(--btn)!important; color:var(--fg)!important; 
           border-radius:10px; border:1px solid var(--border)!important; }
        .stButton>button:hover{ background:var(--btnh)!important; }
        [data-testid="stDataFrame"] div, [data-testid="stDataFrame"] th{
            color:var(--fg)!important; background:var(--card2)!important; border-color:var(--border)!important;
        }
        .st-af, .st-bc, .st-bb{ background:var(--card)!important; color:var(--fg)!important; }
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
# Users + login
# =========================
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    # fallback for local dev
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
            "Add a `[users]` table in **Manage app → Secrets** or in `.streamlit/secrets.toml`."
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

ALL_USERS = list(load_users().keys())
is_admin   = (str(user).strip().lower() in {"arpith","kuldeep"})
is_manager = (str(user).strip() == "Kuldeep")

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

def _get_itinerary(iid: str, projection: Optional[dict] = None) -> dict:
    projection = projection or {}
    doc = None
    try:
        doc = col_itineraries.find_one({"_id": ObjectId(iid)}, projection)
    except Exception:
        doc = col_itineraries.find_one({"itinerary_id": str(iid)}, projection)
    return doc or {}

# =========================
# Final cost helpers
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
# Core fetch (itineraries ⨝ updates)
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_updates_joined() -> pd.DataFrame:
    ups = list(col_updates.find({}, {"_id":0}))
    df_u = pd.DataFrame(ups) if ups else pd.DataFrame(columns=["itinerary_id"])
    its = list(col_itineraries.find({}, {
        "_id":1,"ach_id":1,"client_name":1,"client_mobile":1,"final_route":1,
        "start_date":1,"end_date":1,"representative":1
    }))
    df_i = pd.DataFrame([{
        "itinerary_id": str(r["_id"]),
        "ach_id": r.get("ach_id",""),
        "client_name": r.get("client_name",""),
        "client_mobile": r.get("client_mobile",""),
        "final_route": r.get("final_route",""),
        "start_date": pd.to_datetime(r.get("start_date")).date() if r.get("start_date") else None,
        "end_date": pd.to_datetime(r.get("end_date")).date() if r.get("end_date") else None,
        "representative": r.get("representative",""),
        "_created_utc": ObjectId(str(r["_id"])).generation_time if ObjectId.is_valid(str(r["_id"])) else None
    } for r in its])

    if df_u.empty and df_i.empty:
        return pd.DataFrame()

    df = df_i.merge(df_u, on="itinerary_id", how="left")
    df["status"] = df["status"].fillna("followup")
    df["_booking"] = pd.to_datetime(df.get("booking_date"), errors="coerce", utc=True).dt.tz_convert(None)
    df["_created"] = pd.to_datetime(df.get("_created_utc"), errors="coerce")

    def _client_key(row):
        mob = str(row.get("client_mobile") or "").strip()
        if mob: return f"M:{mob}"
        ach = str(row.get("ach_id") or "").strip()
        nam = str(row.get("client_name") or "").strip()
        return f"A:{ach}|N:{nam}"
    df["_client_key"] = df.apply(_client_key, axis=1)
    return df

def latest_per_client(df: pd.DataFrame, user_filter: Optional[str]=None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if user_filter:
        s = df["status"].fillna("")
        df = df[((s=="followup") & (df["assigned_to"]==user_filter)) |
                ((s=="confirmed") & (df["rep_name"]==user_filter))]
    df = df.sort_values(["_client_key","_booking","_created"], ascending=[True, False, False])
    latest = df.groupby("_client_key", as_index=False).first()
    return latest

def _between_ts(series: pd.Series, start_d: date, end_d: date) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)
    lo = pd.Timestamp(datetime.combine(start_d, dtime.min))
    hi = pd.Timestamp(datetime.combine(end_d, dtime.max))
    return s.ge(lo) & s.le(hi)

# =========================
# Incentives data
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_user_months_with_totals(rep_name: Optional[str]) -> pd.DataFrame:
    q = {
        "status": "confirmed",
        "booking_date": {"$gte": datetime.combine(INCENTIVE_START_DATE, dtime.min)}
    }
    if rep_name and rep_name != "All users":
        q["rep_name"] = rep_name

    cur = list(col_updates.find(q, {"_id":0, "booking_date":1, "incentive":1}))
    if not cur:
        return pd.DataFrame(columns=["Month", "Total Incentive (₹)"])
    df = pd.DataFrame(cur)
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df = df[df["booking_date"].notna()]
    df["Month"] = df["booking_date"].dt.strftime("%Y-%m")
    df["incentive"] = df["incentive"].apply(_to_int)
    out = df.groupby("Month")["incentive"].sum().reset_index().rename(columns={"incentive": "Total Incentive (₹)"})
    out.sort_values("Month", inplace=True)
    return out

@st.cache_data(ttl=60, show_spinner=False)
def fetch_incentive_rows_for_month(rep_name: Optional[str], month_start: date, month_end: date) -> pd.DataFrame:
    """
    Return UNIQUE confirmed rows within month.
    Uniqueness key: (Mobile, Client, Travel date) with the **highest revision** kept.
    If rep_name is None or "All users", include everyone and add 'Rep' column.
    """
    start_window = max(month_start, INCENTIVE_START_DATE)
    if start_window > month_end:
        cols = ["itinerary_id","ACH ID","Client","Mobile","Route","Travel date","Booking date",
                "Final package (₹)","Incentive (₹)","Rep"]
        return pd.DataFrame(columns=cols)

    q = {
        "status": "confirmed",
        "booking_date": {"$gte": datetime.combine(start_window, dtime.min),
                         "$lte": datetime.combine(month_end, dtime.max)}
    }
    if rep_name and rep_name != "All users":
        q["rep_name"] = rep_name

    rows = list(col_updates.find(q, {"_id":0, "itinerary_id":1, "booking_date":1,
                                     "incentive":1, "final_package_cost":1, "rep_name":1}))
    if not rows:
        cols = ["itinerary_id","ACH ID","Client","Mobile","Route","Travel date","Booking date",
                "Final package (₹)","Incentive (₹)","Rep"]
        return pd.DataFrame(columns=cols)

    df_u = pd.DataFrame(rows)
    df_u["itinerary_id"] = df_u["itinerary_id"].astype(str)
    df_u["Rep"] = df_u["rep_name"].fillna("")

    its = list(col_itineraries.find(
        {"_id": {"$in": [ObjectId(x) for x in df_u["itinerary_id"].unique() if ObjectId.is_valid(x)]}},
        {"_id":1, "ach_id":1, "client_name":1, "client_mobile":1, "final_route":1, "start_date":1, "revision_num":1}
    ))
    df_i = pd.DataFrame([{
        "itinerary_id": str(i["_id"]),
        "ACH ID": i.get("ach_id",""),
        "Client": i.get("client_name",""),
        "Mobile": i.get("client_mobile",""),
        "Route": i.get("final_route",""),
        "Travel date": (pd.to_datetime(i.get("start_date")).date() if i.get("start_date") else None),
        "_rev": int(i.get("revision_num", 1) or 1)
    } for i in its])

    df = df_u.merge(df_i, on="itinerary_id", how="left")
    df["Booking date"] = pd.to_datetime(df["booking_date"], errors="coerce").dt.date
    df["Incentive (₹)"] = df["incentive"].apply(_to_int)
    df["Final package (₹)"] = df["final_package_cost"].apply(_to_int)

    df["_key"] = df[["Mobile","Client","Travel date"]].astype(object).agg(tuple, axis=1)
    df = df.sort_values(["_key", "_rev", "Booking date"], ascending=[True, False, False])
    unique_df = df.groupby("_key", as_index=False).first()

    view = unique_df[
        ["itinerary_id","ACH ID","Client","Mobile","Route","Travel date","Booking date",
         "Final package (₹)","Incentive (₹)","Rep"]
    ].sort_values(["Booking date","Client"])
    return view

# =========================
# Update helpers
# =========================
def batch_update_booking_dates(rows: List[dict]) -> int:
    """
    rows: list of {"itinerary_id", "booking_date", "final_package_cost"}
    """
    updated = 0
    for r in rows:
        iid = str(r.get("itinerary_id","")).strip()
        if not iid:
            continue
        bdt = _clean_dt(r.get("booking_date"))
        upd = col_updates.find_one({"itinerary_id": iid}, {"final_package_cost":1, "status":1})
        if not upd or upd.get("status") != "confirmed":
            continue
        fc = _to_int((r.get("final_package_cost") if r.get("final_package_cost") is not None
                      else upd.get("final_package_cost", 0)))
        if fc <= 0:
            fc = _final_cost_for(iid)
        inc = _compute_incentive(fc) if _eligible_for_incentive(bdt) else 0
        col_updates.update_one(
            {"itinerary_id": iid},
            {"$set": {"booking_date": bdt, "final_package_cost": int(fc),
                      "incentive": int(inc), "updated_at": datetime.utcnow()}}
        )
        updated += 1
    return updated

# =============================================================================
# Sidebar summary (admin)
# =============================================================================
with st.sidebar:
    if is_admin:
        st.markdown("### 📅 Monthly confirmed (unique latest)")
        today = _today_utc()
        first_this, _ = month_bounds(today)
        month_pick = st.date_input("Pick any date in month", value=first_this)
        m_start, m_end = month_bounds(month_pick)
        df_base = latest_per_client(fetch_updates_joined(), None)
        if not df_base.empty:
            msk = (df_base["status"]=="confirmed") & _between_ts(df_base["_booking"], m_start, m_end)
            view = df_base.loc[msk, ["client_name","rep_name","final_package_cost"]].copy()
            view.rename(columns={"client_name":"Client","rep_name":"Representative",
                                 "final_package_cost":"Final (₹)"}, inplace=True)
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.caption("Only one row per client (last revision).")

# =============================================================================
# Main tabs
# =============================================================================
tabs = st.tabs(["🗂️ Follow-ups", "📘 All packages", "💰 Incentives", "🧾 Revisions Trail"])

# -------------------------
# TAB 1: Follow-ups (unchanged from your last good version)
# -------------------------
with tabs[0]:
    st.info("Follow-ups tab unchanged here to keep this message focused on the Incentives fix.")

# -------------------------
# TAB 2: All packages (unchanged metrics table)
# -------------------------
with tabs[1]:
    st.info("All packages tab unchanged.")

# -------------------------
# TAB 3: Incentives
# -------------------------
with tabs[2]:
    st.markdown("#### View incentives (booking-date based, policy from **01-Aug-2025**)")

    # Scope selector: single rep OR all reps
    scope_mode = st.radio("Edit scope", ["Single representative", "All representatives"], horizontal=True)
    if scope_mode == "Single representative":
        rep_choice = st.selectbox("Select representative", ["All users"] + ALL_USERS, index=ALL_USERS.index(user) + 1 if user in ALL_USERS else 0)
        scope_rep = None if rep_choice == "All users" else rep_choice
        scope_label = rep_choice if rep_choice != "All users" else "All users"
    else:
        scope_rep = None
        scope_label = "All users"

    month_totals = fetch_user_months_with_totals(scope_rep if scope_rep else None)
    if month_totals.empty:
        st.info("No incentives yet in the selected scope.")
    else:
        st.markdown("**Month-wise totals**")
        st.dataframe(month_totals, use_container_width=True, hide_index=True)

        months = month_totals["Month"].tolist()
        default_month = months[-1] if months else datetime.utcnow().strftime("%Y-%m")
        chosen_month = st.selectbox("Select month", months, index=months.index(default_month))

        yr, mo = map(int, chosen_month.split("-"))
        month_start = date(yr, mo, 1)
        month_end = (pd.Timestamp(month_start) + pd.offsets.MonthEnd(1)).date()

        # ---------- Stable editor state (no reload while typing) ----------
        same_scope = (
            st.session_state.get("inc_edit_mode", False)
            and st.session_state.get("inc_scope_label") == scope_label
            and st.session_state.get("inc_month") == chosen_month
            and isinstance(st.session_state.get("inc_editor_df"), pd.DataFrame)
        )
        if not same_scope:
            details_fresh = fetch_incentive_rows_for_month(scope_rep, month_start, month_end)
            editor_df = details_fresh[[
                "itinerary_id","ACH ID","Client","Mobile","Route","Travel date",
                "Booking date","Final package (₹)","Incentive (₹)","Rep"
            ]].rename(columns={"Booking date":"booking_date",
                               "Final package (₹)":"final_package_cost"}).copy()
            st.session_state["inc_editor_df"] = editor_df
            st.session_state["inc_edit_mode"] = True
            st.session_state["inc_scope_label"] = scope_label
            st.session_state["inc_month"] = chosen_month

        editor_df = st.session_state["inc_editor_df"].copy()

        # Duplicate flag (by Client within current scope)
        editor_df["Duplicate?"] = editor_df["Client"].duplicated(keep=False)

        # Live preview frame (styled duplicates in red)
        preview_df = editor_df.rename(columns={
            "booking_date":"Booking date",
            "final_package_cost":"Final package (₹)"
        }).copy()

        def _style_dupe_clients(df_in: pd.DataFrame) -> pd.io.formats.style.Styler:
            dupe = df_in["Client"].duplicated(keep=False)
            return df_in.style.apply(lambda s: ['color: red' if d else '' for d in dupe], subset=["Client"])

        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("**Customer totals**")
            agg = preview_df.groupby(["Client","Mobile"], as_index=False)["Incentive (₹)"].sum().sort_values("Incentive (₹)", ascending=False)
            st.dataframe(agg, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Package-wise details (unique)**")
            st.dataframe(_style_dupe_clients(preview_df.drop(columns=["itinerary_id"])), use_container_width=True, hide_index=True)

        # ---------- Editable grid ----------
        st.markdown("### ✏️ Admin: Edit booking dates for this month")
        edited = st.data_editor(
            editor_df,
            key="inc_editor",
            use_container_width=True,
            hide_index=True,
            column_config={
                "booking_date": st.column_config.DateColumn("Booking date", format="YYYY-MM-DD"),
                "final_package_cost": st.column_config.NumberColumn("Final package (₹)", step=500, min_value=0),
                "itinerary_id": st.column_config.Column("itinerary_id", help="Internal ID", disabled=True),
                "ACH ID": st.column_config.Column("ACH ID", disabled=True),
                "Client": st.column_config.Column("Client", disabled=True),
                "Mobile": st.column_config.Column("Mobile", disabled=True),
                "Route": st.column_config.Column("Route", disabled=True),
                "Travel date": st.column_config.DateColumn("Travel date", disabled=True),
                "Incentive (₹)": st.column_config.NumberColumn("Incentive (₹)", disabled=True),
                "Rep": st.column_config.Column("Rep", disabled=True),
                "Duplicate?": st.column_config.CheckboxColumn("Duplicate?", disabled=True),
            },
        )
        # keep working copy updated to avoid wipe on rerun
        st.session_state["inc_editor_df"] = edited.copy()

        if st.button("💾 Save booking date changes"):
            try:
                rows = edited[["itinerary_id","booking_date","final_package_cost"]].to_dict(orient="records")
                n = batch_update_booking_dates(rows)
                # reset caches & editor state, then reload fresh scope/month
                fetch_user_months_with_totals.clear()
                fetch_incentive_rows_for_month.clear()
                st.session_state["inc_edit_mode"] = False
                st.session_state.pop("inc_editor_df", None)
                st.success(f"Updated {n} record(s).")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update dates: {e}")

# -------------------------
# TAB 4: Revisions Trail
# -------------------------
with tabs[3]:
    st.markdown("#### View all revisions (read-only) — does not affect counts or incentives")
    qtrail = st.text_input("Search (name / mobile / ACH)", "")
    cur = list(col_itineraries.find({}, {"_id":1,"ach_id":1,"client_name":1,"client_mobile":1,"final_route":1,
                                         "start_date":1,"end_date":1,"upload_date":1,"revision_num":1}))
    if not cur:
        st.info("No itineraries found yet.")
    else:
        df_allrevs = pd.DataFrame([{
            "itinerary_id": str(r["_id"]),
            "ACH ID": r.get("ach_id",""),
            "Client": r.get("client_name",""),
            "Mobile": r.get("client_mobile",""),
            "Route": r.get("final_route",""),
            "Start": pd.to_datetime(r.get("start_date")).date() if r.get("start_date") else None,
            "End": pd.to_datetime(r.get("end_date")).date() if r.get("end_date") else None,
            "Uploaded": pd.to_datetime(r.get("upload_date"), errors="coerce"),
            "Revision": int(r.get("revision_num", 1) or 1),
        } for r in cur])
        if qtrail.strip():
            s = qtrail.strip().lower()
            df_allrevs = df_allrevs[
                df_allrevs["Client"].astype(str).str.lower().str.contains(s) |
                df_allrevs["Mobile"].astype(str).str.lower().str.contains(s) |
                df_allrevs["ACH ID"].astype(str).str.lower().str.contains(s)
            ]
        df_allrevs.sort_values(["Mobile","Start","Revision"], ascending=[True, False, False], inplace=True)
        st.dataframe(df_allrevs.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)
