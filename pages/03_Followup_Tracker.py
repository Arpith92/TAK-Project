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
st.title("📞 Follow-up Tracker")

# =========================
# Incentive policy constants
# =========================
INCENTIVE_START_DATE: date = date(2025, 8, 1)

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
        st.error("❌ Mongo URI missing. Add `mongo_uri` in Secrets.")
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
        st.error(f"❌ Could not connect MongoDB: {e}")
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
        st.error("🔐 Login not configured. Add `[users]` in Secrets.")
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
            st.error("❌ Invalid PIN")
            st.stop()
    return None

user = _login()
if not user:
    st.info("Please sign in to continue.")
    st.stop()

ALL_USERS = list(load_users().keys())
is_admin   = (str(user).strip().lower() in {"arpith","kuldeep"})
is_manager = (str(user).strip() == "Kuldeep")
can_reassign = is_admin or is_manager

# ---------------- Helpers ----------------
def _get_itinerary(iid: str, projection: Optional[dict] = None) -> dict:
    try:
        oid = ObjectId(iid) if ObjectId.is_valid(str(iid)) else iid
        doc = col_itineraries.find_one({"_id": oid}, projection or {})
        return doc or {}
    except Exception as e:
        st.error(f"Error fetching itinerary {iid}: {e}")
        return {}

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

def _ensure_columns(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({k: [v] for k, v in defaults.items()}).iloc[0:0]
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

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

    others = list(col_itineraries.find({"client_mobile": mobile}, {"_id":1}))
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
            "comment": f"Auto-confirmed due to confirmation of another package for the same mobile.",
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
            "utr": "",
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
# Cached fetchers (core)
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_updates_joined() -> pd.DataFrame:
    ups = list(col_updates.find({}, {"_id":0}))
    df_u = pd.DataFrame(ups) if ups else pd.DataFrame(columns=["itinerary_id"])
    its = list(col_itineraries.find({}, {
        "_id":1,"ach_id":1,"client_name":1,"client_mobile":1,"final_route":1,
        "start_date":1,"end_date":1,"representative":1,"upload_date":1
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

    def _ck(row):
        mob = str(row.get("client_mobile") or "").strip()
        if mob: return f"M:{mob}"
        ach = str(row.get("ach_id") or "").strip()
        nam = str(row.get("client_name") or "").strip()
        return f"A:{ach}|N:{nam}"
    df["_client_key"] = df.apply(_ck, axis=1)
    return df

def _filter_for_user(df: pd.DataFrame, who: str) -> pd.DataFrame:
    if is_admin or is_manager:
        return df
    s = df["status"].fillna("")
    mask = ((s == "followup") & (df["assigned_to"] == who)) | ((s == "confirmed") & (df["rep_name"] == who))
    return df[mask]

def latest_per_client(df: pd.DataFrame, user_filter: Optional[str]=None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if user_filter:
        df = _filter_for_user(df, user_filter)
    df = df.sort_values(["_client_key","_booking","_created"], ascending=[True, False, False])
    latest = df.groupby("_client_key", as_index=False).first()
    return latest

def _unique_status_counts(df_latest: pd.DataFrame) -> Dict[str,int]:
    if df_latest.empty:
        return {"confirmed":0,"followup":0,"pending":0,"under_discussion":0,"cancelled":0}
    s = df_latest["status"].fillna("")
    return {
        "confirmed": int((s=="confirmed").sum()),
        "followup": int((s=="followup").sum()),
        "pending": int((s=="pending").sum()),
        "under_discussion": int((s=="under_discussion").sum()),
        "cancelled": int((s=="cancelled").sum()),
    }

def _between_ts(series: pd.Series, start_d: date, end_d: date) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)
    lo = pd.Timestamp(datetime.combine(start_d, dtime.min))
    hi = pd.Timestamp(datetime.combine(end_d, dtime.max))
    return s.ge(lo) & s.le(hi)

def count_confirmed_unique_range(user_filter: Optional[str], start_d: date, end_d: date) -> int:
    df = fetch_updates_joined()
    df_l = latest_per_client(df, user_filter)
    if df_l.empty:
        return 0
    mask = (df_l["status"]=="confirmed") & _between_ts(df_l["_booking"], start_d, end_d)
    return int(mask.sum())

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
    utr: Optional[str] = None, final_package_cost_override: Optional[int] = None
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

    final_status = status if status in ("followup","cancelled","pending","under_discussion") else "confirmed"
    upd = {"itinerary_id": str(iid), "status": final_status, "updated_at": datetime.utcnow()}

    if final_status == "followup":
        upd["assigned_to"] = credit_user

    elif final_status == "confirmed":
        bdt = datetime.combine(booking_date, dtime.min) if booking_date else None
        upd["booking_date"] = bdt
        upd["advance_amount"] = int(advance_amount or 0)
        upd["utr"] = str(utr or "").strip()
        upd["rep_name"] = credit_user
        upd["assigned_to"] = None

        fc = 0
        if final_package_cost_override is not None:
            fc = _to_int(final_package_cost_override)
            base_amt = fc
            disc_amt = 0
            col_expenses.update_one(
                {"itinerary_id": str(iid)},
                {"$set": {
                    "itinerary_id": str(iid),
                    "base_package_cost": int(base_amt),
                    "discount": int(disc_amt),
                    "final_package_cost": int(fc),
                    "package_cost": int(fc),
                    "saved_at": datetime.utcnow(),
                }},
                upsert=True
            )
            _sync_cost_to_updates(iid=str(iid), final_cost=fc, base=base_amt, disc=disc_amt)
        else:
            fc = _final_cost_for(iid)

        inc_val = _compute_incentive(fc) if _eligible_for_incentive(bdt) else 0
        upd.update({
            "package_cost": int(fc),
            "final_package_cost": int(fc),
            "incentive": int(inc_val),
        })

        _auto_confirm_other_packages(
            current_iid=str(iid),
            credit_user=credit_user,
            booking_date=(bdt.date() if bdt else None),
            actor_user=actor_user
        )

    elif final_status in ("pending","under_discussion"):
        upd["assigned_to"] = credit_user

    elif final_status == "cancelled":
        upd["cancellation_reason"] = str(cancellation_reason or "")
        upd["assigned_to"] = None

    col_updates.update_one({"itinerary_id": str(iid)}, {"$set": upd}, upsert=True)

def batch_update_booking_dates(rows: List[dict], actor_user: str) -> int:
    """
    Save edited booking_date and (optionally) final_package_cost.
    Recalculate incentive per policy start date.
    """
    updated = 0
    for r in rows:
        iid = str(r.get("itinerary_id","")).strip()
        if not iid:
            continue
        bdt = _clean_dt(r.get("booking_date"))
        upd = col_updates.find_one({"itinerary_id": iid}, {"final_package_cost":1, "status":1})
        if not upd or upd.get("status") != "confirmed":
            continue  # only confirmed packages are editable
        # accept override if provided, else keep existing (or recompute)
        fc = _to_int(r.get("final_package_cost", upd.get("final_package_cost", 0)))
        if fc <= 0:
            fc = _final_cost_for(iid)
        inc = _compute_incentive(fc) if _eligible_for_incentive(bdt) else 0
        col_updates.update_one(
            {"itinerary_id": iid},
            {"$set": {
                "booking_date": bdt,
                "final_package_cost": int(fc),
                "incentive": int(inc),
                "updated_at": datetime.utcnow()
            }}
        )
        updated += 1
    return updated

# ============ Push back helper ============
def push_back_to_pending(iid: str, actor_user: str, note: str = "Pushed back by admin from Incentives editor") -> None:
    base = _get_itinerary(iid, {"client_name":1,"client_mobile":1,"ach_id":1})
    # audit trail in followups
    col_followups.insert_one({
        "itinerary_id": str(iid),
        "created_at": datetime.utcnow(),
        "created_by": actor_user,
        "status": "pending",
        "comment": note,
        "next_followup_on": None,
        "cancellation_reason": "",
        "credited_to": "",
        "client_name": base.get("client_name",""),
        "client_mobile": base.get("client_mobile",""),
        "ach_id": base.get("ach_id",""),
    })
    # clear booking/incentive to move out of confirmed list
    col_updates.update_one(
        {"itinerary_id": str(iid)},
        {"$set": {
            "status": "pending",
            "assigned_to": None,
            "booking_date": None,
            "advance_amount": 0,
            "utr": "",
            "incentive": 0,
            "rep_name": "",
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )

# =============================================================================
# Sidebar quick summary
# =============================================================================
with st.sidebar:
    if is_admin:
        st.markdown("### 📅 Monthly confirmed (unique latest)")
        today = _today_utc()
        first_this, last_this = month_bounds(today)
        month_pick = st.date_input("Pick any date in month", value=first_this)
        m_start, m_end = month_bounds(month_pick)
        df_base = latest_per_client(fetch_updates_joined(), None)
        df_base = _ensure_columns(df_base, {
            "client_name":"", "rep_name":"", "final_package_cost":0, "_booking": pd.NaT, "status":""
        })
        msk = (df_base["status"]=="confirmed") & _between_ts(df_base["_booking"], m_start, m_end)
        view = df_base.loc[msk, ["client_name","rep_name","final_package_cost"]].copy()
        view.rename(columns={
            "client_name":"Client",
            "rep_name":"Representative",
            "final_package_cost":"Final (₹)"
        }, inplace=True)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("Only one row per client (last revision).")

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
        st.caption("User mode: viewing your assigned & confirmed.")

tabs = st.tabs(["🗂️ Follow-ups", "📘 All packages", "💰 Incentives", "🧾 Revisions Trail"])

# =========================
# TAB 1: Follow-ups
# =========================
with tabs[0]:
    df_join = fetch_updates_joined()
    df_follow = df_join[df_join["status"].fillna("followup") == "followup"].copy()
    df_follow = latest_per_client(df_follow, user_filter)

    df_follow = _ensure_columns(df_follow, {
        "ach_id":"", "client_name":"", "client_mobile":"", "start_date":pd.NaT, "end_date":pd.NaT,
        "final_route":"", "assigned_to":"", "itinerary_id":""
    })

    q = st.text_input("🔎 Search (name / mobile / ACH / route)", "")
    table = df_follow.copy()
    if q.strip():
        s = q.strip().lower()
        table = table[
            table["client_name"].astype(str).str.lower().str.contains(s) |
            table["client_mobile"].astype(str).str.lower().str.contains(s) |
            table["ach_id"].astype(str).str.lower().str.contains(s) |
            table["final_route"].astype(str).str.lower().str.contains(s)
        ]

    table = table[[
        "ach_id","client_name","client_mobile","start_date","end_date",
        "final_route","assigned_to","itinerary_id"
    ]].sort_values(["start_date","client_name"], na_position="last")

    table.rename(columns={
        "ach_id":"ACH ID","client_name":"Client","client_mobile":"Mobile",
        "start_date":"Start","end_date":"End","final_route":"Route","assigned_to":"Assigned to",
    }, inplace=True)

    lcol, rcol = st.columns([2,1])
    with lcol:
        st.dataframe(table.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)
    with rcol:
        options = (table["ACH ID"].fillna("").astype(str) + " | " +
                   table["Client"].fillna("") + " | " +
                   table["Mobile"].fillna("") + " | " +
                   table["itinerary_id"])
        sel = st.selectbox("Open client", options.tolist())
        chosen_id = sel.split(" | ")[-1] if sel else None

    if not chosen_id:
        st.info("Pick a client on the right to view details.")
        st.stop()

    st.divider()
    st.subheader("Details & Update")

    it_doc = col_itineraries.find_one(
        {"_id": ObjectId(chosen_id)},
        {"ach_id":1,"client_name":1,"client_mobile":1,"final_route":1,"total_pax":1,
         "start_date":1,"end_date":1,"representative":1,"itinerary_text":1}
    ) or {}
    upd_doc = col_updates.find_one({"itinerary_id": str(chosen_id)}, {"_id":0}) or {}

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
        })
    with dc2:
        st.markdown("**Current Status**")
        st.write({
            "Status": upd_doc.get("status","followup"),
            "Assigned To": upd_doc.get("assigned_to",""),
            "Booking date": upd_doc.get("booking_date",""),
            "Advance (₹)": upd_doc.get("advance_amount",0),
            "UTR": upd_doc.get("utr",""),
            "Incentive (₹)": upd_doc.get("incentive",0),
            "Rep (credited to)": upd_doc.get("rep_name",""),
            "Final package cost (₹)": _final_cost_for(chosen_id),
        })

    st.markdown("### Confirm booking")
    with st.form("confirm_form"):
        booking_date = st.date_input("Booking date", value=date.today())
        final_pkg_amt = st.number_input("Final package amount (₹)", min_value=0, step=500)
        advance_amt = st.number_input("Advance amount (₹)", min_value=0, step=500)
        utr = st.text_input("UTR / Payment reference*", placeholder="e.g., UPI/NEFT UTR")
        comment = st.text_area("Comment (optional)")
        submitted_confirm = st.form_submit_button("✅ Confirm this package")
    if submitted_confirm:
        if final_pkg_amt <= 0:
            st.error("Enter a valid final package amount."); st.stop()
        if advance_amt <= 0:
            st.error("Enter a valid advance amount."); st.stop()
        if not utr.strip():
            st.error("UTR / Payment reference is required."); st.stop()
        upsert_update_status(
            iid=chosen_id,
            status="confirmed",
            actor_user=user,
            credit_user=user,
            next_followup_on=None,
            booking_date=booking_date,
            comment=comment,
            cancellation_reason=None,
            advance_amount=int(advance_amt),
            utr=utr.strip(),
            final_package_cost_override=int(final_pkg_amt)
        )
        st.success("Package confirmed.")
        fetch_updates_joined.clear()
        st.rerun()
# =========================
# TAB 2: All packages
# =========================
with tabs[1]:
    if is_admin:
        filter_user_all = st.selectbox("Filter packages by user (assigned/credited)", ["All users"] + ALL_USERS, index=0)
        fu = None if filter_user_all == "All users" else filter_user_all
    else:
        st.caption("Showing only your unique latest packages (assigned to you or credited to you).")
        fu = user

    df_latest = latest_per_client(fetch_updates_joined(), fu)
    df_latest = _ensure_columns(df_latest, {
        "ach_id":"", "client_name":"", "client_mobile":"", "final_route":"", "start_date":pd.NaT, "end_date":pd.NaT,
        "status":"", "assigned_to":"", "rep_name":"", "_booking":pd.NaT, "advance_amount":0, "utr":"", "itinerary_id":""
    })

    if df_latest.empty:
        st.info("No packages to display for the selected filter.")
    else:
        counts = _unique_status_counts(df_latest)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("✅ Confirmed (unique)", counts["confirmed"])
        m2.metric("🔵 Follow-up (unique)", counts["followup"])
        m3.metric("🟡 Pending (unique)", counts["pending"])
        m4.metric("🟠 Under discussion (unique)", counts["under_discussion"])
        m5.metric("🔴 Cancelled (unique)", counts["cancelled"])

        id_list = df_latest["itinerary_id"].astype(str).tolist()
        fc_map_all = _final_cost_map(id_list)
        df_latest["final_cost"] = df_latest["itinerary_id"].map(lambda x: fc_map_all.get(str(x), 0))

        q2 = st.text_input("Search (name / mobile / ACH / route)", "")
        view = df_latest.copy()
        if q2.strip():
            s2 = q2.strip().lower()
            view = view[
                view["client_name"].astype(str).str.lower().str.contains(s2) |
                view["client_mobile"].astype(str).str.lower().str.contains(s2) |
                view["ach_id"].astype(str).str.lower().str.contains(s2) |
                view["final_route"].astype(str).str.lower().str.contains(s2)
            ]

        view = _ensure_columns(view, {
            "ach_id":"", "client_name":"", "client_mobile":"", "final_route":"", "start_date":pd.NaT, "end_date":pd.NaT,
            "status":"", "assigned_to":"", "rep_name":"", "_booking":pd.NaT, "advance_amount":0, "utr":"", "final_cost":0, "itinerary_id":""
        })

        view = view[[
            "ach_id","client_name","client_mobile","final_route","start_date","end_date","status",
            "assigned_to","rep_name","_booking","advance_amount","utr","final_cost","itinerary_id"
        ]].copy()
        view.rename(columns={
            "ach_id":"ACH ID","client_name":"Client","client_mobile":"Mobile","final_route":"Route",
            "start_date":"Start","end_date":"End","assigned_to":"Assigned to","rep_name":"Rep (credited)",
            "_booking":"Booking date","final_cost":"Final package cost (₹)","utr":"UTR"
        }, inplace=True)

        st.dataframe(
            view.sort_values(["Booking date","Start","Client"], na_position="last").drop(columns=["itinerary_id"]),
            use_container_width=True, hide_index=True
        )

# =========================
# TAB 3: 💰 Incentives (+ Admin booking-date editor & push back)
# =========================
# =========================
# Incentives fetchers / sums (booking date based)
# =========================
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
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df = df[df["booking_date"].notna()]
    df["Month"] = df["booking_date"].dt.strftime("%Y-%m")
    df["incentive"] = df["incentive"].apply(_to_int)
    out = df.groupby("Month")["incentive"].sum().reset_index().rename(columns={"incentive": "Total Incentive (₹)"})
    out.sort_values("Month", inplace=True)
    return out




with tabs[2]:
    st.markdown("#### View incentives (booking-date based, policy from **01-Aug-2025**)")

    # Scope
    if is_admin:
        scope_choice = st.selectbox("Scope", ["All reps"] + ALL_USERS, index=0)
        rep_scope = None if scope_choice == "All reps" else scope_choice
    else:
        rep_scope = user
        st.caption(f"Showing incentives for **{rep_scope}**")

    # Month picker
    today = _today_utc()
    first_this, _ = month_bounds(today)
    month_pick = st.date_input("Pick month", value=first_this)
    month_start, month_end = month_bounds(month_pick)

    # Build details for scope (all reps or single rep)
    start_window = max(month_start, INCENTIVE_START_DATE)
    if start_window > month_end:
        details = pd.DataFrame(columns=[
            "itinerary_id","ACH ID","Client","Mobile","Route","Travel date","Booking date",
            "Final package (₹)","Incentive (₹)","Rep","Duplicate?"
        ])
    else:
        q = {
            "status": "confirmed",
            "final_package_cost": {"$gt": 0},
            "booking_date": {"$gte": datetime.combine(start_window, dtime.min),
                             "$lte": datetime.combine(month_end, dtime.max)}
        }
        if rep_scope:
            q["rep_name"] = rep_scope

        rows = list(col_updates.find(q, {"_id":0, "itinerary_id":1, "booking_date":1,
                                         "incentive":1, "final_package_cost":1, "rep_name":1}))
        if not rows:
            details = pd.DataFrame(columns=[
                "itinerary_id","ACH ID","Client","Mobile","Route","Travel date","Booking date",
                "Final package (₹)","Incentive (₹)","Rep","Duplicate?"
            ])
        else:
            df_u = pd.DataFrame(rows)
            df_u["itinerary_id"] = df_u["itinerary_id"].astype(str)
            its = list(col_itineraries.find(
                {"_id": {"$in": [ObjectId(x) for x in df_u["itinerary_id"].unique() if ObjectId.is_valid(x)]}},
                {"_id":1,"ach_id":1,"client_name":1,"client_mobile":1,"final_route":1,"start_date":1,"revision_num":1}
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
            df["Final package (₹)"] = df["final_package_cost"].apply(_to_int)
            df["Incentive (₹)"] = df["incentive"].apply(_to_int)
            df["Rep"] = df.get("rep_name", "")

            # unique by (Mobile, Client, Travel date) keeping last revision
            df["_key"] = df[["Mobile","Client","Travel date"]].astype(object).agg(tuple, axis=1)
            df = df.sort_values(["_key","_rev","Booking date"], ascending=[True, False, False])
            unique_df = df.groupby("_key", as_index=False).first()

            details = unique_df[["itinerary_id","ACH ID","Client","Mobile","Route","Travel date",
                                 "Booking date","Final package (₹)","Incentive (₹)","Rep"]].sort_values(["Booking date","Client"])

            # Duplicate flag by client name (within scope+month)
            dup_mask = details.duplicated(subset=["Client"], keep=False)
            details["Duplicate?"] = dup_mask.astype(bool)

    # Month-wise totals (for single rep)
    if rep_scope:
        month_totals = fetch_user_months_with_totals(rep_scope)
        if not month_totals.empty:
            st.markdown("**Month-wise totals (selected rep)**")
            st.dataframe(month_totals, use_container_width=True, hide_index=True)

    if details.empty:
        st.info("No incentives for the selected scope/month.")
    else:
        # Preview tables
        st.markdown("**Customer totals**")
        agg = details.groupby(["Client","Mobile"], as_index=False)["Incentive (₹)"].sum().sort_values("Incentive (₹)", ascending=False)
        c1, c2 = st.columns([1,1])
        with c1:
            st.dataframe(agg, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Package-wise details (unique)**")
            dup_mask = details["Duplicate?"].fillna(False)
            def _styler(d):
                styles = pd.DataFrame("", index=d.index, columns=d.columns)
                styles.loc[dup_mask, :] = "background-color:#5a1414;color:#fff;"
                return styles
            st.dataframe(details.drop(columns=["itinerary_id"]).style.apply(_styler, axis=None),
                         use_container_width=True, hide_index=True)

                 # ---------- Stateful editor buffer ----------
            key_id = f"incbuf::{rep_scope or 'ALL'}::{month_start.strftime('%Y-%m')}"
            if "inc_buffers" not in st.session_state:
                st.session_state["inc_buffers"] = {}
            if key_id not in st.session_state["inc_buffers"]:
                # include Action column default blank
                tmp = details.copy()
                tmp["Action"] = ""  # "", "Push back to pending"
                st.session_state["inc_buffers"][key_id] = {"orig": tmp.copy(), "df": tmp.copy()}
            buf = st.session_state["inc_buffers"][key_id]
            edit_df = buf["df"]

            st.markdown("### ✏️ Admin: Edit booking dates, reps, or push back confirmations")
            editor_view = edit_df[[
                "itinerary_id","ACH ID","Client","Mobile","Route","Travel date",
                "Booking date","Final package (₹)","Incentive (₹)","Rep","Duplicate?","Action"
            ]].copy()
            editor_view.rename(columns={
                "Booking date":"booking_date",
                "Final package (₹)":"final_package_cost"
            }, inplace=True)

            column_config = {
                "booking_date": st.column_config.DateColumn("Booking date", format="YYYY-MM-DD"),
                "final_package_cost": st.column_config.NumberColumn("Final package (₹)", step=500, min_value=0),
                "Duplicate?": st.column_config.CheckboxColumn("Duplicate?", disabled=True),
            }
            if is_admin:
                column_config["Action"] = st.column_config.SelectboxColumn(
                    "Action", options=["", "Push back to pending"]
                )
                column_config["Rep"] = st.column_config.SelectboxColumn(
                    "Representative", options=ALL_USERS
                )
            else:
                column_config["Action"] = st.column_config.TextColumn("Action", disabled=True)
                column_config["Rep"] = st.column_config.TextColumn("Representative", disabled=True)

            # Force data editor to expand full width
st.markdown("""
    <style>
    [data-testid="stDataFrameResizable"] {
        width: 100% !important;
    }
    [data-testid="stHorizontalBlock"] {
        justify-content: flex-start !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.data_editor(
    editor_view,
    key=f"inc_editor_{key_id}",
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
    disabled=[...]
)


            # Persist edits into buffer
            edited_back = edited.rename(columns={
                "booking_date":"Booking date",
                "final_package_cost":"Final package (₹)"
            })
            merge_cols = ["itinerary_id","Booking date","Final package (₹)","Rep","Action"]
            st.session_state["inc_buffers"][key_id]["df"] = edit_df.drop(
                columns=["Booking date","Final package (₹)","Rep","Action"]
            ).merge(edited_back[merge_cols], on="itinerary_id", how="left")

            if st.button("💾 Save changes", type="primary"):
                try:
                    orig = st.session_state["inc_buffers"][key_id]["orig"]
                    cur  = st.session_state["inc_buffers"][key_id]["df"]

                    comp = cur.merge(
                        orig[["itinerary_id","Booking date","Final package (₹)","Rep","Action"]],
                        on="itinerary_id", how="left", suffixes=("", "_orig")
                    )

                    updated = 0
                    pushed  = 0
                    repchg  = 0

                    for _, r in comp.iterrows():
                        iid = str(r["itinerary_id"])
                        # Push back to pending
                        if is_admin and r.get("Action") == "Push back to pending":
                            col_updates.update_one(
                                {"itinerary_id": iid},
                                {"$set": {"status": "pending", "updated_at": datetime.utcnow()}}
                            )
                            col_followups.insert_one({
                                "itinerary_id": iid,
                                "created_at": datetime.utcnow(),
                                "created_by": user,
                                "status": "pending",
                                "comment": "Admin pushed back from confirmed to pending",
                                "credited_to": r.get("Rep",""),
                            })
                            pushed += 1
                            continue

                        # Booking date / cost changes
                        if (r["Booking date"] != r["Booking date_orig"]) or \
                           (r["Final package (₹)"] != r["Final package (₹)_orig"]):
                            rows = [{
                                "itinerary_id": iid,
                                "booking_date": r["Booking date"],
                                "final_package_cost": r["Final package (₹)"]
                            }]
                            updated += batch_update_booking_dates(rows, actor_user=user)

                        # Rep change
                        if is_admin and r["Rep"] != r["Rep_orig"]:
                            col_updates.update_one(
                                {"itinerary_id": iid},
                                {"$set": {"rep_name": r["Rep"], "updated_at": datetime.utcnow()}}
                            )
                            col_followups.insert_one({
                                "itinerary_id": iid,
                                "created_at": datetime.utcnow(),
                                "created_by": user,
                                "status": "confirmed",
                                "comment": f"Admin changed rep from {r['Rep_orig']} to {r['Rep']}",
                                "credited_to": r.get("Rep",""),
                            })
                            repchg += 1

                    # Refresh caches, reset buffer
                    fetch_updates_joined.clear()
                    _final_cost_map.clear()
                    st.session_state["inc_buffers"].pop(key_id, None)

                    msg = []
                    if updated: msg.append(f"updated {updated} booking/cost")
                    if repchg: msg.append(f"changed {repchg} rep(s)")
                    if pushed: msg.append(f"pushed back {pushed} package(s)")
                    if not msg:
                        st.info("No changes to save.")
                    else:
                        st.success("Saved: " + " and ".join(msg))
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to save: {e}")

# =========================
# TAB 4: 🧾 Revisions Trail
# =========================
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
