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
st.title("📞 Follow-up Tracker")

IST = ZoneInfo("Asia/Kolkata")

# =========================
# Safe secrets + Mongo helpers (parity with other pages)
# =========================
CAND_KEYS = ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]

def _visible_secret_keys() -> list[str]:
    try:
        return list(st.secrets.keys())
    except Exception:
        return []

def _find_uri() -> Optional[str]:
    for k in CAND_KEYS:
        try:
            v = st.secrets.get(k)
        except Exception:
            v = None
        if v:
            return v
    for k in CAND_KEYS:
        v = os.getenv(k)
        if v:
            return v
    return None

@st.cache_resource
def _get_client() -> MongoClient:
    uri = _find_uri()
    if not uri:
        st.error(
            "Mongo connection is not configured.\n\n"
            "Add one of these in **Manage app → Settings → Secrets** (recommended: `mongo_uri`).\n"
            "Example:\n"
            'mongo_uri = "mongodb+srv://USER:PASS@host/?retryWrites=true&w=majority"\n'
        )
        st.caption(f"Visible secret keys in this app: {_visible_secret_keys()}")
        env_present = [k for k in CAND_KEYS if os.getenv(k)]
        st.caption(f"Present env vars: {env_present}")
        st.stop()
    client = MongoClient(
        uri,
        appName="TAK_FollowupTracker",
        maxPoolSize=100,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        retryWrites=True,
        tz_aware=True,
    )
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        st.error("Could not connect to MongoDB. Check the URI and Atlas network access.\n\n"
                 f"Details: {e}")
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
# Users loader (Secrets or local toml fallback)
# =========================
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    # fallback to repo .streamlit/secrets.toml (dev only)
    try:
        try:
            import tomllib  # py 3.11+
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, dict) and u:
            with st.sidebar:
                st.warning(
                    "Using users from repo .streamlit/secrets.toml. "
                    "For production, set them in Manage app → Secrets."
                )
            return u
    except Exception:
        pass
    return {}

# =========================
# Enforced PIN login (+ Logout)
# =========================
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
    if not isinstance(users_map, dict) or not users_map:
        with st.sidebar:
            st.caption("Secrets debug")
            st.write("users type:", type(st.secrets.get("users", None)).__name__)
            st.write("keys:", _visible_secret_keys())
        st.error(
            "Login is not configured yet.\n\n"
            "Add to **Manage app → Secrets**:\n"
            'mongo_uri = "mongodb+srv://…"\n\n'
            "[users]\nArpith = \"1234\"\nReena = \"5678\"\nTeena = \"7777\"\nKuldeep = \"8888\"\n"
        )
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
is_admin = (str(user).strip().lower() == "arpith")  # Arpith is admin

# =========================
# Helpers (aligned across pages)
# =========================
def _to_int(x, default=0):
    try:
        if x is None:
            return default
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def _clean_dt(x: object) -> Optional[datetime]:
    if x is None:
        return None
    try:
        ts = pd.to_datetime(x)
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
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
    if not dt:
        return ""
    try:
        return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M UTC")

def _get_itinerary(iid: str, projection: Optional[dict] = None) -> dict:
    """Robust fetch by _id OR itinerary_id (some older data may store string ids)."""
    projection = projection or {}
    doc = None
    try:
        doc = col_itineraries.find_one({"_id": ObjectId(iid)}, projection)
    except Exception:
        doc = col_itineraries.find_one({"itinerary_id": str(iid)}, projection)
    return doc or {}

# -------- Final cost logic (base − discount) ----------
def _final_cost_for(iid: str) -> int:
    """
    Final cost = base_package_cost - discount, preferred explicit 'final_package_cost' in expenses.
    Falls back to itinerary.package_cost if no expense record exists.
    Back-compat: if only 'package_cost' exists in expenses, treat as final.
    """
    exp = col_expenses.find_one(
        {"itinerary_id": str(iid)},
        {"final_package_cost": 1, "base_package_cost": 1, "discount": 1, "package_cost": 1}
    ) or {}

    if "final_package_cost" in exp:
        return _to_int(exp.get("final_package_cost", 0))

    if ("base_package_cost" in exp) or ("discount" in exp):
        base = _to_int(exp.get("base_package_cost", exp.get("package_cost", 0)))
        disc = _to_int(exp.get("discount", 0))
        return max(0, base - disc)

    if "package_cost" in exp:
        return _to_int(exp.get("package_cost", 0))

    it = _get_itinerary(iid, {"package_cost": 1, "discount": 1})
    base = _to_int(it.get("package_cost", 0))
    disc = _to_int(it.get("discount", 0))
    return max(0, base - disc)

def _compute_incentive(final_amt: int) -> int:
    if 5000 < final_amt < 20000:
        return 250
    if final_amt >= 20000:
        return 500
    return 0

# =========================
# Data fetchers (admin-aware)
# =========================
def fetch_assigned_followups(assigned_to: Optional[str]) -> pd.DataFrame:
    """If assigned_to is None -> all followups; else only that user's assigned followups."""
    q = {"status": "followup"}
    if assigned_to is not None:
        q["assigned_to"] = assigned_to
    rows = list(col_updates.find(q, {"_id": 0}))
    if not rows:
        return pd.DataFrame(columns=["itinerary_id", "assigned_to", "status"])
    df_u = pd.DataFrame(rows)
    df_u["itinerary_id"] = df_u["itinerary_id"].astype(str)

    its = list(col_itineraries.find({}, {
        "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
        "start_date": 1, "end_date": 1, "final_route": 1, "total_pax": 1,
        "representative": 1, "itinerary_text": 1
    }))
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        for k in ("start_date", "end_date"):
            try:
                r[k] = pd.to_datetime(r.get(k)).date()
            except Exception:
                r[k] = None
    df_i = pd.DataFrame(its).drop(columns=["_id"]) if its else pd.DataFrame()
    return df_u.merge(df_i, on="itinerary_id", how="left")

def fetch_packages_with_updates(filter_user: Optional[str]) -> pd.DataFrame:
    """
    All packages + latest update snapshot.
    If filter_user provided: show rows where assigned_to==user OR rep_name==user.
    """
    its = list(col_itineraries.find({}, {
        "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
        "start_date": 1, "end_date": 1, "final_route": 1, "total_pax": 1
    }))
    if not its:
        return pd.DataFrame()
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        r["start_date"] = pd.to_datetime(r.get("start_date")).date() if r.get("start_date") else None
        r["end_date"] = pd.to_datetime(r.get("end_date")).date() if r.get("end_date") else None
    df_i = pd.DataFrame(its).drop(columns=["_id"])
    ups = list(col_updates.find({}, {"_id":0}))
    df_u = pd.DataFrame(ups) if ups else pd.DataFrame(columns=["itinerary_id"])
    df = df_i.merge(df_u, on="itinerary_id", how="left")
    if filter_user:
        df = df[(df["assigned_to"] == filter_user) | (df["rep_name"] == filter_user)]
    return df

def fetch_latest_followup_log_map(itinerary_ids: List[str]) -> Dict[str, dict]:
    if not itinerary_ids:
        return {}
    cur = col_followups.find({"itinerary_id": {"$in": itinerary_ids}})
    latest: Dict[str, dict] = {}
    for d in cur:
        iid = str(d.get("itinerary_id"))
        ts = _clean_dt(d.get("created_at")) or datetime.min
        if iid not in latest or ts > latest[iid].get("_ts", datetime.min):
            d["_ts"] = ts
            latest[iid] = d
    return latest

def fetch_confirmed_incentives(user_filter: Optional[str], start_d: date, end_d: date) -> int:
    q = {
        "status": "confirmed",
        "booking_date": {
            "$gte": datetime.combine(start_d, dtime.min),
            "$lte": datetime.combine(end_d, dtime.max),
        },
    }
    if user_filter:
        q["rep_name"] = user_filter
    cur = col_updates.find(q, {"_id": 0, "incentive": 1})
    return sum(_to_int(d.get("incentive", 0)) for d in cur)

def count_confirmed(user_filter: Optional[str], start_d: Optional[date]=None, end_d: Optional[date]=None) -> int:
    q = {"status": "confirmed"}
    if user_filter:
        q["rep_name"] = user_filter
    if start_d and end_d:
        q["booking_date"] = {
            "$gte": datetime.combine(start_d, dtime.min),
            "$lte": datetime.combine(end_d, dtime.max),
        }
    return col_updates.count_documents(q)

# =========================
# Reassign follow-ups
# =========================
def _latest_next_followup_date(iid: str) -> Optional[datetime]:
    d = col_followups.find_one(
        {"itinerary_id": str(iid)},
        sort=[("created_at", -1)],
        projection={"next_followup_on": 1}
    )
    return d.get("next_followup_on") if d else None

def reassign_followup(iid: str, from_user: str, to_user: str) -> None:
    next_dt = _latest_next_followup_date(iid)
    # Immutable log of transfer
    log_doc = {
        "itinerary_id": str(iid),
        "created_at": datetime.utcnow(),
        "created_by": from_user,
        "status": "followup",
        "comment": f"Reassigned from {from_user} to {to_user}",
        "next_followup_on": next_dt
    }
    # enrich with client meta (for better trail queries)
    base = _get_itinerary(iid, {"client_name":1,"client_mobile":1,"ach_id":1})
    if base:
        log_doc.update({
            "client_name": base.get("client_name", ""),
            "client_mobile": base.get("client_mobile", ""),
            "ach_id": base.get("ach_id", ""),
        })
    col_followups.insert_one(log_doc)

    # Latest snapshot
    col_updates.update_one(
        {"itinerary_id": str(iid)},
        {"$set": {
            "status": "followup",
            "assigned_to": to_user,
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )

# =========================
# Updaters (admin can credit another user)
# =========================
def upsert_update_status(
    iid: str,
    status: str,               # "followup" | "confirmed" | "cancelled"
    actor_user: str,           # who is saving (e.g., Arpith)
    credit_user: str,          # whose queue/incentive to credit
    next_followup_on: Optional[date],
    booking_date: Optional[date],
    comment: str,
    cancellation_reason: Optional[str],
    advance_amount: Optional[int],
) -> None:
    # 1) Immutable log
    log_doc = {
        "itinerary_id": str(iid),
        "created_at": datetime.utcnow(),
        "created_by": actor_user,             # who actually did it
        "status": status,
        "comment": str(comment or ""),
        "next_followup_on": (
            datetime.combine(next_followup_on, dtime.min) if next_followup_on else None
        ),
        "cancellation_reason": (str(cancellation_reason or "") if status == "cancelled" else ""),
        "credited_to": credit_user            # on-behalf-of
    }
    base = _get_itinerary(iid, {"client_name":1,"client_mobile":1,"ach_id":1})
    if base:
        log_doc.update({
            "client_name": base.get("client_name", ""),
            "client_mobile": base.get("client_mobile", ""),
            "ach_id": base.get("ach_id", ""),
        })
    col_followups.insert_one(log_doc)

    # 2) Latest snapshot (incentive uses FINAL COST)
    final_status = status if status in ("followup", "cancelled") else "confirmed"
    upd = {
        "itinerary_id": str(iid),
        "status": final_status,
        "updated_at": datetime.utcnow(),
    }
    if final_status == "followup":
        upd["assigned_to"] = credit_user
        upd.pop("rep_name", None)
        upd.pop("incentive", None)
        upd.pop("booking_date", None)
        upd.pop("advance_amount", None)
    elif final_status == "confirmed":
        if booking_date:
            upd["booking_date"] = datetime.combine(booking_date, dtime.min)
        if advance_amount is not None:
            upd["advance_amount"] = int(advance_amount)
        final_amt = _final_cost_for(iid)
        upd["incentive"] = _compute_incentive(final_amt)
        upd["rep_name"] = credit_user      # credit confirmation to selected user
        upd["assigned_to"] = None
    elif final_status == "cancelled":
        upd["cancellation_reason"] = str(cancellation_reason or "")
        upd["assigned_to"] = None

    col_updates.update_one({"itinerary_id": str(iid)}, {"$set": upd}, upsert=True)

def save_final_package_cost(iid: str, base_amount: int, discount: int, actor_user: str, credit_user: Optional[str]=None) -> None:
    """
    Persist base, discount, and final cost. Also overwrite legacy 'package_cost' with final for consistency.
    If already confirmed, recompute incentive using the final cost.
    """
    base = _to_int(base_amount)
    disc = _to_int(discount)
    final_cost = max(0, base - disc)

    doc = {
        "itinerary_id": str(iid),
        "base_package_cost": int(base),
        "discount": int(disc),
        "final_package_cost": int(final_cost),
        "package_cost": int(final_cost),  # keep legacy key aligned everywhere
        "saved_at": datetime.utcnow(),
    }
    col_expenses.update_one({"itinerary_id": str(iid)}, {"$set": doc}, upsert=True)

    # If confirmed, refresh incentive (credit to explicit user if provided)
    upd = col_updates.find_one({"itinerary_id": str(iid)}, {"status": 1, "rep_name": 1})
    if upd and upd.get("status") == "confirmed":
        inc = _compute_incentive(int(final_cost))
        rep = credit_user or upd.get("rep_name") or actor_user
        col_updates.update_one(
            {"itinerary_id": str(iid)},
            {"$set": {"incentive": inc, "rep_name": rep}}
        )

# =============================================================================
# UI
# =============================================================================

# Sidebar filters (admin can choose user)
with st.sidebar:
    if is_admin:
        view_user = st.selectbox("Filter follow-ups by user", ["All users"] + ALL_USERS, index=0)
        user_filter = None if view_user == "All users" else view_user
        st.caption("Admin mode: view all or filter by user. You can record updates on behalf of a user.")
    else:
        st.caption("User mode: viewing your assigned follow-ups.")
        view_user = user
        user_filter = user

tabs = st.tabs(["🗂️ Follow-ups", "📘 All packages"])

# =========================
# TAB 1: Follow-ups
# =========================
with tabs[0]:
    df_assigned = fetch_assigned_followups(user_filter)
    if not df_assigned.empty:
        # Optional quick-search
        q = st.text_input("🔎 Search (name / mobile / ACH / route)", "")
        if q.strip():
            s = q.strip().lower()
            df_assigned = df_assigned[
                df_assigned["client_name"].astype(str).str.lower().str.contains(s) |
                df_assigned["client_mobile"].astype(str).str.lower().str.contains(s) |
                df_assigned["ach_id"].astype(str).str.lower().str.contains(s) |
                df_assigned["final_route"].astype(str).str.lower().str.contains(s)
            ]

    itinerary_ids = df_assigned["itinerary_id"].astype(str).tolist() if not df_assigned.empty else []
    latest_map = fetch_latest_followup_log_map(itinerary_ids)

    # derive "next follow-up" & last comment from latest log
    if not df_assigned.empty:
        df_assigned["next_followup_on"] = df_assigned["itinerary_id"].map(
            lambda x: (latest_map.get(str(x), {}) or {}).get("next_followup_on")
        )
        df_assigned["next_followup_on"] = df_assigned["next_followup_on"].apply(
            lambda x: pd.to_datetime(x).date() if pd.notna(x) else None
        )
        df_assigned["last_comment"] = df_assigned["itinerary_id"].map(
            lambda x: (latest_map.get(str(x), {}) or {}).get("comment", "")
        )

    today = _today_utc()
    tmr = today + timedelta(days=1)
    in7 = today + timedelta(days=7)

    total_pkgs = 0 if df_assigned.empty else len(df_assigned)
    due_today = 0 if df_assigned.empty else int((df_assigned["next_followup_on"] == today).sum())
    due_tomorrow = 0 if df_assigned.empty else int((df_assigned["next_followup_on"] == tmr).sum())
    due_week = 0 if df_assigned.empty else int(((df_assigned["next_followup_on"] >= today) &
                                                (df_assigned["next_followup_on"] <= in7)).sum())

    first_this, last_this = month_bounds(today)
    first_last, last_last = month_bounds(first_this - timedelta(days=1))
    this_month_incentive = fetch_confirmed_incentives(user_filter, first_this, last_this)
    last_month_incentive  = fetch_confirmed_incentives(user_filter, first_last, last_last)
    confirmed_this_month  = count_confirmed(user_filter, first_this, last_this)
    confirmed_all_time    = count_confirmed(user_filter)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Assigned follow-ups" if user_filter else "Follow-ups (all users)", total_pkgs)
    c2.metric("Due today", due_today)
    c3.metric("Due tomorrow", due_tomorrow)
    c4.metric("Due in next 7 days", due_week)
    label_inc = "My incentive" if user_filter else "Incentive (all users)"
    c5.metric(label_inc, f"₹ {this_month_incentive:,}", help=f"Last month: ₹ {last_month_incentive:,}")

    c6, c7 = st.columns(2)
    label_conf1 = "Confirmed this month" if user_filter else "Confirmed this month (all users)"
    label_conf2 = "Confirmed (all time)" if user_filter else "Confirmed (all time, all users)"
    c6.metric(label_conf1, confirmed_this_month)
    c7.metric(label_conf2, confirmed_all_time)

    st.divider()

    st.subheader("Follow-ups list")
    if df_assigned.empty:
        st.info("No follow-ups found for the selected filter.")
        st.stop()

    table = df_assigned[[
        "ach_id","client_name","client_mobile","start_date","end_date",
        "final_route","assigned_to","next_followup_on","last_comment","itinerary_id"
    ]].copy().sort_values(["next_followup_on","start_date"], na_position="last")
    table.rename(columns={
        "ach_id":"ACH ID",
        "client_name":"Client",
        "client_mobile":"Mobile",
        "start_date":"Start",
        "end_date":"End",
        "final_route":"Route",
        "assigned_to":"Assigned to",
        "next_followup_on":"Next follow-up",
        "last_comment":"Last comment",
    }, inplace=True)

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(table.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)

        # lightweight CSV export
        if st.button("⬇️ Export current list (CSV)"):
            out = io.StringIO()
            table.drop(columns=["itinerary_id"]).to_csv(out, index=False)
            st.download_button(
                "Download CSV",
                data=out.getvalue().encode("utf-8"),
                file_name=f"followups_{today}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with right:
        options = (table["ACH ID"].fillna("").astype(str) + " | " +
                   table["Client"].fillna("") + " | " +
                   table["Mobile"].fillna("") + " | " +
                   table["itinerary_id"])
        sel = st.selectbox("Open client", options.tolist())
        chosen_id = sel.split(" | ")[-1] if sel else None

    if not chosen_id:
        st.stop()

    st.divider()
    st.subheader("Details & Update")

    it_doc = _get_itinerary(chosen_id) or {}
    upd_doc = col_updates.find_one({"itinerary_id": str(chosen_id)}, {"_id": 0}) or {}

    # Created timestamps (nice to have)
    try:
        created_dt_utc = ObjectId(str(chosen_id)).generation_time
    except Exception:
        created_dt_utc = None
    created_ist_str = _fmt_ist(created_dt_utc)
    created_utc_str = created_dt_utc.strftime("%Y-%m-%d %H:%M %Z") if created_dt_utc else ""

    dc1, dc2 = st.columns([1, 1])
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

    # ---- Reassign control (admin only) ----
    if is_admin:
        st.markdown("### Reassign this follow-up")
        current_assignee = upd_doc.get("assigned_to")
        candidates = [u for u in ALL_USERS if u != current_assignee] or ALL_USERS
        to_user = st.selectbox("Move to user", candidates, key="reassign_to")
        if st.button("➡️ Reassign now"):
            try:
                reassign_followup(chosen_id, from_user=user, to_user=to_user)
                st.success(f"Moved to {to_user}.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not reassign: {e}")

    # ---- Final Package Cost editor (base − discount) ----
    st.markdown("### Final package cost")

    exp_doc = col_expenses.find_one(
        {"itinerary_id": str(chosen_id)},
        {"base_package_cost":1, "discount":1, "final_package_cost":1, "package_cost":1}
    ) or {}
    base_default = _to_int(exp_doc.get("base_package_cost", 0)) or _to_int(it_doc.get("package_cost", 0))
    disc_default = _to_int(exp_doc.get("discount", 0))

    if is_admin:
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
            save_final_package_cost(chosen_id, int(base_amount), int(discount), actor_user=user, credit_user=credit_user_cost)
            st.success("Final package cost saved. Incentive updated if already confirmed.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not save package cost: {e}")

    with st.expander("Show full itinerary text"):
        st.text_area("Itinerary shared with client",
                    value=it_doc.get("itinerary_text",""), height=260, disabled=True)

    # Follow-up trail
    st.markdown("### Follow-up trail")
    trail = list(col_followups.find({"itinerary_id": str(chosen_id)}).sort("created_at", -1))
    if trail:
        df_trail = pd.DataFrame([{
            "When (UTC)": t.get("created_at"),
            "When (IST)": _fmt_ist(_clean_dt(t.get("created_at"))),
            "By": t.get("created_by"),
            "Credited to": t.get("credited_to", ""),
            "Status": t.get("status"),
            "Next follow-up": (
                pd.to_datetime(t.get("next_followup_on")).date()
                if t.get("next_followup_on") else None
            ),
            "Comment": t.get("comment",""),
            "Cancel reason": t.get("cancellation_reason",""),
        } for t in trail])
        st.dataframe(df_trail, use_container_width=True, hide_index=True)

        # Export trail
        if st.button("⬇️ Export trail (CSV)"):
            out2 = io.StringIO()
            df_trail.to_csv(out2, index=False)
            st.download_button(
                "Download CSV",
                data=out2.getvalue().encode("utf-8"),
                file_name=f"followup_trail_{chosen_id}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.caption("No follow-up logs yet for this client.")

    st.markdown("---")
    st.markdown("### Add follow-up update")

    # Admin can record on behalf of any user
    if is_admin and ALL_USERS:
        default_idx = ALL_USERS.index(upd_doc.get("assigned_to")) if upd_doc.get("assigned_to") in ALL_USERS else 0
        record_as = st.selectbox("Record this update on behalf of", ALL_USERS, index=default_idx)
    else:
        record_as = user

    with st.form("followup_form"):
        status_choice = st.selectbox("Status", ["followup required", "confirmed", "cancelled"])
        comment = st.text_area("Comment", placeholder="Write your update…")

        next_date = None
        cancel_reason = None
        booking_date = None
        advance_amt = None

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
        st.rerun()

# =========================
# TAB 2: All packages (admin overview)
# =========================
with tabs[1]:
    if is_admin:
        filter_user_all = st.selectbox("Filter packages by user (assigned/credited)", ["All users"] + ALL_USERS, index=0)
        fu = None if filter_user_all == "All users" else filter_user_all
    else:
        st.caption("Showing packages related to you (assigned to you or credited to you).")
        fu = user

    df_all = fetch_packages_with_updates(fu)
    if df_all.empty:
        st.info("No packages to display for the selected filter.")
    else:
        # quick search
        q2 = st.text_input("Search in packages (name / mobile / ACH / route)", "")
        if q2.strip():
            s2 = q2.strip().lower()
            df_all = df_all[
                df_all["client_name"].astype(str).str.lower().str.contains(s2) |
                df_all["client_mobile"].astype(str).str.lower().str.contains(s2) |
                df_all["ach_id"].astype(str).str.lower().str.contains(s2) |
                df_all["final_route"].astype(str).str.lower().str.contains(s2)
            ]

        view = df_all[[
            "ach_id","client_name","client_mobile","final_route","total_pax",
            "start_date","end_date","status","assigned_to","rep_name","booking_date","advance_amount","incentive","itinerary_id"
        ]].copy()
        view.rename(columns={
            "ach_id":"ACH ID","client_name":"Client","client_mobile":"Mobile","final_route":"Route","total_pax":"Pax",
            "start_date":"Start","end_date":"End","assigned_to":"Assigned to","rep_name":"Rep (credited)","booking_date":"Booking date"
        }, inplace=True)
        st.dataframe(
            view.sort_values(["Booking date","Start","Client"], na_position="last").drop(columns=["itinerary_id"]),
            use_container_width=True, hide_index=True
        )

        # export
        if st.button("⬇️ Export table (CSV)"):
            out3 = io.StringIO()
            view.drop(columns=["itinerary_id"]).to_csv(out3, index=False)
            st.download_button(
                "Download CSV",
                data=out3.getvalue().encode("utf-8"),
                file_name=f"packages_{_today_utc()}.csv",
                mime="text/csv",
                use_container_width=True
            )
