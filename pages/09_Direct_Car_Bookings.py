from __future__ import annotations
from datetime import datetime, date
from typing import Optional
import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from bson import ObjectId  # 👈 for deletes

# -------------------------------------------------
# Page
# -------------------------------------------------
st.set_page_config(page_title="Direct Car Bookings", layout="wide")
st.title("🚖 Direct Car Bookings (Cash / Employee Collection)")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def safe_dt_from_date(d: date) -> datetime:
    """Convert a python date into a naive datetime at 00:00:00."""
    return datetime.combine(d, datetime.min.time())

def month_bounds(any_day: date) -> tuple[datetime, datetime]:
    """Return [start_dt, end_dt) datetimes for the month of any_day."""
    start_d = any_day.replace(day=1)
    if start_d.month == 12:
        next_month_d = start_d.replace(year=start_d.year + 1, month=1, day=1)
    else:
        next_month_d = start_d.replace(month=start_d.month + 1, day=1)
    return safe_dt_from_date(start_d), safe_dt_from_date(next_month_d)

def to_display_date(x) -> str:
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, date):
        return x.strftime("%Y-%m-%d")
    return str(x or "")

def _as_oid(s: str) -> Optional[ObjectId]:
    try:
        return ObjectId(s)
    except Exception:
        return None

# -------------------------------------------------
# Mongo Connection
# -------------------------------------------------
CAND_KEYS = ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]

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
def _get_client() -> Optional[MongoClient]:
    uri = _find_uri()
    if not uri:
        return None
    client = MongoClient(uri, appName="TAK_DirectCars", tz_aware=True)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        return None
    return client

# -------------------------------------------------
# Users & Login (same style as other pages)
# -------------------------------------------------
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    # fallback for dev
    try:
        try:
            import tomllib
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, dict):
            with st.sidebar:
                st.warning("Using users from repo .streamlit/secrets.toml. For production, set them in Manage app → Secrets.")
            return u
    except Exception:
        pass
    return {}

def all_employees() -> list[str]:
    return sorted(load_users().keys())

def _login() -> Optional[str]:
    # if logged-in already
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
        st.sidebar.error("⚠️ Login is not configured in Secrets.")
        st.stop()

    st.sidebar.markdown("### 🔐 Login")
    name = st.sidebar.selectbox("User", list(users_map.keys()), key="login_user")
    pin = st.sidebar.text_input("PIN", type="password", key="login_pin")
    if st.sidebar.button("Sign in"):
        if str(users_map.get(name, "")).strip() == str(pin).strip():
            st.session_state["user"] = name
            st.rerun()
        else:
            st.sidebar.error("Invalid PIN")
            st.stop()
    return None

user = _login()
if not user:
    st.stop()

client = _get_client()
if not client:
    st.error("❌ Could not connect to MongoDB. Check your URI in Secrets.")
    st.stop()

db = client["TAK_DB"]
col_cars = db["direct_car_bookings"]
col_split = db["expense_splitwise"]

# -------------------------------------------------
# Manual refresh (avoid auto-refreshing)
# -------------------------------------------------
with st.sidebar:
    if st.button("🔄 Refresh tables"):
        st.session_state["refresh_now"] = True
    else:
        st.session_state["refresh_now"] = st.session_state.get("refresh_now", False)

# -------------------------------------------------
# Add Booking Form
# -------------------------------------------------
st.subheader("➕ Add Direct Car Booking")

with st.form("car_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        when = st.date_input("Date", value=date.today())
    with c2:
        client_name = st.text_input("Client Name (optional)")
    with c3:
        trip_plan = st.text_input("Trip Plan (optional)")

    c4, c5, c6 = st.columns(3)
    with c4:
        amount = st.number_input("Amount (₹)", min_value=0, step=500)
    with c5:
        car_type = st.selectbox("Car Type", ["Sedan", "Ertiga"])
    with c6:
        recv_in = st.radio("Received In", ["Company Account", "Personal Account"], horizontal=True)

    emp_list: list[str] = []
    if recv_in == "Personal Account":
        emp_list = st.multiselect("Payment received by (employee(s))", all_employees())

    notes = st.text_area("Notes", placeholder="Any remarks…")
    submitted = st.form_submit_button("💾 Save booking", use_container_width=True)

if submitted:
    if amount <= 0:
        st.error("Amount must be > 0")
    elif recv_in == "Personal Account" and not emp_list:
        st.error("Please select at least one employee")
    else:
        # Insert booking (sanitize ALL types)
        safe_doc = {
            "date": safe_dt_from_date(when),          # datetime
            "client_name": str(client_name or ""),
            "trip_plan": str(trip_plan or ""),
            "amount": int(amount),
            "car_type": str(car_type),
            "received_in": str(recv_in),
            "employees": [str(e) for e in (emp_list or [])],
            "notes": str(notes or ""),
            "created_by": str(user),
            "created_at": datetime.utcnow(),
        }
        ins = col_cars.insert_one(safe_doc)
        booking_id = ins.inserted_id  # 👈 for linking settlements

        # If money received in personal account, log settlements that reduce outstanding
        if recv_in == "Personal Account" and emp_list:
            per_emp_amt = int(amount / len(emp_list))
            for emp in emp_list:
                col_split.insert_one({
                    "kind": "settlement",
                    "created_at": datetime.utcnow(),
                    "created_by": str(user),
                    "date": safe_dt_from_date(when),
                    "employee": str(emp),
                    "amount": int(per_emp_amt),
                    "ref": f"Direct Car ({car_type})",
                    "notes": f"Direct booking for {client_name or 'N/A'}",
                    "dc_booking_id": str(booking_id),  # 👈 link to allow safe deletion
                })

        st.success("✅ Booking saved successfully")
        # Do not rerun automatically; keep page stable

# -------------------------------------------------
# Delete utilities
# -------------------------------------------------
def delete_booking_and_linked_settlements(booking_id_str: str) -> tuple[int, int]:
    """
    Deletes the booking and any settlements created from it.
    Returns: (bookings_deleted, settlements_deleted)
    """
    oid = _as_oid(booking_id_str)
    if not oid:
        return (0, 0)

    # fetch booking (for UI/reporting or fallback heuristics if needed)
    booking = col_cars.find_one({"_id": oid})
    if not booking:
        return (0, 0)

    # delete linked settlements via dc_booking_id
    res_set = col_split.delete_many({"kind": "settlement", "dc_booking_id": str(oid)})

    # delete the booking itself
    res_book = col_cars.delete_one({"_id": oid})

    return (res_book.deleted_count or 0, res_set.deleted_count or 0)

# -------------------------------------------------
# Recent Bookings (load only when asked OR after a save)
# -------------------------------------------------
st.subheader("📜 Recent Direct Car Bookings")
if st.session_state.get("refresh_now", False) or submitted:
    recent_docs = list(col_cars.find().sort("date", -1).limit(20))
else:
    # first render: light query (still shows something without flicker)
    recent_docs = list(col_cars.find().sort("date", -1).limit(20))

if recent_docs:
    # ---- Manage/Delete card list (with confirm) ----
    st.markdown("#### 🗑️ Delete / Undo (Recent)")
    for d in recent_docs:
        vid = str(d.get("_id", ""))
        vdate = to_display_date(d.get("date"))
        vclient = d.get("client_name", "") or "—"
        vamt = int(d.get("amount", 0) or 0)
        vrcv = d.get("received_in", "")
        vemp = ", ".join(d.get("employees", []) or [])
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.2, 1.2])
            c1.write(f"**Date:** {vdate}")
            c2.write(f"**Client:** {vclient}")
            c3.write(f"**Amount:** ₹{vamt:,}")
            c4.write(f"**Received In:** {vrcv}")
            st.caption(f"Employees: {vemp or '—'}  |  _id: {vid}")

            cc1, cc2 = st.columns([0.35, 0.65])
            with cc1:
                confirm = st.checkbox("Confirm", key=f"del_confirm_{vid}")
            with cc2:
                if st.button("🗑️ Delete this booking", key=f"del_btn_{vid}", use_container_width=True, disabled=not confirm):
                    bdel, sdel = delete_booking_and_linked_settlements(vid)
                    if bdel:
                        st.success(f"Deleted booking {vid}. Linked settlements removed: {sdel}.")
                        st.session_state["refresh_now"] = True
                        st.rerun()
                    else:
                        st.warning("Nothing deleted (booking not found).")

    # ---- Simple table view below ----
    for d in recent_docs:
        d["_id"] = str(d.get("_id", ""))
        d["date"] = to_display_date(d.get("date"))
    df_recent = pd.DataFrame(recent_docs)
    cols_recent = ["date", "client_name", "trip_plan", "amount", "car_type",
                   "received_in", "employees", "notes"]
    for c in cols_recent:
        if c not in df_recent:
            df_recent[c] = ""
    st.dataframe(df_recent[cols_recent], use_container_width=True, hide_index=True)
else:
    st.info("No bookings yet.")

# -------------------------------------------------
# Monthly Report (with totals)
# -------------------------------------------------
st.subheader("📅 Monthly Car Bookings Report")
sel_month = st.date_input("Select Month", value=date.today())

if sel_month:
    start_dt, end_dt = month_bounds(sel_month)  # both datetime -> BSON-safe
    # query strictly with datetimes to avoid InvalidDocument
    cursor = col_cars.find({"date": {"$gte": start_dt, "$lt": end_dt}}).sort("date", 1)
    month_rows = list(cursor)

    if month_rows:
        # normalize for display
        for r in month_rows:
            r["_id"] = str(r.get("_id", ""))
            r["date"] = to_display_date(r.get("date"))

        dfm = pd.DataFrame(month_rows)
        dfm.index = dfm.index + 1
        dfm.rename_axis("Sr No", inplace=True)

        show_cols = ["date", "car_type", "client_name", "trip_plan", "amount", "received_in", "employees", "notes"]
        for c in show_cols:
            if c not in dfm:
                dfm[c] = ""

        st.dataframe(dfm[show_cols], use_container_width=True)

        # Totals
        total_all = int(pd.to_numeric(dfm["amount"], errors="coerce").fillna(0).sum())
        total_bank = int(pd.to_numeric(dfm.loc[dfm["received_in"] == "Company Account", "amount"], errors="coerce").fillna(0).sum())
        total_personal = int(pd.to_numeric(dfm.loc[dfm["received_in"] == "Personal Account", "amount"], errors="coerce").fillna(0).sum())

        st.markdown(
            f"**Total this month (Package Cost): ₹{total_all:,}**  \n"
            f"🏦 Cash received in **Bank/Company**: ₹{total_bank:,}  \n"
            f"👤 Cash received in **Personal**: ₹{total_personal:,}"
        )

        # --- Quick delete (picker) ---
        st.markdown("#### 🗑️ Quick delete (this month)")
        del_options = [
            f"{row['_id']} — {row['date']} | {row.get('client_name','')} | ₹{int(row.get('amount',0)):,} | {row.get('received_in','')}"
            for _, row in dfm.iterrows()
        ]
        pick = st.selectbox("Pick a booking to delete", del_options, index=0) if del_options else None
        if pick:
            bid = pick.split(" — ")[0]
            c1, c2 = st.columns([0.3, 0.7])
            with c1:
                confirm_mon = st.checkbox("Confirm", key=f"mon_del_confirm_{bid}")
            with c2:
                if st.button("🗑️ Delete selected booking", disabled=not confirm_mon, key=f"mon_del_btn_{bid}"):
                    bdel, sdel = delete_booking_and_linked_settlements(bid)
                    if bdel:
                        st.success(f"Deleted booking {bid}. Linked settlements removed: {sdel}.")
                        st.session_state["refresh_now"] = True
                        st.rerun()
                    else:
                        st.warning("Nothing deleted (booking not found).")
    else:
        st.info("No bookings for this month.")
