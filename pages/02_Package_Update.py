# pages/02_Package_Update.py
from __future__ import annotations

import math, os, io
from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, List, Dict

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# ------------------------------------------------------------------
# Access guard
# ------------------------------------------------------------------
if st.session_state.get("user") in ("Teena", "Kuldeep"):
    st.stop()

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Package Update", layout="wide")
st.title("📦 Package Update (Admin)")

# Optional calendar
CALENDAR_AVAILABLE = True
try:
    from streamlit_calendar import calendar
except Exception:
    CALENDAR_AVAILABLE = False

# ------------------------------------------------------------------
# Admin-only gate
# ------------------------------------------------------------------
def require_admin():
    ADMIN_PASS_DEFAULT = "Arpith&92"
    ADMIN_PASS = str(st.secrets.get("admin_pass", ADMIN_PASS_DEFAULT))
    with st.sidebar:
        st.markdown("### Admin access")
        p = st.text_input("Enter admin password", type="password", placeholder="enter pass")
    if (p or "").strip() != ADMIN_PASS.strip():
        st.stop()
    st.session_state["user"] = "Admin"
    st.session_state["is_admin"] = True

require_admin()

# ------------------------------------------------------------------
# MongoDB (fast + cached)
# ------------------------------------------------------------------
CAND_KEYS = ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]
IST = ZoneInfo("Asia/Kolkata")

def _find_uri() -> Optional[str]:
    # Prefer Secrets; fallback to env
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
    uri = _find_uri() or st.secrets["mongo_uri"]
    client = MongoClient(
        uri,
        appName="TAK_PackageUpdate",
        maxPoolSize=100,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        retryWrites=True,
        tz_aware=True,
    )
    client.admin.command("ping")
    return client

db = _get_client()["TAK_DB"]
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_expenses    = db["expenses"]
col_vendorpay   = db["vendor_payments"]
col_vendors     = db["vendors"]
col_followups   = db.get("followups")  # optional; not required here

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _to_dt_or_none(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        ts = pd.to_datetime(x)
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, date):
            return datetime.combine(ts, dtime.min)
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None

def _to_int(x, default=0):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return default
        s = str(x).replace(",", "")
        return int(round(float(s)))
    except Exception:
        return default

def _clean_for_mongo(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool, bytes)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, datetime):
        return obj
    if isinstance(obj, date):
        return datetime.combine(obj, dtime.min)
    if isinstance(obj, dict):
        return {str(k): _clean_for_mongo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean_for_mongo(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    return str(obj)

def _str_or_blank(x):
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return "" if x is None else str(x)

def _created_utc(iid: str) -> datetime | None:
    try:
        return ObjectId(str(iid)).generation_time  # tz-aware UTC
    except Exception:
        return None

def _fmt_ist(dt: datetime | None) -> str:
    if not dt:
        return ""
    try:
        return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M UTC")

def to_date_or_none(x):
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

# ------------------------------------------------------------------
# Cached loaders (projections only)
# ------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_itineraries_df() -> pd.DataFrame:
    rows = list(col_itineraries.find(
        {},
        {
            "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
            "representative": 1, "final_route": 1, "total_pax": 1,
            "start_date": 1, "end_date": 1, "upload_date": 1,
            "package_cost": 1, "discount": 1, "itinerary_text": 1
        }
    ))
    if not rows:
        return pd.DataFrame()
    for r in rows:
        r["itinerary_id"] = str(r.get("_id"))
        r["ach_id"] = r.get("ach_id", "")
        for k in ("start_date", "end_date", "upload_date"):
            v = r.get(k)
            r[k] = pd.to_datetime(v).to_pydatetime() if v else None
        r["package_cost_num"] = _to_int(r.get("package_cost"))
        r["client_mobile"] = r.get("client_mobile", "")
        r["client_name"] = r.get("client_name", "")
        r["representative"] = r.get("representative", "")
        r["final_route"] = r.get("final_route", "")
        r["total_pax"] = _to_int(r.get("total_pax", 0))
        r["_id"] = None
    return pd.DataFrame(rows)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_updates_df() -> pd.DataFrame:
    rows = list(col_updates.find(
        {},
        {"_id": 0, "itinerary_id": 1, "status": 1, "booking_date": 1,
         "advance_amount": 1, "assigned_to": 1, "incentive": 1, "rep_name": 1}
    ))
    if not rows:
        return pd.DataFrame(columns=["itinerary_id","status","booking_date","advance_amount","assigned_to","incentive","rep_name"])
    for r in rows:
        r["itinerary_id"] = str(r.get("itinerary_id"))
        r["status"] = r.get("status", "pending")
        r["booking_date"] = to_date_or_none(r.get("booking_date"))
        r["advance_amount"] = _to_int(r.get("advance_amount", 0))
        r["assigned_to"] = r.get("assigned_to", "")
        r["incentive"] = _to_int(r.get("incentive", 0))
        r["rep_name"] = r.get("rep_name", "")
    return pd.DataFrame(rows)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_expenses_df() -> pd.DataFrame:
    rows = list(col_expenses.find(
        {},
        {"_id": 0, "itinerary_id": 1, "base_package_cost": 1, "discount": 1,
         "final_package_cost": 1, "package_cost": 1, "total_expenses": 1, "profit": 1}
    ))
    if not rows:
        return pd.DataFrame(columns=["itinerary_id","total_expenses","profit","package_cost","base_package_cost","discount","final_package_cost"])
    for r in rows:
        r["itinerary_id"] = str(r.get("itinerary_id"))
        r["base_package_cost"] = _to_int(r.get("base_package_cost", 0))
        r["discount"] = _to_int(r.get("discount", 0))
        r["final_package_cost"] = _to_int(r.get("final_package_cost", r.get("package_cost", 0)))
        r["total_expenses"] = _to_int(r.get("total_expenses", 0))
        r["profit"] = _to_int(r.get("profit", 0))
    return pd.DataFrame(rows)

@st.cache_data(ttl=120, show_spinner=False)
def list_vendor_names(category: str) -> List[str]:
    names = [v.get("name","").strip() for v in col_vendors.find({"category": category}, {"name":1}, sort=[("name", 1)])]
    return [n for n in names if n]

def create_vendor(name: str, city: str, category: str) -> bool:
    name = (name or "").strip()
    city = (city or "").strip()
    category = (category or "").strip()
    if not name or not category:
        return False
    doc = {
        "name": name,
        "city": city,
        "category": category,
        "created_at": datetime.utcnow(),
    }
    try:
        col_vendors.update_one(
            {"name": name, "category": category},
            {"$setOnInsert": _clean_for_mongo(doc)},
            upsert=True
        )
        # bust cache for this category
        fetch_vendors_cache_clear(category)
        return True
    except Exception:
        return False

def fetch_vendors_cache_clear(category: str):
    list_vendor_names.clear()  # type: ignore

# ------------------------------------------------------------------
# Dedupe logic (to align with Dashboard when needed)
# ------------------------------------------------------------------
def group_latest_by_mobile(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    df_all = df_all.copy()
    df_all["upload_date"] = pd.to_datetime(df_all["upload_date"])
    df_all.sort_values(["client_mobile","upload_date"], ascending=[True, False], inplace=True)
    latest_rows = df_all.groupby("client_mobile", as_index=False).first()
    hist_map: Dict[str, List[str]] = {}
    for mob, grp in df_all.groupby("client_mobile"):
        ids = grp["itinerary_id"].tolist()
        hist_map[mob] = ids[1:] if len(ids) > 1 else []
    latest_rows["history_ids"] = latest_rows["client_mobile"].map(hist_map).apply(lambda x: x or [])
    return latest_rows

# ------------------------------------------------------------------
# Build page data (vectorized merges + final_cost)
# ------------------------------------------------------------------
df_it = fetch_itineraries_df()
if df_it.empty:
    st.info("No packages found yet. Upload a file in the main app first.")
    st.stop()

df_up  = fetch_updates_df()
df_exp = fetch_expenses_df()

df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")

# numeric & dates
if "advance_amount" not in df.columns:
    df["advance_amount"] = 0
df["advance_amount"] = pd.to_numeric(df["advance_amount"], errors="coerce").fillna(0).astype(int)
for col_ in ["start_date", "end_date", "booking_date", "upload_date"]:
    if col_ in df.columns:
        df[col_] = df[col_].apply(to_date_or_none)

# attach expenses (for final cost, profit)
df = df.merge(
    df_exp[["itinerary_id","base_package_cost","discount","final_package_cost","total_expenses","profit"]],
    on="itinerary_id", how="left"
)
# vectorized final cost (prefer explicit, else compute, else itinerary)
df["final_cost"] = pd.to_numeric(df["final_package_cost"], errors="coerce").fillna(0).astype(int)
need_comp = df["final_cost"].eq(0)
if need_comp.any():
    base = pd.to_numeric(df["base_package_cost"], errors="coerce").fillna(0).astype(int)
    disc = pd.to_numeric(df["discount"], errors="coerce").fillna(0).astype(int)
    comp = (base - disc).clip(lower=0)
    it_base = pd.to_numeric(df["package_cost_num"], errors="coerce").fillna(0).astype(int)
    it_disc = pd.to_numeric(df["discount"], errors="coerce").fillna(0).astype(int)
    comp2 = (it_base - it_disc).clip(lower=0)
    df["final_cost"] = df["final_cost"].mask(df["final_cost"].eq(0), comp).mask(df["final_cost"].eq(0), comp2)
df["total_expenses"] = pd.to_numeric(df["total_expenses"], errors="coerce").fillna(0).astype(int)
df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(df["final_cost"] - df["total_expenses"]).astype(int)

# ------------------------------------------------------------------
# 🔧 Consistency with Dashboard
# ------------------------------------------------------------------
with st.sidebar:
    match_dash = st.checkbox("Match Dashboard (unique by mobile for KPIs/lists)", value=True,
                             help="Enable to dedupe by latest package per mobile (like Dashboard default).")

# ------------------------------------------------------------------
# 💸 Vendor dues summary (cached scan)
# ------------------------------------------------------------------
def vendor_dues_summary():
    from collections import defaultdict
    sums = defaultdict(int)
    counts = defaultdict(int)
    docs = col_vendorpay.find({}, {"_id":0, "items":1})
    for d in docs:
        for it in d.get("items", []):
            vendor = (it.get("vendor") or "").strip()
            if not vendor:
                continue
            bal = it.get("balance")
            if bal is None:
                bal = _to_int(it.get("finalization_cost", 0)) - (_to_int(it.get("adv1_amt", 0)) + _to_int(it.get("adv2_amt", 0)) + _to_int(it.get("final_amt", 0)))
            bal = max(_to_int(bal), 0)
            if bal <= 0:
                continue
            sums[vendor] += bal
            counts[vendor] += 1

    rows = []
    for vendor, bal in sorted(sums.items(), key=lambda x: -x[1]):
        vdoc = col_vendors.find_one({"name": vendor}, {"category":1, "city":1}) or {}
        rows.append({
            "Vendor": vendor,
            "Category": vdoc.get("category", ""),
            "City": vdoc.get("city", ""),
            "Bookings": counts[vendor],
            "Balance (₹)": int(bal),
        })
    dfv = pd.DataFrame(rows)
    total_pending = int(sum(sums.values()))
    num_vendors = len(sums)
    return dfv, total_pending, num_vendors

df_dues, total_dues, vendors_with_dues = vendor_dues_summary()
d1, d2 = st.columns(2)
d1.metric("💸 Vendor payouts pending (₹)", f"{total_dues:,}")
d2.metric("🏷️ Vendors with dues", vendors_with_dues)
if not df_dues.empty:
    st.dataframe(df_dues, use_container_width=True, hide_index=True)
st.divider()

# ------------------------------------------------------------------
# 🧾 Vendor Directory
# ------------------------------------------------------------------
VENDOR_CATEGORIES = ["Bhasmarathi", "Car", "Hotel", "Poojan", "Others"]

st.subheader("🧾 Vendor Directory")
with st.expander("➕ Add vendor"):
    v1, v2, v3, v4 = st.columns([1.4, 1, 1.2, 1.0])
    with v1:
        new_vendor_name = st.text_input("Vendor name")
    with v2:
        new_vendor_city = st.text_input("City")
    with v3:
        new_vendor_cat = st.selectbox("Nature of service", VENDOR_CATEGORIES, index=2)
    with v4:
        st.caption(" ")
        if st.button("Save vendor"):
            ok = create_vendor(new_vendor_name, new_vendor_city, new_vendor_cat)
            if ok:
                st.success("Vendor saved.")
                st.rerun()
            else:
                st.warning("Please enter at least vendor name and category.")

with st.expander("📋 Current vendors (by category)"):
    tabs = st.tabs(VENDOR_CATEGORIES)
    for i, cat in enumerate(VENDOR_CATEGORIES):
        with tabs[i]:
            opt = list_vendor_names(cat)
            df_v = pd.DataFrame({"Vendor": opt}) if opt else pd.DataFrame({"Vendor": []})
            st.dataframe(df_v, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------
# Summary KPIs (optionally unique by mobile)
# ------------------------------------------------------------------
df_for_counts = group_latest_by_mobile(df) if match_dash else df.copy()

pending_count           = int((df_for_counts["status"] == "pending").sum())
under_discussion_count  = int((df_for_counts["status"] == "under_discussion").sum())
followup_count          = int((df_for_counts["status"] == "followup").sum())
cancelled_count         = int((df_for_counts["status"] == "cancelled").sum())
confirmed_count         = int((df_for_counts["status"] == "confirmed").sum())

# Confirmed – expense pending (unique mobile logic aligned with df_for_counts)
have_expense_ids = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
confirmed_latest = df_for_counts[df_for_counts["status"] == "confirmed"].copy()
confirmed_expense_pending = confirmed_latest[~confirmed_latest["itinerary_id"].isin(have_expense_ids)].shape[0]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("✅ Confirmed", confirmed_count)
k2.metric("🟡 Pending", int(pending_count))
k3.metric("🟠 Under discussion", int(under_discussion_count))
k4.metric("🔵 Follow-up", int(followup_count))
k5.metric("🔴 Cancelled", int(cancelled_count))

st.caption("Tip: Turn ON 'Match Dashboard' in the sidebar to make counts identical to Dashboard’s default view.")
st.divider()

# ------------------------------------------------------------------
# 🗂️ Package by timeline (with admin delete)
# ------------------------------------------------------------------
st.subheader("🗂️ Package by timeline")
with st.expander("Show recently created (last 50)"):
    created_df = df[["itinerary_id","ach_id","client_name","client_mobile","representative","upload_date"]].copy()
    created_df["created_utc"] = created_df["itinerary_id"].apply(_created_utc)
    created_df = created_df.dropna(subset=["created_utc"]).sort_values("created_utc", ascending=False).head(50)
    created_df["Created (IST)"] = created_df["created_utc"].apply(_fmt_ist)
    created_df["Created (UTC)"] = created_df["created_utc"].apply(lambda d: d.strftime("%Y-%m-%d %H:%M %Z"))
    created_df["delete"] = False
    show_cols = ["delete","ach_id","client_name","client_mobile","representative","Created (IST)","Created (UTC)","itinerary_id"]
    created_view = created_df[show_cols].rename(columns={"itinerary_id":"_itinerary_id"})
    edited = st.data_editor(
        created_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "delete": st.column_config.CheckboxColumn("Select"),
            "_itinerary_id": st.column_config.TextColumn("Itinerary ID", disabled=True),
        },
        key="created_timeline_editor",
    )

    if st.button("🗑️ Delete selected package(s)"):
        to_del = edited[edited["delete"] == True]
        if to_del.empty:
            st.warning("No rows selected for deletion.")
        else:
            st.session_state["_pending_delete_rows"] = to_del[["_itinerary_id","client_name"]].to_dict("records")
            st.rerun()

if "_pending_delete_rows" in st.session_state and st.session_state["_pending_delete_rows"]:
    rows = st.session_state["_pending_delete_rows"]
    names = ", ".join([f"{r['client_name']}" for r in rows])
    st.warning(
        f"Confirm delete for package(s): **{names}** ? This will remove entries from "
        "*itineraries*, *package_updates*, *expenses*, and *vendor_payments*.",
        icon="⚠️"
    )
    cdel1, cdel2 = st.columns([1,1])
    with cdel1:
        if st.button("✅ Yes, delete now"):
            deleted = 0
            for r in rows:
                iid = r["_itinerary_id"]
                try:
                    col_itineraries.delete_one({"_id": ObjectId(iid)})
                except Exception:
                    col_itineraries.delete_one({"itinerary_id": iid})
                col_updates.delete_many({"itinerary_id": iid})
                col_expenses.delete_many({"itinerary_id": iid})
                col_vendorpay.delete_many({"itinerary_id": iid})
                deleted += 1
            st.session_state["_pending_delete_rows"] = []
            st.success(f"Deleted {deleted} package(s).")
            st.rerun()
    with cdel2:
        if st.button("❌ Cancel"):
            st.session_state["_pending_delete_rows"] = []
            st.info("Deletion cancelled.")
            st.rerun()

st.divider()

# ------------------------------------------------------------------
# 1) Status Update (NO follow-ups here)
# ------------------------------------------------------------------
st.subheader("1) Update Status")

view_mode = st.radio("View mode", ["Latest per client (by mobile)", "All packages"], horizontal=True)
editable = group_latest_by_mobile(df) if view_mode == "Latest per client (by mobile)" else df.copy()

# Only show pipeline items
editable = editable[editable["status"].isin(["pending","under_discussion"])].copy()

if editable.empty:
    st.success("Nothing to update right now. 🎉")
else:
    must_cols = [
        "ach_id","itinerary_id","client_name","client_mobile","final_route","total_pax",
        "start_date","end_date","package_cost_num","status","booking_date","advance_amount","assigned_to"
    ]
    for c in must_cols:
        if c not in editable.columns:
            editable[c] = None

    for c in ["ach_id","itinerary_id","client_name","final_route","client_mobile"]:
        editable[c] = editable[c].astype(str).fillna("")
    editable["package_cost"] = editable.get("package_cost_num", 0)
    editable["total_pax"] = pd.to_numeric(editable["total_pax"], errors="coerce").fillna(0).astype(int)
    editable["advance_amount"] = pd.to_numeric(editable["advance_amount"], errors="coerce").fillna(0).astype(int)
    for c in ["start_date","end_date","booking_date"]:
        editable[c] = editable[c].apply(to_date_or_none)

    if "assigned_to" not in editable.columns:
        editable["assigned_to"] = ""
    editable["assigned_to"] = editable["assigned_to"].apply(_str_or_blank)

    if "select" not in editable.columns:
        editable.insert(0, "select", False)

    show_cols = [
        "select",
        "ach_id","itinerary_id","client_name","client_mobile","final_route","total_pax",
        "start_date","end_date","package_cost","status","booking_date","advance_amount","assigned_to"
    ]
    editable = editable[show_cols].sort_values(["start_date","client_name"], na_position="last")

    st.caption("Tick **Select** on the rows you want, then use **Bulk update** below to change Status and/or Assign To.")

    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "status": st.column_config.SelectboxColumn(
                "Status", options=["pending","under_discussion","followup","confirmed","cancelled"]
            ),
            "assigned_to": st.column_config.SelectboxColumn(
                "Assign To", options=["", "Arpith","Reena","Teena","Kuldeep"]
            ),
            "booking_date": st.column_config.DateColumn("Booking date", format="YYYY-MM-DD"),
            "advance_amount": st.column_config.NumberColumn("Advance (₹)", min_value=0, step=500),
        },
        key="status_editor_v2"
    )

    with st.expander("🔁 Bulk update selected rows"):
        b1, b2, b3, b4 = st.columns([1,1,1,1])
        with b1:
            bulk_status = st.selectbox("Set Status", ["pending","under_discussion","followup","confirmed","cancelled"])
        with b2:
            bulk_assignee = st.selectbox("Assign To (for follow-up)", ["", "Arpith","Reena","Teena","Kuldeep"])
        with b3:
            bulk_date = st.date_input("Booking date (for confirmed)", value=None)
        with b4:
            bulk_adv = st.number_input("Advance (₹)", min_value=0, step=500, value=0)

        if st.button("Apply to selected"):
            sel_df = edited[edited["select"] == True]
            if sel_df.empty:
                st.warning("No rows ticked.")
            else:
                applied, skipped = 0, 0
                for _, r in sel_df.iterrows():
                    bdate = None
                    if bulk_status == "confirmed":
                        if not bulk_date:
                            skipped += 1
                            continue
                        bdate = pd.to_datetime(bulk_date).date().isoformat()

                    assignee = _str_or_blank(bulk_assignee) if bulk_status == "followup" else None
                    if bulk_status == "followup" and not assignee:
                        skipped += 1
                        continue

                    try:
                        # write latest snapshot
                        up = {
                            "itinerary_id": str(r["itinerary_id"]),
                            "status": bulk_status,
                            "updated_at": datetime.utcnow(),
                            "advance_amount": _to_int(bulk_adv),
                            "assigned_to": assignee if bulk_status == "followup" else None,
                        }
                        if bulk_status == "confirmed" and bdate:
                            up["booking_date"] = _to_dt_or_none(bdate)
                        else:
                            up["booking_date"] = None
                        col_updates.update_one({"itinerary_id": str(r["itinerary_id"])}, {"$set": _clean_for_mongo(up)}, upsert=True)
                        applied += 1
                    except Exception:
                        skipped += 1

                st.success(f"Bulk updated: {applied} ✓")
                if skipped:
                    st.warning(f"Skipped: {skipped}")
                fetch_updates_df.clear()  # refresh cache
                st.rerun()

    if st.button("💾 Save row-by-row edits"):
        saved, errors = 0, 0
        for _, r in edited.iterrows():
            itinerary_id = r["itinerary_id"]
            status = r["status"]
            assignee = _str_or_blank(r.get("assigned_to")).strip()
            bdate = r.get("booking_date")
            adv   = r.get("advance_amount", 0)

            if status == "followup" and not assignee:
                errors += 1; continue
            if status == "confirmed":
                if bdate is None or (isinstance(bdate, str) and not bdate):
                    errors += 1; continue
                bdate = pd.to_datetime(bdate).date().isoformat()
            else:
                bdate = None

            try:
                up = {
                    "itinerary_id": str(itinerary_id),
                    "status": status,
                    "updated_at": datetime.utcnow(),
                    "advance_amount": _to_int(adv),
                    "assigned_to": assignee if status == "followup" else None,
                    "booking_date": _to_dt_or_none(bdate) if bdate else None
                }
                col_updates.update_one({"itinerary_id": str(itinerary_id)}, {"$set": _clean_for_mongo(up)}, upsert=True)
                saved += 1
            except Exception:
                errors += 1
        if saved:
            st.success(f"Saved {saved} update(s).")
        if errors:
            st.warning(f"{errors} row(s) skipped (missing assignee for follow-up or booking date for confirmed).")
        fetch_updates_df.clear()
        st.rerun()

    if view_mode == "Latest per client (by mobile)":
        st.markdown("### Client-wise history")
        latest = group_latest_by_mobile(df)
        latest.sort_values("client_name", inplace=True)
        for _, row in latest.iterrows():
            hist_ids = row.get("history_ids", []) or []
            label = f"➕ Show packages — {row.get('client_name','')} ({row.get('client_mobile','')})"
            with st.expander(label, expanded=False):
                if not hist_ids:
                    st.caption("No older packages for this client.")
                else:
                    hist = df[df["itinerary_id"].isin(hist_ids)].copy()
                    hist = hist[["ach_id","itinerary_id","upload_date","status","start_date","end_date","package_cost_num","final_route"]]
                    hist.sort_values("upload_date", ascending=False, inplace=True)
                    st.dataframe(hist, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# 2) Expenses & Vendor Payments (Confirmed Only)
# ------------------------------------------------------------------
st.subheader("2) Expenses & Vendor Payments (Confirmed Only)")

df_up_now = fetch_updates_df()
df_now = df_it.merge(df_up_now, on="itinerary_id", how="left")
df_now["status"] = df_now["status"].fillna("pending")
if "advance_amount" not in df_now.columns:
    df_now["advance_amount"] = 0
df_now["advance_amount"] = pd.to_numeric(df_now["advance_amount"], errors="coerce").fillna(0).astype(int)
for c in ["booking_date", "start_date", "end_date"]:
    if c in df_now.columns:
        df_now[c] = df_now[c].apply(to_date_or_none)

confirmed = df_now[df_now["status"] == "confirmed"].copy()

if confirmed.empty:
    st.info("No confirmed packages yet.")
else:
    have_expense = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
    confirmed["expense_entered"] = confirmed["itinerary_id"].isin(have_expense)

    # Use precomputed final_cost from df (join back)
    final_map = df.set_index("itinerary_id")["final_cost"].to_dict()
    confirmed["final_cost"] = confirmed["itinerary_id"].map(final_map).fillna(0).astype(int)

    search = st.text_input("🔎 Search confirmed clients (name/mobile/ACH ID)")
    view_tbl = confirmed.copy()
    if search.strip():
        s = search.strip().lower()
        view_tbl = view_tbl[
            view_tbl["client_name"].astype(str).str.lower().str.contains(s) |
            view_tbl["client_mobile"].astype(str).str.lower().str.contains(s) |
            view_tbl["ach_id"].astype(str).str.lower().str.contains(s)
        ]

    left, right = st.columns([2,1])
    with left:
        show_cols = ["ach_id","itinerary_id","client_name","client_mobile","final_route","total_pax",
                     "final_cost","advance_amount","booking_date","expense_entered"]
        view = view_tbl[show_cols].rename(columns={"final_cost":"Final package cost (₹)"})
        st.dataframe(view.sort_values("booking_date"), use_container_width=True)
    with right:
        st.markdown("**Select a confirmed package to manage:**")
        options = (confirmed["ach_id"].fillna("") + " | " +
                   confirmed["client_name"].fillna("") + " | " +
                   confirmed["booking_date"].fillna("").astype(str) + " | " +
                   confirmed["itinerary_id"])
        sel = st.selectbox("Choose package", options.tolist() if not options.empty else [])
        chosen_id = sel.split(" | ")[-1] if sel else None

    # ------- helpers for estimates & vendor payments -------
    def get_estimates(itinerary_id: str) -> dict:
        doc = col_expenses.find_one({"itinerary_id": str(itinerary_id)},
                                    {"_id":0, "estimates":1, "estimates_locked":1}) or {}
        return doc

    def save_estimates(itinerary_id: str, estimates: dict, lock: bool):
        payload = {
            "itinerary_id": str(itinerary_id),
            "estimates": _clean_for_mongo(estimates),
            "estimates_locked": bool(lock),
            "estimates_updated_at": datetime.utcnow()
        }
        col_expenses.update_one({"itinerary_id": str(itinerary_id)}, {"$set": _clean_for_mongo(payload)}, upsert=True)
        fetch_expenses_df.clear()

    def get_vendor_pay_doc(itinerary_id: str) -> dict:
        return col_vendorpay.find_one({"itinerary_id": str(itinerary_id)}) or {}

    def save_vendor_pay(itinerary_id: str, items: List[dict], final_done: bool):
        doc = {
            "itinerary_id": str(itinerary_id),
            "final_done": bool(final_done),
            "items": _clean_for_mongo(items),
            "updated_at": datetime.utcnow(),
        }
        col_vendorpay.update_one({"itinerary_id": str(itinerary_id)}, {"$set": _clean_for_mongo(doc)}, upsert=True)

    def push_back_status(itinerary_id: str, new_status: str = "under_discussion"):
        doc = {
            "status": new_status,
            "assigned_to": None,
            "booking_date": None,
            "advance_amount": 0,
            "incentive": 0,
            "rep_name": "",
            "updated_at": datetime.utcnow(),
        }
        col_updates.update_one({"itinerary_id": str(itinerary_id)}, {"$set": doc}, upsert=True)
        fetch_updates_df.clear()

    def save_expense_summary(
        itinerary_id: str,
        client_name: str,
        booking_date,
        base_amount: int,
        discount: int,
        notes: str = ""
    ):
        vp = get_vendor_pay_doc(itinerary_id)
        items = vp.get("items", [])
        total_expenses = 0
        for it in items:
            fc = _to_int(it.get("finalization_cost", 0))
            if fc > 0:
                total_expenses += fc
            else:
                total_expenses += _to_int(it.get("adv1_amt", 0)) + _to_int(it.get("adv2_amt", 0)) + _to_int(it.get("final_amt", 0))

        base = _to_int(base_amount)
        disc = _to_int(discount)
        final_cost = max(0, base - disc)
        profit = final_cost - total_expenses

        doc = {
            "itinerary_id": str(itinerary_id),
            "client_name": str(client_name or ""),
            "booking_date": _to_dt_or_none(booking_date),
            "base_package_cost": base,
            "discount": disc,
            "final_package_cost": final_cost,
            "package_cost": final_cost,  # legacy
            "total_expenses": _to_int(total_expenses),
            "profit": _to_int(profit),
            "notes": str(notes or ""),
            "saved_at": datetime.utcnow(),
        }
        col_expenses.update_one({"itinerary_id": str(itinerary_id)}, {"$set": _clean_for_mongo(doc)}, upsert=True)
        fetch_expenses_df.clear()
        return profit, total_expenses, final_cost

    # ------- UI for selected confirmed package -------
    if chosen_id:
        row = df[df["itinerary_id"] == chosen_id].iloc[0]
        client_name  = row.get("client_name","")
        booking_date = row.get("booking_date","")

        st.markdown("#### ↩️ Admin: Push back to Update Status")
        st.caption("Send this package back to the pipeline so it reappears in Section 1 for editing.")
        colpb1, colpb2 = st.columns([2,1])
        with colpb1:
            revert_to = st.selectbox("Set status to", ["under_discussion", "pending"], index=0)
        with colpb2:
            if st.button("Push back now"):
                try:
                    push_back_status(chosen_id, revert_to)
                    st.success(f"Moved to **{revert_to}**. It will now show in Section 1 → Update Status.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not push back: {e}")

        # ---- Expense Estimates (edit once) ----
        st.markdown("#### Expense Estimates (edit once)")
        est_doc = get_estimates(chosen_id)
        locked = bool(est_doc.get("estimates_locked", False))
        estimates = est_doc.get("estimates", {
            "Car": {"vendor": "", "amount": 0},
            "Hotel": {"vendor": "", "amount": 0},
            "Bhasmarathi": {"vendor": "", "amount": 0},
            "Poojan": {"vendor": "", "amount": 0},
            "PhotoFrame": {"vendor": "", "amount": 0},
        })

        with st.form("estimates_form", clear_on_submit=False):
            cols = st.columns(5)
            cats = ["Car","Hotel","Bhasmarathi","Poojan","PhotoFrame"]
            new_names, new_cities = {}, {}
            for i, cat in enumerate(cats):
                with cols[i]:
                    st.caption(cat)
                    look_cat = cat if cat != "PhotoFrame" else "Others"
                    vendor_options = list_vendor_names(look_cat)
                    options = (vendor_options + ["Create new..."]) if vendor_options else ["Create new..."]
                    cur_vendor = estimates.get(cat,{}).get("vendor","")
                    idx = options.index(cur_vendor) if (cur_vendor and cur_vendor in options) else 0
                    selv = st.selectbox(
                        f"{cat} Vendor", options,
                        index=idx, key=f"est_v_{cat}", disabled=locked
                    )
                    if selv == "Create new..." and not locked:
                        new_names[cat] = st.text_input(f"New {cat} vendor name", key=f"new_v_{cat}")
                        new_cities[cat] = st.text_input(f"{cat} vendor city", key=f"new_c_{cat}")
                        vname = (new_names[cat] or "").strip()
                    else:
                        vname = selv or ""
                    amt = st.number_input(
                        f"{cat} Estimate (₹)", min_value=0, step=100,
                        value=_to_int(estimates.get(cat,{}).get("amount",0)),
                        disabled=locked, key=f"est_a_{cat}"
                    )
                    estimates[cat] = {"vendor": vname, "amount": _to_int(amt)}
            lock_now = st.checkbox("Lock estimates (cannot edit later here)", value=locked, disabled=locked)
            save_est = st.form_submit_button("💾 Save Estimates", disabled=locked)

        if save_est:
            for cat, name in new_names.items():
                name = (name or "").strip()
                if name:
                    city = (new_cities.get(cat) or "").strip()
                    cat_dir = cat if cat != "PhotoFrame" else "Others"
                    create_vendor(name, city, cat_dir)
            save_estimates(chosen_id, estimates, lock_now)
            st.success("Estimates saved.")
            st.rerun()

        # ---- Package Summary (final cost)
        st.markdown("#### Package Summary")
        exp_row = df_exp[df_exp["itinerary_id"] == chosen_id].head(1)
        base_default = int(exp_row["base_package_cost"].iat[0]) if not exp_row.empty else _to_int(row.get("package_cost_num"))
        disc_default = int(exp_row["discount"].iat[0]) if not exp_row.empty else 0

        c1c, c2c, c3c = st.columns(3)
        with c1c:
            base_amount = st.number_input("Quoted/Initial amount (₹)", min_value=0, step=500, value=int(base_default))
        with c2c:
            discount = st.number_input("Discount (₹)", min_value=0, step=500, value=int(disc_default))
        with c3c:
            st.metric("Final package cost", f"₹ {max(0, int(base_amount) - int(discount)):,}")

        notes = st.text_area("Notes (optional)", value="")

        if st.button("💾 Save Summary (compute totals & profit)"):
            profit, total_expenses, final_cost = save_expense_summary(
                chosen_id, client_name, booking_date, base_amount, discount, notes
            )
            st.success(f"Saved. Final cost: ₹{final_cost:,} • Total expenses: ₹{total_expenses:,} • Profit: ₹{profit:,}")
            st.rerun()

        st.markdown("---")

        # ---- Vendor Payments
        st.markdown("### Vendor Payments")
        vp_doc = get_vendor_pay_doc(chosen_id)
        items = vp_doc.get("items", [])
        final_done = bool(vp_doc.get("final_done", False))
        st.caption("Update vendor-wise payments. Vendor name is taken from Estimates. Mark **Final done** to lock further edits.")

        est_doc = get_estimates(chosen_id)
        estimates = est_doc.get("estimates", {})

        with st.form("vendor_pay_form", clear_on_submit=False):
            c_cat = st.selectbox("Category", ["Hotel","Car","Bhasmarathi","Poojan","PhotoFrame"], index=0, disabled=final_done)
            est_vendor = (estimates.get(c_cat, {}) or {}).get("vendor", "")
            st.text_input("Vendor (from Estimates)", value=est_vendor, disabled=True,
                          help="To change vendor, edit above in Expense Estimates.")

            final_cost_v = st.number_input("Finalization cost (₹)", min_value=0, step=100, disabled=final_done)
            a1, a2 = st.columns(2)
            with a1:
                adv1_amt = st.number_input("Advance-1 (₹)", min_value=0, step=100, disabled=final_done)
                adv1_date = st.date_input("Advance-1 date", value=None, disabled=final_done)
                final_amt = st.number_input("Final paid (₹)", min_value=0, step=100, disabled=final_done)
                final_date = st.date_input("Final paid date", value=None, disabled=final_done)
            with a2:
                adv2_amt = st.number_input("Advance-2 (₹)", min_value=0, step=100, disabled=final_done)
                adv2_date = st.date_input("Advance-2 date", value=None, disabled=final_done)
                lock_done = st.checkbox("Final done (lock further edits)", value=final_done)

            submitted_vp = st.form_submit_button("➕ Add/Update Vendor Payment", disabled=final_done)

        if submitted_vp and not final_done:
            vname = str(est_vendor or "").strip()
            bal = max(_to_int(final_cost_v) - (_to_int(adv1_amt) + _to_int(adv2_amt) + _to_int(final_amt)), 0)
            entry = {
                "category": c_cat,
                "vendor": vname,
                "finalization_cost": _to_int(final_cost_v),
                "adv1_amt": _to_int(adv1_amt),
                "adv1_date": adv1_date.isoformat() if adv1_date else None,
                "adv2_amt": _to_int(adv2_amt),
                "adv2_date": adv2_date.isoformat() if adv2_date else None,
                "final_amt": _to_int(final_amt),
                "final_date": final_date.isoformat() if final_date else None,
                "balance": _to_int(bal),
            }
            # upsert by category+vendor
            updated = False
            for i, it in enumerate(items):
                if it.get("category")==c_cat and it.get("vendor")==vname:
                    items[i] = entry
                    updated = True
                    break
            if not updated:
                items.append(entry)
            save_vendor_pay(chosen_id, items, lock_done)
            st.success("Vendor payment saved.")
            st.rerun()

        if items:
            show = pd.DataFrame(items)
            st.dataframe(show, use_container_width=True)
        else:
            st.caption("No vendor payments added yet.")

st.divider()

# ------------------------------------------------------------------
# 3) Calendar – Confirmed Packages
# ------------------------------------------------------------------
st.subheader("3) Calendar – Confirmed Packages")
view = st.radio("View", ["By Booking Date", "By Travel Dates"], horizontal=True)

confirmed_view = df[df["status"] == "confirmed"].copy()
if confirmed_view.empty:
    st.info("No confirmed packages to show on calendar.")
else:
    events = []
    for _, r in confirmed_view.iterrows():
        title = f"{r.get('client_name','')}_{r.get('total_pax','')}pax"
        ev = {"title": title, "id": r["itinerary_id"]}
        if view == "By Booking Date":
            if pd.isna(r.get("booking_date")):
                continue
            ev["start"] = pd.to_datetime(r["booking_date"]).strftime("%Y-%m-%d")
        else:
            if pd.isna(r.get("start_date")) or pd.isna(r.get("end_date")):
                continue
            ev["start"] = pd.to_datetime(r["start_date"]).strftime("%Y-%m-%d")
            end_ = pd.to_datetime(r["end_date"]) + pd.Timedelta(days=1)
            ev["end"] = end_.strftime("%Y-%m-%d")
        events.append(ev)

    selected_id = None
    if CALENDAR_AVAILABLE:
        opts = {"initialView": "dayGridMonth", "height": 620, "eventDisplay": "block"}
        result = calendar(options=opts, events=events, key=f"pkg_cal_{'booking' if view=='By Booking Date' else 'travel'}")
        if result and isinstance(result, dict) and result.get("eventClick"):
            try:
                selected_id = result["eventClick"]["event"]["id"]
            except Exception:
                selected_id = None
    else:
        st.caption("Calendar component not installed. Showing a simple list instead.")
        display = pd.DataFrame(events).rename(columns={"title": "Package", "start": "Start", "end": "End"})
        st.dataframe(display.sort_values(["Start","End"]), use_container_width=True)
        if not confirmed_view.empty:
            selected_id = st.selectbox(
                "Open package details",
                (confirmed_view["itinerary_id"] + " | " + confirmed_view["client_name"]).tolist()
            )
            if selected_id:
                selected_id = selected_id.split(" | ")[0]

    if selected_id:
        st.divider()
        st.subheader("📦 Package Details")

        it = df_it[df_it["itinerary_id"] == selected_id].iloc[0].to_dict()
        upd = df_up[df_up["itinerary_id"] == selected_id].iloc[0].to_dict() if not df_up.empty and (df_up["itinerary_id"]==selected_id).any() else {}
        exp = df_exp[df_exp["itinerary_id"] == selected_id].iloc[0].to_dict() if not df_exp.empty and (df_exp["itinerary_id"]==selected_id).any() else {}

        created_dt_utc = _created_utc(selected_id)
        created_ist_str = _fmt_ist(created_dt_utc)
        created_utc_str = created_dt_utc.strftime("%Y-%m-%d %H:%M %Z") if created_dt_utc else ""

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Basic**")
            st.write({
                "ACH ID": it.get("ach_id", ""),
                "Client": it.get("client_name",""),
                "Mobile": it.get("client_mobile",""),
                "Route": it.get("final_route",""),
                "Pax": it.get("total_pax",""),
                "Travel": f"{it.get('start_date','')} → {it.get('end_date','')}",
                "Created (IST)": created_ist_str,
                "Created (UTC)": created_utc_str,
            })
        with c2:
            st.markdown("**Status & Money**")
            final_cost_display = int(df[df["itinerary_id"] == selected_id]["final_cost"].iloc[0])
            base_cost_display = _to_int(exp.get("base_package_cost", it.get("package_cost_num", 0)))
            discount_display  = _to_int(exp.get("discount", 0))
            st.write({
                "Status": upd.get("status",""),
                "Assigned To": upd.get("assigned_to",""),
                "Booking date": upd.get("booking_date",""),
                "Advance (₹)": upd.get("advance_amount",0),
                "Incentive (₹)": upd.get("incentive",0),
                "Representative": upd.get("rep_name","") or it.get("representative",""),
                "Quoted amount (₹)": base_cost_display,
                "Discount (₹)": discount_display,
                "Final package cost (₹)": final_cost_display,
                "Total expenses (₹)": exp.get("total_expenses", 0),
                "Profit (₹)": exp.get("profit", 0),
            })

        st.markdown("**Itinerary text**")
        st.text_area("Shared with client", value=(it.get("itinerary_text","") or ""), height=260, disabled=True)
