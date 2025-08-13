# pages/02_Package_Update.py
from __future__ import annotations

import math
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# Deny for these users
if st.session_state.get("user") in ("Teena", "Kuldeep"):
    st.stop()  # silently deny

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Package Update", layout="wide")
st.title("📦 Package Update (Admin)")

# Optional: pretty calendar
CALENDAR_AVAILABLE = True
try:
    from streamlit_calendar import calendar
except Exception:
    CALENDAR_AVAILABLE = False


# ----------------------------
# --- Admin-only gate (no PIN login here) ---
# ----------------------------
def require_admin():
    ADMIN_PASS_DEFAULT = "Arpith&92"  # set your default here
    ADMIN_PASS = str(st.secrets.get("admin_pass", ADMIN_PASS_DEFAULT))

    with st.sidebar:
        st.markdown("### Admin access")
        p = st.text_input("Enter admin password", type="password", placeholder="enter pass")

    if (p or "").strip() != ADMIN_PASS.strip():
        st.stop()

    # Force identity for this page
    st.session_state["user"] = "Admin"
    st.session_state["is_admin"] = True

require_admin()


# ----------------------------
# MongoDB Setup (with helpful errors)
# ----------------------------
try:
    MONGO_URI = st.secrets["mongo_uri"]
except KeyError:
    st.error(
        "❌ MongoDB is not configured.\n\n"
        "In **Manage app → Secrets**, add:\n"
        'mongo_uri = "mongodb+srv://USERNAME:PASSWORD@CLUSTER/?retryWrites=true&w=majority"\n'
    )
    st.stop()

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
except Exception as e:
    st.error(f"❌ Could not connect to MongoDB. Details: {e}")
    st.stop()

db = client["TAK_DB"]
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_expenses    = db["expenses"]
col_vendorpay   = db["vendor_payments"]
col_vendors     = db["vendors"]   # NEW: vendor directory (name, city, category)


# ----------------------------
# Helpers
# ----------------------------
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

# ---- Created-at helpers (use ObjectId timestamp) ----
IST = ZoneInfo("Asia/Kolkata")
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

# ---- Final cost logic (base − discount) used everywhere on this page ----
def _final_cost_for(itinerary_id: str) -> int:
    """
    Final cost = base_package_cost - discount.
    Fallbacks:
      - if only expenses.package_cost exists, treat it as final
      - else use itinerary.package_cost - itinerary.discount
    """
    exp = col_expenses.find_one(
        {"itinerary_id": str(itinerary_id)},
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

    it = col_itineraries.find_one({"_id": ObjectId(itinerary_id)}, {"package_cost": 1, "discount": 1}) or {}
    base = _to_int(it.get("package_cost", 0))
    disc = _to_int(it.get("discount", 0))
    return max(0, base - disc)


def to_int_money(x):
    return _to_int(x, 0)

def current_fy_two_digits(today: date | None = None) -> int:
    d = today or date.today()
    year = d.year if d.month >= 4 else d.year - 1
    return year % 100

def next_ach_id(fy_2d: int) -> str:
    prefix = f"ACH-{fy_2d:02d}-"
    docs = col_itineraries.find({"ach_id": {"$regex": f"^{prefix}\\d{{3}}$"}}, {"ach_id": 1})
    max_no = 0
    for d in docs:
        try:
            n = int(d["ach_id"].split("-")[-1])
            if n > max_no:
                max_no = n
        except Exception:
            pass
    return f"{prefix}{max_no+1:03d}"

def backfill_ach_ids():
    fy = current_fy_two_digits()
    cursor = col_itineraries.find({"$or": [{"ach_id": {"$exists": False}}, {"ach_id": ""}]})
    for doc in cursor:
        new_id = next_ach_id(fy)
        col_itineraries.update_one({"_id": doc["_id"]}, {"$set": {"ach_id": new_id}})

def fetch_itineraries_df():
    backfill_ach_ids()
    rows = list(col_itineraries.find({}))
    if not rows:
        return pd.DataFrame()
    for r in rows:
        r["itinerary_id"] = str(r.get("_id"))
        r["ach_id"] = r.get("ach_id", "")
        for k in ("start_date", "end_date", "upload_date"):
            try:
                v = r.get(k)
                r[k] = pd.to_datetime(v).to_pydatetime() if pd.notna(v) else None
            except Exception:
                r[k] = None
        r["package_cost_num"] = to_int_money(r.get("package_cost"))
        r["client_mobile"] = r.get("client_mobile", "")
        r["client_name"] = r.get("client_name", "")
        r["representative"] = r.get("representative", "")
        r["final_route"] = r.get("final_route", "")
        r["total_pax"] = r.get("total_pax", 0)
    return pd.DataFrame(rows)

def fetch_updates_df():
    rows = list(col_updates.find({}, {"_id": 0}))
    if not rows:
        return pd.DataFrame(columns=["itinerary_id","status","booking_date","advance_amount","assigned_to","incentive","rep_name"])
    for r in rows:
        if r.get("booking_date"):
            try:
                r["booking_date"] = pd.to_datetime(r["booking_date"]).date()
            except Exception:
                r["booking_date"] = None
        r["advance_amount"] = to_int_money(r.get("advance_amount", 0))
    return pd.DataFrame(rows)

def fetch_expenses_df():
    rows = list(col_expenses.find({}, {"_id":0}))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["itinerary_id","total_expenses","profit","package_cost"])

def find_itinerary_doc(selected_id: str):
    it = None
    try:
        it = col_itineraries.find_one({"_id": ObjectId(selected_id)})
    except Exception:
        it = None
    if it is None:
        it = (col_itineraries.find_one({"itinerary_id": selected_id}) or
              col_itineraries.find_one({"ach_id": selected_id}))
    return it

def to_date_or_none(x):
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

def group_latest_by_mobile(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    df_all = df_all.copy()
    df_all["upload_date"] = pd.to_datetime(df_all["upload_date"])
    df_all.sort_values(["client_mobile","upload_date"], ascending=[True, False], inplace=True)
    latest_rows = df_all.groupby("client_mobile", as_index=False).first()
    hist_map = {}
    for mob, grp in df_all.groupby("client_mobile"):
        ids = grp["itinerary_id"].tolist()
        hist_map[mob] = ids[1:] if len(ids) > 1 else []
    latest_rows["history_ids"] = latest_rows["client_mobile"].map(hist_map).apply(lambda x: x or [])
    return latest_rows


# ----------------------------
# Vendors directory (DB-driven)
# ----------------------------
VENDOR_CATEGORIES = ["Bhasmarathi", "Car", "Hotel", "Poojan", "Others"]

def list_vendor_names(category: str) -> list[str]:
    q = {"category": category}
    names = [v.get("name","").strip() for v in col_vendors.find(q, {"name":1}, sort=[("name", 1)])]
    return [n for n in names if n]  # clean empties

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
        return True
    except Exception:
        return False


# ----------------------------
# Auto-assign brand-new packages to Follow-up for their representative
# ----------------------------
def ensure_followup_assigned_for_new_packages():
    """
    For each itinerary with a representative but NO package_updates yet:
      Create a package_update doc:
        status='followup', assigned_to=<representative>, updated_at=now
    """
    cur = col_itineraries.find({}, {"_id":1, "representative":1})
    for it in cur:
        iid = str(it["_id"])
        rep = _str_or_blank(it.get("representative", "")).strip()
        if not rep:
            continue
        if col_updates.count_documents({"itinerary_id": iid}, limit=1) > 0:
            continue
        doc = {
            "itinerary_id": iid,
            "status": "followup",
            "assigned_to": rep,
            "advance_amount": 0,
            "booking_date": None,
            "incentive": 0,
            "rep_name": "",
            "updated_at": datetime.utcnow(),
        }
        try:
            col_updates.update_one({"itinerary_id": iid}, {"$setOnInsert": _clean_for_mongo(doc)}, upsert=True)
        except Exception:
            pass


# ----------------------------
# Estimates & Vendor Payments
# ----------------------------
def get_estimates(itinerary_id: str) -> dict:
    doc = col_expenses.find_one({"itinerary_id": str(itinerary_id)},
                                {"_id":0, "estimates":1, "estimates_locked":1})
    return doc or {}

def save_estimates(itinerary_id: str, estimates: dict, lock: bool):
    payload = {
        "itinerary_id": str(itinerary_id),
        "estimates": _clean_for_mongo(estimates),
        "estimates_locked": bool(lock),
        "estimates_updated_at": datetime.utcnow()
    }
    payload = _clean_for_mongo(payload)
    col_expenses.update_one({"itinerary_id": str(itinerary_id)}, {"$set": payload}, upsert=True)

def get_vendor_pay_doc(itinerary_id: str) -> dict:
    doc = col_vendorpay.find_one({"itinerary_id": str(itinerary_id)}) or {}
    return doc

def save_vendor_pay(itinerary_id: str, items: list[dict], final_done: bool):
    doc = {
        "itinerary_id": str(itinerary_id),
        "final_done": bool(final_done),
        "items": _clean_for_mongo(items),
        "updated_at": datetime.utcnow(),
    }
    doc = _clean_for_mongo(doc)
    col_vendorpay.update_one({"itinerary_id": str(itinerary_id)}, {"$set": doc}, upsert=True)

def push_back_status(itinerary_id: str, new_status: str = "under_discussion"):
    """
    Move a confirmed package back to the sales pipeline.
    Clears booking/incentive so it reappears in Section 1 for edits.
    """
    doc = {
        "status": new_status,          # "under_discussion" or "pending"
        "assigned_to": None,
        "booking_date": None,
        "advance_amount": 0,
        "incentive": 0,
        "rep_name": "",
        "updated_at": datetime.utcnow(),
    }
    col_updates.update_one({"itinerary_id": str(itinerary_id)}, {"$set": doc}, upsert=True)

def _final_cost_for(itinerary_id: str) -> int:
    exp = col_expenses.find_one(
        {"itinerary_id": str(itinerary_id)},
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
    it = col_itineraries.find_one({"_id": ObjectId(itinerary_id)}, {"package_cost": 1, "discount": 1}) or {}
    base = _to_int(it.get("package_cost", 0))
    disc = _to_int(it.get("discount", 0))
    return max(0, base - disc)

def upsert_status(itinerary_id, status, booking_date, advance_amount, assigned_to=None):
    # compute incentive if confirming -> use FINAL cost
    incentive = 0
    rep_name = ""
    if status == "confirmed":
        it = find_itinerary_doc(itinerary_id)
        rep_name = (it or {}).get("representative", "")
        final_amt = _final_cost_for(itinerary_id)
        if 5000 < final_amt < 20000:
            incentive = 250
        elif final_amt >= 20000:
            incentive = 500

    doc = {
        "itinerary_id": str(itinerary_id),
        "status": status,
        "updated_at": datetime.utcnow(),
        "advance_amount": _to_int(advance_amount or 0),
        "assigned_to": assigned_to if status == "followup" else None,
        "incentive": int(incentive),
        "rep_name": rep_name,
    }
    if status == "confirmed" and booking_date:
        doc["booking_date"] = _to_dt_or_none(booking_date)
    else:
        doc["booking_date"] = None

    col_updates.update_one({"itinerary_id": str(itinerary_id)}, {"$set": _clean_for_mongo(doc)}, upsert=True)

def save_expense_summary(
    itinerary_id: str,
    client_name: str,
    booking_date,
    base_amount: int,
    discount: int,
    notes: str = ""
):
    """
    Persist base, discount, and FINAL cost.
    Also compute total vendor expenses and profit = FINAL - total_expenses.
    Keep legacy expenses.package_cost aligned to FINAL for old views.
    """
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
        # store detailed + align legacy key
        "base_package_cost": base,
        "discount": disc,
        "final_package_cost": final_cost,
        "package_cost": final_cost,  # legacy compatibility
        "total_expenses": _to_int(total_expenses),
        "profit": _to_int(profit),
        "notes": str(notes or ""),
        "saved_at": datetime.utcnow(),
    }
    col_expenses.update_one({"itinerary_id": str(itinerary_id)}, {"$set": _clean_for_mongo(doc)}, upsert=True)
    return profit, total_expenses, final_cost


# ----------------------------
# Load & Prep data
# ----------------------------
# 1) make sure brand-new packages are auto-assigned to follow-up for their representative
ensure_followup_assigned_for_new_packages()

# 2) load app data
df_it = fetch_itineraries_df()
if df_it.empty:
    st.info("No packages found yet. Upload a file in the main app first.")
    st.stop()

df_up  = fetch_updates_df()
df_exp = fetch_expenses_df()

# Merge updates
df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")

# Ensure advance_amount exists & numeric
if "advance_amount" not in df.columns:
    df["advance_amount"] = 0
df["advance_amount"] = pd.to_numeric(df["advance_amount"], errors="coerce").fillna(0).astype(int)

# Normalize date columns
for col_ in ["start_date", "end_date", "booking_date"]:
    if col_ in df.columns:
        df[col_] = df[col_].apply(to_date_or_none)


# ----------------------------
# Vendor Directory (create + quick view)
# ----------------------------
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
                st.experimental_rerun()
            else:
                st.warning("Please enter at least vendor name and category.")

with st.expander("📋 Current vendors (by category)"):
    tabs = st.tabs(VENDOR_CATEGORIES)
    for i, cat in enumerate(VENDOR_CATEGORIES):
        with tabs[i]:
            opt = list_vendor_names(cat)
            df_v = pd.DataFrame({"Vendor": opt}) if opt else pd.DataFrame({"Vendor": []})
            st.dataframe(df_v, use_container_width=True, hide_index=True)


# ----------------------------
# Summary KPIs  (unique by client_mobile)
# ----------------------------
latest_for_counts = group_latest_by_mobile(df.copy())

pending_count           = int((latest_for_counts["status"] == "pending").sum())
under_discussion_count  = int((latest_for_counts["status"] == "under_discussion").sum())
followup_count          = int((latest_for_counts["status"] == "followup").sum())
cancelled_count         = int((latest_for_counts["status"] == "cancelled").sum())

# Confirmed – expense pending (unique by mobile)
have_expense_ids = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
confirmed_latest = latest_for_counts[latest_for_counts["status"] == "confirmed"].copy()
confirmed_expense_pending = confirmed_latest[~confirmed_latest["itinerary_id"].isin(have_expense_ids)].shape[0]

k1, k2, kf, k3, k4 = st.columns(5)
k1.metric("🟡 Pending", int(pending_count))
k2.metric("🟠 Under discussion", int(under_discussion_count))
kf.metric("🔵 Follow-up", int(followup_count))
k3.metric("🟧 Confirmed – expense pending", int(confirmed_expense_pending))
k4.metric("🔴 Cancelled", int(cancelled_count))

st.divider()

# ----------------------------
# 🕒 Created-at section (new)
# ----------------------------
st.subheader("🕒 Package created time")
with st.expander("Show recently created (last 25)"):
    created_df = df[["itinerary_id","ach_id","client_name","client_mobile"]].copy()
    created_df["created_utc"] = created_df["itinerary_id"].apply(_created_utc)
    created_df = created_df.dropna(subset=["created_utc"]).sort_values("created_utc", ascending=False).head(25)
    created_df["Created (IST)"] = created_df["created_utc"].apply(_fmt_ist)
    created_df["Created (UTC)"] = created_df["created_utc"].apply(lambda d: d.strftime("%Y-%m-%d %H:%M %Z"))
    show_cols = ["ach_id","client_name","client_mobile","Created (IST)","Created (UTC)"]
    st.dataframe(created_df[show_cols], use_container_width=True, hide_index=True)

st.divider()


# ----------------------------
# 1) Status Update (NO follow-ups shown here)
# ----------------------------
st.subheader("1) Update Status")
view_mode = st.radio("View mode", ["Latest per client (by mobile)", "All packages"], horizontal=True)

if view_mode == "Latest per client (by mobile)":
    latest = group_latest_by_mobile(df)
    editable = latest.copy()
else:
    editable = df.copy()

# Only show items still in sales pipeline (remove followup from this page)
editable = editable[editable["status"].isin(["pending","under_discussion"])].copy()

if editable.empty:
    st.success("Nothing to update right now. 🎉")
else:
    must_cols = [
        "ach_id","itinerary_id","client_name","client_mobile","final_route","total_pax",
        "start_date","end_date","package_cost","status","booking_date","advance_amount","assigned_to"
    ]
    for c in must_cols:
        if c not in editable.columns:
            editable[c] = None

    for c in ["ach_id","itinerary_id","client_name","final_route","package_cost","client_mobile"]:
        editable[c] = editable[c].astype(str).fillna("")
    editable["total_pax"] = pd.to_numeric(editable["total_pax"], errors="coerce").fillna(0).astype(int)
    editable["advance_amount"] = pd.to_numeric(editable["advance_amount"], errors="coerce").fillna(0).astype(int)
    for c in ["start_date","end_date","booking_date"]:
        editable[c] = editable[c].apply(to_date_or_none)

    # normalize assigned_to so NaN won't break saves
    if "assigned_to" not in editable.columns:
        editable["assigned_to"] = ""
    editable["assigned_to"] = editable["assigned_to"].apply(_str_or_blank)

    # --- checkbox column for robust selection ---
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

    # --- BULK update using the checkbox selection ---
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
                        upsert_status(
                            r["itinerary_id"],
                            bulk_status,
                            bdate,
                            bulk_adv,
                            assigned_to=assignee
                        )
                        applied += 1
                    except Exception:
                        skipped += 1

                st.success(f"Bulk updated: {applied} ✓")
                if skipped:
                    st.warning(f"Skipped: {skipped}")
                st.rerun()

    # --- Save any row-by-row edits made directly in the grid ---
    if st.button("💾 Save row-by-row edits"):
        saved, errors = 0, 0
        for _, r in edited.iterrows():
            itinerary_id = r["itinerary_id"]
            status = r["status"]
            assignee = _str_or_blank(r.get("assigned_to")).strip()
            bdate = r.get("booking_date")
            adv   = r.get("advance_amount", 0)

            if status == "followup" and not assignee:
                errors += 1
                continue
            if status == "confirmed":
                if bdate is None or (isinstance(bdate, str) and not bdate):
                    errors += 1
                    continue
                bdate = pd.to_datetime(bdate).date().isoformat()
            else:
                bdate = None

            try:
                upsert_status(
                    itinerary_id, status, bdate, adv,
                    assigned_to=assignee if status == "followup" else None
                )
                saved += 1
            except Exception:
                errors += 1
        if saved:
            st.success(f"Saved {saved} update(s).")
        if errors:
            st.warning(f"{errors} row(s) skipped (missing assignee for follow-up or booking date for confirmed).")
        st.rerun()

    # (Latest view) history block still available
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
                    hist = hist[["ach_id","itinerary_id","upload_date","status","start_date","end_date","package_cost","final_route"]]
                    hist.sort_values("upload_date", ascending=False, inplace=True)
                    st.dataframe(hist, use_container_width=True)

st.divider()


# ----------------------------
# 2) Expenses & Vendor Payments (Confirmed Only)
# ----------------------------
st.subheader("2) Expenses & Vendor Payments (Confirmed Only)")

df_up = fetch_updates_df()
df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")
if "advance_amount" not in df.columns:
    df["advance_amount"] = 0
df["advance_amount"] = pd.to_numeric(df["advance_amount"], errors="coerce").fillna(0).astype(int)
for c in ["booking_date", "start_date", "end_date"]:
    if c in df.columns:
        df[c] = df[c].apply(to_date_or_none)

confirmed = df[df["status"] == "confirmed"].copy()

if confirmed.empty:
    st.info("No confirmed packages yet.")
else:
    have_expense = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
    confirmed["expense_entered"] = confirmed["itinerary_id"].isin(have_expense)

    # Add FINAL cost column for display (computed live)
    confirmed["final_cost"] = confirmed["itinerary_id"].apply(_final_cost_for)

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

    if chosen_id:
        row = confirmed[confirmed["itinerary_id"] == chosen_id].iloc[0]
        client_name  = row.get("client_name","")
        booking_date = row.get("booking_date","")

        # --- Admin: push a confirmed package back to "Update Status" ---
        st.markdown("#### ↩️ Admin: Push back to Update Status")
        st.caption("Send this package back to the pipeline so it reappears in Section 1 for editing.")
        colpb1, colpb2 = st.columns([2,1])
        with colpb1:
            revert_to = st.selectbox("Set status to", ["under_discussion", "pending"], index=0,
                                     help="Choose where it should appear in Section 1.")
        with colpb2:
            if st.button("Push back now", help="Move out of Confirmed and clear booking/incentive."):
                try:
                    push_back_status(chosen_id, revert_to)
                    st.success(f"Moved to **{revert_to}**. It will now show in Section 1 → Update Status.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not push back: {e}")

        # ===== Expense Estimates (edit once) using DB vendors, with inline creation =====
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
            new_names = {}
            new_cities = {}
            for i, cat in enumerate(cats):
                with cols[i]:
                    st.caption(cat)
                    vendor_options = list_vendor_names(cat if cat!="PhotoFrame" else "Others")
                    options = vendor_options + ["Create new..."]
                    cur_vendor = estimates.get(cat,{}).get("vendor","")
                    idx = options.index(cur_vendor) if (cur_vendor and cur_vendor in options) else 0
                    selv = st.selectbox(
                        f"{cat} Vendor", options if options else ["Create new..."],
                        index=idx if options else 0,
                        key=f"est_v_{cat}", disabled=locked
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
            # create any newly typed vendors in DB
            for cat, name in new_names.items():
                name = (name or "").strip()
                if name:
                    city = (new_cities.get(cat) or "").strip()
                    # map PhotoFrame -> Others in directory
                    cat_dir = cat if cat != "PhotoFrame" else "Others"
                    create_vendor(name, city, cat_dir)
            save_estimates(chosen_id, estimates, lock_now)
            st.success("Estimates saved.")
            st.rerun()

        # ---------------- Final package cost editor (base − discount) ----------------
        st.markdown("#### Package Summary")

        exp_doc = col_expenses.find_one(
            {"itinerary_id": str(chosen_id)},
            {"base_package_cost":1, "discount":1, "final_package_cost":1, "package_cost":1}
        ) or {}
        base_default = _to_int(exp_doc.get("base_package_cost", 0)) or _to_int(row.get("package_cost") or row.get("package_cost_num"))
        disc_default = _to_int(exp_doc.get("discount", 0))

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

        # ===== Vendor Payments =====
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


# ----------------------------
# 3) Calendar – Confirmed Packages
# ----------------------------
st.subheader("3) Calendar – Confirmed Packages")
view = st.radio("View", ["By Booking Date", "By Travel Dates"], horizontal=True)

confirmed = df[df["status"] == "confirmed"].copy()
if confirmed.empty:
    st.info("No confirmed packages to show on calendar.")
else:
    events = []
    for _, r in confirmed.iterrows():
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
        if not confirmed.empty:
            selected_id = st.selectbox(
                "Open package details",
                (confirmed["itinerary_id"] + " | " + confirmed["client_name"]).tolist()
            )
            if selected_id:
                selected_id = selected_id.split(" | ")[0]

    if selected_id:
        st.divider()
        st.subheader("📦 Package Details")

        it = find_itinerary_doc(selected_id)
        upd = col_updates.find_one({"itinerary_id": str(selected_id)}, {"_id":0})
        exp = col_expenses.find_one({"itinerary_id": str(selected_id)}, {"_id":0})

        created_dt_utc = _created_utc(selected_id)
        created_ist_str = _fmt_ist(created_dt_utc)
        created_utc_str = created_dt_utc.strftime("%Y-%m-%d %H:%M %Z") if created_dt_utc else ""

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Basic**")
            st.write({
                "ACH ID": it.get("ach_id", "") if it else "",
                "Client": it.get("client_name","") if it else "",
                "Mobile": it.get("client_mobile","") if it else "",
                "Route": it.get("final_route","") if it else "",
                "Pax": it.get("total_pax","") if it else "",
                "Travel": f"{it.get('start_date','')} → {it.get('end_date','')}" if it else "",
                "Created (IST)": created_ist_str,
                "Created (UTC)": created_utc_str,
            })
        with c2:
            st.markdown("**Status & Money**")
            final_cost_display = _final_cost_for(selected_id)
            base_cost_display = _to_int((exp or {}).get("base_package_cost", (it or {}).get("package_cost", 0))) if exp or it else 0
            discount_display  = _to_int((exp or {}).get("discount", (it or {}).get("discount", 0))) if exp or it else 0
            st.write({
                "Status": upd.get("status","") if upd else "",
                "Assigned To": upd.get("assigned_to","") if upd else "",
                "Booking date": upd.get("booking_date","") if upd else "",
                "Advance (₹)": upd.get("advance_amount",0) if upd else 0,
                "Incentive (₹)": upd.get("incentive",0) if upd else 0,
                "Representative": upd.get("rep_name","") if upd else (it.get("representative","") if it else ""),
                "Quoted amount (₹)": base_cost_display,
                "Discount (₹)": discount_display,
                "Final package cost (₹)": final_cost_display,
                "Total expenses (₹)": (exp.get("total_expenses", 0) if exp else 0),
                "Profit (₹)": (exp.get("profit", 0) if exp else 0),
            })

        st.markdown("**Itinerary text**")
        st.text_area("Shared with client", value=(it.get("itinerary_text","") if it else ""), height=260, disabled=True)
