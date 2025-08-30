# pages/02_Package_Update.py
from __future__ import annotations

import math, os, io
import calendar as pycal  # avoid shadowing streamlit_calendar.calendar
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
st.session_state.setdefault("user", "Unknown")
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
        serverSelectionTimeoutMS=6000,
        connectTimeoutMS=6000,
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
# ✅ FIX: get collections with bracket syntax (safe even if not created yet)
col_split       = db["expense_splitwise"]
col_att         = db["driver_attendance"]

# Driver costing constants (can be moved to Secrets if needed)
BASE_SALARY   = int(st.secrets.get("driver_base_salary", 12000))
LEAVE_DEDUCT  = int(st.secrets.get("driver_leave_deduct", 400))
OVERTIME_ADD  = int(st.secrets.get("driver_overtime_add", 300))
TAK_PREFIX    = str(st.secrets.get("tak_car_prefix", "TAK"))

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
try:
    from tak_audit import audit_pageview
    audit_pageview(st.session_state.get("user", "Unknown"), "02_Package_Update")
except Exception:
    pass

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

def _fmt_ist(dt_: datetime | None) -> str:
    if not dt_:
        return ""
    try:
        return dt_.astimezone(IST).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt_.strftime("%Y-%m-%d %H:%M UTC")

def to_date_or_none(x):
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

def _ensure_cols(df: pd.DataFrame, cols: list[str], fill=None) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

def _month_bounds(d: date) -> Tuple[date, date]:
    first = d.replace(day=1)
    last = d.replace(day=pycal.monthrange(d.year, d.month)[1])
    return first, last

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
            # new app.py totals (for reference if needed)
            "package_total": 1, "package_after_referral": 1,
            "actual_total": 1, "profit_total": 1,
            # legacy names used in older pages
            "package_cost": 1, "discount": 1,
            "itinerary_text": 1,
            # revisions
            "revision_num": 1
        }
    ))
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        rec = dict(r)
        rec["itinerary_id"] = str(rec.pop("_id"))
        for k in ("start_date","end_date","upload_date"):
            v = rec.get(k)
            rec[k] = pd.to_datetime(v).to_pydatetime() if v else None
        # prefer new totals from app.py
        pkg_total = _to_int(rec.get("package_total", rec.get("package_cost", 0)))
        pkg_after = _to_int(rec.get("package_after_referral", 0))
        if pkg_after == 0 and _to_int(rec.get("discount", 0)) > 0:
            pkg_after = max(pkg_total - _to_int(rec.get("discount", 0)), 0)
        rec["package_cost_num"] = pkg_total
        # compute discount if absent
        if "discount" not in rec or rec["discount"] is None:
            rec["discount"] = max(pkg_total - pkg_after, 0)
        rec["discount"] = _to_int(rec.get("discount", 0))
        rec["revision_num"] = _to_int(rec.get("revision_num", 0))
        out.append(rec)
    df = pd.DataFrame(out)
    return _ensure_cols(df, ["ach_id","representative","itinerary_text"], "")

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
         "final_package_cost": 1, "package_cost": 1,
         "vendor_total": 1, "splitwise_total": 1, "driver_total": 1,
         "total_expenses": 1, "profit": 1, "notes": 1}
    ))
    if not rows:
        return pd.DataFrame(columns=["itinerary_id","total_expenses","profit","package_cost",
                                     "base_package_cost","discount","final_package_cost",
                                     "vendor_total","splitwise_total","driver_total","notes"])
    for r in rows:
        r["itinerary_id"] = str(r.get("itinerary_id"))
        for k in ("base_package_cost","discount","final_package_cost","package_cost",
                  "vendor_total","splitwise_total","driver_total","total_expenses","profit"):
            r[k] = _to_int(r.get(k, 0))
    return pd.DataFrame(rows)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_vendorpay_df() -> pd.DataFrame:
    docs = list(col_vendorpay.find({}, {"_id":0}))
    rows = []
    for d in docs:
        iid = str(d.get("itinerary_id"))
        final_done = bool(d.get("final_done", False))
        for it in d.get("items", []):
            rows.append({
                "itinerary_id": iid,
                "final_done": final_done,
                "category": it.get("category",""),
                "vendor": it.get("vendor",""),
                "finalization_cost": _to_int(it.get("finalization_cost", 0)),
                "adv1_amt": _to_int(it.get("adv1_amt", 0)),
                "adv1_date": to_date_or_none(it.get("adv1_date")),
                "adv2_amt": _to_int(it.get("adv2_amt", 0)),
                "adv2_date": to_date_or_none(it.get("adv2_date")),
                "final_amt": _to_int(it.get("final_amt", 0)),
                "final_date": to_date_or_none(it.get("final_date")),
                "balance": _to_int(it.get("balance", 0)),
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_vendor_master() -> pd.DataFrame:
    """Vendor master list from `vendors` collection."""
    docs = list(col_vendors.find({}, {"_id":0}))
    if not docs:
        return pd.DataFrame(columns=["vendor","category","active","contact"])
    out = []
    for d in docs:
        out.append({
            "vendor": _str_or_blank(d.get("vendor")),
            "category": _str_or_blank(d.get("category")),
            "active": bool(d.get("active", True)),
            "contact": _str_or_blank(d.get("contact")),
        })
    dfv = pd.DataFrame(out)
    return dfv[dfv["active"] == True] if not dfv.empty else dfv

# ======== PROFIT CALCULATOR (extended) ========
def _safe_int(x):
    try: return int(round(float(str(x).replace(",", ""))))
    except: return 0

def final_cost_for(iid: str) -> int:
    exp = col_expenses.find_one({"itinerary_id": str(iid)},
                                {"final_package_cost":1,"base_package_cost":1,"discount":1,"package_cost":1}) or {}
    if "final_package_cost" in exp:
        return _safe_int(exp.get("final_package_cost", 0))
    if any(k in exp for k in ("base_package_cost","discount","package_cost")):
        base = _safe_int(exp.get("base_package_cost", exp.get("package_cost", 0)))
        disc = _safe_int(exp.get("discount", 0))
        return max(0, base - disc)
    it = col_itineraries.find_one({"_id": ObjectId(iid)}, {"package_cost":1,"discount":1}) or {}
    base = _safe_int(it.get("package_cost", 0))
    disc = _safe_int(it.get("discount", 0))
    return max(0, base - disc)

def vendor_cost_for(iid: str) -> int:
    rec = col_vendorpay.find_one({"itinerary_id": str(iid)}, {"_id":0, "items":1}) or {}
    total = 0
    for it in (rec.get("items") or []):
        fc = _safe_int(it.get("finalization_cost", 0))
        if fc > 0:
            total += fc
        else:
            total += _safe_int(it.get("adv1_amt", 0)) + _safe_int(it.get("adv2_amt", 0)) + _safe_int(it.get("final_amt", 0))
    return total

def splitwise_cost_for(iid: str) -> int:
    q = {"itinerary_id": str(iid), "kind": "expense"}
    amt = 0
    for r in col_split.find(q, {"amount":1}):
        amt += _safe_int(r.get("amount", 0))
    return amt

def _driver_month_gross(driver: str, y: int, m: int) -> Tuple[int,int,int,int]:
    """Return (gross, days_in_month, leave_days, ot_units_all_month) from attendance."""
    first = date(y, m, 1)
    last  = date(y, m, pycal.monthrange(y, m)[1])
    cur = list(col_att.find(
        {"driver": driver,
         "date": {"$gte": datetime.combine(first, dtime.min), "$lte": datetime.combine(last, dtime.max)}},
        {"status":1,"outstation_overnight":1,"overnight_client":1,"bhasmarathi":1}
    ))
    leave_days = sum(1 for r in cur if (r.get("status","Present") == "Leave"))
    ot_all = 0
    for r in cur:
        ot_all += int(bool(r.get("outstation_overnight", False)))
        ot_all += int(bool(r.get("overnight_client", False)))
        ot_all += int(bool(r.get("bhasmarathi", False)))
    days_in_month = (last - first).days + 1
    gross = BASE_SALARY - leave_days * LEAVE_DEDUCT + ot_all * OVERTIME_ADD
    return int(gross), int(days_in_month), int(leave_days), int(ot_all)

def driver_cost_for_client(iid: str) -> int:
    """
    If TAK car used AND attendance row linked to this itinerary:
      Sum across (driver, yyyy-mm) buckets:
        daily_rate = (monthly_gross / days_in_month)
        driver_cost += daily_rate * client_days_in_that_month + client_ot_units * OVERTIME_ADD
    """
    rows = list(col_att.find(
        {"cust_itinerary_id": str(iid), "car": {"$regex": f"^{TAK_PREFIX}", "$options": "i"}},
        {"driver":1,"date":1,"outstation_overnight":1,"overnight_client":1,"bhasmarathi":1,"billable_ot_units":1}
    ))
    if not rows:
        return 0
    buckets: Dict[Tuple[str,int,int], List[dict]] = {}
    for r in rows:
        dt = pd.to_datetime(r.get("date")).to_pydatetime()
        key = (str(r.get("driver","")), dt.year, dt.month)
        buckets.setdefault(key, []).append(r)

    total = 0
    for (drv, y, m), items in buckets.items():
        gross, dim, _leave, _ot_all = _driver_month_gross(drv, y, m)
        daily_rate = gross / max(dim, 1)
        client_days = len(items)
        client_ot = 0
        for r in items:
            bu = _to_int(r.get("billable_ot_units", 0))
            if bu > 0:
                client_ot += bu
            else:
                client_ot += int(bool(r.get("outstation_overnight", False)))
                client_ot += int(bool(r.get("overnight_client", False)))
                client_ot += int(bool(r.get("bhasmarathi", False)))
        total += int(round(daily_rate * client_days)) + client_ot * OVERTIME_ADD
    return int(total)

def profit_breakup(iid: str) -> dict:
    fc = final_cost_for(iid)
    vc = vendor_cost_for(iid)
    sc = splitwise_cost_for(iid)
    dr = driver_cost_for_client(iid)
    tot = vc + sc + dr
    return {
        "final_cost": fc,
        "vendor_total": vc,
        "splitwise_total": sc,
        "driver_total": dr,
        "total_expenses": tot,
        "actual_profit": max(fc - tot, 0)
    }

# ------------------------------------------------------------------
# Dedupe logic & revision view
# ------------------------------------------------------------------
def group_latest_by_mobile(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    df_all = df_all.copy()
    df_all["upload_date"] = pd.to_datetime(df_all["upload_date"])
    df_all["revision_num"] = pd.to_numeric(df_all["revision_num"], errors="coerce").fillna(0).astype(int)
    df_all.sort_values(["client_mobile", "start_date", "revision_num", "upload_date"],
                       ascending=[True, True, False, False], inplace=True)
    latest_per_pkg = df_all.groupby(["client_mobile","start_date"], as_index=False).first()
    latest_per_pkg.sort_values(["client_mobile","upload_date"], ascending=[True, False], inplace=True)
    latest_by_mobile = latest_per_pkg.groupby("client_mobile", as_index=False).first()
    hist_map: Dict[str, List[str]] = {}
    for mob, grp in latest_per_pkg.groupby("client_mobile"):
        ids = grp["itinerary_id"].tolist()
        hist_map[mob] = ids[1:] if len(ids) > 1 else []
    latest_by_mobile["history_ids"] = latest_by_mobile["client_mobile"].map(hist_map).apply(lambda x: x or [])
    return latest_by_mobile

def build_revision_table(df_all: pd.DataFrame, mobile: str) -> pd.DataFrame:
    sub = df_all[df_all["client_mobile"] == mobile].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["upload_date"] = pd.to_datetime(sub["upload_date"])
    sub["revision_num"] = pd.to_numeric(sub["revision_num"], errors="coerce").fillna(0).astype(int)
    cols = ["ach_id","itinerary_id","start_date","end_date","revision_num","upload_date","final_route","package_cost_num","discount","representative"]
    for c in cols:
        if c not in sub.columns:
            sub[c] = None
    sub.sort_values(["start_date","revision_num","upload_date"], ascending=[True, False, False], inplace=True)
    return sub[cols]

def _oid_time(iid: str) -> datetime:
    ts = _created_utc(iid)
    if ts is None:
        return datetime.min.replace(tzinfo=None)
    return ts

def latest_confirmed_unique_by_mobile(df_now: pd.DataFrame) -> pd.DataFrame:
    if df_now.empty:
        return df_now
    d = df_now.copy()
    d["booking_date"] = pd.to_datetime(d.get("booking_date")).dt.date
    d["_bd_sort"] = pd.to_datetime(d["booking_date"], errors="coerce")
    d["_ud_sort"] = pd.to_datetime(d.get("upload_date"), errors="coerce")
    d["_oid_sort"] = d["itinerary_id"].apply(_oid_time)
    d.sort_values(
        ["client_mobile", "_bd_sort", "_ud_sort", "_oid_sort"],
        ascending=[True, False, False, False],
        inplace=True
    )
    latest = d.groupby("client_mobile", as_index=False).first()
    latest.drop(columns=["_bd_sort","_ud_sort","_oid_sort"], inplace=True, errors="ignore")
    return latest

# ------------------------------------------------------------------
# Build page data (vectorized merges + final_cost)
# ------------------------------------------------------------------
df_it = fetch_itineraries_df()
if df_it.empty:
    st.info("No packages found yet. Create in the main app first.")
    st.stop()

df_up   = fetch_updates_df()
df_exp  = fetch_expenses_df()
df_vpay = fetch_vendorpay_df()
df_vend_master = fetch_vendor_master()

df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")

# numeric & dates
df = _ensure_cols(df, ["advance_amount"], 0)
df["advance_amount"] = pd.to_numeric(df["advance_amount"], errors="coerce").fillna(0).astype(int)
for col_ in ["start_date", "end_date", "booking_date", "upload_date"]:
    if col_ in df.columns:
        df[col_] = df[col_].apply(to_date_or_none)

# attach expenses (for final cost, profit)
df = df.merge(
    df_exp[["itinerary_id","base_package_cost","discount","final_package_cost","total_expenses","profit"]],
    on="itinerary_id", how="left"
)

# ---- SAFETY: ensure required numeric columns exist before using them ----
df = _ensure_cols(df, ["base_package_cost","discount","final_package_cost","package_cost_num"], 0)

# compute final cost (prefer explicit final, else compute from base-discount, else from itinerary)
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

df["total_expenses"] = pd.to_numeric(df.get("total_expenses", 0), errors="coerce").fillna(0).astype(int)
df["profit"] = pd.to_numeric(df.get("profit", df["final_cost"] - df["total_expenses"]), errors="coerce") \
                  .fillna(df["final_cost"] - df["total_expenses"]).astype(int)

# ------------------------------------------------------------------
# Sidebar: option
# ------------------------------------------------------------------
with st.sidebar:
    match_dash = st.checkbox("Match Dashboard (unique by mobile)", value=True)

# ------------------------------------------------------------------
# 💸 Vendor dues summary (quick)
# ------------------------------------------------------------------
def vendor_dues_summary_from_df(vdf: pd.DataFrame) -> pd.DataFrame:
    if vdf.empty:
        return pd.DataFrame(columns=["Vendor","Category","Bookings","Balance (₹)"])
    grp = vdf.groupby(["vendor","category"], as_index=False)["balance"].sum()
    cnt = vdf.groupby(["vendor","category"], as_index=False)["itinerary_id"].nunique().rename(columns={"itinerary_id":"Bookings"})
    out = grp.merge(cnt, on=["vendor","category"], how="left")
    out.rename(columns={"vendor":"Vendor","category":"Category","balance":"Balance (₹)"}, inplace=True)
    out.sort_values("Balance (₹)", ascending=False, inplace=True)
    return out

df_dues = vendor_dues_summary_from_df(df_vpay[df_vpay["balance"] > 0]) if not df_vpay.empty else pd.DataFrame()
d1, d2 = st.columns(2)
d1.metric("💸 Vendor payouts pending (₹)", f"{int(df_dues['Balance (₹)'].sum()) if not df_dues.empty else 0:,}")
d2.metric("🏷️ Vendors with dues", int(df_dues["Vendor"].nunique()) if not df_dues.empty else 0)
if not df_dues.empty:
    st.dataframe(df_dues, use_container_width=True, hide_index=True)
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

have_expense_ids = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
confirmed_latest = df_for_counts[df_for_counts["status"] == "confirmed"].copy()
confirmed_expense_pending = confirmed_latest[~confirmed_latest["itinerary_id"].isin(have_expense_ids)].shape[0]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("✅ Confirmed", confirmed_count)
k2.metric("🟡 Pending", int(pending_count))
k3.metric("🟠 Under discussion", int(under_discussion_count))
k4.metric("🔵 Follow-up", int(followup_count))
k5.metric("🔴 Cancelled", int(cancelled_count))
st.caption("Tip: Use ‘Match Dashboard’ to align counts with the Dashboard page.")
st.divider()

# ------------------------------------------------------------------
# 🗂️ Packages: recently created (with delete)
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
# 1) Update Status (pipeline)
# ------------------------------------------------------------------
st.subheader("1) Update Status")

view_mode = st.radio("View mode", ["Latest per client (by mobile)", "All packages"], horizontal=True)
editable = group_latest_by_mobile(df) if view_mode == "Latest per client (by mobile)" else df.copy()
editable = editable[editable["status"].isin(["pending","under_discussion"])].copy()

if editable.empty:
    st.success("Nothing to update right now. 🎉")
else:
    must_cols = [
        "ach_id","itinerary_id","client_name","client_mobile","final_route","total_pax",
        "start_date","end_date","package_cost_num","status","booking_date","advance_amount","assigned_to","revision_num"
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
        "start_date","end_date","package_cost","revision_num",
        "status","booking_date","advance_amount","assigned_to"
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
            "revision_num": st.column_config.NumberColumn("Rev", min_value=0, step=1, disabled=True),
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
                        up = {
                            "itinerary_id": str(r["itinerary_id"]),
                            "status": bulk_status,
                            "updated_at": datetime.utcnow(),
                            "advance_amount": _to_int(bulk_adv),
                            "assigned_to": assignee if bulk_status == "followup" else None,
                        }
                        up["booking_date"] = _to_dt_or_none(bdate) if (bulk_status == "confirmed" and bdate) else None
                        col_updates.update_one({"itinerary_id": str(r["itinerary_id"])}, {"$set": _clean_for_mongo(up)}, upsert=True)
                        applied += 1
                    except Exception:
                        skipped += 1

                st.success(f"Bulk updated: {applied} ✓")
                if skipped:
                    st.warning(f"Skipped: {skipped}")
                fetch_updates_df.clear()
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

    # --------- Client-wise history with revisions ----------
    if view_mode == "Latest per client (by mobile)":
        st.markdown("### Client-wise history (with revisions)")
        latest = group_latest_by_mobile(df)
        latest.sort_values("client_name", inplace=True)
        for _, row in latest.iterrows():
            hist_ids = row.get("history_ids", []) or []
            label = f"➕ {row.get('client_name','')} ({row.get('client_mobile','')})"
            with st.expander(label, expanded=False):
                mob = row.get("client_mobile","")
                rev_table = build_revision_table(df, mob)
                if rev_table.empty:
                    st.caption("No older packages for this client.")
                else:
                    st.dataframe(rev_table, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------
# 2) Final Cost & Vendor Payments (Confirmed Only)  + Header Expenses
# ------------------------------------------------------------------
st.subheader("2) Final Cost & Vendor Payments (Confirmed Only)")

df_now = df_it.merge(fetch_updates_df(), on="itinerary_id", how="left")
df_now["status"] = df_now["status"].fillna("pending")
df_now = _ensure_cols(df_now, ["advance_amount"], 0)
df_now["advance_amount"] = pd.to_numeric(df_now["advance_amount"], errors="coerce").fillna(0).astype(int)
for c in ["booking_date","start_date","end_date","upload_date"]:
    if c in df_now.columns:
        df_now[c] = df_now[c].apply(to_date_or_none)

# ✅ Keep ONLY ONE confirmed package per client (mobile): the latest
confirmed_all = df_now[df_now["status"] == "confirmed"].copy()
confirmed = latest_confirmed_unique_by_mobile(confirmed_all)

if confirmed.empty:
    st.info("No confirmed packages yet.")
else:
    fin_map = df.set_index("itinerary_id")["final_cost"].to_dict()
    exp_map = df.set_index("itinerary_id")[["total_expenses","profit"]].to_dict(orient="index")
    confirmed["final_cost"] = confirmed["itinerary_id"].map(fin_map).fillna(0).astype(int)
    confirmed["total_expenses"] = confirmed["itinerary_id"].map(lambda x: exp_map.get(x,{}).get("total_expenses",0)).fillna(0).astype(int)
    confirmed["profit"] = confirmed["itinerary_id"].map(lambda x: exp_map.get(x,{}).get("profit",0)).fillna(0).astype(int)

    search = st.text_input("🔎 Search confirmed clients (name/mobile/ACH ID)")
    view_tbl = confirmed.copy()
    if search.strip():
        s = search.strip().lower()
        view_tbl = view_tbl[
            view_tbl["client_name"].astype(str).str.lower().str.contains(s, na=False) |
            view_tbl["client_mobile"].astype(str).str.lower().str.contains(s, na=False) |
            view_tbl["ach_id"].astype(str).str.lower().str.contains(s, na=False)
        ]

    left, right = st.columns([2,1])
    with left:
        show_cols = ["ach_id","itinerary_id","client_name","client_mobile","final_route","total_pax",
                     "final_cost","advance_amount","booking_date"]
        view = view_tbl[show_cols].rename(columns={"final_cost":"Final package cost (₹)"})
        st.dataframe(view.sort_values("booking_date"), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Select a confirmed package to manage:**")
        options = (confirmed["ach_id"].fillna("") + " | " +
                   confirmed["client_name"].fillna("") + " | " +
                   confirmed["booking_date"].fillna("").astype(str) + " | " +
                   confirmed["itinerary_id"])
        sel = st.selectbox("Choose package", options.tolist() if not options.empty else [])
        chosen_id = sel.split(" | ")[-1] if sel else None

    # ------- Save one-time final package summary (with full expenses) -------
    def save_expense_summary(
        itinerary_id: str,
        client_name: str,
        booking_date,
        base_amount: int,
        discount: int,
        notes: str = ""
    ):
        vendor_total    = vendor_cost_for(itinerary_id)
        splitwise_total = splitwise_cost_for(itinerary_id)
        driver_total    = driver_cost_for_client(itinerary_id)

        base = _to_int(base_amount)
        disc = _to_int(discount)
        final_cost = max(0, base - disc)
        total_expenses = vendor_total + splitwise_total + driver_total
        profit = max(final_cost - total_expenses, 0)

        doc = {
            "itinerary_id": str(itinerary_id),
            "client_name": str(client_name or ""),
            "booking_date": _to_dt_or_none(booking_date),
            "base_package_cost": base,
            "discount": disc,
            "final_package_cost": final_cost,
            "package_cost": final_cost,  # legacy
            "vendor_total": _to_int(vendor_total),
            "splitwise_total": _to_int(splitwise_total),
            "driver_total": _to_int(driver_total),
            "total_expenses": _to_int(total_expenses),
            "profit": _to_int(profit),
            "notes": str(notes or ""),
            "saved_at": datetime.utcnow(),
        }
        col_expenses.update_one({"itinerary_id": str(itinerary_id)}, {"$set": _clean_for_mongo(doc)}, upsert=True)
        fetch_expenses_df.clear()
        return profit, total_expenses, final_cost, vendor_total, splitwise_total, driver_total

    # ------- Vendor payments helpers -------
    def get_vendor_rows(itinerary_id: str) -> pd.DataFrame:
        vdf = fetch_vendorpay_df()
        return vdf[vdf["itinerary_id"] == itinerary_id].copy() if not vdf.empty else pd.DataFrame()

    def save_vendor_rows(itinerary_id: str, rows_df: pd.DataFrame, final_done: bool):
        rows_df = rows_df.copy()
        need_cols = ["category","vendor","finalization_cost","adv1_amt","adv1_date","adv2_amt","adv2_date","final_amt","final_date","balance"]
        for c in need_cols:
            if c not in rows_df.columns:
                rows_df[c] = 0 if c not in ("category","vendor","adv1_date","adv2_date","final_date") else ""
        items = []
        for _, r in rows_df.iterrows():
            items.append({
                "category": r.get("category",""),
                "vendor": r.get("vendor",""),
                "finalization_cost": _to_int(r.get("finalization_cost", 0)),
                "adv1_amt": _to_int(r.get("adv1_amt", 0)),
                "adv1_date": r.get("adv1_date").isoformat() if pd.notna(r.get("adv1_date")) and r.get("adv1_date") else None,
                "adv2_amt": _to_int(r.get("adv2_amt", 0)),
                "adv2_date": r.get("adv2_date").isoformat() if pd.notna(r.get("adv2_date")) and r.get("adv2_date") else None,
                "final_amt": _to_int(r.get("final_amt", 0)),
                "final_date": r.get("final_date").isoformat() if pd.notna(r.get("final_date")) and r.get("final_date") else None,
                "balance": max(_to_int(r.get("finalization_cost", 0)) - (_to_int(r.get("adv1_amt", 0)) + _to_int(r.get("adv2_amt", 0)) + _to_int(r.get("final_amt", 0))), 0),
            })
        doc = {
            "itinerary_id": str(itinerary_id),
            "final_done": bool(final_done),
            "items": _clean_for_mongo(items),
            "updated_at": datetime.utcnow(),
        }
        col_vendorpay.update_one({"itinerary_id": str(itinerary_id)}, {"$set": _clean_for_mongo(doc)}, upsert=True)
        fetch_vendorpay_df.clear()

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

    # ------- UI for selected confirmed package -------
    if chosen_id:
        row = df[df["itinerary_id"] == chosen_id].iloc[0].to_dict()
        client_name  = row.get("client_name","")
        booking_date = row.get("booking_date","")
        advance      = _to_int(df_up[df_up["itinerary_id"]==chosen_id]["advance_amount"].fillna(0).sum()) if not df_up.empty else 0

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

        # ---- Final Package Summary ----
        st.markdown("#### Final Package Summary")
        exp_row = df_exp[df_exp["itinerary_id"] == chosen_id].head(1)
        base_default = int(exp_row["base_package_cost"].iat[0]) if not exp_row.empty else _to_int(row.get("package_cost_num"))
        disc_default = int(exp_row["discount"].iat[0]) if not exp_row.empty else _to_int(row.get("discount", 0))
        current_final = int(df[df["itinerary_id"] == chosen_id]["final_cost"].iloc[0])

        c1c, c2c, c3c = st.columns(3)
        with c1c:
            base_amount = st.number_input("Quoted/Initial amount (₹)", min_value=0, step=500, value=int(base_default))
        with c2c:
            discount = st.number_input("Discount (₹)", min_value=0, step=500, value=int(disc_default))
        with c3c:
            computed_final = max(0, int(base_amount) - int(discount))
            st.metric("Final package (computed)", f"₹ {computed_final:,}")

        # Live header totals (Vendor, Splitwise, Driver, Total, Profit)
        br_now = profit_breakup(chosen_id)
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Vendor (₹)", f"{br_now['vendor_total']:,}")
        h2.metric("Splitwise (₹)", f"{br_now['splitwise_total']:,}")
        h3.metric("Driver (₹)", f"{br_now['driver_total']:,}")
        h4.metric("Total expenses (₹)", f"{br_now['total_expenses']:,}")
        h5.metric("Profit (₹)", f"{max(computed_final - br_now['total_expenses'], 0):,}")

        notes = st.text_area("Notes (optional)", value=str(exp_row["notes"].iat[0]) if (not exp_row.empty and "notes" in exp_row.columns) else "")

        if st.button("💾 Save Final Package & Expenses"):
            profit, total_expenses, final_cost, vtot, swtot, drtot = save_expense_summary(
                chosen_id, client_name, booking_date, base_amount, discount, notes
            )
            st.success(f"Saved. Final: ₹{final_cost:,} • Vendor: ₹{vtot:,} • Splitwise: ₹{swtot:,} • Driver: ₹{drtot:,} • Total: ₹{total_expenses:,} • Profit: ₹{profit:,}")
            st.rerun()

        # ---- Vendor Payments ----
        st.markdown("### Vendor Payments")
        st.caption("Add or edit vendor rows freely. Balance auto-calculates. You can lock with **Final done** if needed.")

        with st.expander("➕ Add vendor to master"):
            vc1, vc2, vc3 = st.columns([1,1,2])
            with vc1:
                new_cat = st.selectbox("Category", ["Hotel","Car","Bhasmarathi","Poojan","PhotoFrame","Others"])
            with vc2:
                new_vendor = st.text_input("Vendor name")
            with vc3:
                new_contact = st.text_input("Contact (optional)")
            if st.button("Add vendor to master"):
                if (new_vendor or "").strip():
                    try:
                        col_vendors.update_one(
                            {"vendor": new_vendor.strip(), "category": new_cat},
                            {"$set": {"vendor": new_vendor.strip(), "category": new_cat, "active": True, "contact": new_contact.strip(), "updated_at": datetime.utcnow()}},
                            upsert=True
                        )
                        fetch_vendor_master.clear()
                        st.success("Vendor added.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not add vendor: {e}")
                else:
                    st.warning("Enter a vendor name.")

        vendor_options = sorted(df_vend_master["vendor"].unique().tolist()) if not df_vend_master.empty else []

        current_rows = get_vendor_rows(chosen_id)
        if current_rows.empty:
            current_rows = pd.DataFrame([{
                "category":"Hotel","vendor":"","finalization_cost":0,
                "adv1_amt":0,"adv1_date":None,"adv2_amt":0,"adv2_date":None,
                "final_amt":0,"final_date":None,"balance":0,"final_done":False
            }])
        final_done_state = bool(current_rows["final_done"].iloc[0]) if "final_done" in current_rows.columns and not current_rows.empty else False

        edited_vendors = st.data_editor(
            current_rows.drop(columns=["itinerary_id","final_done"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "category": st.column_config.SelectboxColumn("Category", options=["Hotel","Car","Bhasmarathi","Poojan","PhotoFrame","Others"]),
                "vendor": st.column_config.SelectboxColumn("Vendor", options=[""] + vendor_options),
                "finalization_cost": st.column_config.NumberColumn("Finalization (₹)", min_value=0, step=100),
                "adv1_amt": st.column_config.NumberColumn("Adv-1 (₹)", min_value=0, step=100),
                "adv1_date": st.column_config.DateColumn("Adv-1 Date"),
                "adv2_amt": st.column_config.NumberColumn("Adv-2 (₹)", min_value=0, step=100),
                "adv2_date": st.column_config.DateColumn("Adv-2 Date"),
                "final_amt": st.column_config.NumberColumn("Final paid (₹)", min_value=0, step=100),
                "final_date": st.column_config.DateColumn("Final Date"),
                "balance": st.column_config.NumberColumn("Balance (₹)", disabled=True),
            },
            key=f"vendor_editor_{chosen_id}"
        )

        colfd1, colfd2 = st.columns([1,1])
        with colfd1:
            final_done_new = st.checkbox("Final done (lock further edits)", value=final_done_state)
        with colfd2:
            if st.button("💾 Save Vendor Payments"):
                if not edited_vendors.empty:
                    for i in edited_vendors.index:
                        edited_vendors.loc[i, "balance"] = max(
                            _to_int(edited_vendors.loc[i, "finalization_cost"]) -
                            (_to_int(edited_vendors.loc[i, "adv1_amt"]) + _to_int(edited_vendors.loc[i, "adv2_amt"]) + _to_int(edited_vendors.loc[i, "final_amt"])),
                            0
                        )
                save_vendor_rows(chosen_id, edited_vendors, final_done_new)
                br2 = profit_breakup(chosen_id)
                st.success(f"Vendor payments saved. Vendor: ₹{br2['vendor_total']:,} • Splitwise: ₹{br2['splitwise_total']:,} • Driver: ₹{br2['driver_total']:,} • Total: ₹{br2['total_expenses']:,} • Profit: ₹{max(final_cost_for(chosen_id)-br2['total_expenses'],0):,}")
                st.rerun()

        # ---- Client & Vendor Payment Summaries ----
        st.markdown("### 📑 Payment Summaries")
        final_now = final_cost_for(chosen_id)
        received  = _to_int(df_up[df_up["itinerary_id"]==chosen_id]["advance_amount"].fillna(0).sum()) if not df_up.empty else 0
        balance   = max(final_now - received, 0)
        cS1, cS2, cS3 = st.columns(3)
        cS1.metric("Final package (₹)", f"{final_now:,}")
        cS2.metric("Received from client (₹)", f"{received:,}")
        cS3.metric("Pending from client (₹)", f"{balance:,}")

        vtab = get_vendor_rows(chosen_id)
        if not vtab.empty:
            vtab = vtab[["category","vendor","finalization_cost","adv1_amt","adv2_amt","final_amt","balance","adv1_date","adv2_date","final_date"]]
            vtab.rename(columns={
                "category":"Category","vendor":"Vendor","finalization_cost":"Finalization (₹)",
                "adv1_amt":"Adv-1 (₹)","adv2_amt":"Adv-2 (₹)","final_amt":"Final paid (₹)","balance":"Balance (₹)",
                "adv1_date":"Adv-1 Date","adv2_date":"Adv-2 Date","final_date":"Final Date"
            }, inplace=True)
            st.dataframe(vtab.sort_values(["Category","Vendor"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No vendor rows yet.")

st.divider()

# ------------------------------------------------------------------
# 3) Calendar – Confirmed Packages
# ------------------------------------------------------------------
st.subheader("3) Calendar – Confirmed Packages")
view = st.radio("View", ["By Booking Date", "By Travel Dates"], horizontal=True)

confirmed_view = latest_confirmed_unique_by_mobile(df[df["status"] == "confirmed"].copy())
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
            selected = st.selectbox(
                "Open package details",
                (confirmed_view["itinerary_id"] + " | " + confirmed_view["client_name"]).tolist()
            )
            if selected:
                selected_id = selected.split(" | ")[0]

    if selected_id:
        st.divider()
        st.subheader("📦 Package Details")

        it = df_it[df_it["itinerary_id"] == selected_id].iloc[0].to_dict()
        upd = df_up[df_up["itinerary_id"] == selected_id].iloc[0].to_dict() if not df_up.empty and (df_up["itinerary_id"]==selected_id).any() else {}
        exp = df_exp[df_exp["itinerary_id"] == selected_id].iloc[0].to_dict() if not df_exp.empty and (df_exp["itinerary_id"]==selected_id).any() else {}

        created_dt_utc = _created_utc(selected_id)
        created_ist_str = _fmt_ist(created_dt_utc)
        created_utc_str = created_dt_utc.strftime("%Y-%m-%d %H:%M %Z") if created_dt_utc else ""

        br = profit_breakup(selected_id)

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
                "Revision": it.get("revision_num", 0),
            })
        with c2:
            st.markdown("**Status & Money**")
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
                "Final package (₹)": br["final_cost"],
                "Vendor (₹)": br["vendor_total"],
                "Splitwise (₹)": br["splitwise_total"],
                "Driver (₹)": br["driver_total"],
                "Total expenses (₹)": br["total_expenses"],
                "Profit (₹)": br["actual_profit"],
            })

        st.markdown("**Itinerary text**")
        st.text_area("Shared with client", value=(it.get("itinerary_text","") or ""), height=260, disabled=True)
