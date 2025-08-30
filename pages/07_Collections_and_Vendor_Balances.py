# pages/07_Collections_and_Vendor_Balances.py
from __future__ import annotations

import os, io
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Collections & Vendor Balances", layout="wide")
st.title("💰 Collections (Customer) & Vendor Balances — Confirmed Packages Only")

IST = ZoneInfo("Asia/Kolkata")

# =========================================================
# Admin gate (same pattern as other admin pages)
# =========================================================
def require_admin():
    ADMIN_PASS_DEFAULT = "Arpith&92"
    ADMIN_PASS = str(st.secrets.get("admin_pass", ADMIN_PASS_DEFAULT))
    with st.sidebar:
        st.markdown("### Admin access")
        p = st.text_input("Enter admin password", type="password", placeholder="enter pass", key="admin_pass_input")
    if (p or "").strip() != ADMIN_PASS.strip():
        st.stop()
    st.session_state["user"] = "Admin"
    st.session_state["is_admin"] = True

require_admin()

# Optional audit (if available)
try:
    from tak_audit import audit_pageview
    audit_pageview(st.session_state.get("user", "Unknown"), page="07_Collections_and_Vendor_Balances")
except Exception:
    pass

# =========================================================
# Mongo (robust URI finder)
# =========================================================
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
    uri = _find_uri() or st.secrets["mongo_uri"]
    client = MongoClient(
        uri,
        appName="TAK_Collections_Vendors",
        serverSelectionTimeoutMS=6000,
        connectTimeoutMS=6000,
        tz_aware=True,
        maxPoolSize=100,
    )
    client.admin.command("ping")
    return client

db = _get_client()["TAK_DB"]
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_expenses    = db["expenses"]
col_vendorpay   = db["vendor_payments"]          # legacy per-item fields (adv1/adv2/final)
col_vendors     = db["vendors"]

# NEW lightweight collections
col_cust_txn    = db["customer_payments"]        # {itinerary_id, amount, date, mode, utr, collected_by, note, term_label, term_percent, created_at}
col_vendor_txn  = db["vendor_payment_txns"]      # {itinerary_id, vendor, category, amount, date, utr, mode, collected_by, type, note, term_label, term_percent, created_at}
col_reminders   = db["payment_reminders"]        # {for: 'customer'|'vendor', itinerary_id?, vendor?, due_date, amount, note, status}
col_terms       = db["payment_terms"]            # {scope:'customer'|'vendor', itinerary_id, vendor?, category?, plan:[{label, percent}], created_at}

# =========================================================
# Helpers
# =========================================================
def _to_int(x, default=0) -> int:
    try:
        if x is None: return default
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def _d(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

def month_bounds(d: date) -> Tuple[date, date]:
    first = d.replace(day=1)
    last = (first + pd.offsets.MonthEnd(1)).date()
    return first, last

def _now_utc():
    return datetime.utcnow()

def ensure_cols(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    """Guarantee required columns exist with safe defaults."""
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    return df

# Final cost resolver (in sync with other pages)
def _final_cost_for(iid: str) -> int:
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
    it = col_itineraries.find_one({"_id": ObjectId(iid)}, {"package_cost": 1, "discount": 1}) or {}
    base = _to_int(it.get("package_cost", 0))
    disc = _to_int(it.get("discount", 0))
    return max(0, base - disc)

# =========================================================
# Latest packages only (unique per client_mobile + start_date) — then filter to CONFIRMED only
# =========================================================
@st.cache_data(ttl=120, show_spinner=False)
def load_latest_itineraries_df() -> pd.DataFrame:
    """
    Picks only the latest revision for each (client_mobile, start_date).
    """
    pipeline = [
        {"$sort": {"client_mobile": 1, "start_date": 1, "revision_num": -1, "upload_date": -1}},
        {"$group": {
            "_id": {"m": "$client_mobile", "s": "$start_date"},
            "doc": {"$first": "$$ROOT"}
        }},
        {"$replaceRoot": {"newRoot": "$doc"}},
        {"$project": {
            "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
            "final_route": 1, "total_pax": 1, "representative": 1,
            "start_date": 1, "revision_num": 1, "upload_date": 1
        }},
    ]
    rows = list(col_itineraries.aggregate(pipeline))
    for r in rows:
        r["itinerary_id"] = str(r.pop("_id"))
        r["start_date"] = str(r.get("start_date", ""))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "itinerary_id","ach_id","client_name","client_mobile","final_route","total_pax",
        "representative","start_date","revision_num","upload_date"
    ])

@st.cache_data(ttl=120, show_spinner=False)
def load_confirmed_snapshot() -> pd.DataFrame:
    ups = list(col_updates.find(
        {"status": "confirmed"},
        {"_id":0, "itinerary_id":1, "status":1, "booking_date":1, "advance_amount":1, "rep_name":1}
    ))
    for u in ups:
        u["itinerary_id"] = str(u.get("itinerary_id", ""))
        u["booking_date"] = _d(u.get("booking_date"))
        u["advance_amount"] = _to_int(u.get("advance_amount", 0))
        u["rep_name"] = u.get("rep_name", "")
    return pd.DataFrame(ups) if ups else pd.DataFrame(columns=[
        "itinerary_id","booking_date","advance_amount","rep_name"
    ])

@st.cache_data(ttl=120, show_spinner=False)
def load_customer_payments() -> pd.DataFrame:
    rows = []
    cur = col_cust_txn.find({}, {"_id":0})
    for d in cur:
        d["itinerary_id"] = str(d.get("itinerary_id",""))
        d["amount"] = _to_int(d.get("amount",0))
        d["date"] = _d(d.get("date"))
        rows.append(d)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "itinerary_id","amount","date","mode","utr","collected_by","term_label","term_percent","note","created_at"
    ])

@st.cache_data(ttl=120, show_spinner=False)
def load_vendor_payment_txns() -> pd.DataFrame:
    rows = []
    cur = col_vendor_txn.find({}, {"_id":0})
    for d in cur:
        d["itinerary_id"] = str(d.get("itinerary_id",""))
        d["amount"] = _to_int(d.get("amount",0))
        d["date"] = _d(d.get("date"))
        d["type"] = (d.get("type") or "").strip()  # 'adv'|'final'|'other'
        d["vendor"] = (d.get("vendor") or "").strip()
        d["category"] = (d.get("category") or "").strip()
        rows.append(d)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "itinerary_id","vendor","category","amount","date","mode","utr","collected_by","type","term_label","term_percent","note","created_at"
    ])

@st.cache_data(ttl=120, show_spinner=False)
def load_vendor_payments_legacy() -> pd.DataFrame:
    rows = []
    cur = col_vendorpay.find({}, {"_id": 0, "itinerary_id": 1, "final_done": 1, "items": 1, "updated_at": 1})
    for d in cur:
        iid = str(d.get("itinerary_id", ""))
        for it in d.get("items", []) or []:
            cat = (it.get("category") or "").strip()
            vendor = (it.get("vendor") or "").strip()
            fc = _to_int(it.get("finalization_cost", 0))
            a1 = _to_int(it.get("adv1_amt", 0)); a1d = _d(it.get("adv1_date"))
            a2 = _to_int(it.get("adv2_amt", 0)); a2d = _d(it.get("adv2_date"))
            fa = _to_int(it.get("final_amt", 0)); fad = _d(it.get("final_date"))
            bal = it.get("balance")
            if bal is None:
                bal = max(fc - (a1 + a2 + fa), 0)
            bal = _to_int(bal, 0)
            rows.append({
                "itinerary_id": iid,
                "category": cat,
                "vendor": vendor,
                "finalization_cost": fc,
                "adv1_amt": a1, "adv1_date": a1d,
                "adv2_amt": a2, "adv2_date": a2d,
                "final_amt": fa, "final_date": fad,
                "paid_total": a1 + a2 + fa,
                "balance": bal,
                "final_done": bool(d.get("final_done", False)),
                "updated_at": d.get("updated_at")
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "itinerary_id","category","vendor","finalization_cost",
        "adv1_amt","adv1_date","adv2_amt","adv2_date","final_amt","final_date",
        "paid_total","balance","final_done","updated_at"
    ])

@st.cache_data(ttl=120, show_spinner=False)
def load_vendor_directory() -> pd.DataFrame:
    vs = list(col_vendors.find({}, {"_id":0, "name":1, "city":1, "category":1, "contact":1}))
    return pd.DataFrame(vs) if vs else pd.DataFrame(columns=["name","city","category","contact"])

@st.cache_data(ttl=120, show_spinner=False)
def load_terms(scope: str, itinerary_id: str, vendor: Optional[str]=None, category: Optional[str]=None) -> List[Dict[str, Any]]:
    q = {"scope": scope, "itinerary_id": itinerary_id}
    if scope == "vendor":
        q["vendor"] = vendor or ""
        q["category"] = (category or "").strip()
    doc = col_terms.find_one(q, {"_id":0, "plan":1})
    return (doc or {}).get("plan", [])

# =========================================================
# Filters
# =========================================================
today = date.today()
mstart, mend = month_bounds(today)

with st.container():
    fl1, fl2, fl3, fl4 = st.columns([1.2, 1.2, 1.6, 2.0])
    with fl1:
        basis = st.selectbox("Customer date basis", ["Booking date", "Custom range"], index=0, key="cust_date_basis")
    with fl2:
        if basis == "Booking date":
            start_c, end_c = mstart, mend
        else:
            start_c, end_c = today - timedelta(days=30), today
        start_c = st.date_input("From (customers)", value=start_c, key="cust_from")
    with fl3:
        end_c = st.date_input("To (customers)", value=end_c, key="cust_to")
        if end_c < start_c:
            end_c = start_c
    with fl4:
        search_txt = st.text_input("Search customer (name/mobile/ACH/route)", "", key="cust_search_text")

vl1, vl2 = st.columns([1.2, 2.8])
with vl1:
    vendor_date_mode = st.selectbox("Vendor filter", ["Any payment in range", "Ignore dates"], index=0, key="vendor_filter_mode")
with vl2:
    start_v = st.date_input("Vendor payments from", value=mstart, key="vendor_from")
    end_v   = st.date_input("Vendor payments to", value=mend, key="vendor_to")
    if end_v < start_v:
        end_v = start_v

st.divider()

# =========================================================
# Data prep (latest-only packages) — then filter to CONFIRMED only
# =========================================================
df_latest = load_latest_itineraries_df()  # unique latest packages
df_conf   = load_confirmed_snapshot()     # ONLY confirmed
df_cpay   = load_customer_payments()
df_vdir   = load_vendor_directory()

# Keep ONLY confirmed packages (inner-join on itinerary_id)
if not df_latest.empty and not df_conf.empty:
    df_cust = df_latest.merge(df_conf, on="itinerary_id", how="inner")   # <- confirmed only
else:
    df_cust = pd.DataFrame(columns=[
        "itinerary_id","ach_id","client_name","client_mobile","final_route","total_pax",
        "representative","start_date","revision_num","upload_date",
        "booking_date","advance_amount","rep_name"
    ])

# Compute FINAL cost, RECEIVED, PENDING for confirmed-only
if not df_cust.empty:
    finals = [ _final_cost_for(i) for i in df_cust["itinerary_id"] ]
    df_cust["final_cost"] = pd.to_numeric(pd.Series(finals, index=df_cust.index), errors="coerce").fillna(0).astype(int)

    # Customer received from txns (+ legacy advance_amount)
    if not df_cpay.empty:
        recv = df_cpay.groupby("itinerary_id", as_index=False)["amount"].sum().rename(columns={"amount":"received_txn"})
        df_cust = df_cust.merge(recv, on="itinerary_id", how="left")
    else:
        df_cust["received_txn"] = 0
    df_cust["received_legacy"] = pd.to_numeric(df_cust.get("advance_amount", 0), errors="coerce").fillna(0).astype(int)
    df_cust["received"] = (df_cust["received_txn"].fillna(0) + df_cust["received_legacy"]).astype(int)
    df_cust["pending"]  = (df_cust["final_cost"] - df_cust["received"]).clip(lower=0).astype(int)

    # last payment info
    if not df_cpay.empty:
        last_pay = (
            df_cpay.sort_values(["itinerary_id","date"], ascending=[True, True])
                   .groupby("itinerary_id")
                   .agg(last_pay_date=("date","last"), last_utr=("utr","last"))
                   .reset_index()
        )
        df_cust = df_cust.merge(last_pay, on="itinerary_id", how="left")

# Ensure columns exist
df_cust = ensure_cols(df_cust, {
    "ach_id":"", "client_name":"", "client_mobile":"", "final_route":"", "total_pax":0,
    "booking_date": None, "rep_name":"", "final_cost":0, "received":0, "pending":0,
    "last_pay_date": None, "last_utr":"", "itinerary_id":""
})

# Vendor transactions (new) + legacy
df_vtxn = load_vendor_payment_txns()
df_vlegacy = load_vendor_payments_legacy()

# Build unified vendor lines
if df_vlegacy.empty and df_vtxn.empty:
    df_v = pd.DataFrame(columns=[
        "itinerary_id","vendor","category","finalization_cost","paid_total","balance",
        "last_pay_date","last_utr","source","city"
    ])
else:
    # Start with legacy lines (already balanced)
    df_v = df_vlegacy.copy()
    df_v["last_pay_date"] = df_v[["adv1_date","adv2_date","final_date"]].max(axis=1)
    df_v["last_utr"] = None
    df_v["source"] = "legacy"

    # Add txn-based lines (compute paid_total, last date/utr)
    if not df_vtxn.empty:
        ag = (
            df_vtxn.groupby(["itinerary_id","vendor","category"], dropna=False)
                   .agg(
                        paid_total=("amount","sum"),
                        last_pay_date=("date","max"),
                        last_utr=("utr","last")
                    ).reset_index()
        )
        # Try to fetch "finalization_cost" from legacy if present
        if not df_vlegacy.empty:
            base_fc = df_vlegacy[["itinerary_id","vendor","category","finalization_cost"]].drop_duplicates()
            ag = ag.merge(base_fc, on=["itinerary_id","vendor","category"], how="left")
        else:
            ag["finalization_cost"] = 0

        ag["balance"] = (pd.to_numeric(ag["finalization_cost"], errors="coerce").fillna(0) - ag["paid_total"]).clip(lower=0).astype(int)
        ag["source"] = "txns"
        df_v = pd.concat([df_v, ag], ignore_index=True, sort=False)

# Restrict df_v to CONFIRMED itineraries only
if not df_v.empty and not df_cust.empty:
    confirmed_ids = set(df_cust["itinerary_id"].unique())
    df_v = df_v[df_v["itinerary_id"].isin(confirmed_ids)].copy()

# Add vendor directory info
df_vdir = df_vdir if 'df_vdir' in locals() else load_vendor_directory()
if not df_v.empty and not df_vdir.empty:
    vdir = (
        df_vdir[["name","city","category"]]
        .rename(columns={"name":"vendor"})
        .drop_duplicates(["vendor","category"])
    )
    df_v = df_v.merge(vdir, on=["vendor","category"], how="left")

# Ensure vendor columns exist
df_v = ensure_cols(df_v, {
    "itinerary_id":"", "vendor":"", "category":"", "city":"", "finalization_cost":0,
    "paid_total":0, "balance":0, "last_pay_date": None, "last_utr":"", "source":""
})

# =========================================================
# Apply filters (confirmed packages only)
# =========================================================
if not df_cust.empty:
    mask_range = df_cust["booking_date"].apply(lambda x: isinstance(x, date) and (start_c <= x <= end_c))
    df_cust = df_cust[mask_range | df_cust["booking_date"].isna()]
    if (search_txt or "").strip():
        s = search_txt.strip().lower()
        df_cust = df_cust[
            df_cust["client_name"].astype(str).str.lower().str.contains(s) |
            df_cust["client_mobile"].astype(str).str.lower().str.contains(s) |
            df_cust["ach_id"].astype(str).str.lower().str.contains(s) |
            df_cust["final_route"].astype(str).str.lower().str.contains(s)
        ]

# Vendors: filter by any txn date in range
if not df_v.empty and vendor_date_mode == "Any payment in range":
    def in_range(row) -> bool:
        d = row.get("last_pay_date")
        return isinstance(d, date) and (start_v <= d <= end_v)
    df_v = df_v[df_v.apply(in_range, axis=1)].copy()

st.success("Showing CONFIRMED packages only — latest revision per (client_mobile, start_date).")

# =========================================================
# 🔧 Payment Terms (Customer & Vendor)
# =========================================================
st.subheader("🧩 Define Payment Terms / Plan")

with st.expander("🧑‍💼 Customer Payment Terms", expanded=False):
    if df_cust.empty:
        st.info("No confirmed packages.")
    else:
        opts = (df_cust["ach_id"].fillna("") + " | " + df_cust["client_name"].fillna("") +
                " | " + df_cust["booking_date"].astype(str) + " | " + df_cust["itinerary_id"])
        pick = st.selectbox("Select package", opts.tolist(), key="cust_terms_pkg")
        sel_iid = pick.split(" | ")[-1] if pick else None

        tmpl = st.selectbox("Template", ["-- Select --", "50-20-30", "25-25-25-25", "100"], key="cust_terms_template")
        custom = st.text_input("Custom percentages (comma-separated, e.g., 40,30,20,10)", key="cust_terms_custom")

        if st.button("Save Customer Terms", key="save_cust_terms_btn"):
            if not sel_iid:
                st.error("Pick a package.")
            else:
                if tmpl == "50-20-30":
                    plan = [{"label": "Break 1", "percent": 50}, {"label": "Break 2", "percent": 20}, {"label": "Break 3", "percent": 30}]
                elif tmpl == "25-25-25-25":
                    plan = [{"label": f"Break {i}", "percent": 25} for i in range(1,5)]
                elif tmpl == "100":
                    plan = [{"label": "Full", "percent": 100}]
                else:
                    # parse custom
                    try:
                        parts = [int(p.strip()) for p in custom.split(",") if p.strip() != ""]
                        if sum(parts) != 100: raise ValueError("Sum must be 100")
                        plan = [{"label": f"Break {i+1}", "percent": parts[i]} for i in range(len(parts))]
                    except Exception as e:
                        st.error(f"Invalid custom plan: {e}")
                        plan = []
                if plan:
                    col_terms.update_one(
                        {"scope":"customer","itinerary_id": sel_iid},
                        {"$set": {"scope":"customer","itinerary_id": sel_iid, "plan": plan, "created_at": _now_utc()}},
                        upsert=True
                    )
                    st.success("Customer payment terms saved.")

with st.expander("🏷️ Vendor Payment Terms (per vendor/category)", expanded=False):
    if df_cust.empty:
        st.info("No confirmed packages.")
    else:
        opts = (df_cust["ach_id"].fillna("") + " | " + df_cust["client_name"].fillna("") +
                " | " + df_cust["booking_date"].astype(str) + " | " + df_cust["itinerary_id"])
        pick = st.selectbox("Select package", opts.tolist(), key="vend_terms_pkg")
        sel_iid = pick.split(" | ")[-1] if pick else None

        vend_names = sorted(col_vendors.distinct("name"))
        vsel = st.selectbox("Vendor", ["--"] + vend_names, key="vend_terms_vendor")
        vcat = st.text_input("Category (Hotel/Taxi/Guide...)", key="vend_terms_cat")

        tmpl = st.selectbox("Template", ["-- Select --", "50-50", "40-60", "30-30-40"], key="vend_terms_template")
        custom = st.text_input("Custom percentages (comma-separated, sum 100)", key="vend_terms_custom")

        if st.button("Save Vendor Terms", key="save_vend_terms_btn"):
            if not sel_iid or vsel == "--":
                st.error("Pick package & vendor.")
            else:
                if tmpl == "50-50":
                    plan = [{"label": "Break 1", "percent": 50}, {"label": "Break 2", "percent": 50}]
                elif tmpl == "40-60":
                    plan = [{"label": "Break 1", "percent": 40}, {"label": "Break 2", "percent": 60}]
                elif tmpl == "30-30-40":
                    plan = [{"label": "Break 1", "percent": 30}, {"label": "Break 2", "percent": 30}, {"label": "Break 3", "percent": 40}]
                else:
                    try:
                        parts = [int(p.strip()) for p in custom.split(",") if p.strip() != ""]
                        if sum(parts) != 100: raise ValueError("Sum must be 100")
                        plan = [{"label": f"Break {i+1}", "percent": parts[i]} for i in range(len(parts))]
                    except Exception as e:
                        st.error(f"Invalid custom plan: {e}")
                        plan = []
                if plan:
                    col_terms.update_one(
                        {"scope":"vendor","itinerary_id": sel_iid, "vendor": vsel.strip(), "category": vcat.strip()},
                        {"$set": {"scope":"vendor","itinerary_id": sel_iid, "vendor": vsel.strip(), "category": vcat.strip(),
                                  "plan": plan, "created_at": _now_utc()}},
                        upsert=True
                    )
                    st.success("Vendor payment terms saved.")

st.divider()

# =========================================================
# Entry forms (create vendors / record payments / reminders)
# =========================================================
with st.expander("➕ Create / Update Vendor", expanded=False):
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    with vcol1: v_name = st.text_input("Vendor name*", key="vendor_name_input")
    with vcol2: v_city = st.text_input("City", key="vendor_city_input")
    with vcol3: v_cat  = st.text_input("Category", key="vendor_category_input")
    with vcol4: v_contact = st.text_input("Contact (phone/email)", key="vendor_contact_input")
    if st.button("Save vendor", key="vendor_save_btn"):
        if not v_name.strip():
            st.error("Vendor name is required.")
        else:
            col_vendors.update_one(
                {"name": v_name.strip()},
                {"$set": {"name": v_name.strip(), "city": v_city.strip(), "category": v_cat.strip(), "contact": v_contact.strip()}},
                upsert=True
            )
            st.success("Vendor saved/updated.")

with st.expander("🧾 Record Customer Receipt (multi-break, mode/UTR/receiver, optional reminder)", expanded=False):
    if df_cust.empty:
        st.info("No confirmed packages.")
    else:
        options = (df_cust["ach_id"].fillna("") + " | " + df_cust["client_name"].fillna("") +
                   " | " + df_cust["booking_date"].astype(str) + " | " + df_cust["itinerary_id"])
        pick = st.selectbox("Select package", options.tolist(), key="rec_cust_pkg")
        sel_iid = pick.split(" | ")[-1] if pick else None

        # Load terms plan (if any)
        plan = load_terms("customer", sel_iid) if sel_iid else []
        term_labels = ["--"] + [f'{x["label"]} ({x["percent"]}%)' for x in plan] if plan else ["--"]

        c1,c2,c3,c4 = st.columns([1,1,1,1.5])
        with c1: amt = st.number_input("Amount (₹)", min_value=0, step=500, key="cust_amount_input")
        with c2: dtp = st.date_input("Payment date", value=today, key="cust_payment_date")
        with c3: mode = st.selectbox("Mode", ["UPI","NEFT/RTGS","IMPS","Cash","Card","Other"], key="cust_mode_select")
        with c4: utr = st.text_input("UTR / Ref no.", key="cust_utr_input")
        collected_by = st.text_input("If Cash: collected by (employee)", key="cust_collected_by")
        tsel = st.selectbox("Map to term (optional)", term_labels, key="cust_term_select")
        note = st.text_input("Note (optional)", key="cust_note_input")
        mk_rem = st.checkbox("Also create reminder for remaining balance", value=False, key="cust_make_reminder")

        if st.button("Add receipt", key="cust_add_receipt_btn"):
            if not sel_iid:
                st.error("Pick a package.")
            elif amt <= 0:
                st.error("Enter a positive amount.")
            else:
                term_label = None; term_percent = None
                if plan and tsel != "--":
                    idx = term_labels.index(tsel) - 1
                    term_label = plan[idx]["label"]; term_percent = plan[idx]["percent"]
                col_cust_txn.insert_one({
                    "itinerary_id": sel_iid,
                    "amount": int(amt),
                    "date": datetime(dtp.year, dtp.month, dtp.day),
                    "mode": mode, "utr": utr.strip(), "collected_by": (collected_by or "").strip(),
                    "term_label": term_label, "term_percent": term_percent,
                    "note": note.strip(),
                    "created_at": _now_utc()
                })
                st.success("Customer receipt recorded.")

                # Optional reminder for remaining balance (based on final_cost)
                if mk_rem:
                    final_cost = _final_cost_for(sel_iid)
                    paid_so_far = int((df_cpay[df_cpay["itinerary_id"]==sel_iid]["amount"].sum() if not df_cpay.empty else 0) + amt)
                    bal = max(final_cost - paid_so_far, 0)
                    if bal > 0:
                        col_reminders.insert_one({
                            "for":"customer","itinerary_id": sel_iid,
                            "amount": bal, "due_date": datetime(dtp.year, dtp.month, dtp.day) + timedelta(days=3),
                            "note": f"Auto reminder for remaining balance after receipt ₹{amt:,}",
                            "status":"open","created_at": _now_utc()
                        })
                        st.info(f"Reminder created for remaining ₹{bal:,}.")

with st.expander("🏷️ Record Vendor Payment (multi-break, mode/UTR/receiver, optional reminder)", expanded=False):
    if df_cust.empty:
        st.info("No confirmed packages.")
    else:
        options = (df_cust["ach_id"].fillna("") + " | " + df_cust["client_name"].fillna("") +
                   " | " + df_cust["booking_date"].astype(str) + " | " + df_cust["itinerary_id"])
        pick = st.selectbox("Select package", options.tolist(), key="rec_vendor_pkg")
        sel_iid = pick.split(" | ")[-1] if pick else None

        vend_names = sorted(col_vendors.distinct("name"))
        c1,c2,c3,c4 = st.columns([1.3,1,1,1.2])
        with c1: vsel = st.selectbox("Vendor*", ["--"] + vend_names, key="vendor_select_input")
        with c2: cat  = st.text_input("Category (Hotel/Taxi/Guide)", key="vendor_cat_input")
        with c3: vtype = st.selectbox("Type", ["adv","final","other"], key="vendor_type_select")
        with c4: vdate = st.date_input("Paid on", value=today, key="vendor_paid_on_input")
        r1, r2, r3 = st.columns([1,1.4,1.4])
        with r1: vamt = st.number_input("Amount (₹)", min_value=0, step=500, key="vendor_amount_input")
        with r2: vmode = st.selectbox("Mode", ["UPI","NEFT/RTGS","IMPS","Cash","Card","Other"], key="vendor_mode_select")
        with r3: vutr = st.text_input("UTR / Ref no.", key="vendor_utr_input")
        vrecv = st.text_input("If Cash: paid to (person)", key="vendor_paid_to_input")
        vnote = st.text_input("Note", key="vendor_note_input")

        # Load vendor-specific terms (if any)
        vplan = load_terms("vendor", sel_iid, vsel if vsel!="--" else None, cat) if sel_iid else []
        vterm_labels = ["--"] + [f'{x["label"]} ({x["percent"]}%)' for x in vplan] if vplan else ["--"]
        vtsel = st.selectbox("Map to term (optional)", vterm_labels, key="vendor_term_select")
        mk_rem_v = st.checkbox("Also create reminder for remaining vendor balance", value=False, key="vend_make_reminder")

        if st.button("Add vendor payment", key="vendor_add_payment_btn"):
            if not sel_iid:
                st.error("Pick a package.")
            elif vsel == "--":
                st.error("Pick a vendor.")
            elif vamt <= 0:
                st.error("Enter a positive amount.")
            else:
                term_label = None; term_percent = None
                if vplan and vtsel != "--":
                    idx = vterm_labels.index(vtsel) - 1
                    term_label = vplan[idx]["label"]; term_percent = vplan[idx]["percent"]

                col_vendor_txn.insert_one({
                    "itinerary_id": sel_iid,
                    "vendor": vsel.strip(),
                    "category": cat.strip(),
                    "amount": int(vamt),
                    "date": datetime(vdate.year, vdate.month, vdate.day),
                    "mode": vmode,
                    "utr": vutr.strip(),
                    "collected_by": (vrecv or "").strip(),  # who received, if cash
                    "type": vtype,
                    "term_label": term_label, "term_percent": term_percent,
                    "note": vnote.strip(),
                    "created_at": _now_utc()
                })
                st.success("Vendor payment recorded.")

                # Optional reminder for remaining vendor balance (based on finalization_cost if known)
                if mk_rem_v:
                    # find finalization cost reference from legacy lines (if available)
                    ref = df_vlegacy
                    fc = 0
                    if not ref.empty:
                        mask = (ref["itinerary_id"]==sel_iid)&(ref["vendor"]==vsel)&(ref["category"]==cat)
                        if ref[mask].shape[0]>0:
                            fc = int(pd.to_numeric(ref[mask]["finalization_cost"], errors="coerce").fillna(0).max())
                    paid_so_far = int((df_vtxn[(df_vtxn["itinerary_id"]==sel_iid)&(df_vtxn["vendor"]==vsel)&(df_vtxn["category"]==cat)]["amount"].sum() if not df_vtxn.empty else 0) + vamt)
                    bal = max(fc - paid_so_far, 0)
                    if bal > 0:
                        col_reminders.insert_one({
                            "for":"vendor","itinerary_id": sel_iid, "vendor": vsel.strip(),
                            "amount": bal, "due_date": datetime(vdate.year, vdate.month, vdate.day) + timedelta(days=3),
                            "note": f"Auto reminder for remaining vendor balance after pay ₹{vamt:,}",
                            "status":"open","created_at": _now_utc()
                        })
                        st.info(f"Reminder created for remaining vendor balance ₹{bal:,}.")

st.divider()

# =========================================================
# ⚖️ Customer collections — overview (Confirmed Only)
# =========================================================
st.subheader("👤 Customer Collections — Confirmed Only")

if df_cust.empty:
    st.info("No confirmed packages found.")
else:
    total_final   = int(pd.to_numeric(df_cust["final_cost"], errors="coerce").fillna(0).sum())
    total_received= int(pd.to_numeric(df_cust["received"], errors="coerce").fillna(0).sum())
    total_pending = int(pd.to_numeric(df_cust["pending"], errors="coerce").fillna(0).sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Final (₹)", f"{total_final:,}")
    k2.metric("Received (₹)", f"{total_received:,}")
    k3.metric("Pending (₹)", f"{total_pending:,}")

    # Table per customer
    cust_cols = ["ach_id","client_name","client_mobile","final_route","total_pax",
                 "booking_date","rep_name","final_cost","received","pending","last_pay_date","last_utr","itinerary_id"]
    df_cust = ensure_cols(df_cust, {c: "" for c in cust_cols})
    view_c = df_cust[cust_cols].rename(columns={
        "ach_id":"ACH ID","client_name":"Customer","client_mobile":"Mobile",
        "final_route":"Route","total_pax":"Pax","booking_date":"Booked on",
        "rep_name":"Rep (credited)","final_cost":"Final (₹)","received":"Received (₹)",
        "pending":"Pending (₹)","last_pay_date":"Last pay date","last_utr":"Last UTR"
    }).sort_values(["Booked on","Customer"], na_position="last")
    st.dataframe(view_c.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)

    # Date-wise customer receipts
    if not df_cpay.empty:
        cust_by_day = (
            df_cpay[df_cpay["itinerary_id"].isin(df_cust["itinerary_id"])]
            .groupby("date", as_index=False)
            .agg(Received=("amount","sum"))
            .rename(columns={"date":"Date"})
            .sort_values("Date")
        )
        st.markdown("**Date-wise customer receipts**")
        st.dataframe(cust_by_day, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# 🧾 Vendor dues — overview (Confirmed Only) + Date-wise vendor payments
# =========================================================
st.subheader("🏷️ Vendor Dues / Payments — Confirmed Only")

if df_v.empty:
    st.info("No vendor payment records match the current filter.")
else:
    # Summary per vendor (combine legacy + txns already)
    vendor_sum = (
        df_v.groupby(["vendor","category","city"], dropna=False)
            .agg(
                Bookings=("itinerary_id","nunique"),
                Finalization=("finalization_cost","sum"),
                Paid=("paid_total","sum"),
                Balance=("balance","sum"),
                LastPaymentDate=("last_pay_date","max")
            )
            .reset_index()
            .sort_values(["Balance","Finalization"], ascending=[False, False])
    )
    total_vendor_balance = int(pd.to_numeric(vendor_sum["Balance"], errors="coerce").fillna(0).sum())
    total_vendors_with_dues = int((pd.to_numeric(vendor_sum["Balance"], errors="coerce").fillna(0) > 0).sum())
    v1, v2 = st.columns(2)
    v1.metric("Total Vendor Balance (₹)", f"{total_vendor_balance:,}")
    v2.metric("Vendors with dues", total_vendors_with_dues)

    st.markdown("**Vendor summary**")
    show_vs = vendor_sum.rename(columns={
        "vendor":"Vendor","category":"Category","city":"City",
        "Finalization":"Finalized (₹)","Paid":"Paid (₹)","Balance":"Balance (₹)","LastPaymentDate":"Last payment"
    })
    st.dataframe(show_vs, use_container_width=True, hide_index=True)

    with st.expander("Show line items (vendor payments per package)"):
        # Enrich line items with customer meta
        meta_needed_cols = ["itinerary_id","ach_id","client_name","client_mobile","booking_date","final_route"]
        meta = df_cust[meta_needed_cols].drop_duplicates("itinerary_id") if not df_cust.empty else pd.DataFrame(columns=meta_needed_cols)
        line = df_v.merge(meta, on="itinerary_id", how="left")
        line = ensure_cols(line, {
            "vendor":"", "category":"", "city":"", "ach_id":"", "client_name":"", "client_mobile":"",
            "booking_date":None, "final_route":"", "finalization_cost":0, "paid_total":0, "balance":0,
            "last_pay_date":None, "last_utr":"", "source":""
        })
        line = line[[
            "vendor","category","city","itinerary_id","ach_id","client_name","client_mobile","booking_date","final_route",
            "finalization_cost","paid_total","balance","last_pay_date","last_utr","source"
        ]].rename(columns={
            "vendor":"Vendor","category":"Category","city":"City","ach_id":"ACH ID",
            "client_name":"Customer","client_mobile":"Mobile","booking_date":"Booked on","final_route":"Route",
            "finalization_cost":"Finalized (₹)","paid_total":"Paid total (₹)","balance":"Balance (₹)",
            "last_pay_date":"Last pay date","last_utr":"Last UTR","source":"From"
        }).sort_values(["Vendor","Booked on","Customer"], na_position="last")
        st.dataframe(line, use_container_width=True, hide_index=True)

    # Date-wise vendor payments (from txns)
    if not df_vtxn.empty:
        vend_by_day = (
            df_vtxn[df_vtxn["itinerary_id"].isin(df_cust["itinerary_id"])]
            .groupby("date", as_index=False)
            .agg(Paid=("amount","sum"))
            .rename(columns={"date":"Date"})
            .sort_values("Date")
        )
        st.markdown("**Date-wise vendor payments**")
        st.dataframe(vend_by_day, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# 🧭 Per-package explorer (plan vs actual, UTRs, cash collector)
# =========================================================
st.subheader("🧭 Per-package explorer (Plan vs Actual)")

left, right = st.columns([2, 1])
with right:
    if not df_cust.empty:
        opts = (
            df_cust["ach_id"].fillna("") + " | " +
            df_cust["client_name"].fillna("") + " | " +
            df_cust["booking_date"].astype(str).fillna("") + " | " +
            df_cust["itinerary_id"]
        )
        pick = st.selectbox("Open package", opts.tolist(), key="pkg_explorer_select")
        sel_id = pick.split(" | ")[-1] if pick else None
    else:
        sel_id = None

with left:
    if sel_id:
        row = df_cust[df_cust["itinerary_id"] == sel_id].iloc[0].to_dict()
        st.write({
            "ACH ID": row.get("ach_id",""),
            "Customer": row.get("client_name",""),
            "Mobile": row.get("client_mobile",""),
            "Route": row.get("final_route",""),
            "Pax": row.get("total_pax",""),
            "Booked on": row.get("booking_date",""),
            "Rep (credited)": row.get("rep_name",""),
            "Final (₹)": row.get("final_cost",0),
            "Received (₹)": row.get("received",0),
            "Pending (₹)": row.get("pending",0),
            "Last pay date": row.get("last_pay_date",""),
            "Last UTR": row.get("last_utr",""),
        })

        # Customer PLAN vs ACTUAL
        st.caption("Customer payment plan (if defined)")
        cplan = load_terms("customer", sel_id)
        if cplan:
            final_cost = _final_cost_for(sel_id)
            plan_df = pd.DataFrame([{
                "Break": p["label"], "Percent": p["percent"], "Planned (₹)": int(round(final_cost * p["percent"]/100.0))
            } for p in cplan])
            paid_map = {}
            txc = df_cpay[df_cpay["itinerary_id"] == sel_id].copy() if not df_cpay.empty else pd.DataFrame()
            if not txc.empty:
                for _, r in txc.iterrows():
                    key = (r.get("term_label") or "").strip()
                    paid_map[key] = paid_map.get(key, 0) + int(r.get("amount", 0))
            plan_df["Paid (₹)"] = plan_df["Break"].map(paid_map).fillna(0).astype(int)
            plan_df["Balance (₹)"] = (plan_df["Planned (₹)"] - plan_df["Paid (₹)"]).clip(lower=0)
            st.dataframe(plan_df, use_container_width=True, hide_index=True)
        else:
            st.write("No customer payment plan defined.")

        st.caption("Customer payment transactions (with mode/UTR/cash collector)")
        txc = df_cpay[df_cpay["itinerary_id"] == sel_id].copy() if not df_cpay.empty else pd.DataFrame()
        if not txc.empty:
            txc = txc[["date","amount","mode","utr","collected_by","term_label","term_percent","note"]].rename(columns={
                "date":"Date","amount":"Amount (₹)","mode":"Mode","utr":"UTR / Ref","collected_by":"Collected by",
                "term_label":"Term","term_percent":"%"
            }).sort_values("Date")
            st.dataframe(txc, use_container_width=True, hide_index=True)
        else:
            st.write("No customer receipts recorded yet.")

        # Vendor PLAN vs ACTUAL for each vendor/category
        st.caption("Vendor payment plan (if defined) & summary")
        if not df_vtxn.empty:
            vend_pairs = df_vtxn[df_vtxn["itinerary_id"] == sel_id][["vendor","category"]].drop_duplicates()
            if vend_pairs.empty:
                st.write("No vendor payments yet.")
            else:
                for _, vp in vend_pairs.iterrows():
                    vname, vcat = vp["vendor"], vp["category"]
                    st.write(f"**{vname} — {vcat}**")
                    vplan = load_terms("vendor", sel_id, vname, vcat)
                    if vplan:
                        # try to fetch finalization cost baseline
                        ref = df_vlegacy
                        fc = 0
                        if not ref.empty:
                            mask = (ref["itinerary_id"]==sel_id)&(ref["vendor"]==vname)&(ref["category"]==vcat)
                            if ref[mask].shape[0]>0:
                                fc = int(pd.to_numeric(ref[mask]["finalization_cost"], errors="coerce").fillna(0).max())
                        plan_df = pd.DataFrame([{
                            "Break": p["label"], "Percent": p["percent"], "Planned (₹)": int(round(fc * p["percent"]/100.0))
                        } for p in vplan])
                        paid_map = {}
                        sub = df_vtxn[(df_vtxn["itinerary_id"]==sel_id)&(df_vtxn["vendor"]==vname)&(df_vtxn["category"]==vcat)]
                        for _, r in sub.iterrows():
                            key = (r.get("term_label") or "").strip()
                            paid_map[key] = paid_map.get(key, 0) + int(r.get("amount", 0))
                        plan_df["Paid (₹)"] = plan_df["Break"].map(paid_map).fillna(0).astype(int)
                        plan_df["Balance (₹)"] = (plan_df["Planned (₹)"] - plan_df["Paid (₹)"]).clip(lower=0)
                        st.dataframe(plan_df, use_container_width=True, hide_index=True)
                    else:
                        st.write("No vendor payment plan defined for this vendor/category.")

        st.caption("Individual vendor transactions (mode/UTR/receiver)")
        txv = df_vtxn[df_vtxn["itinerary_id"] == sel_id].copy() if not df_vtxn.empty else pd.DataFrame()
        if not txv.empty:
            txv = txv[["vendor","category","type","date","amount","mode","utr","collected_by","term_label","term_percent","note"]].rename(columns={
                "vendor":"Vendor","category":"Category","type":"Type","date":"Date",
                "amount":"Amount (₹)","mode":"Mode","utr":"UTR / Ref","collected_by":"Paid to",
                "term_label":"Term","term_percent":"%"
            }).sort_values(["Vendor","Date"])
            st.dataframe(txv, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# 🔔 Open reminders
# =========================================================
st.subheader("🔔 Open Payment Reminders")
rem = list(col_reminders.find({"status": {"$ne":"closed"}}, {"_id":1,"for":1,"itinerary_id":1,"vendor":1,"amount":1,"due_date":1,"note":1}))
if rem:
    df_rem = pd.DataFrame([{
        "id": str(x["_id"]),
        "for": x.get("for"),
        "itinerary_id": x.get("itinerary_id"),
        "vendor": x.get("vendor"),
        "amount": _to_int(x.get("amount",0)),
        "due_date": _d(x.get("due_date")),
        "note": x.get("note","")
    } for x in rem])
    # enrich with customer meta
    if not df_cust.empty and "itinerary_id" in df_rem.columns:
        df_rem = df_rem.merge(
            df_cust[["itinerary_id","ach_id","client_name","client_mobile"]].drop_duplicates("itinerary_id"),
            on="itinerary_id", how="left"
        )
    df_rem = ensure_cols(df_rem, {"ach_id":"", "client_name":"", "client_mobile":"", "vendor":"", "amount":0, "due_date": None, "note":""})
    df_show = df_rem.rename(columns={
        "for":"For","ach_id":"ACH ID","client_name":"Customer","client_mobile":"Mobile",
        "vendor":"Vendor","amount":"Amount (₹)","due_date":"Due date","note":"Note"
    })[["For","ACH ID","Customer","Mobile","Vendor","Amount (₹)","Due date","Note","id"]]
    st.dataframe(df_show.drop(columns=["id"]), use_container_width=True, hide_index=True)

    rid = st.text_input("Reminder ID to close", key="rem_close_id_input")
    if st.button("Close reminder", key="rem_close_btn"):
        try:
            col_reminders.update_one({"_id": ObjectId(rid)}, {"$set": {"status":"closed","closed_at": _now_utc()}})
            st.success("Reminder closed.")
        except Exception as e:
            st.error(f"Invalid ID or error: {e}")
else:
    st.caption("No open reminders.")

st.divider()

# =========================================================
# ⬇️ Excel export (multi-sheet, confirmed only)
# =========================================================
st.subheader("⬇️ Export Excel (current filters)")
def export_excel_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        # Customer summary
        if not df_cust.empty:
            df_out = df_cust[[
                "ach_id","client_name","client_mobile","final_route","total_pax",
                "booking_date","rep_name","final_cost","received","pending","last_pay_date","last_utr","itinerary_id"
            ]].rename(columns={
                "ach_id":"ACH_ID","client_name":"Customer","client_mobile":"Mobile","final_route":"Route","total_pax":"Pax",
                "booking_date":"Booked_on","rep_name":"Rep_credited","final_cost":"Final","received":"Received","pending":"Pending",
                "last_pay_date":"Last_pay_date","last_utr":"Last_UTR"
            }).sort_values(["Booked_on","Customer"], na_position="last")
            df_out.to_excel(xw, index=False, sheet_name="Customer_Summary")

            by_day = (
                df_cust[~df_cust["booking_date"].isna()]
                .groupby("booking_date", as_index=False)
                .agg(Final=("final_cost","sum"), Received=("received","sum"), Pending=("pending","sum"))
                .rename(columns={"booking_date":"Date"})
                .sort_values("Date")
            )
            by_day.to_excel(xw, index=False, sheet_name="Customer_By_Date")
        else:
            pd.DataFrame(columns=["No data"]).to_excel(xw, index=False, sheet_name="Customer_Summary")
            pd.DataFrame(columns=["No data"]).to_excel(xw, index=False, sheet_name="Customer_By_Date")

        # Vendor summary + line items
        if not df_v.empty:
            vendor_sum = (
                df_v.groupby(["vendor","category","city"], dropna=False)
                    .agg(
                        Bookings=("itinerary_id","nunique"),
                        Finalization=("finalization_cost","sum"),
                        Paid=("paid_total","sum"),
                        Balance=("balance","sum"),
                        LastPaymentDate=("last_pay_date","max")
                    ).reset_index()
            )
            vendor_sum.rename(columns={"vendor":"Vendor","category":"Category","city":"City"}, inplace=True)
            vendor_sum.to_excel(xw, index=False, sheet_name="Vendor_Summary")

            meta_needed_cols = ["itinerary_id","ach_id","client_name","client_mobile","booking_date","final_route"]
            meta = df_cust[meta_needed_cols].drop_duplicates("itinerary_id") if not df_cust.empty else pd.DataFrame(columns=meta_needed_cols)
            line = df_v.merge(meta, on="itinerary_id", how="left")
            line = ensure_cols(line, {
                "vendor":"", "category":"", "city":"", "ach_id":"", "client_name":"", "client_mobile":"",
                "booking_date":None, "final_route":"", "finalization_cost":0, "paid_total":0, "balance":0,
                "last_pay_date":None, "last_utr":"", "source":""
            })
            line = line[[
                "vendor","category","city","itinerary_id","ach_id","client_name","client_mobile","booking_date","final_route",
                "finalization_cost","paid_total","balance","last_pay_date","last_utr","source"
            ]].rename(columns={
                "vendor":"Vendor","category":"Category","city":"City","ach_id":"ACH_ID",
                "client_name":"Customer","client_mobile":"Mobile","booking_date":"Booked_on","final_route":"Route",
                "finalization_cost":"Finalized","paid_total":"Paid_total","balance":"Balance",
                "last_pay_date":"Last_pay_date","last_utr":"Last_UTR","source":"From"
            })
            line.to_excel(xw, index=False, sheet_name="Vendor_LineItems")
        else:
            pd.DataFrame(columns=["No data"]).to_excel(xw, index=False, sheet_name="Vendor_Summary")
            pd.DataFrame(columns=["No data"]).to_excel(xw, index=False, sheet_name="Vendor_LineItems")

        # Raw transactions for reference (include mode/UTR/collector & term mapping)
        if not df_cpay.empty:
            df_cpay.rename(columns={"date":"Date","amount":"Amount","mode":"Mode","utr":"UTR","collected_by":"Collected_by",
                                    "term_label":"Term","term_percent":"Percent","note":"Note"}, inplace=False).to_excel(
                xw, index=False, sheet_name="Customer_Txns"
            )
        else:
            pd.DataFrame(columns=["No customer txns"]).to_excel(xw, index=False, sheet_name="Customer_Txns")

        if not df_vtxn.empty:
            df_vtxn.rename(columns={"date":"Date","amount":"Amount","mode":"Mode","utr":"UTR","collected_by":"Paid_to",
                                    "type":"Type","term_label":"Term","term_percent":"Percent","note":"Note"}, inplace=False).to_excel(
                xw, index=False, sheet_name="Vendor_Txns"
            )
        else:
            pd.DataFrame(columns=["No vendor txns"]).to_excel(xw, index=False, sheet_name="Vendor_Txns")

    return buf.getvalue()

st.download_button(
    "Download Excel",
    data=export_excel_bytes(),
    file_name=f"collections_vendors_{date.today()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
    key="download_excel_btn"
)
