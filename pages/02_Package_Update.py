# pages/02_Package_Update.py
from __future__ import annotations

import os
from datetime import datetime, date
from typing import Optional
import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

st.set_page_config(page_title="Package Update (Vendors & Profit)", layout="wide")
st.title("🧺 Package Update — Vendor Finalization & Profit")

# ---------- Mongo ----------
CAND_KEYS = ["mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"]

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
        st.error("Mongo URI not configured. Add `mongo_uri` in Secrets.")
        st.stop()
    cli = MongoClient(uri, appName="TAK_PackageUpdate",
                      serverSelectionTimeoutMS=6000, connectTimeoutMS=6000, tz_aware=True)
    try:
        cli.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        st.error(f"Mongo connection error: {e}")
        st.stop()
    return cli

db = _get_client()["TAK_DB"]
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_expenses    = db["expenses"]
col_vendorpay   = db["vendor_payments"]
col_vendors     = db["vendors"]
col_split       = db["expense_splitwise"]

# ---------- Helpers ----------
def _i(x, d=0):
    try:
        if x is None: return d
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return d

def _d(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)): return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

# ======== Profit calculus ========
def final_cost_for(iid: str) -> int:
    exp = col_expenses.find_one({"itinerary_id": str(iid)},
                                {"final_package_cost":1,"base_package_cost":1,"discount":1,"package_cost":1}) or {}
    if "final_package_cost" in exp: 
        return _i(exp.get("final_package_cost", 0))
    if ("base_package_cost" in exp) or ("discount" in exp) or ("package_cost" in exp):
        base = _i(exp.get("base_package_cost", exp.get("package_cost", 0)))
        disc = _i(exp.get("discount", 0))
        return max(0, base - disc)
    it = col_itineraries.find_one({"_id": ObjectId(iid)}, {"package_cost":1,"discount":1}) or {}
    return max(0, _i(it.get("package_cost", 0)) - _i(it.get("discount", 0)))

def vendor_cost_for(iid: str) -> int:
    """Sum of vendor finalization_cost across items for this itinerary."""
    rec = col_vendorpay.find_one({"itinerary_id": str(iid)}, {"_id":0, "items":1}) or {}
    tot = 0
    for it in (rec.get("items") or []):
        tot += _i(it.get("finalization_cost", 0))
    return tot

def splitwise_cost_for(iid: str) -> int:
    """Sum Splitwise expenses linked to this itinerary (kind='expense')."""
    s = 0
    for r in col_split.find({"itinerary_id": str(iid), "kind": "expense"}, {"amount":1}):
        s += _i(r.get("amount", 0))
    return s

def profit_breakup(iid: str) -> dict:
    fc = final_cost_for(iid)
    vc = vendor_cost_for(iid)
    sc = splitwise_cost_for(iid)
    return {"final_cost": fc, "vendor_cost": vc, "splitwise_cost": sc, "actual_profit": max(fc - (vc + sc), 0)}

# ---------- Load confirmed itineraries ----------
@st.cache_data(ttl=90, show_spinner=False)
def confirmed_itineraries() -> pd.DataFrame:
    ups = list(col_updates.find({"status":"confirmed"},
                                {"_id":0,"itinerary_id":1,"booking_date":1,"rep_name":1}))
    if not ups: return pd.DataFrame()
    for u in ups:
        u["itinerary_id"] = str(u.get("itinerary_id"))
        u["booking_date"] = _d(u.get("booking_date"))
    df_u = pd.DataFrame(ups)

    its = list(col_itineraries.find({}, {"_id":1,"ach_id":1,"client_name":1,"client_mobile":1,
                                         "final_route":1,"total_pax":1,"start_date":1,"end_date":1}))
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        r["start_date"] = _d(r.get("start_date"))
        r["end_date"]   = _d(r.get("end_date"))
        r.pop("_id", None)
    df_i = pd.DataFrame(its)

    return df_i.merge(df_u, on="itinerary_id", how="inner")

df = confirmed_itineraries()
if df.empty:
    st.info("No confirmed packages.")
    st.stop()

# ---------- Pick a package ----------
q = st.text_input("🔎 Search (name / mobile / ACH / route)")
view = df.copy()
if q.strip():
    s = q.strip().lower()
    view = view[
        view["client_name"].astype(str).str.lower().str.contains(s) |
        view["client_mobile"].astype(str).str.lower().str.contains(s) |
        view["ach_id"].astype(str).str.lower().str.contains(s) |
        view["final_route"].astype(str).str.lower().str.contains(s)
    ]

opts = (view["ach_id"].fillna("") + " | " + view["client_name"].fillna("") + " | " +
        view["booking_date"].astype(str).fillna("") + " | " + view["itinerary_id"])
choice = st.selectbox("Select package", opts.tolist())
iid = choice.split(" | ")[-1] if choice else None
if not iid: st.stop()

meta = view[view["itinerary_id"]==iid].iloc[0].to_dict()
topL, topR = st.columns([2,1])
with topL:
    st.write({
        "ACH ID": meta.get("ach_id",""),
        "Customer": meta.get("client_name",""),
        "Mobile": meta.get("client_mobile",""),
        "Route": meta.get("final_route",""),
        "Pax": meta.get("total_pax",""),
        "Travel": f"{meta.get('start_date','')} → {meta.get('end_date','')}",
    })
with topR:
    br = profit_breakup(iid)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final (₹)", f"{br['final_cost']:,}")
    c2.metric("Vendors (₹)", f"{br['vendor_cost']:,}")
    c3.metric("Splitwise (₹)", f"{br['splitwise_cost']:,}")
    c4.metric("Actual Profit (₹)", f"{br['actual_profit']:,}")

st.divider()

# ---------- Vendor directory (category-scoped) ----------
@st.cache_data(ttl=120, show_spinner=False)
def vendor_dir() -> pd.DataFrame:
    vs = list(col_vendors.find({}, {"_id":0,"name":1,"category":1,"city":1}))
    return pd.DataFrame(vs) if vs else pd.DataFrame(columns=["name","category","city"])

vdir = vendor_dir()
CATS = sorted(vdir["category"].dropna().unique().tolist()) if not vdir.empty else \
       ["Hotel","Car","Sightseeing","Other"]

st.subheader("🏷️ Vendor finalization")
with st.form("vendor_add"):
    c1, c2, c3, c4 = st.columns([1,1.6,1,1])
    with c1:
        cat = st.selectbox("Category", CATS, index=0)
    with c2:
        # ✅ Only vendors for the chosen category
        vopts = (vdir[vdir["category"]==cat]["name"].dropna().unique().tolist()
                 if not vdir.empty else [])
        vendor = st.selectbox("Vendor", vopts, index=0 if vopts else None)
    with c3:
        fcost = st.number_input("Finalization cost (₹)", min_value=0, step=500, value=0)
    with c4:
        locked = st.selectbox("Lock group?", ["No","Yes"], index=0)

    s1, s2, s3 = st.columns(3)
    with s1:
        adv1_amt = st.number_input("Adv1 (₹)", min_value=0, step=500, value=0)
        adv1_date = st.date_input("Adv1 date", value=date.today())
    with s2:
        adv2_amt = st.number_input("Adv2 (₹)", min_value=0, step=500, value=0)
        adv2_date = st.date_input("Adv2 date", value=None)
    with s3:
        final_amt = st.number_input("Final (₹)", min_value=0, step=500, value=0)
        final_date = st.date_input("Final date", value=None)

    submitted = st.form_submit_button("➕ Add / Append vendor item")

if submitted:
    item = {
        "category": cat, "vendor": vendor or "",
        "finalization_cost": int(fcost),
        "adv1_amt": int(adv1_amt), "adv1_date": adv1_date,
        "adv2_amt": int(adv2_amt), "adv2_date": adv2_date,
        "final_amt": int(final_amt), "final_date": final_date,
        "balance": max(int(fcost) - (int(adv1_amt)+int(adv2_amt)+int(final_amt)), 0)
    }
    doc = col_vendorpay.find_one({"itinerary_id": str(iid)}, {"_id":1})
    if doc:
        col_vendorpay.update_one(
            {"_id": doc["_id"]},
            {"$push": {"items": item}, "$set": {"final_done": (locked=="Yes"), "updated_at": datetime.utcnow()}}
        )
    else:
        col_vendorpay.insert_one({
            "itinerary_id": str(iid),
            "items": [item],
            "final_done": (locked=="Yes"),
            "updated_at": datetime.utcnow()
        })
    st.success("Vendor item saved.")
    st.rerun()

# ---------- Existing items ----------
rec = col_vendorpay.find_one({"itinerary_id": str(iid)}, {"_id":0})
if not rec:
    st.info("No vendor items yet.")
else:
    items = rec.get("items") or []
    dfv = pd.DataFrame([{
        "Category": it.get("category",""),
        "Vendor": it.get("vendor",""),
        "Finalization (₹)": _i(it.get("finalization_cost",0)),
        "Adv1 (₹)": _i(it.get("adv1_amt",0)), "Adv1 date": _d(it.get("adv1_date")),
        "Adv2 (₹)": _i(it.get("adv2_amt",0)), "Adv2 date": _d(it.get("adv2_date")),
        "Final (₹)": _i(it.get("final_amt",0)), "Final date": _d(it.get("final_date")),
        "Balance (₹)": _i(it.get("balance", max(_i(it.get("finalization_cost",0))
                         - (_i(it.get("adv1_amt",0)) + _i(it.get("adv2_amt",0)) + _i(it.get("final_amt",0))), 0)))
    } for it in items])
    st.dataframe(dfv, use_container_width=True, hide_index=True)

st.divider()

# ---------- Live profit after all deductions ----------
br = profit_breakup(iid)
l1, l2, l3, l4 = st.columns(4)
l1.metric("Final package (₹)", f"{br['final_cost']:,}")
l2.metric("Vendors total (₹)", f"{br['vendor_cost']:,}")
l3.metric("Splitwise linked (₹)", f"{br['splitwise_cost']:,}")
l4.metric("Actual Profit (₹)", f"{br['actual_profit']:,}")
