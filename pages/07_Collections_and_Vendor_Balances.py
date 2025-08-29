# pages/07_Collections_and_Vendor_Balances.py
from __future__ import annotations

import os, io
from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, List, Dict

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Collections & Vendor Balances", layout="wide")
st.title("💰 Collections (Customer) & Vendor Balances")

IST = ZoneInfo("Asia/Kolkata")

# =========================================================
# Admin gate (same pattern as other admin pages)
# =========================================================
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

from tak_audit import audit_pageview
audit_pageview(st.session_state.get("user", "Unknown"), page="07_Collections_and_Vendor_Balances")

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
col_vendorpay   = db["vendor_payments"]
col_vendors     = db["vendors"]

# =========================================================
# Helpers
# =========================================================
def _to_int(x, default=0) -> int:
    try:
        if x is None: return default
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def _d(x) -> Optional[date]:
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
# Cached loads (only projections needed)
# =========================================================
@st.cache_data(ttl=120, show_spinner=False)
def load_confirmed_packages() -> pd.DataFrame:
    # Updates: confirmed snapshot (booking_date + advance_amount + rep_name)
    ups = list(col_updates.find(
        {"status": "confirmed"},
        {"_id": 0, "itinerary_id": 1, "status": 1, "booking_date": 1, "advance_amount": 1, "rep_name": 1}
    ))
    if not ups:
        return pd.DataFrame()
    for u in ups:
        u["itinerary_id"] = str(u.get("itinerary_id"))
        u["booking_date"] = _d(u.get("booking_date"))
        u["advance_amount"] = _to_int(u.get("advance_amount", 0))
        u["rep_name"] = u.get("rep_name", "")
    df_u = pd.DataFrame(ups)

    # Base itinerary info to display
    its = list(col_itineraries.find({}, {
        "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
        "final_route": 1, "total_pax": 1, "representative": 1
    }))
    if not its:
        return pd.DataFrame(columns=["itinerary_id","ach_id","client_name","client_mobile","final_route","total_pax","representative",
                                     "booking_date","advance_amount","rep_name","final_cost","received","pending"])
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        r.pop("_id", None)
    df_i = pd.DataFrame(its)

    df = df_u.merge(df_i, on="itinerary_id", how="left")

    # Compute final cost quickly
    finals = [ _final_cost_for(iid) for iid in df["itinerary_id"] ]
    # ✅ make it a Series (so we can fillna) and align indices
    df["final_cost"] = pd.to_numeric(pd.Series(finals, index=df.index), errors="coerce").fillna(0).astype(int)

    # Received from customer = advance_amount (current schema)
    df["received"] = pd.to_numeric(df["advance_amount"], errors="coerce").fillna(0).astype(int)
    df["pending"]  = (df["final_cost"] - df["received"]).clip(lower=0).astype(int)
    return df

@st.cache_data(ttl=120, show_spinner=False)
def load_vendor_payments() -> pd.DataFrame:
    """
    Flattens vendor_payments.items to one row per vendor-item with amounts & dates.
    """
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
    vs = list(col_vendors.find({}, {"_id":0, "name":1, "city":1, "category":1}))
    return pd.DataFrame(vs) if vs else pd.DataFrame(columns=["name","city","category"])

# =========================================================
# Filters
# =========================================================
today = date.today()
mstart, mend = month_bounds(today)

with st.container():
    fl1, fl2, fl3, fl4 = st.columns([1.2, 1.2, 1.6, 2.0])
    with fl1:
        basis = st.selectbox("Customer date basis", ["Booking date", "Custom range"], index=0)
    with fl2:
        if basis == "Booking date":
            start_c, end_c = mstart, mend
        else:
            start_c, end_c = today - timedelta(days=30), today
        start_c = st.date_input("From (customers)", value=start_c)
    with fl3:
        end_c = st.date_input("To (customers)", value=end_c)
        if end_c < start_c:
            end_c = start_c
    with fl4:
        search_txt = st.text_input("Search customer (name/mobile/ACH/route)", "")

vl1, vl2 = st.columns([1.2, 2.8])
with vl1:
    vendor_date_mode = st.selectbox("Vendor filter", ["Any payment in range", "Ignore dates"], index=0)
with vl2:
    start_v = st.date_input("Vendor payments from", value=mstart)
    end_v   = st.date_input("Vendor payments to", value=mend)
    if end_v < start_v:
        end_v = start_v

st.divider()

# =========================================================
# Data prep
# =========================================================
df_cust = load_confirmed_packages()
df_vraw = load_vendor_payments()
df_vdir = load_vendor_directory()

# Join vendor city/category where possible
if not df_vraw.empty and not df_vdir.empty:
    df_vraw = df_vraw.merge(
        df_vdir.rename(columns={"name":"vendor"})[["vendor","city","category"]].drop_duplicates("vendor"),
        on="vendor", how="left", suffixes=("", "_dir")
    )

# Filter customers
if not df_cust.empty:
    df_cust = df_cust[df_cust["booking_date"].apply(_d).between(start_c, end_c)]
    if search_txt.strip():
        s = search_txt.strip().lower()
        df_cust = df_cust[
            df_cust["client_name"].astype(str).str.lower().str.contains(s) |
            df_cust["client_mobile"].astype(str).str.lower().str.contains(s) |
            df_cust["ach_id"].astype(str).str.lower().str.contains(s) |
            df_cust["final_route"].astype(str).str.lower().str.contains(s)
        ]

# Filter vendors by any payment date falling in range (adv1/adv2/final)
if not df_vraw.empty and vendor_date_mode == "Any payment in range":
    def in_range(row) -> bool:
        for c in ("adv1_date", "adv2_date", "final_date"):
            d = row.get(c)
            if isinstance(d, date) and (start_v <= d <= end_v):
                return True
        return False
    df_v = df_vraw[df_vraw.apply(in_range, axis=1)].copy()
else:
    df_v = df_vraw.copy()

# =========================================================
# ⚖️ Customer collections — overview
# =========================================================
st.subheader("👤 Customer Collections (Confirmed)")

if df_cust.empty:
    st.info("No confirmed packages found in the selected range.")
else:
    total_final   = int(df_cust["final_cost"].sum())
    total_received= int(df_cust["received"].sum())
    total_pending = int(df_cust["pending"].sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Final (₹)", f"{total_final:,}")
    k2.metric("Received (₹)", f"{total_received:,}")
    k3.metric("Pending (₹)", f"{total_pending:,}")

    # Table per customer
    cust_cols = ["ach_id","client_name","client_mobile","final_route","total_pax",
                 "booking_date","rep_name","final_cost","received","pending","itinerary_id"]
    view_c = df_cust[cust_cols].rename(columns={
        "ach_id":"ACH ID","client_name":"Customer","client_mobile":"Mobile",
        "final_route":"Route","total_pax":"Pax","booking_date":"Booked on",
        "rep_name":"Rep (credited)","final_cost":"Final (₹)","received":"Received (₹)","pending":"Pending (₹)"
    }).sort_values(["Booked on","Customer"], na_position="last")
    st.dataframe(view_c.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)

    # Date-wise (booking_date)
    by_day = (
        df_cust.groupby("booking_date", as_index=False)
        .agg(Final=("final_cost","sum"), Received=("received","sum"), Pending=("pending","sum"))
        .sort_values("booking_date")
    )
    st.markdown("**Date-wise summary**")
    st.dataframe(by_day.rename(columns={"booking_date":"Date"}), use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# 🧾 Vendor dues — overview
# =========================================================
st.subheader("🏷️ Vendor Dues / Payments")

if df_v.empty:
    st.info("No vendor payment records match the current filter.")
else:
    # Summary per vendor
    vendor_sum = (
        df_v.groupby(["vendor","category","city"], dropna=False)
            .agg(
                Bookings=("itinerary_id","nunique"),
                Finalization=("finalization_cost","sum"),
                Paid=("paid_total","sum"),
                Balance=("balance","sum"),
                LastPaymentDate=("final_date","max")
            )
            .reset_index()
            .sort_values(["Balance","Finalization"], ascending=[False, False])
    )
    total_vendor_balance = int(vendor_sum["Balance"].sum())
    total_vendors_with_dues = int((vendor_sum["Balance"] > 0).sum())
    v1, v2 = st.columns(2)
    v1.metric("Total Vendor Balance (₹)", f"{total_vendor_balance:,}")
    v2.metric("Vendors with dues", total_vendors_with_dues)

    st.markdown("**Vendor summary**")
    show_vs = vendor_sum.rename(columns={
        "vendor":"Vendor","category":"Category","city":"City",
        "Finalization":"Finalized (₹)","Paid":"Paid (₹)","Balance":"Balance (₹)"
    })
    st.dataframe(show_vs, use_container_width=True, hide_index=True)

    with st.expander("Show line items (vendor payments per package)"):
        # Enrich line items with customer meta for readability
        if not df_cust.empty:
            meta = df_cust[["itinerary_id","ach_id","client_name","client_mobile","booking_date","final_route"]].drop_duplicates("itinerary_id")
            line = df_v.merge(meta, on="itinerary_id", how="left")
        else:
            line = df_v.copy()
        line = line[[
            "vendor","category","city","itinerary_id","ach_id","client_name","client_mobile","booking_date","final_route",
            "finalization_cost","adv1_amt","adv1_date","adv2_amt","adv2_date","final_amt","final_date","paid_total","balance","final_done"
        ]].rename(columns={
            "vendor":"Vendor","category":"Category","city":"City","ach_id":"ACH ID",
            "client_name":"Customer","client_mobile":"Mobile","booking_date":"Booked on","final_route":"Route",
            "finalization_cost":"Finalized (₹)","adv1_amt":"Adv1 (₹)","adv1_date":"Adv1 date",
            "adv2_amt":"Adv2 (₹)","adv2_date":"Adv2 date","final_amt":"Final (₹)","final_date":"Final date",
            "paid_total":"Paid total (₹)","balance":"Balance (₹)","final_done":"Locked"
        }).sort_values(["Vendor","Booked on","Customer"])
        st.dataframe(line, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# 📦 Customer vs Vendor — per package quick explorer
# =========================================================
st.subheader("🧭 Per-package explorer")

left, right = st.columns([2, 1])
with right:
    if not df_cust.empty:
        opts = (
            df_cust["ach_id"].fillna("") + " | " +
            df_cust["client_name"].fillna("") + " | " +
            df_cust["booking_date"].astype(str).fillna("") + " | " +
            df_cust["itinerary_id"]
        )
        pick = st.selectbox("Open package", opts.tolist())
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
        })
        # vendor lines for this package
        detail = df_v[df_v["itinerary_id"] == sel_id].copy() if not df_v.empty else pd.DataFrame()
        if not detail.empty:
            st.caption("Vendor payments for this package")
            show = detail[[
                "vendor","category","city",
                "finalization_cost","adv1_amt","adv1_date","adv2_amt","adv2_date","final_amt","final_date",
                "paid_total","balance","final_done"
            ]].rename(columns={
                "vendor":"Vendor","category":"Category","city":"City",
                "finalization_cost":"Finalized (₹)","adv1_amt":"Adv1 (₹)","adv1_date":"Adv1 date",
                "adv2_amt":"Adv2 (₹)","adv2_date":"Adv2 date","final_amt":"Final (₹)","final_date":"Final date",
                "paid_total":"Paid total (₹)","balance":"Balance (₹)","final_done":"Locked"
            })
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.caption("No vendor payment records yet for this package.")

st.divider()

# =========================================================
# ⬇️ Excel export (multi-sheet)
# =========================================================
st.subheader("⬇️ Export Excel (current filters)")
def export_excel_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        # Customer summary
        if not df_cust.empty:
            df_out = df_cust[[
                "ach_id","client_name","client_mobile","final_route","total_pax",
                "booking_date","rep_name","final_cost","received","pending","itinerary_id"
            ]].rename(columns={
                "ach_id":"ACH_ID","client_name":"Customer","client_mobile":"Mobile","final_route":"Route","total_pax":"Pax",
                "booking_date":"Booked_on","rep_name":"Rep_credited","final_cost":"Final","received":"Received","pending":"Pending"
            }).sort_values(["Booked_on","Customer"], na_position="last")
            df_out.to_excel(xw, index=False, sheet_name="Customer_Summary")

            by_day = (
                df_cust.groupby("booking_date", as_index=False)
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
                        LastPaymentDate=("final_date","max")
                    ).reset_index()
            )
            vendor_sum.rename(columns={"vendor":"Vendor","category":"Category","city":"City"}, inplace=True)
            vendor_sum.to_excel(xw, index=False, sheet_name="Vendor_Summary")

            line = df_v.copy()
            # Enrich with ACH/Customer for readability
            if not df_cust.empty:
                meta = df_cust[["itinerary_id","ach_id","client_name","client_mobile","booking_date","final_route"]].drop_duplicates("itinerary_id")
                line = line.merge(meta, on="itinerary_id", how="left")

            line = line[[
                "vendor","category","city","itinerary_id","ach_id","client_name","client_mobile","booking_date","final_route",
                "finalization_cost","adv1_amt","adv1_date","adv2_amt","adv2_date","final_amt","final_date","paid_total","balance","final_done"
            ]].rename(columns={
                "vendor":"Vendor","category":"Category","city":"City","ach_id":"ACH_ID",
                "client_name":"Customer","client_mobile":"Mobile","booking_date":"Booked_on","final_route":"Route",
                "finalization_cost":"Finalized","adv1_amt":"Adv1_amt","adv1_date":"Adv1_date",
                "adv2_amt":"Adv2_amt","adv2_date":"Adv2_date","final_amt":"Final_amt","final_date":"Final_date",
                "paid_total":"Paid_total","balance":"Balance","final_done":"Locked"
            })
            line.to_excel(xw, index=False, sheet_name="Vendor_LineItems")
        else:
            pd.DataFrame(columns=["No data"]).to_excel(xw, index=False, sheet_name="Vendor_Summary")
            pd.DataFrame(columns=["No data"]).to_excel(xw, index=False, sheet_name="Vendor_LineItems")

    return buf.getvalue()

st.download_button(
    "Download Excel",
    data=export_excel_bytes(),
    file_name=f"collections_vendors_{date.today()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)
