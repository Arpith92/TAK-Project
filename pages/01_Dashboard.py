# pages/01_Dashboard.py
from __future__ import annotations

import io, csv, os, re
from datetime import datetime, date, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Tuple, Optional

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# =========================
# Guard & Page config
# =========================
# If user not set, default to Unknown (prevents KeyError on direct page open)
st.session_state.setdefault("user", "Unknown")

# Restrict certain users (as in your original)
if st.session_state.get("user") in ("Teena", "Kuldeep"):
    st.stop()

st.set_page_config(page_title="TAK Dashboard", layout="wide")
st.markdown("## 📊 TAK – Operations Dashboard")

# Optional pageview audit (kept as-is; safely import)
try:
    from tak_audit import audit_pageview
    audit_pageview(st.session_state.get("user", "Unknown"), "01_Dashboard")
except Exception:
    pass

# Optional calendar
CALENDAR_AVAILABLE = True
try:
    from streamlit_calendar import calendar
except Exception:
    CALENDAR_AVAILABLE = False

# =========================
# Admin gate (consistent with app)
# =========================
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

# =========================
# Mongo (fast, cached)
# =========================
IST = ZoneInfo("Asia/Kolkata")
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
def get_client() -> MongoClient:
    uri = _find_uri() or st.secrets["mongo_uri"]
    return MongoClient(
        uri,
        appName="TAK_Dashboard",
        maxPoolSize=100,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        retryWrites=True,
        tz_aware=True,
    )

@st.cache_resource
def get_db():
    client = get_client()
    client.admin.command("ping")
    return client["TAK_DB"]

db = get_db()
col_it = db["itineraries"]
col_up = db["package_updates"]
col_ex = db["expenses"]
col_fu = db["followups"]

# =========================
# Helpers
# =========================
def _to_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def _norm_date(x):
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

def fy_bounds(d: date) -> Tuple[date, date]:
    start = date(d.year if d.month >= 4 else d.year - 1, 4, 1)
    return start, d

def _created_utc_from_oid(iid: str):
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

def _ensure_cols(df: pd.DataFrame, cols: list[str], fill=None) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

# =========================
# Fast cached readers
# =========================
@st.cache_data(ttl=120, show_spinner=False)
def load_itineraries_df() -> pd.DataFrame:
    # Pull essential fields incl. new schema columns (from updated app.py)
    docs = list(col_it.find(
        {},
        {
            "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
            "representative": 1, "final_route": 1, "total_pax": 1,
            "upload_date": 1, "start_date": 1, "end_date": 1,
            # new totals
            "package_total": 1, "package_after_referral": 1,
            "actual_total": 1, "profit_total": 1,
            "referred_by": 1, "referral_discount_pct": 1,
            # legacy
            "package_cost": 1, "discount": 1,
            # revision metadata
            "revision_num": 1
        }
    ))
    if not docs:
        return pd.DataFrame()

    # Keep only latest revision per (client_mobile, start_date)
    best: dict[tuple[str, str], tuple[int, datetime, dict]] = {}
    for r in docs:
        key = (str(r.get("client_mobile","")), str(r.get("start_date","")))
        rev = _to_int(r.get("revision_num", 0))
        up  = r.get("upload_date")
        cur = best.get(key)
        if cur is None:
            best[key] = (rev, up, r)
        else:
            prev_rev, prev_up, prev_doc = cur
            if rev > prev_rev:
                best[key] = (rev, up, r)
            elif rev == prev_rev:
                try:
                    if pd.to_datetime(up) > pd.to_datetime(prev_up):
                        best[key] = (rev, up, r)
                except Exception:
                    pass
    docs = [v[2] for v in best.values()]

    out = []
    for r in docs:
        rec = dict(r)
        rec["itinerary_id"] = str(r["_id"])
        rec["_id"] = None

        rec["upload_date"] = _norm_date(rec.get("upload_date"))
        rec["start_date"]  = _norm_date(rec.get("start_date"))
        rec["end_date"]    = _norm_date(rec.get("end_date"))
        rec["total_pax"]   = _to_int(rec.get("total_pax", 0))
        rec["created_utc"] = _created_utc_from_oid(rec["itinerary_id"])
        rec["created_ist"] = _fmt_ist(rec["created_utc"])

        # Back-compat for dashboard calculations:
        # Prefer new fields from app.py, fallback to legacy if needed
        pkg_total = _to_int(rec.get("package_total", rec.get("package_cost", 0)))
        pkg_after = _to_int(rec.get("package_after_referral", 0))
        if pkg_after == 0 and _to_int(rec.get("referral_discount_pct", 0)) > 0:
            try:
                pct = _to_int(rec.get("referral_discount_pct", 0))
                pkg_after = max(pkg_total - int(round(pkg_total * pct / 100.0)), 0)
            except Exception:
                pkg_after = 0

        rec["package_cost"] = pkg_total
        rec["discount"]     = max(pkg_total - pkg_after, 0)

        # Expose profit/actual if needed elsewhere
        rec["actual_total"] = _to_int(rec.get("actual_total", 0))
        rec["profit_total"] = _to_int(rec.get("profit_total", 0))

        out.append(rec)

    df = pd.DataFrame(out)
    # Ensure optional columns exist
    df = _ensure_cols(df, ["ach_id", "representative"], "")
    return df

@st.cache_data(ttl=120, show_spinner=False)
def load_updates_df() -> pd.DataFrame:
    docs = list(col_up.find(
        {},
        {"_id": 0, "itinerary_id": 1, "status": 1, "assigned_to": 1,
         "booking_date": 1, "advance_amount": 1, "rep_name": 1, "incentive": 1}
    ))
    if not docs:
        return pd.DataFrame(columns=["itinerary_id","status","assigned_to","booking_date","advance_amount","rep_name","incentive"])
    for r in docs:
        r["itinerary_id"] = str(r.get("itinerary_id"))
        r["status"] = r.get("status", "pending")
        r["assigned_to"] = r.get("assigned_to", "")
        r["rep_name"] = r.get("rep_name", "")
        r["booking_date"] = _norm_date(r.get("booking_date"))
        r["advance_amount"] = _to_int(r.get("advance_amount", 0))
        r["incentive"] = _to_int(r.get("incentive", 0))
    return pd.DataFrame(docs)

@st.cache_data(ttl=120, show_spinner=False)
def load_expenses_df() -> pd.DataFrame:
    docs = list(col_ex.find(
        {},
        {"_id": 0, "itinerary_id": 1, "base_package_cost": 1, "discount": 1,
         "final_package_cost": 1, "package_cost": 1, "total_expenses": 1, "profit": 1}
    ))
    if not docs:
        return pd.DataFrame(columns=["itinerary_id","base_package_cost","discount","final_package_cost","total_expenses","profit"])
    for r in docs:
        r["itinerary_id"] = str(r.get("itinerary_id"))
        r["base_package_cost"] = _to_int(r.get("base_package_cost", 0))
        r["discount"] = _to_int(r.get("discount", 0))
        # prefer explicit final; fallback to legacy package_cost
        r["final_package_cost"] = _to_int(r.get("final_package_cost", r.get("package_cost", 0)))
        r["total_expenses"] = _to_int(r.get("total_expenses", 0))
        r["profit"] = _to_int(r.get("profit", 0))
    return pd.DataFrame(docs)

@st.cache_data(ttl=120, show_spinner=False)
def load_followups_for_ids(iids: list[str]) -> pd.DataFrame:
    if not iids:
        return pd.DataFrame()
    docs = list(col_fu.find({"itinerary_id": {"$in": iids}}, {"_id": 0}))
    return pd.DataFrame(docs)

# =========================
# Build master DF (vectorized)
# =========================
df_it = load_itineraries_df()
if df_it.empty:
    st.info("No data yet. Create packages in the main app.")
    st.stop()

df_up = load_updates_df()
df_ex = load_expenses_df()

df_all = df_it.merge(df_up, on="itinerary_id", how="left")
df_all = df_all.merge(df_ex, on="itinerary_id", how="left", suffixes=("", "_ex"))

# defaults & types
df_all["status"] = df_all["status"].fillna("pending")
for c in ("total_expenses","profit","incentive","advance_amount","total_pax"):
    if c in df_all.columns:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce").fillna(0).astype(int)

# compute final cost efficiently (no row-wise DB hits)
final_cost = pd.to_numeric(df_all.get("final_package_cost", 0), errors="coerce").fillna(0).astype(int)

# If not present in expenses, compute from itinerary base & discount
fallback_need = final_cost.eq(0)
if fallback_need.any():
    base = pd.to_numeric(df_all.get("package_cost", 0), errors="coerce").fillna(0).astype(int)
    disc = pd.to_numeric(df_all.get("discount", 0), errors="coerce").fillna(0).astype(int)
    comp = (base - disc).clip(lower=0)
    final_cost = final_cost.mask(final_cost.eq(0), comp)

df_all["final_cost"] = final_cost.astype(int)

# =========================
# Top bar: Global search
# =========================
top_l, top_m, top_r = st.columns([3,1,2])
with top_r:
    search_txt = st.text_input("🔎 Quick search", placeholder="Client name / Mobile / ACH ID", label_visibility="collapsed")

# =========================
# Filters
# =========================
flt1, flt2, flt3, flt4, flt5 = st.columns([1.1, 1.1, 1.8, 1.8, 1.2])
with flt1:
    basis = st.selectbox("Date basis", ["Upload date", "Booking date", "Travel start date"])
with flt2:
    dedupe_mobile = st.checkbox("Unique by mobile", value=True, help="Count only latest package per mobile in KPIs.")
with flt3:
    preset = st.selectbox("Quick range", ["This month", "Last month", "This FY", "This year", "Last 90 days", "Custom"])
with flt4:
    today = date.today()
    if preset == "This month":
        start, end = month_bounds(today)
    elif preset == "Last month":
        ft, _ = month_bounds(today)
        start, end = month_bounds(ft - timedelta(days=1))
    elif preset == "This FY":
        start, end = fy_bounds(today)
    elif preset == "This year":
        start, end = date(today.year, 1, 1), today
    elif preset == "Last 90 days":
        start, end = today - timedelta(days=90), today
    else:
        start, end = today - timedelta(days=30), today
    dr = st.date_input("Date range", (start, end))
    if isinstance(dr, tuple) and len(dr) == 2:
        start, end = dr
with flt5:
    reps = sorted([r for r in df_all.get("representative", pd.Series(dtype=str)).dropna().unique().tolist() if r]) or []
    rep_filter = st.multiselect("Representative", reps, default=reps)

date_col = {"Upload date":"upload_date", "Booking date":"booking_date", "Travel start date":"start_date"}[basis]
df = df_all.copy()
df[date_col] = df[date_col].apply(_norm_date)
df = df[df[date_col].between(start, end)]
if rep_filter:
    df = df[df["representative"].isin(rep_filter)]
if (search_txt or "").strip():
    s = search_txt.strip().lower()
    # Ensure columns exist for robust search
    df = _ensure_cols(df, ["ach_id","client_name","client_mobile"], "")
    df = df[
        df["client_name"].astype(str).str.lower().str.contains(s, na=False) |
        df["client_mobile"].astype(str).str.lower().str.contains(s, na=False) |
        df["ach_id"].astype(str).str.lower().str.contains(s, na=False)
    ]

# unique by mobile if enabled (latest by created_utc)
if dedupe_mobile and not df.empty:
    df = df.sort_values(["client_mobile","created_utc"], ascending=[True, False]) \
           .groupby("client_mobile", as_index=False).first()

# masks
is_confirmed = df["status"].eq("confirmed")
is_pending   = df["status"].eq("pending")
is_udisc     = df["status"].eq("under_discussion")
is_cancel    = df["status"].eq("cancelled")
is_followup  = df["status"].eq("followup")
is_enquiry   = is_pending | is_udisc | is_followup

# =========================
# KPI Row (structured)
# =========================
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("✅ Confirmed", int(is_confirmed.sum()))
k2.metric("🟡 Enquiries", int(is_enquiry.sum()))
k3.metric("🟠 Under discussion", int(is_udisc.sum()))
k4.metric("🔵 Follow-up", int(is_followup.sum()))
k5.metric("🔴 Cancelled", int(is_cancel.sum()))
# expenses pending among confirmed
have_cost_ids = set(df.loc[df["total_expenses"] > 0, "itinerary_id"])
conf_ids = set(df.loc[is_confirmed, "itinerary_id"])
k6.metric("🧾 Expense entry pending", int(max(len(conf_ids) - len(conf_ids & have_cost_ids), 0)))
st.divider()

# =========================
# Status Split (simple)
# =========================
st.subheader("Status mix")
mix = pd.DataFrame({
    "status":["confirmed","under_discussion","followup","pending","cancelled"],
    "count":[int(is_confirmed.sum()), int(is_udisc.sum()), int(is_followup.sum()), int(is_pending.sum()), int(is_cancel.sum())]
})
mix = mix[mix["count"]>0]
st.dataframe(mix, use_container_width=True, hide_index=True)
st.divider()

# =========================
# 📈 Daily trends (Confirmed vs Enquiries)
# =========================
st.subheader("📈 Daily trends (Confirmed vs Enquiries)")
df_c = df_all.copy()
df_c["booking_date"] = df_c["booking_date"].apply(_norm_date)
df_c = df_c[df_c["booking_date"].between(start, end) & df_c["status"].eq("confirmed")]
daily_conf = df_c.groupby("booking_date")["itinerary_id"].nunique().rename("Confirmed").reset_index()

df_e = df_all.copy()
df_e["upload_date"] = df_e["upload_date"].apply(_norm_date)
df_e = df_e[df_e["upload_date"].between(start, end) & df_e["status"].isin(["pending","under_discussion","followup"])]
daily_enq = df_e.groupby("upload_date")["itinerary_id"].nunique().rename("Enquiries").reset_index()

trend = pd.DataFrame({"date": pd.date_range(start, end)})
trend["date"] = trend["date"].dt.date
trend = trend.merge(daily_conf.rename(columns={"booking_date":"date"}), on="date", how="left") \
             .merge(daily_enq.rename(columns={"upload_date":"date"}), on="date", how="left") \
             .fillna(0)
st.line_chart(trend.set_index("date"))
st.divider()

# =========================
# 💰 Financials — Structured
# =========================
st.subheader("💰 Revenue snapshot (confirmed in range)")

conf_in_range = df[df["status"].eq("confirmed")].copy()

# Totals for the selected range
sum_final = int(conf_in_range["final_cost"].sum())
sum_exp   = int(conf_in_range["total_expenses"].sum())
sum_prof  = int(conf_in_range["profit"].sum())

# Totals (till date) — confirmed any time
df_confirmed_all = df_all[df_all["status"].eq("confirmed")].copy()
total_final_all = int(df_confirmed_all["final_cost"].sum())
total_profit_all = int(df_confirmed_all["profit"].sum())

# Totals (this month) — booking_date in this month
mstart, mend = month_bounds(date.today())
df_confirmed_month = df_all[
    df_all["status"].eq("confirmed") &
    df_all["booking_date"].apply(_norm_date).between(mstart, mend)
].copy()
total_final_month = int(df_confirmed_month["final_cost"].sum())
total_profit_month = int(df_confirmed_month["profit"].sum())

c1,c2,c3 = st.columns(3)
c1.metric("Final package (₹) — in range", f"{sum_final:,}")
c2.metric("Expenses (₹) — in range", f"{sum_exp:,}")
c3.metric("Profit (₹) — in range", f"{sum_prof:,}")

c4,c5 = st.columns(2)
c4.metric("Total Final (₹) — till date", f"{total_final_all:,}")
c5.metric("Total Profit (₹) — till date", f"{total_profit_all:,}")

c6,c7 = st.columns(2)
c6.metric("Total Final (₹) — this month", f"{total_final_month:,}")
c7.metric("Total Profit (₹) — this month", f"{total_profit_month:,}")

# Table: who confirmed & at what final cost (confirmed in current filter range)
if not conf_in_range.empty:
    df_tmp = _ensure_cols(conf_in_range.copy(), ["ach_id","rep_name"], "")
    view_cols = ["ach_id","client_name","client_mobile","rep_name","booking_date","final_cost","profit","itinerary_id"]
    table_fin = df_tmp[view_cols].copy()
    table_fin.rename(columns={
        "rep_name": "Confirmed by",
        "final_cost": "Final package (₹)"
    }, inplace=True)
    table_fin.sort_values(["booking_date","Final package (₹)"], ascending=[True, False], inplace=True)
    st.dataframe(table_fin.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)
else:
    st.caption("No confirmed packages in the selected range.")

st.divider()

# =========================
# Top routes & Pax distribution
# =========================
cA, cB = st.columns(2)
with cA:
    st.subheader("🗺️ Top routes")
    top_routes = df.groupby("final_route")["itinerary_id"].nunique().reset_index().sort_values("itinerary_id", ascending=False).head(12)
    top_routes.columns = ["Route","Packages"]
    st.dataframe(top_routes, use_container_width=True, hide_index=True)

with cB:
    st.subheader("👥 Pax distribution")
    pax = df[df["total_pax"] > 0]["total_pax"]
    st.bar_chart(pax.value_counts().sort_index())
st.divider()

# =========================
# 📅 Calendars
# =========================
st.subheader("📅 Calendars")
cal_tab1, cal_tab2 = st.tabs(["Confirmed – Booking dates", "Travel dates"])
with cal_tab1:
    confirmed = df_all[df_all["status"].eq("confirmed")].copy()
    confirmed["booking_date"] = confirmed["booking_date"].apply(_norm_date)
    confirmed = confirmed.dropna(subset=["booking_date"])
    events = [{"title": f"{r.client_name}_{r.total_pax}pax", "id": r.itinerary_id, "start": str(r.booking_date)}
              for r in confirmed.itertuples()]
    if CALENDAR_AVAILABLE:
        opts = {"initialView": "dayGridMonth", "height": 620, "eventDisplay": "block"}
        calendar(options=opts, events=events, key="cal_book")
    else:
        show = pd.DataFrame(events).rename(columns={"title":"Package","start":"Date"})
        st.dataframe(show.sort_values("Date"), use_container_width=True, hide_index=True)

with cal_tab2:
    confirmed = df_all[df_all["status"].eq("confirmed")].copy()
    confirmed["start_date"] = confirmed["start_date"].apply(_norm_date)
    confirmed["end_date"] = confirmed["end_date"].apply(_norm_date)
    confirmed = confirmed.dropna(subset=["start_date","end_date"])
    events = []
    for r in confirmed.itertuples():
        end_ = r.end_date + timedelta(days=1)  # exclusive end
        events.append({"title": f"{r.client_name}_{r.total_pax}pax", "id": r.itinerary_id,
                       "start": str(r.start_date), "end": end_.strftime("%Y-%m-%d")})
    if CALENDAR_AVAILABLE:
        opts = {"initialView": "dayGridMonth", "height": 620, "eventDisplay": "block"}
        calendar(options=opts, events=events, key="cal_travel")
    else:
        show = pd.DataFrame(events).rename(columns={"title":"Package","start":"Start","end":"End"})
        st.dataframe(show.sort_values(["Start","End"]), use_container_width=True, hide_index=True)
st.divider()

# =========================
# 🧭 Client explorer
# =========================
st.subheader("🧭 Client explorer")
opt_df = df_all.copy()
opt_df = _ensure_cols(opt_df, ["ach_id","client_name","client_mobile","created_ist"], "")
opt_df["label"] = (
    opt_df["ach_id"].fillna("") + " | " +
    opt_df["client_name"].fillna("") + " | " +
    opt_df["client_mobile"].fillna("") + " | " +
    opt_df["created_ist"].fillna("")
)
choice = st.selectbox("Open client/package", [""] + opt_df["label"].tolist())
sel_id = None
if choice:
    idx = opt_df.index[opt_df["label"] == choice].tolist()
    if idx:
        sel_id = opt_df.loc[idx[0], "itinerary_id"]

if sel_id:
    row = df_all[df_all["itinerary_id"] == sel_id].iloc[0].to_dict()
    st.markdown("#### Package details")
    c1,c2 = st.columns(2)
    with c1:
        st.write({
            "ACH ID": row.get("ach_id",""),
            "Client": row.get("client_name",""),
            "Mobile": row.get("client_mobile",""),
            "Route": row.get("final_route",""),
            "Pax": row.get("total_pax",""),
            "Representative": row.get("representative",""),
            "Created (IST)": row.get("created_ist",""),
            "Upload date": row.get("upload_date",""),
            "Travel": f"{row.get('start_date','')} → {row.get('end_date','')}",
        })
    with c2:
        st.write({
            "Status": row.get("status",""),
            "Booking date": row.get("booking_date",""),
            "Advance (₹)": row.get("advance_amount",0),
            "Incentive (₹)": row.get("incentive",0),
            "Final package (₹)": row.get("final_cost",0),
            "Expenses (₹)": row.get("total_expenses",0),
            "Profit (₹)": row.get("profit",0),
            "Rep (credited)": row.get("rep_name",""),
        })

    # Per-client Excel export (light I/O)
    def _client_excel_bytes(iid: str) -> bytes | None:
        it_doc = df_all[df_all["itinerary_id"]==iid].copy()
        up_docs = pd.DataFrame(list(col_up.find({"itinerary_id": str(iid)}, {"_id":0})))
        fu_docs = pd.DataFrame(list(col_fu.find({"itinerary_id": str(iid)}, {"_id":0})))
        ex_doc = pd.DataFrame(list(col_ex.find({"itinerary_id": str(iid)}, {"_id":0})))
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                it_doc.to_excel(xw, index=False, sheet_name="Itinerary")
                if not up_docs.empty: up_docs.to_excel(xw, index=False, sheet_name="Updates")
                if not fu_docs.empty: fu_docs.to_excel(xw, index=False, sheet_name="Followups")
                if not ex_doc.empty: ex_doc.to_excel(xw, index=False, sheet_name="Expenses")
            return buf.getvalue()
        except Exception:
            try:
                out = io.StringIO()
                it_doc.to_csv(out, index=False)
                return out.getvalue().encode("utf-8")
            except Exception:
                return None

    data_bytes = _client_excel_bytes(sel_id)
    if data_bytes:
        st.download_button(
            "⬇️ Download this client (Excel)",
            data=data_bytes,
            file_name=f"client_{row.get('ach_id','') or sel_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.caption("Could not build Excel; please try again later.")
st.divider()

# =========================
# 📞 Follow-ups by package
# =========================
st.subheader("📞 Follow-ups by package")
iids = df["itinerary_id"].astype(str).unique().tolist()
df_fu_bulk = load_followups_for_ids(iids)
if df_fu_bulk.empty:
    st.caption("No follow-ups for the current filters.")
else:
    df_fu_bulk["created_at"] = pd.to_datetime(df_fu_bulk["created_at"])
    latest = df_fu_bulk.sort_values("created_at").groupby("itinerary_id").last().reset_index()
    attempts = (df_fu_bulk["status"]=="followup").groupby(df_fu_bulk["itinerary_id"]).sum().rename("followup_attempts")
    confirmed_flag = (df_fu_bulk["status"]=="confirmed").groupby(df_fu_bulk["itinerary_id"]).any().rename("confirmed_from_followup")

    head = df[["itinerary_id","ach_id","client_name","client_mobile","representative"]].drop_duplicates("itinerary_id").set_index("itinerary_id")
    pkg_fu = head.join([attempts, confirmed_flag]).reset_index()
    pkg_fu["followup_attempts"] = pkg_fu["followup_attempts"].fillna(0).astype(int)
    pkg_fu["confirmed_from_followup"] = pkg_fu["confirmed_from_followup"].fillna(False)

    sfu = st.text_input("Search in follow-ups", value="", placeholder="Client / Mobile / ACH")
    view = pkg_fu.copy()
    if sfu.strip():
        ss = sfu.strip().lower()
        view = view[
            view["client_name"].astype(str).str.lower().str.contains(ss, na=False) |
            view["client_mobile"].astype(str).str.lower().str.contains(ss, na=False) |
            view["ach_id"].astype(str).str.lower().str.contains(ss, na=False)
        ]
    show_cols = ["ach_id","client_name","client_mobile","representative","followup_attempts","confirmed_from_followup"]
    st.dataframe(view[show_cols].sort_values(["confirmed_from_followup","followup_attempts"], ascending=[False, False]),
                 use_container_width=True, hide_index=True)

    st.markdown("**Trails (click ➕ to expand)**")
    for iid, grp in df_fu_bulk.sort_values("created_at", ascending=False).groupby("itinerary_id"):
        meta = head.loc[iid].to_dict() if iid in head.index else {}
        label = f"{meta.get('ach_id','')} — {meta.get('client_name','')} ({meta.get('client_mobile','')})"
        with st.expander(label, expanded=False):
            trail = grp[["created_at","created_by","status","next_followup_on","comment"]].copy()
            trail["next_followup_on"] = pd.to_datetime(trail["next_followup_on"], errors="coerce")
            trail.rename(columns={
                "created_at":"When",
                "created_by":"By",
                "status":"Status",
                "next_followup_on":"Next follow-up",
                "comment":"Comment"
            }, inplace=True)
            st.dataframe(trail.sort_values("When", ascending=False), use_container_width=True, hide_index=True)
st.divider()

# =========================
# 📞 Follow-up activity (range overview)
# =========================
st.subheader("📞 Follow-up activity (range overview)")
start_dt = datetime.combine(start, dtime.min)
end_dt   = datetime.combine(end, dtime.max)
logs = list(col_fu.find({"created_at": {"$gte": start_dt, "$lte": end_dt}}, {"_id":0}))
df_fu_range = pd.DataFrame(logs)

if df_fu_range.empty:
    st.caption("No follow-up logs in selected range.")
else:
    df_fu_range["created_at"] = pd.to_datetime(df_fu_range["created_at"])
    df_fu_range["next_followup_on"] = pd.to_datetime(df_fu_range.get("next_followup_on"), errors="coerce")
    total_fu = len(df_fu_range)
    total_confirmed_from_fu = int((df_fu_range["status"]=="confirmed").sum())
    fc1, fc2 = st.columns(2)
    fc1.metric("Total follow-up entries", total_fu)
    fc2.metric("Confirmed from follow-up", total_confirmed_from_fu)
    with st.expander("Show follow-up table"):
        show_cols = ["created_at","created_by","ach_id","client_name","client_mobile","status","next_followup_on","comment"]
        for m in show_cols:
            if m not in df_fu_range.columns: df_fu_range[m] = None
        st.dataframe(df_fu_range[show_cols].sort_values("created_at", ascending=False),
                     use_container_width=True, hide_index=True)
st.divider()

# =========================
# 🎯 Incentives overview
# =========================
st.subheader("🎯 Incentives overview")
pi1, pi2, pi3 = st.columns([1.2, 1.2, 2.6])
with pi1:
    inc_mode = st.selectbox("Period", ["This month","Last month","This FY","This year","Custom range"])
with pi2:
    t = date.today()
    if inc_mode == "This month":
        inc_start, inc_end = month_bounds(t)
    elif inc_mode == "Last month":
        ft,_ = month_bounds(t); inc_start, inc_end = month_bounds(ft - timedelta(days=1))
    elif inc_mode == "This FY":
        inc_start, inc_end = fy_bounds(t)
    elif inc_mode == "This year":
        inc_start, inc_end = date(t.year,1,1), t
    else:
        inc_start, inc_end = t.replace(day=1), t
with pi3:
    if inc_mode == "Custom range":
        dr2 = st.date_input("Choose incentive range", (inc_start, inc_end), key="incdr")
        if isinstance(dr2, tuple) and len(dr2)==2:
            inc_start, inc_end = dr2

q = {
    "status": "confirmed",
    "booking_date": {
        "$gte": datetime.combine(inc_start, dtime.min),
        "$lte": datetime.combine(inc_end, dtime.max)
    }
}
inc_rows = list(col_up.find(q, {"_id":0, "rep_name":1, "incentive":1}))
df_inc = pd.DataFrame(inc_rows)
if df_inc.empty:
    st.caption("No confirmed packages in this period.")
else:
    df_inc["rep_name"] = df_inc["rep_name"].fillna("").replace("", "Unassigned")
    df_inc["incentive"] = pd.to_numeric(df_inc["incentive"], errors="coerce").fillna(0).astype(int)
    rep_summary = df_inc.groupby("rep_name").agg(Confirmed=("incentive","count"), Incentive=("incentive","sum")).reset_index()
    cI1,cI2 = st.columns(2)
    cI1.metric("Total confirmed", int(rep_summary["Confirmed"].sum()))
    cI2.metric("Total incentives (₹)", f"{int(rep_summary['Incentive'].sum()):,}")
    st.dataframe(rep_summary.rename(columns={"rep_name":"Representative"}), use_container_width=True, hide_index=True)
st.divider()

# =========================
# ⬇️ Export (Excel) – filtered view + references
# =========================
st.subheader("⬇️ Export filtered data")
def export_filtered_bytes() -> bytes | None:
    start_dt = datetime.combine(start, dtime.min)
    end_dt   = datetime.combine(end, dtime.max)
    fu_rng = list(col_fu.find({"created_at": {"$gte": start_dt, "$lte": end_dt}}, {"_id":0}))
    df_fu_rng = pd.DataFrame(fu_rng)
    iids = df["itinerary_id"].astype(str).unique().tolist()
    raw_up = pd.DataFrame(list(col_up.find({"itinerary_id": {"$in": iids}}, {"_id":0})))
    raw_ex = pd.DataFrame(list(col_ex.find({"itinerary_id": {"$in": iids}}, {"_id":0})))
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df.to_excel(xw, index=False, sheet_name="Filtered_Packages")
            if not df_fu_rng.empty: df_fu_rng.to_excel(xw, index=False, sheet_name="Followups_in_range")
            if not raw_up.empty:   raw_up.to_excel(xw, index=False, sheet_name="Updates_raw")
            if not raw_ex.empty:   raw_ex.to_excel(xw, index=False, sheet_name="Expenses_raw")
        return buf.getvalue()
    except Exception:
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(df.columns.tolist())
        for _, row in df.iterrows():
            writer.writerow([row.get(c, "") for c in df.columns])
        return out.getvalue().encode("utf-8")

st.download_button(
    "Download Excel (current filters)",
    data=export_filtered_bytes(),
    file_name=f"TAK_dashboard_{date.today()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)
