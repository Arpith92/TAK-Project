# pages/03_Dashboard.py
from __future__ import annotations

# --- Block certain users (same as other pages) ---
import streamlit as st
if st.session_state.get("user") in ("Teena", "Kuldeep"):
    st.stop()  # silently deny

# ---- Optional compatibility shim if Streamlit downgrades Rich later ----
try:
    import rich  # noqa
    from packaging.version import Version
    import subprocess, sys
    if Version(st.__version__) < Version("1.42.0"):
        if Version(rich.__version__) >= Version("14.0.0"):
            subprocess.run([sys.executable, "-m", "pip", "install", "rich==13.9.4"], check=True)
            st.warning("Adjusted rich to 13.9.4 for compatibility. Rerunning…")
            st.experimental_rerun()
except Exception:
    pass

# ---- Page config ----
st.set_page_config(page_title="TAK Dashboard", layout="wide")

# ---- Optional calendar widget ----
CALENDAR_AVAILABLE = True
try:
    from streamlit_calendar import calendar
except Exception:
    CALENDAR_AVAILABLE = False

# ---- Plotly for modern interactive charts ----
PLOTLY = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    PLOTLY = False

from datetime import datetime, date, timedelta, time as dtime
from zoneinfo import ZoneInfo
from bson import ObjectId
from pymongo import MongoClient
import pandas as pd
import io

# ----------------------------
# Admin gate (same style as other admin pages)
# ----------------------------
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

# ----------------------------
# Mongo connection (with friendly errors)
# ----------------------------
try:
    MONGO_URI = st.secrets["mongo_uri"]
except KeyError:
    st.error("❌ Add `mongo_uri` in Manage app → Settings → Secrets for this app.")
    st.stop()

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
except Exception as e:
    st.error(f"❌ Could not connect to MongoDB: {e}")
    st.stop()

db = client["TAK_DB"]
col_it = db["itineraries"]
col_up = db["package_updates"]
col_ex = db["expenses"]
col_fu = db["followups"]
# (vendor_payments is not required here, but you could add if needed)

# ----------------------------
# Helpers
# ----------------------------
IST = ZoneInfo("Asia/Kolkata")

def _to_int(x, default=0):
    try:
        if x is None: return default
        s = str(x).replace(",", "")
        return int(round(float(s)))
    except Exception:
        return default

def _norm_date(x):
    try:
        if x is None or pd.isna(x): return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

def _created_utc_from_oid(iid: str) -> datetime | None:
    try:
        return ObjectId(str(iid)).generation_time  # tz-aware UTC
    except Exception:
        return None

def _fmt_ist(dt: datetime | None) -> str:
    if not dt: return ""
    try:
        return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M UTC")

def month_bounds(d: date):
    first = d.replace(day=1)
    last = (first + pd.offsets.MonthEnd(1)).date()
    return first, last

def fy_bounds(d: date):
    start = date(d.year if d.month >= 4 else d.year - 1, 4, 1)
    return start, d

def _final_cost_from_docs(it_row: dict, ex_row: dict | None) -> int:
    """FINAL = base_package_cost - discount, with sensible fallbacks."""
    ex_row = ex_row or {}
    if "final_package_cost" in ex_row:
        return _to_int(ex_row.get("final_package_cost", 0))
    if ("base_package_cost" in ex_row) or ("discount" in ex_row):
        base = _to_int(ex_row.get("base_package_cost", ex_row.get("package_cost", 0)))
        disc = _to_int(ex_row.get("discount", 0))
        return max(0, base - disc)
    if "package_cost" in ex_row:
        return _to_int(ex_row.get("package_cost", 0))
    # fall back to itinerary fields
    base = _to_int((it_row or {}).get("package_cost", 0))
    disc = _to_int((it_row or {}).get("discount", 0))
    return max(0, base - disc)

def _load_all():
    it = list(col_it.find({}))
    up = list(col_up.find({}))
    ex = list(col_ex.find({}))
    # normalize itineraries
    for r in it:
        r["itinerary_id"] = str(r["_id"])
        r["ach_id"] = r.get("ach_id", "")
        r["client_name"] = r.get("client_name", "")
        r["client_mobile"] = r.get("client_mobile", "")
        r["representative"] = r.get("representative", "")
        r["final_route"] = r.get("final_route", "")
        r["total_pax"] = _to_int(r.get("total_pax", 0))
        r["upload_date"] = _norm_date(r.get("upload_date"))
        r["start_date"] = _norm_date(r.get("start_date"))
        r["end_date"] = _norm_date(r.get("end_date"))
        r["created_utc"] = _created_utc_from_oid(r["itinerary_id"])
        r["_id"] = None
    # normalize updates
    for r in up:
        r["itinerary_id"] = str(r.get("itinerary_id"))
        r["status"] = r.get("status", "pending")
        r["assigned_to"] = r.get("assigned_to", "")
        r["booking_date"] = _norm_date(r.get("booking_date"))
        r["advance_amount"] = _to_int(r.get("advance_amount", 0))
        r["rep_name"] = r.get("rep_name", "")
        r["incentive"] = _to_int(r.get("incentive", 0))
        r["_id"] = None
    # normalize expenses
    for r in ex:
        r["itinerary_id"] = str(r.get("itinerary_id"))
        r["base_package_cost"] = _to_int(r.get("base_package_cost", 0))
        r["discount"] = _to_int(r.get("discount", 0))
        r["final_package_cost"] = _to_int(r.get("final_package_cost", r.get("package_cost", 0)))
        r["total_expenses"] = _to_int(r.get("total_expenses", 0))
        r["profit"] = _to_int(r.get("profit", 0))
        r["_id"] = None
    df_it = pd.DataFrame(it) if it else pd.DataFrame()
    df_up = pd.DataFrame(up) if up else pd.DataFrame(columns=[
        "itinerary_id","status","assigned_to","booking_date","advance_amount","rep_name","incentive"
    ])
    df_ex = pd.DataFrame(ex) if ex else pd.DataFrame(columns=[
        "itinerary_id","base_package_cost","discount","final_package_cost","total_expenses","profit"
    ])
    # merge
    df = df_it.merge(df_up, on="itinerary_id", how="left", suffixes=("","_up"))
    df = df.merge(df_ex, on="itinerary_id", how="left", suffixes=("","_ex"))
    # default status
    df["status"] = df["status"].fillna("pending")
    # compute final cost per row
    if not df.empty:
        def _calc_final(row):
            itrow = {k: row.get(k) for k in df_it.columns} if not df_it.empty else {}
            exrow = {k: row.get(k) for k in df_ex.columns} if not df_ex.empty else {}
            return _final_cost_from_docs(itrow, exrow)
        df["final_cost"] = df.apply(_calc_final, axis=1).astype(int)
        df["created_ist"] = df["created_utc"].apply(_fmt_ist)
    # ensure numeric cols
    for c in ("total_expenses","profit","incentive","advance_amount","total_pax"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

# ----------------------------
# Header + Global search
# ----------------------------
st.markdown("## 📊 TAK – Operations Dashboard")

top_l, top_r = st.columns([3,2])
with top_r:
    search_txt = st.text_input("🔎 Quick search", placeholder="Client name / Mobile / ACH ID", label_visibility="collapsed")

df_all = _load_all()
if df_all.empty:
    st.info("No data yet. Upload packages in the main app.")
    st.stop()

# ----------------------------
# Filters
# ----------------------------
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
        last_prev = ft - timedelta(days=1)
        start, end = month_bounds(last_prev)
    elif preset == "This FY":
        start, end = fy_bounds(today)
    elif preset == "This year":
        start, end = date(today.year,1,1), today
    elif preset == "Last 90 days":
        start, end = today - timedelta(days=90), today
    else:
        start, end = today - timedelta(days=30), today
    dr = st.date_input("Date range", (start, end))
    if isinstance(dr, tuple) and len(dr) == 2:
        start, end = dr
with flt5:
    reps = sorted([r for r in df_all["representative"].dropna().unique().tolist() if r]) or []
    rep_filter = st.multiselect("Representative", reps, default=reps)

# pick date column
date_col = {"Upload date":"upload_date","Booking date":"booking_date","Travel start date":"start_date"}[basis]

df = df_all.copy()
df[date_col] = df[date_col].apply(_norm_date)
df = df[df[date_col].between(start, end)]
if rep_filter:
    df = df[df["representative"].isin(rep_filter)]
# global search
if (search_txt or "").strip():
    s = search_txt.strip().lower()
    df = df[
        df["client_name"].astype(str).str.lower().str.contains(s) |
        df["client_mobile"].astype(str).str.lower().str.contains(s) |
        df["ach_id"].astype(str).str.lower().str.contains(s)
    ]

# "unique by mobile" view if enabled (latest by created_utc)
if dedupe_mobile and not df.empty:
    df = df.sort_values(["client_mobile","created_utc"], ascending=[True, False]) \
           .groupby("client_mobile", as_index=False).first()

# convenience masks
is_confirmed = df["status"].eq("confirmed")
is_pending = df["status"].eq("pending")
is_udisc = df["status"].eq("under_discussion")
is_cancel = df["status"].eq("cancelled")
is_followup = df["status"].eq("followup")
is_enquiry = is_pending | is_udisc | is_followup  # sales pipeline

# ----------------------------
# KPI row
# ----------------------------
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("✅ Confirmed", int(is_confirmed.sum()))
k2.metric("🟡 Enquiries", int(is_enquiry.sum()))
k3.metric("🟠 Under discussion", int(is_udisc.sum()))
k4.metric("🔵 Follow-up", int(is_followup.sum()))
k5.metric("🔴 Cancelled", int(is_cancel.sum()))
# expenses pending among confirmed
have_cost = set(df.loc[df["total_expenses"]>0, "itinerary_id"])
k6.metric("🧾 Expense entry pending", int(max(is_confirmed.sum() - len(set(df.loc[is_confirmed,"itinerary_id"]) & have_cost), 0)))

st.divider()

# ----------------------------
# Donut: status split
# ----------------------------
st.subheader("Status mix")
mix = pd.DataFrame({
    "status":["confirmed","under_discussion","followup","pending","cancelled"],
    "count":[int(is_confirmed.sum()), int(is_udisc.sum()), int(is_followup.sum()), int(is_pending.sum()), int(is_cancel.sum())]
})
mix = mix[mix["count"]>0]
if PLOTLY and not mix.empty:
    fig = px.pie(mix, values="count", names="status", hole=0.55)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.dataframe(mix, use_container_width=True, hide_index=True)

st.divider()

# ----------------------------
# Daily trends: Confirmed vs Enquiries
# ----------------------------
st.subheader("📈 Daily trends (Confirmed vs Enquiries)")
# confirmed by booking_date, enquiries by upload_date (when created)
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

if PLOTLY and not trend.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["Confirmed"], mode="lines+markers", name="Confirmed"))
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["Enquiries"], mode="lines+markers", name="Enquiries"))
    fig.update_layout(xaxis_title=None, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(trend.set_index("date"))

st.divider()

# ----------------------------
# Revenue snapshot (confirmed in range)
# ----------------------------
st.subheader("💰 Revenue snapshot (confirmed in range)")
conf_in_range = df[df["status"].eq("confirmed")].copy()
sum_final = int(conf_in_range["final_cost"].sum())
sum_exp   = int(conf_in_range["total_expenses"].sum())
sum_prof  = int(conf_in_range["profit"].sum())
c1,c2,c3 = st.columns(3)
c1.metric("Final package (₹)", f"{sum_final:,}")
c2.metric("Expenses (₹)", f"{sum_exp:,}")
c3.metric("Profit (₹)", f"{sum_prof:,}")

if PLOTLY and not conf_in_range.empty:
    by_rep = conf_in_range.groupby(conf_in_range["rep_name"].replace("", "Unassigned")).agg(
        Final=("final_cost","sum"), Profit=("profit","sum"), Count=("itinerary_id","nunique")
    ).reset_index().sort_values("Final", ascending=False)
    fig = px.bar(by_rep, x="rep_name", y="Final", text="Count", labels={"rep_name":"Representative"})
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------
# Top routes (bar) & Pax distribution (hist)
# ----------------------------
cA, cB = st.columns(2)
with cA:
    st.subheader("🗺️ Top routes")
    top_routes = df.groupby("final_route")["itinerary_id"].nunique().reset_index().sort_values("itinerary_id", ascending=False).head(12)
    top_routes.columns = ["Route","Packages"]
    if PLOTLY and not top_routes.empty:
        fig = px.bar(top_routes, x="Packages", y="Route", orientation="h", text="Packages")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(top_routes, use_container_width=True, hide_index=True)
with cB:
    st.subheader("👥 Pax distribution")
    pax = df[df["total_pax"]>0]["total_pax"]
    if PLOTLY and not pax.empty:
        fig = px.histogram(pax, nbins=10)
        fig.update_layout(xaxis_title="Total pax", yaxis_title="Packages")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pax.value_counts().sort_index())

st.divider()

# ----------------------------
# Calendars (Confirmed by Booking / Travel)
# ----------------------------
st.subheader("📅 Calendars")
cal_tab1, cal_tab2 = st.tabs(["Confirmed – Booking dates", "Travel dates"])
with cal_tab1:
    confirmed = df_all[df_all["status"].eq("confirmed")].copy()
    confirmed["booking_date"] = confirmed["booking_date"].apply(_norm_date)
    confirmed = confirmed.dropna(subset=["booking_date"])
    events = [{"title": f"{r.client_name}_{r.total_pax}pax", "id": r.itinerary_id, "start": str(r.booking_date)} 
              for r in confirmed.itertuples()]
    if CALENDAR_AVAILABLE:
        opts = {"initialView":"dayGridMonth", "height": 620, "eventDisplay":"block"}
        calendar(options=opts, events=events, key="cal_book")
    else:
        st.caption("Calendar component not installed; showing list.")
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
        opts = {"initialView":"dayGridMonth", "height": 620, "eventDisplay":"block"}
        calendar(options=opts, events=events, key="cal_travel")
    else:
        st.caption("Calendar component not installed; showing list.")
        show = pd.DataFrame(events).rename(columns={"title":"Package","start":"Start","end":"End"})
        st.dataframe(show.sort_values(["Start","End"]), use_container_width=True, hide_index=True)

st.divider()

# ----------------------------
# Client explorer (full detail + export)
# ----------------------------
st.subheader("🧭 Client explorer")
# Build options
opt_df = df_all.copy()
opt_df["label"] = (opt_df["ach_id"].fillna("") + " | " +
                   opt_df["client_name"].fillna("") + " | " +
                   opt_df["client_mobile"].fillna("") + " | " +
                   opt_df["created_ist"].fillna(""))
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

    # Per-client Excel export
    def _client_excel_bytes(iid: str) -> bytes:
        it_doc = df_all[df_all["itinerary_id"]==iid].copy()
        up_docs = pd.DataFrame(list(col_up.find({"itinerary_id": str(iid)}, {"_id":0})))
        fu_docs = pd.DataFrame(list(col_fu.find({"itinerary_id": str(iid)}, {"_id":0})))
        ex_doc = pd.DataFrame(list(col_ex.find({"itinerary_id": str(iid)}, {"_id":0})))
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            it_doc.to_excel(xw, index=False, sheet_name="Itinerary")
            if not up_docs.empty: up_docs.to_excel(xw, index=False, sheet_name="Updates")
            if not fu_docs.empty: fu_docs.to_excel(xw, index=False, sheet_name="Followups")
            if not ex_doc.empty: ex_doc.to_excel(xw, index=False, sheet_name="Expenses")
        return buf.getvalue()

    st.download_button(
        "⬇️ Download this client (Excel)",
        data=_client_excel_bytes(sel_id),
        file_name=f"client_{row.get('ach_id','') or sel_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.divider()

# ----------------------------
# Follow-up activity (range)
# ----------------------------
st.subheader("📞 Follow-up activity (range)")
start_dt = datetime.combine(start, dtime.min)
end_dt   = datetime.combine(end,   dtime.max)
logs = list(col_fu.find({"created_at": {"$gte": start_dt, "$lte": end_dt}}, {"_id":0}))
df_fu = pd.DataFrame(logs)
if df_fu.empty:
    st.caption("No follow-up logs in selected range.")
else:
    df_fu["created_at"] = pd.to_datetime(df_fu["created_at"])
    df_fu["next_followup_on"] = pd.to_datetime(df_fu.get("next_followup_on"), errors="coerce")
    # Quick KPIs
    total_fu = len(df_fu)
    total_confirmed_from_fu = int((df_fu["status"]=="confirmed").sum())
    fc1, fc2 = st.columns(2)
    fc1.metric("Total follow-up entries", total_fu)
    fc2.metric("Confirmed from follow-up", total_confirmed_from_fu)

    if PLOTLY:
        by_user = df_fu.groupby(df_fu["created_by"].replace("", "Unknown"))["status"].count().reset_index(name="entries")
        fig = px.bar(by_user, x="created_by", y="entries", labels={"created_by":"User","entries":"Entries"})
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show follow-up table"):
        show_cols = ["created_at","created_by","ach_id","client_name","client_mobile","status","next_followup_on","comment"]
        missing = [c for c in show_cols if c not in df_fu.columns]
        for m in missing: df_fu[m] = None
        st.dataframe(df_fu[show_cols].sort_values("created_at", ascending=False), use_container_width=True, hide_index=True)

st.divider()

# ----------------------------
# Incentives overview
# ----------------------------
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
    if PLOTLY:
        fig = px.bar(rep_summary, x="Representative", y="Incentive", text="Confirmed")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------
# Export (Excel) – filtered data & reference tables
# ----------------------------
st.subheader("⬇️ Export filtered data")
def export_filtered_bytes() -> bytes:
    buf = io.BytesIO()
    # Build follow-ups in same range
    fu_rng = list(col_fu.find({"created_at": {"$gte": start_dt, "$lte": end_dt}}, {"_id":0}))
    df_fu_rng = pd.DataFrame(fu_rng)
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="Filtered_Packages")
        if not df_fu_rng.empty: df_fu_rng.to_excel(xw, index=False, sheet_name="Followups_in_range")
        # Also dump raw per-table filtered by itinerary ids
        iids = df["itinerary_id"].astype(str).unique().tolist()
        raw_up = pd.DataFrame(list(col_up.find({"itinerary_id": {"$in": iids}}, {"_id":0})))
        raw_ex = pd.DataFrame(list(col_ex.find({"itinerary_id": {"$in": iids}}, {"_id":0})))
        if not raw_up.empty: raw_up.to_excel(xw, index=False, sheet_name="Updates_raw")
        if not raw_ex.empty: raw_ex.to_excel(xw, index=False, sheet_name="Expenses_raw")
    return buf.getvalue()

st.download_button(
    "Download Excel (current filters)",
    data=export_filtered_bytes(),
    file_name=f"TAK_dashboard_{date.today()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)
