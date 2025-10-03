from __future__ import annotations

from datetime import datetime, date
from typing import Optional, Dict, List, Tuple
import os

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# =========================
# Page
# =========================
st.set_page_config(page_title="Splitwise Reimbursements", layout="wide")
st.title("🧾 Splitwise-style Reimbursements")

CATEGORIES = ["Car", "Hotel", "Bhasmarathi", "Poojan", "PhotoFrame", "Other"]
IST_TTL = 60  # short cache TTL so updates reflect quickly

# =========================
# Dark mode + Defer loads
# =========================
with st.sidebar:
    dark_mode = st.toggle("🌙 Dark mode", value=False, help="Switch between dark and normal theme")
    st.markdown("---")
    defer_loads = st.toggle("⚡ Defer heavy loads", value=True,
                            help="Skip loading big tables until you press Refresh or after a Save.")
    refresh_now = st.button("🔄 Refresh data now", use_container_width=True)

if dark_mode:
    st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] { background: #0f1115 !important; color: #e5e7eb !important; }
        [data-testid="stHeader"] { background: #0f1115 !important; }
        .stMarkdown, .stText, .stDataFrame, .stMetric { color: #e5e7eb !important; }
        div[data-baseweb="select"] * { color: #e5e7eb !important; }
        .st-af { background: #111318 !important; }
        .st-bc, .st-bb { background: #151822 !important; }
        .stButton > button { background:#1f2937;color:#e5e7eb;border-radius:8px;border:1px solid #374151; }
        .stButton > button:hover { background:#374151; }
        </style>
    """, unsafe_allow_html=True)

if refresh_now:
    st.session_state["force_refresh"] = True

def _defer_guard(msg: str) -> bool:
    if defer_loads and not st.session_state.get("force_refresh", False):
        st.info(f"Deferred: {msg}\n\nClick **🔄 Refresh data now** (sidebar) to load.")
        return True
    return False

def _clear_force_refresh():
    if st.session_state.get("force_refresh"):
        st.session_state.pop("force_refresh", None)

# =========================
# Secrets & Mongo
# =========================
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
def _get_client() -> MongoClient:
    uri = _find_uri()
    if not uri:
        st.error("Mongo connection is not configured.\n\nAdd one of these in Manage app → Settings → Secrets (recommended: `mongo_uri`).")
        st.stop()
    client = MongoClient(
        uri,
        appName="TAK_Splitwise",
        maxPoolSize=100,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        retryWrites=True,
        tz_aware=True,
    )
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        st.error(f"Could not connect to MongoDB. Details: {e}")
        st.stop()
    return client

@st.cache_resource
def get_db():
    return _get_client()["TAK_DB"]

db = get_db()
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_split       = db["expense_splitwise"]
col_cars        = db["direct_car_bookings"]

# =========================
# Users & login
# =========================
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    try:
        try:
            import tomllib
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, dict) and u:
            with st.sidebar:
                st.warning("Using users from repo .streamlit/secrets.toml. For production, set them in Manage app → Secrets.")
            return u
    except Exception:
        pass
    return {}

ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith", "Kuldeep"]))

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
    if not users_map:
        st.error("Login not configured yet. Add to Manage app → Secrets:\n\nmongo_uri = \"...\"\n\n[users]\nArpith=\"1234\"...")
        st.stop()

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns(2)
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
is_admin = user in ADMIN_USERS

# =========================
# Helpers
# =========================
def _to_int(x, default=0):
    try:
        if x is None: return default
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def norm_date(x):
    try:
        if x is None or pd.isna(x): return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

@st.cache_data(ttl=IST_TTL, show_spinner=False)
def all_employees() -> List[str]:
    return [e for e in sorted(load_users().keys()) if e]

def entry_to_row(d: dict) -> dict:
    return {
        "Kind": d.get("kind",""),
        "Date": norm_date(d.get("date")),
        "Employee": d.get("payer") or d.get("employee") or "",
        "Customer": d.get("customer_name",""),
        "ACH ID": d.get("ach_id",""),
        "Category": d.get("category",""),
        "Subheader": d.get("subheader",""),
        "Amount (₹)": _to_int(d.get("amount",0)),
        "Notes": d.get("notes",""),
        "Ref": d.get("ref",""),
        "itinerary_id": d.get("itinerary_id",""),
        "_id": str(d.get("_id","")),
        "created_by": d.get("created_by",""),
        "created_at": d.get("created_at"),
    }
# =========================
# 🔹 Direct Car integration
# =========================
def directcar_to_rows(start: Optional[date], end: Optional[date]) -> List[dict]:
    q = {}
    if start and end:
        q["date"] = {"$gte": datetime.combine(start, datetime.min.time()),
                     "$lte": datetime.combine(end, datetime.max.time())}
    docs = list(col_cars.find(q))
    rows=[]
    for d in docs:
        base = {
            "kind":"expense",
            "date": d.get("date"),
            "customer_name": d.get("client_name",""),
            "category":"Car",
            "subheader": d.get("trip_plan",""),
            "amount": d.get("amount",0),
            "notes": d.get("notes",""),
            "ref":"Direct Car",
            "_id": str(d.get("_id",""))
        }
        if d.get("received_in")=="Company Account":
            base["payer"]="Company"
            rows.append(entry_to_row(base))
        else:
            for emp in d.get("employees",[]):
                base2=base.copy(); base2["payer"]=emp
                rows.append(entry_to_row(base2))
    return rows

# =========================
# Confirmed packages helper
# =========================
@st.cache_data(ttl=IST_TTL, show_spinner=False)
def confirmed_itineraries_df_unique_clients() -> pd.DataFrame:
    """Get unique confirmed itineraries (latest revision per client+date)."""
    q = {"status": "confirmed"}
    cur = col_itineraries.find(q, {
        "_id": 1, "client_name": 1, "client_mobile": 1,
        "start_date": 1, "ach_id": 1, "created_at": 1
    })
    docs = list(cur)
    if not docs:
        return pd.DataFrame(columns=["itinerary_id","client_name","client_mobile","ach_id","start_date"])
    rows=[]
    for d in docs:
        rows.append({
            "itinerary_id": str(d.get("_id","")),
            "client_name": d.get("client_name",""),
            "client_mobile": d.get("client_mobile",""),
            "ach_id": d.get("ach_id",""),
            "start_date": norm_date(d.get("start_date")),
            "created_at": d.get("created_at"),
        })
    df=pd.DataFrame(rows)
    # keep latest per client_mobile + start_date
    df=df.sort_values("created_at").drop_duplicates(
        subset=["client_mobile","start_date"], keep="last"
    )
    return df

def pack_options(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty: return []
    labels=[]
    for _,r in df.iterrows():
        labels.append(
            f"{r.get('client_name','')} ({r.get('client_mobile','')}) "
            f"| {r.get('start_date','')} | {r.get('ach_id','')} | {r.get('itinerary_id','')}"
        )
    return labels

# =========================
# Data fetchers
# =========================
def fetch_entries(start: Optional[date] = None, end: Optional[date] = None,
                  employee: Optional[str] = None, itinerary_id: Optional[str] = None) -> pd.DataFrame:
    q: Dict = {}
    if start and end:
        q["date"] = {"$gte": datetime.combine(start, datetime.min.time()),
                     "$lte": datetime.combine(end, datetime.max.time())}
    if employee:
        q["$or"] = [{"payer": employee}, {"employee": employee}]
    if itinerary_id:
        q["itinerary_id"] = str(itinerary_id)
    cur = col_split.find(q, projection={"_id":1,"kind":1,"date":1,"payer":1,"employee":1,
                                        "customer_name":1,"ach_id":1,"category":1,"subheader":1,
                                        "amount":1,"notes":1,"ref":1,"itinerary_id":1,
                                        "created_by":1,"created_at":1}).sort("date", 1)
    rows = [entry_to_row(d) for d in cur]
    # 🔹 merge direct car bookings
    rows += directcar_to_rows(start, end)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(entry_to_row({}).keys()))

def totals_for_employee(emp: str, start: Optional[date]=None, end: Optional[date]=None) -> Tuple[int,int,int]:
    df = fetch_entries(start, end, employee=emp)
    exp_sum = int(df[df["Kind"]=="expense"]["Amount (₹)"].sum())
    pay_sum = int(df[df["Kind"]=="settlement"]["Amount (₹)"].sum())
    return exp_sum, pay_sum, (exp_sum - pay_sum)

# =========================
# DB ops
# =========================
def add_expense(*, payer: str, itinerary_id: str, customer_name: str, ach_id: str,
                category: str, subheader: str, amount: int, when: date, notes: str) -> None:
    col_split.insert_one({
        "kind": "expense", "created_at": datetime.utcnow(), "created_by": user,
        "date": datetime.combine(when, datetime.min.time()), "payer": payer,
        "itinerary_id": itinerary_id, "customer_name": customer_name, "ach_id": ach_id,
        "category": category or "Other", "subheader": subheader or "",
        "amount": int(amount), "notes": notes or "",
    })

def add_settlement(*, employee: str, amount: int, when: date, ref: str, notes: str) -> None:
    col_split.insert_one({
        "kind": "settlement", "created_at": datetime.utcnow(), "created_by": user,
        "date": datetime.combine(when, datetime.min.time()), "employee": employee,
        "amount": int(amount), "ref": ref or "", "notes": notes or "",
    })

def add_direct_car_settlement(*, employee: str, amount: int, when: date, car_type: str, client: str):
    col_split.insert_one({
        "kind": "settlement", "created_at": datetime.utcnow(), "created_by": user,
        "date": datetime.combine(when, datetime.min.time()), "employee": employee,
        "amount": int(amount), "ref": f"Direct Car ({car_type})",
        "notes": f"Direct booking for {client or 'N/A'}",
    })

# =========================
# Filters / controls
# =========================
if _defer_guard("Filters and confirmed customer list"):
    _clear_force_refresh()
    st.stop()

df_confirmed_unique = confirmed_itineraries_df_unique_clients()

with st.container():
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 2.2])
    with f1:
        today = date.today()
        month_start = today.replace(day=1)
        start = st.date_input("From", value=month_start)
    with f2:
        end = st.date_input("To", value=today)
        if end < start:
            end = start
    with f3:
        emp_options = all_employees()
        default_emp = [user] if user in emp_options else []
        emp_filter = st.multiselect("Employees", options=emp_options, default=default_emp)
    with f4:
        search_txt = st.text_input("Search confirmed client/mobile/ACH", placeholder="Type to filter package list…")

# =========================
# KPIs for current user
# =========================
if not _defer_guard("My balances KPIs"):
    st.subheader("My balances")
    exp_m, pay_m, bal_m = totals_for_employee(user, start, end)
    exp_all, pay_all, bal_all = totals_for_employee(user, None, None)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Paid this period", f"₹ {exp_m:,}")
    k2.metric("Settled this period", f"₹ {pay_m:,}")
    k3.metric("Balance this period", f"₹ {bal_m:,}")
    k4.metric("Paid (all time)", f"₹ {exp_all:,}")
    k5.metric("Settled (all time)", f"₹ {pay_all:,}")
    k6.metric("Balance (all time)", f"₹ {bal_all:,}")

st.divider()

# =========================
# Add expense
# =========================
st.subheader("➕ Add expense")
mode = st.radio("Expense type", ["Linked to confirmed package", "Other expense (no package)"], horizontal=True)

with st.form("add_expense_form", clear_on_submit=False):
    if mode == "Linked to confirmed package":
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            iid, cust_name, ach = "", "", ""
            opt_labels = pack_options(df_confirmed_unique)
            if opt_labels:
                sel = st.selectbox("Confirmed package / customer", opt_labels)
                if sel:
                    rid = sel.split(" | ")[-1].strip()
                    row = df_confirmed_unique[df_confirmed_unique["itinerary_id"]==rid].iloc[0]
                    iid, cust_name, ach = rid, row["client_name"], row["ach_id"]
        with c2:
            when = st.date_input("Date", value=date.today())
        with c3:
            amount = st.number_input("Amount (₹)", min_value=0, step=100, value=0)
        c4, c5 = st.columns(2)
        with c4:
            category = st.selectbox("Category", CATEGORIES, index=0)
        with c5:
            subheader = st.text_input("Subheader (detail)")
        notes = st.text_area("Notes (optional)")
    else:
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            cust_name = st.text_input("Beneficiary / Customer (free text)")
        with c2:
            when = st.date_input("Date", value=date.today(), key="oth_date")
        with c3:
            amount = st.number_input("Amount (₹)", min_value=0, step=100, value=0, key="oth_amt")
        c4, c5 = st.columns(2)
        with c4:
            category = st.selectbox("Category", CATEGORIES, index=5, key="oth_cat")
        with c5:
            subheader = st.text_input("Subheader (detail)", key="oth_sub")
        notes = st.text_area("Notes (optional)", key="oth_notes")
        iid, ach = "", ""

    submitted = st.form_submit_button("Save expense")

if submitted:
    if amount <= 0:
        st.error("Amount must be > 0.")
    else:
        add_expense(
            payer=user, itinerary_id=str(iid or ""), customer_name=cust_name,
            ach_id=str(ach or ""), category=category, subheader=subheader,
            amount=int(amount), when=when, notes=notes
        )
        st.success("Expense added.")
        st.session_state["force_refresh"] = True
        st.rerun()

st.divider()

# =========================
# Admin settlements
# =========================
if is_admin:
    st.subheader("💵 Admin – Settle employee balance")
    with st.form("settlement_form", clear_on_submit=False):
        a1, a2, a3, a4 = st.columns([1.4,1,1,1.6])
        with a1:
            emp_to_pay = st.selectbox("Employee", all_employees(), index=0)
        with a2:
            pay_date = st.date_input("Date", value=date.today(), key="pay_dt")
        with a3:
            pay_amt = st.number_input("Amount (₹)", min_value=0, step=500, value=0)
        with a4:
            ref = st.text_input("Ref / Mode", placeholder="UPI/NEFT/Receipt #")
        notes_s = st.text_area("Notes (optional)")
        pay_btn = st.form_submit_button("Record settlement")
    if pay_btn and emp_to_pay and pay_amt>0:
        add_settlement(employee=emp_to_pay, amount=int(pay_amt), when=pay_date, ref=ref, notes=notes_s)
        st.success("Settlement recorded.")
        st.session_state["force_refresh"] = True
        st.rerun()

st.divider()

# =========================
# Package ledger
# =========================
st.subheader("📦 Package ledger (confirmed only)")
if not _defer_guard("Package ledger"):
    opt_labels = pack_options(df_confirmed_unique)
    if opt_labels:
        sel = st.selectbox("Pick a confirmed package to view ledger", opt_labels)
        if sel:
            rid = sel.split(" | ")[-1].strip()
            df_pkg = fetch_entries(start=None, end=None, itinerary_id=rid)
            if df_pkg.empty:
                st.info("No entries yet for this package.")
            else:
                exp_tbl = df_pkg[df_pkg["Kind"]=="expense"].copy()
                exp_tbl = exp_tbl[["Date","Employee","Category","Subheader","Amount (₹)","Notes"]].sort_values("Date")
                st.dataframe(exp_tbl, use_container_width=True, hide_index=True)
                with st.expander("Summaries"):
                    by_emp = exp_tbl.groupby("Employee", as_index=False)["Amount (₹)"].sum()
                    by_cat = exp_tbl.groupby("Category", as_index=False)["Amount (₹)"].sum()
                    c1, c2 = st.columns(2)
                    c1.dataframe(by_emp, use_container_width=True, hide_index=True)
                    c2.dataframe(by_cat, use_container_width=True, hide_index=True)

st.divider()

# =========================
# Team balances
# =========================
st.subheader("👥 Balances")
if not _defer_guard("Team balances"):
    emps = emp_filter if emp_filter else all_employees()
    rows=[]
    for e in emps:
        em = totals_for_employee(e, start, end)
        ea = totals_for_employee(e, None, None)
        rows.append({
            "Employee": e,
            "Paid (period)": em[0],
            "Settled (period)": em[1],
            "Balance (period)": em[2],
            "Paid (all time)": ea[0],
            "Settled (all time)": ea[1],
            "Balance (all time)": ea[2],
        })
    df_bal=pd.DataFrame(rows)
    st.dataframe(df_bal, use_container_width=True, hide_index=True)

st.divider()

# =========================
# My entries
# =========================
st.subheader("📜 My entries (in selected period)")
if not _defer_guard("My entries table"):
    df_me = fetch_entries(start, end, employee=user, itinerary_id=None)
    if df_me.empty:
        st.info("No entries in this period.")
    else:
        show_cols=["Kind","Date","Customer","ACH ID","Category","Subheader","Amount (₹)","Notes","Ref"]
        st.dataframe(df_me[show_cols].sort_values(["Date","Kind"]),
                     use_container_width=True, hide_index=True)

# =========================
# End cycle clear
# =========================
_clear_force_refresh()
