# pages/04_Splitwise_Reimbursements.py
from __future__ import annotations

from datetime import datetime, date
from typing import Optional, Dict, List

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# ----------------------------
# Page & constants
# ----------------------------
st.set_page_config(page_title="Splitwise Reimbursements", layout="wide")
st.title("🧾 Splitwise-style Reimbursements")

CATEGORIES = ["Car", "Hotel", "Bhasmarathi", "Poojan", "PhotoFrame", "Other"]

# ----------------------------
# Secrets & Mongo
# ----------------------------
MONGO_URI = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
db = client["TAK_DB"]
col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_split       = db["expense_splitwise"]     # this page writes here

# ----------------------------
# Users & login
# ----------------------------
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    try:
        try:
            import tomllib  # 3.11+
        except Exception:  # pragma: no cover
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        return data.get("users", {}) or {}
    except Exception:
        return {}

ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith"]))

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
        st.error("Login not configured. Add `mongo_uri` and a [users] table in **Manage app → Secrets**.")
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

from tak_audit import audit_pageview
audit_pageview(st.session_state.get("user", "Admin"), "04_Splitwise_Reimbursements")


# ----------------------------
# Helpers
# ----------------------------
def _to_int(x, default=0):
    try:
        if x is None:
            return default
        return int(round(float(str(x).replace(",", ""))))
    except Exception:
        return default

def norm_date(x):
    try:
        if x is None or pd.isna(x):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

def all_employees() -> List[str]:
    return [e for e in sorted(load_users().keys()) if e]

def confirmed_itineraries_df() -> pd.DataFrame:
    """
    Only itineraries that are CONFIRMED in package_updates.
    """
    # pull confirmed ids
    upd = list(col_updates.find({"status":"confirmed"}, {"_id":0, "itinerary_id":1}))
    if not upd:
        return pd.DataFrame(columns=["itinerary_id","ach_id","client_name","client_mobile","start_date","end_date","final_route"])
    ids = [str(u["itinerary_id"]) for u in upd if u.get("itinerary_id")]

    its = list(col_itineraries.find(
        {"_id": {"$in": [ObjectId(i) for i in ids if len(i)==24]}},
        {"_id":1,"ach_id":1,"client_name":1,"client_mobile":1,"start_date":1,"end_date":1,"final_route":1}
    ))
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        r["start_date"] = norm_date(r.get("start_date"))
        r["end_date"]   = norm_date(r.get("end_date"))
    df = pd.DataFrame(its) if its else pd.DataFrame(columns=[
        "itinerary_id","ach_id","client_name","client_mobile","start_date","end_date","final_route"
    ])
    return df.drop(columns=["_id"], errors="ignore")

def pack_label(r: pd.Series) -> str:
    return f"{(r.get('ach_id') or '').strip()} | {(r.get('client_name') or '').strip()} | {(r.get('client_mobile') or '').strip()} | {(r.get('itinerary_id') or '').strip()}"

def pack_options(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    return [pack_label(r) for _, r in df.iterrows()]

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

# ----------------------------
# DB ops
# ----------------------------
def add_expense(
    *, payer: str, itinerary_id: str, customer_name: str, ach_id: str,
    category: str, subheader: str, amount: int, when: date, notes: str
) -> None:
    col_split.insert_one({
        "kind": "expense",
        "created_at": datetime.utcnow(),
        "created_by": user,
        "date": datetime.combine(when, datetime.min.time()),
        "payer": payer,
        "itinerary_id": itinerary_id,
        "customer_name": customer_name,
        "ach_id": ach_id,
        "category": category or "Other",
        "subheader": subheader or "",
        "amount": int(amount),
        "notes": notes or "",
    })

def add_settlement(*, employee: str, amount: int, when: date, ref: str, notes: str) -> None:
    col_split.insert_one({
        "kind": "settlement",
        "created_at": datetime.utcnow(),
        "created_by": user,
        "date": datetime.combine(when, datetime.min.time()),
        "employee": employee,
        "amount": int(amount),
        "ref": ref or "",
        "notes": notes or "",
    })

def fetch_entries(start: Optional[date] = None, end: Optional[date] = None,
                  employee: Optional[str] = None, itinerary_id: Optional[str] = None) -> pd.DataFrame:
    q: Dict = {}
    if start and end:
        q["date"] = {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end, datetime.max.time()),
        }
    if employee:
        q["$or"] = [{"payer": employee}, {"employee": employee}]
    if itinerary_id:
        q["itinerary_id"] = str(itinerary_id)
    cur = col_split.find(q).sort("date", 1)
    rows = [entry_to_row(d) for d in cur]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(entry_to_row({}).keys()))

def totals_for_employee(emp: str, start: Optional[date]=None, end: Optional[date]=None):
    q_exp: Dict = {"kind":"expense","payer":emp}
    q_pay: Dict = {"kind":"settlement","employee":emp}
    if start and end:
        span = {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end, datetime.max.time()),
        }
        q_exp["date"] = span
        q_pay["date"] = span
    exp_sum = sum(_to_int(d.get("amount",0)) for d in col_split.find(q_exp, {"amount":1}))
    pay_sum = sum(_to_int(d.get("amount",0)) for d in col_split.find(q_pay, {"amount":1}))
    return exp_sum, pay_sum, (exp_sum - pay_sum)

# ----------------------------
# Filters / controls
# ----------------------------
df_confirmed = confirmed_itineraries_df()

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
        emp_filter = st.multiselect("Employees", options=emp_options, default=[user])
    with f4:
        search_txt = st.text_input("Search confirmed client/mobile/ACH", placeholder="Type to filter package list…")

def choose_package(label="Select confirmed package", key="pkg_pick"):
    options = df_confirmed.copy()
    if search_txt.strip():
        s = search_txt.strip().lower()
        options = options[
            options["client_name"].astype(str).str.lower().str.contains(s) |
            options["client_mobile"].astype(str).str.lower().str.contains(s) |
            options["ach_id"].astype(str).str.lower().str.contains(s)
        ]
    opt_labels = pack_options(options)
    sel = st.selectbox(label, opt_labels, index=0 if opt_labels else None, key=key)
    if not sel:
        return None, "", ""
    rid = sel.split(" | ")[-1]
    row = options[options["itinerary_id"]==rid].iloc[0]
    return rid, row.get("client_name",""), row.get("ach_id","")

# ----------------------------
# KPIs for current user
# ----------------------------
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

# ----------------------------
# Add expense
# ----------------------------
st.subheader("➕ Add expense")
mode = st.radio("Expense type", ["Linked to confirmed package", "Other expense (no package)"], horizontal=True)

with st.form("add_expense_form", clear_on_submit=False):
    if mode == "Linked to confirmed package":
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            iid, cust_name, ach = choose_package("Confirmed package / customer", key="add_pkg")
        with c2:
            when = st.date_input("Date", value=date.today())
        with c3:
            amount = st.number_input("Amount (₹)", min_value=0, step=100, value=0)
        c4, c5 = st.columns([1,1])
        with c4:
            category = st.selectbox("Category", CATEGORIES, index=0)
        with c5:
            subheader = st.text_input("Subheader (detail)", placeholder="e.g., Airport transfer / Room upgrade")
        notes = st.text_area("Notes (optional)", placeholder="Anything helpful for accounting")
    else:
        # Other expense flow (no package)
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            cust_name = st.text_input("Beneficiary / Customer (free text)", placeholder="e.g., Office supplies / Misc")
        with c2:
            when = st.date_input("Date", value=date.today(), key="oth_date")
        with c3:
            amount = st.number_input("Amount (₹)", min_value=0, step=100, value=0, key="oth_amt")
        c4, c5 = st.columns([1,1])
        with c4:
            category = st.selectbox("Category", CATEGORIES, index=5, key="oth_cat")  # default Other
        with c5:
            subheader = st.text_input("Subheader (detail)", placeholder="e.g., Courier / Printouts", key="oth_sub")
        notes = st.text_area("Notes (optional)", placeholder="Anything helpful for accounting", key="oth_notes")
        iid, ach = "", ""  # not tied to any itinerary

    submitted = st.form_submit_button("Save expense")

if submitted:
    if amount <= 0:
        st.error("Amount must be > 0.")
    elif mode == "Linked to confirmed package" and not iid:
        st.error("Please choose a confirmed package/customer.")
    else:
        add_expense(
            payer=user, itinerary_id=str(iid or ""), customer_name=cust_name, ach_id=str(ach or ""),
            category=category, subheader=subheader, amount=int(amount), when=when, notes=notes
        )
        st.success("Expense added.")
        st.rerun()

# ----------------------------
# Admin: settle balances
# ----------------------------
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
        notes_s = st.text_area("Notes (optional)", placeholder="e.g., July reimbursement")
        pay_btn = st.form_submit_button("Record settlement")
    if pay_btn:
        if not emp_to_pay or pay_amt <= 0:
            st.error("Choose employee and enter amount > 0.")
        else:
            add_settlement(employee=emp_to_pay, amount=int(pay_amt), when=pay_date, ref=ref, notes=notes_s)
            st.success("Settlement recorded.")
            st.rerun()

st.divider()

# ----------------------------
# Package ledger (confirmed only)
# ----------------------------
st.subheader("📦 Package ledger (confirmed only)")
iid_l, cust_l, ach_l = choose_package("Pick a confirmed package to view ledger", key="ledger_pick")
if iid_l:
    df_pkg = fetch_entries(start=None, end=None, itinerary_id=iid_l)
    if df_pkg.empty:
        st.info("No entries yet for this package.")
    else:
        exp_tbl = df_pkg[df_pkg["Kind"]=="expense"].copy()
        exp_tbl = exp_tbl[["Date","Employee","Category","Subheader","Amount (₹)","Notes"]].sort_values("Date")
        st.dataframe(exp_tbl, use_container_width=True, hide_index=True)
        with st.expander("Show summaries"):
            by_emp = exp_tbl.groupby("Employee", as_index=False)["Amount (₹)"].sum()
            by_cat = exp_tbl.groupby("Category", as_index=False)["Amount (₹)"].sum()
            csum1, csum2 = st.columns(2)
            with csum1:
                st.markdown("**By employee**")
                st.dataframe(by_emp, use_container_width=True, hide_index=True)
            with csum2:
                st.markdown("**By category**")
                st.dataframe(by_cat, use_container_width=True, hide_index=True)
        st.caption("Settlements shown below are global for the employee (not tied to one package).")
        st.dataframe(df_pkg[df_pkg["Kind"]=="settlement"][["Date","Employee","Amount (₹)","Ref","Notes"]],
                     use_container_width=True, hide_index=True)

st.divider()

# ----------------------------
# Team balances table (period + all-time)
# ----------------------------
st.subheader("👥 Balances")
search_emp = st.text_input("Search employee (optional)", key="emp_search")
emps = emp_filter if emp_filter else all_employees()
if search_emp.strip():
    s = search_emp.strip().lower()
    emps = [e for e in emps if s in e.lower()]

rows = []
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

df_bal = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
    "Employee","Paid (period)","Settled (period)","Balance (period)","Paid (all time)","Settled (all time)","Balance (all time)"
])
df_bal = df_bal.sort_values(["Balance (period)","Balance (all time)"], ascending=False)
st.dataframe(df_bal, use_container_width=True, hide_index=True)

st.divider()

# ----------------------------
# My entries (in selected period)
# ----------------------------
st.subheader("📜 My entries (in selected period)")
df_me = fetch_entries(start, end, employee=user, itinerary_id=None)
if df_me.empty:
    st.info("No entries in this period.")
else:
    show_cols = ["Kind","Date","Customer","ACH ID","Category","Subheader","Amount (₹)","Notes","Ref"]
    st.dataframe(df_me[show_cols].sort_values(["Date","Kind"]), use_container_width=True, hide_index=True)
