# pages/04_Splitwise_Reimbursements.py
from __future__ import annotations

from datetime import datetime, date
from typing import Optional, Dict, List, Tuple
import io
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
    defer_loads = st.toggle(
        "⚡ Defer heavy loads",
        value=True,
        help="Skip loading big tables until you press Refresh or after a Save.",
    )
    refresh_now = st.button("🔄 Refresh data now", use_container_width=True, key="btn_refresh_now")

# apply a lightweight dark theme (background + core components)
if dark_mode:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] { background: #0f1115 !important; color: #e5e7eb !important; }
        [data-testid="stHeader"] { background: #0f1115 !important; }
        .stMarkdown, .stText, .stDataFrame, .stMetric { color: #e5e7eb !important; }
        div[data-baseweb="select"] * { color: #e5e7eb !important; }
        .st-af { background: #111318 !important; } /* expander */
        .st-bc, .st-bb { background: #151822 !important; } /* cards */
        .stButton > button { background:#1f2937;color:#e5e7eb;border-radius:8px;border:1px solid #374151; }
        .stButton > button:hover { background:#374151; }
        </style>
        """,
        unsafe_allow_html=True,
    )

if refresh_now:
    st.session_state["force_refresh"] = True


def _defer_guard(msg: str) -> bool:
    """Return True if we should SKIP heavy fetches based on defer setting."""
    if defer_loads and not st.session_state.get("force_refresh", False):
        st.info(f"Deferred: {msg}

Click **🔄 Refresh data now** (sidebar) to load.")
        return True
    return False


def _clear_force_refresh():
    if st.session_state.get("force_refresh"):
        st.session_state.pop("force_refresh", None)


# =========================
# Secrets & Mongo (parity)
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
        st.error(
            "Mongo connection is not configured.

"
            "Add one of these in **Manage app → Settings → Secrets** (recommended: `mongo_uri`).",
        )
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
col_updates = db["package_updates"]
col_split = db["expense_splitwise"]  # this page writes here

# =========================
# Users & login (parity)
# =========================

def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    # fallback to repo .streamlit/secrets.toml for dev
    try:
        try:
            import tomllib  # 3.11+
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, dict) and u:
            with st.sidebar:
                st.warning(
                    "Using users from repo .streamlit/secrets.toml. "
                    "For production, set them in Manage app → Secrets.",
                )
            return u
    except Exception:
        pass
    return {}


# Admins/managers (Kuldeep included)
ADMIN_USERS = set(st.secrets.get("admin_users", ["Arpith", "Kuldeep"]))


def _login() -> Optional[str]:
    with st.sidebar:
        if st.session_state.get("user"):
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out", key="btn_logout"):
                st.session_state.pop("user", None)
                st.rerun()
    if st.session_state.get("user"):
        return st.session_state["user"]

    users_map = load_users()
    if not users_map:
        st.error(
            "Login is not configured yet.

"
            "Add to **Manage app → Secrets**:
"
            'mongo_uri = "mongodb+srv://…"

'
            "[users]
Arpith = \"1234\"
Reena = \"5678\"
Teena = \"7777\"
Kuldeep = \"8888\"
",
        )
        st.stop()

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns(2)
    with c1:
        name = st.selectbox("User", list(users_map.keys()), key="login_user")
    with c2:
        pin = st.text_input("PIN", type="password", key="login_pin")
    if st.button("Sign in", key="btn_signin"):
        if str(users_map.get(name, "")).strip() == str(pin).strip():
            st.session_state["user"] = name
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid PIN")
            st.stop()
    return None


user = _login()
if not user:
    st.stop()
is_admin = user in ADMIN_USERS

# Audit (safe import)
try:
    from tak_audit import audit_pageview

    audit_pageview(st.session_state.get("user", "Unknown"), "04_Splitwise_Reimbursements")
except Exception:
    pass

# =========================
# Helpers (parity)
# =========================

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


@st.cache_data(ttl=IST_TTL, show_spinner=False)
def all_employees() -> List[str]:
    return [e for e in sorted(load_users().keys()) if e]


def pack_label(r: pd.Series) -> str:
    ach = str(r.get("ach_id", "") or "").strip()
    name = str(r.get("client_name", "") or "").strip()
    mob = str(r.get("client_mobile", "") or "").strip()
    iid = str(r.get("itinerary_id", "") or "").strip()
    return f"{ach} | {name} | {mob} | {iid}"


def pack_options(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    for c in ["ach_id", "client_name", "client_mobile", "itinerary_id"]:
        if c not in df.columns:
            df[c] = ""
    return [pack_label(r) for _, r in df.iterrows()]


def entry_to_row(d: dict) -> dict:
    return {
        "Kind": d.get("kind", ""),
        "Date": norm_date(d.get("date")),
        "Employee": d.get("payer") or d.get("employee") or "",
        "Collector": d.get("collector", ""),
        "Customer": d.get("customer_name", ""),
        "ACH ID": d.get("ach_id", ""),
        "Category": d.get("category", ""),
        "Subheader": d.get("subheader", ""),
        "Amount (₹)": _to_int(d.get("amount", 0)),
        "Mode": d.get("mode", ""),
        "UTR": d.get("utr", ""),
        "Ref": d.get("ref", ""),
        "Notes": d.get("notes", ""),
        "itinerary_id": d.get("itinerary_id", ""),
        "_id": str(d.get("_id", "")),
        "created_by": d.get("created_by", ""),
        "created_at": d.get("created_at"),
    }


# =========================
# Data fetchers
# =========================
@st.cache_data(ttl=IST_TTL, show_spinner=False)
def _confirmed_updates_map() -> Dict[str, dict]:
    """
    Map of itinerary_id -> {'booking_date': datetime or None}
    for confirmed packages.
    """
    cur = col_updates.find(
        {"status": "confirmed"},
        {"_id": 0, "itinerary_id": 1, "booking_date": 1},
    )
    out = {}
    for d in cur:
        iid = str(d.get("itinerary_id"))
        bd = pd.to_datetime(d.get("booking_date")) if d.get("booking_date") else None
        out[iid] = {"booking_date": (bd.to_pydatetime() if isinstance(bd, pd.Timestamp) else bd)}
    return out


@st.cache_data(ttl=IST_TTL, show_spinner=False)
def confirmed_itineraries_df_unique_clients() -> pd.DataFrame:
    """
    Only itineraries that are CONFIRMED, deduped to show a single 'final revision' per client.
    Final revision chosen by:
      1) latest booking_date (if present)
      2) fallback to latest ObjectId generation_time (creation)
    Client identity key: prefer client_mobile, else ACH+name combo.
    """
    upd_map = _confirmed_updates_map()
    ids = list(upd_map.keys())
    if not ids:
        return pd.DataFrame(
            columns=[
                "itinerary_id",
                "ach_id",
                "client_name",
                "client_mobile",
                "start_date",
                "end_date",
                "final_route",
            ]
        )

    # Fetch itineraries by _id or by legacy 'itinerary_id'
    obj_ids = [ObjectId(i) for i in ids if len(i) == 24]
    docs_by_oid = []
    if obj_ids:
        docs_by_oid = list(
            col_itineraries.find(
                {"_id": {"$in": obj_ids}},
                {
                    "_id": 1,
                    "ach_id": 1,
                    "client_name": 1,
                    "client_mobile": 1,
                    "start_date": 1,
                    "end_date": 1,
                    "final_route": 1,
                },
            )
        )

    str_ids = [i for i in ids if len(i) != 24]
    docs_by_str = []
    if str_ids:
        docs_by_str = list(
            col_itineraries.find(
                {"itinerary_id": {"$in": str_ids}},
                {
                    "_id": 1,
                    "itinerary_id": 1,
                    "ach_id": 1,
                    "client_name": 1,
                    "client_mobile": 1,
                    "start_date": 1,
                    "end_date": 1,
                    "final_route": 1,
                },
            )
        )

    rows: List[dict] = []
    # Normalize OID docs
    for r in docs_by_oid:
        iid = str(r["_id"])
        created_ts = ObjectId(iid).generation_time if ObjectId.is_valid(iid) else None
        rows.append(
            {
                "itinerary_id": iid,
                "ach_id": r.get("ach_id", ""),
                "client_name": r.get("client_name", ""),
                "client_mobile": r.get("client_mobile", ""),
                "start_date": norm_date(r.get("start_date")),
                "end_date": norm_date(r.get("end_date")),
                "final_route": r.get("final_route", ""),
                "_created": created_ts,
                "_booking": upd_map.get(iid, {}).get("booking_date"),
            }
        )
    # Normalize string-id docs
    for r in docs_by_str:
        iid = str(r.get("itinerary_id") or r.get("_id"))
        created_ts = ObjectId(iid).generation_time if ObjectId.is_valid(iid) else None
        rows.append(
            {
                "itinerary_id": iid,
                "ach_id": r.get("ach_id", ""),
                "client_name": r.get("client_name", ""),
                "client_mobile": r.get("client_mobile", ""),
                "start_date": norm_date(r.get("start_date")),
                "end_date": norm_date(r.get("end_date")),
                "final_route": r.get("final_route", ""),
                "_created": created_ts,
                "_booking": upd_map.get(iid, {}).get("booking_date"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "itinerary_id",
                "ach_id",
                "client_name",
                "client_mobile",
                "start_date",
                "end_date",
                "final_route",
            ]
        )

    df = pd.DataFrame(rows)

    # Build client identity key (prefer mobile)
    def _ck(r):
        mob = str(r.get("client_mobile") or "").strip()
        if mob:
            return f"M:{mob}"
        ach = str(r.get("ach_id") or "").strip()
        nam = str(r.get("client_name") or "").strip()
        return f"A:{ach}|N:{nam}"

    df["_client_key"] = df.apply(_ck, axis=1)

    # Rank: first by booking_date (desc), then by created (desc)
    df["_booking"] = pd.to_datetime(df["_booking"])  # may be NaT
    df["_created"] = pd.to_datetime(df["_created"])  # may be NaT
    df.sort_values(by=["_client_key", "_booking", "_created"], ascending=[True, False, False], inplace=True)

    # Keep only the latest for each client
    latest = df.groupby("_client_key", as_index=False).first()

    # Final projected columns
    out = latest[["itinerary_id", "ach_id", "client_name", "client_mobile", "start_date", "end_date", "final_route"]].copy()
    return out.reset_index(drop=True)


def fetch_entries(
    start: Optional[date] = None,
    end: Optional[date] = None,
    employee: Optional[str] = None,
    itinerary_id: Optional[str] = None,
) -> pd.DataFrame:
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
    cur = col_split.find(
        q,
        projection={
            "_id": 1,
            "kind": 1,
            "date": 1,
            "payer": 1,
            "employee": 1,
            "collector": 1,
            "mode": 1,
            "utr": 1,
            "customer_name": 1,
            "ach_id": 1,
            "category": 1,
            "subheader": 1,
            "amount": 1,
            "notes": 1,
            "ref": 1,
            "itinerary_id": 1,
            "created_by": 1,
            "created_at": 1,
        },
    ).sort("date", 1)
    rows = [entry_to_row(d) for d in cur]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(entry_to_row({}).keys()))


# NEW: bulk fetch supporting multi-employee filter (admin)
def fetch_entries_any(
    start: Optional[date] = None,
    end: Optional[date] = None,
    employees: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
) -> pd.DataFrame:
    q: Dict = {}
    if start and end:
        q["date"] = {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end, datetime.max.time()),
        }
    if employees:
        q["$or"] = [{"payer": {"$in": employees}}, {"employee": {"$in": employees}}, {"collector": {"$in": employees}}]
    if kinds:
        q["kind"] = {"$in": kinds}
    cur = col_split.find(
        q,
        projection={
            "_id": 1,
            "kind": 1,
            "date": 1,
            "payer": 1,
            "employee": 1,
            "collector": 1,
            "mode": 1,
            "utr": 1,
            "customer_name": 1,
            "ach_id": 1,
            "category": 1,
            "subheader": 1,
            "amount": 1,
            "notes": 1,
            "ref": 1,
            "itinerary_id": 1,
            "created_by": 1,
            "created_at": 1,
        },
    ).sort("date", 1)
    rows = [entry_to_row(d) for d in cur]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(entry_to_row({}).keys()))
    return df


def totals_for_employee(
    emp: str, start: Optional[date] = None, end: Optional[date] = None
) -> Tuple[int, int, int]:
    q_exp: Dict = {"kind": "expense", "payer": emp}
    q_pay: Dict = {"kind": "settlement", "employee": emp}
    if start and end:
        span = {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end, datetime.max.time()),
        }
        q_exp["date"] = span
        q_pay["date"] = span
    exp_sum = sum(_to_int(d.get("amount", 0)) for d in col_split.find(q_exp, {"amount": 1}))
    pay_sum = sum(_to_int(d.get("amount", 0)) for d in col_split.find(q_pay, {"amount": 1}))
    return exp_sum, pay_sum, (exp_sum - pay_sum)


# =========================
# DB ops
# =========================

def add_expense(
    *,
    payer: str,
    itinerary_id: str,
    customer_name: str,
    ach_id: str,
    category: str,
    subheader: str,
    amount: int,
    when: date,
    notes: str,
) -> None:
    col_split.insert_one(
        {
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
        }
    )


def add_settlement(*, employee: str, amount: int, when: date, ref: str, notes: str) -> None:
    col_split.insert_one(
        {
            "kind": "settlement",
            "created_at": datetime.utcnow(),
            "created_by": user,
            "date": datetime.combine(when, datetime.min.time()),
            "employee": employee,
            "amount": int(amount),
            "ref": ref or "",
            "notes": notes or "",
        }
    )


# NEW: cash received from clients or others

def add_cash_received(
    *,
    collector: str,
    itinerary_id: str,
    customer_name: str,
    ach_id: str,
    amount: int,
    when: date,
    mode: str,
    utr: str,
    ref: str,
    notes: str,
) -> None:
    col_split.insert_one(
        {
            "kind": "cash_received",
            "created_at": datetime.utcnow(),
            "created_by": user,
            "date": datetime.combine(when, datetime.min.time()),
            "collector": collector,  # who collected the cash
            "mode": mode or "Cash",
            "utr": utr or "",
            "ref": ref or "",
            "itinerary_id": itinerary_id,
            "customer_name": customer_name,
            "ach_id": ach_id,
            "category": "Cash Received",
            "subheader": "",
            "amount": int(amount),
            "notes": notes or "",
        }
    )


# =========================
# Filters / controls (deferred)
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
        start = st.date_input("From", value=month_start, key="flt_from")
    with f2:
        end = st.date_input("To", value=today, key="flt_to")
        if end < start:
            end = start
    with f3:
        emp_options = all_employees()
        default_emp = [user] if user in emp_options else []
        emp_filter = st.multiselect("Employees", options=emp_options, default=default_emp, key="flt_emps")
    with f4:
        search_txt = st.text_input(
            "Search confirmed client/mobile/ACH",
            placeholder="Type to filter package list…",
            key="flt_search",
        )


def choose_package(label="Select confirmed package", key="pkg_pick") -> Tuple[Optional[str], str, str]:
    options = df_confirmed_unique.copy()
    if options is None or options.empty:
        st.info("No confirmed packages found.")
        return None, "", ""

    if search_txt.strip():
        s = search_txt.strip().lower()
        for c in ["client_name", "client_mobile", "ach_id"]:
            if c not in options.columns:
                options[c] = ""
        options = options[
            options["client_name"].astype(str).str.lower().str.contains(s, na=False)
            | options["client_mobile"].astype(str).str.lower().str.contains(s, na=False)
            | options["ach_id"].astype(str).str.lower().str.contains(s, na=False)
        ]

    opt_labels = pack_options(options)
    if not opt_labels:
        st.info("No matching confirmed packages.")
        return None, "", ""

    sel = st.selectbox(label, opt_labels, index=0, key=key)
    if not sel:
        return None, "", ""

    rid = sel.split(" | ")[-1].strip()
    row = options[options["itinerary_id"].astype(str).str.strip() == rid]
    if row.empty:
        return rid, "", ""
    row = row.iloc[0]
    return rid, str(row.get("client_name", "") or ""), str(row.get("ach_id", "") or "")


# =========================
# KPIs for current user (deferred)
# =========================
if _defer_guard("My balances KPIs"):
    pass
else:
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
mode = st.radio(
    "Expense type",
    ["Linked to confirmed package", "Other expense (no package)"]
    ,
    horizontal=True,
    key="rad_expense_type",
)

with st.form("add_expense_form", clear_on_submit=False):
    if mode == "Linked to confirmed package":
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            iid, cust_name, ach = choose_package("Confirmed package / customer", key="add_pkg")
        with c2:
            when = st.date_input("Date", value=date.today(), key="exp_date_linked")
        with c3:
            amount = st.number_input("Amount (₹)", min_value=0, step=100, value=0, key="exp_amt_linked")
        c4, c5 = st.columns([1, 1])
        with c4:
            category = st.selectbox("Category", CATEGORIES, index=0, key="exp_cat_linked")
        with c5:
            subheader = st.text_input(
                "Subheader (detail)",
                placeholder="e.g., Airport transfer / Room upgrade",
                key="exp_sub_linked",
            )
        notes = st.text_area(
            "Notes (optional)",
            placeholder="Anything helpful for accounting",
            key="exp_notes_linked",
        )
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            cust_name = st.text_input(
                "Beneficiary / Customer (free text)",
                placeholder="e.g., Office supplies / Misc",
                key="oth_name",
            )
        with c2:
            when = st.date_input("Date", value=date.today(), key="oth_date")
        with c3:
            amount = st.number_input("Amount (₹)", min_value=0, step=100, value=0, key="oth_amt")
        c4, c5 = st.columns([1, 1])
        with c4:
            category = st.selectbox("Category", CATEGORIES, index=5, key="oth_cat")  # default Other
        with c5:
            subheader = st.text_input(
                "Subheader (detail)",
                placeholder="e.g., Courier / Printouts",
                key="oth_sub",
            )
        notes = st.text_area(
            "Notes (optional)",
            placeholder="Anything helpful for accounting",
            key="oth_notes",
        )
        iid, ach = "", ""  # not tied to any itinerary

    submitted = st.form_submit_button("Save expense", key="btn_save_expense")

if submitted:
    if amount <= 0:
        st.error("Amount must be > 0.")
    elif mode == "Linked to confirmed package" and not iid:
        st.error("Please choose a confirmed package/customer.")
    else:
        add_expense(
            payer=user,
            itinerary_id=str(iid or ""),
            customer_name=cust_name,
            ach_id=str(ach or ""),
            category=category,
            subheader=subheader,
            amount=int(amount),
            when=when,
            notes=notes,
        )
        st.success("Expense added.")
        # Force reload after save only
        st.session_state["force_refresh"] = True
        st.rerun()

st.divider()

# =========================
# NEW: Cash received recorder (any user or admin)
# =========================
st.subheader("💰 Record cash received from client / other")
rec_mode = st.radio(
    "Receipt type",
    ["Linked to confirmed package", "Other (no package)"],
    horizontal=True,
    key="rad_receipt_type",
)

with st.form("add_cash_form", clear_on_submit=False):
    if rec_mode == "Linked to confirmed package":
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            iid_r, cust_name_r, ach_r = choose_package("Confirmed package / customer", key="recv_pkg")
        with c2:
            when_r = st.date_input("Date", value=date.today(), key="recv_date_linked")
        with c3:
            amount_r = st.number_input("Amount (₹)", min_value=0, step=500, value=0, key="recv_amt_linked")
        c4, c5, c6 = st.columns([1, 1, 1.2])
        with c4:
            mode_r = st.selectbox("Mode", ["Cash", "Bank"], index=0, key="recv_mode_linked")
        with c5:
            utr_r = st.text_input("UTR (if Bank)", key="recv_utr_linked")
        with c6:
            ref_r = st.text_input("Ref / Receipt #", key="recv_ref_linked")
        notes_r = st.text_area("Notes (optional)", key="recv_notes_linked")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            cust_name_r = st.text_input(
                "Payer name (free text)", placeholder="e.g., Walk-in / Vendor / Specific person", key="recv_name_other"
            )
        with c2:
            when_r = st.date_input("Date", value=date.today(), key="recv_date_other")
        with c3:
            amount_r = st.number_input("Amount (₹)", min_value=0, step=500, value=0, key="recv_amt_other")
        c4, c5, c6 = st.columns([1, 1, 1.2])
        with c4:
            mode_r = st.selectbox("Mode", ["Cash", "Bank"], index=0, key="recv_mode_other")
        with c5:
            utr_r = st.text_input("UTR (if Bank)", key="recv_utr_other")
        with c6:
            ref_r = st.text_input("Ref / Receipt #", key="recv_ref_other")
        notes_r = st.text_area("Notes (optional)", key="recv_notes_other")
        iid_r, ach_r = "", ""

    submit_recv = st.form_submit_button("Save receipt", key="btn_save_receipt")

if submit_recv:
    if amount_r <= 0:
        st.error("Amount must be > 0.")
    elif rec_mode == "Linked to confirmed package" and not iid_r:
        st.error("Please choose a confirmed package/customer.")
    else:
        add_cash_received(
            collector=user,
            itinerary_id=str(iid_r or ""),
            customer_name=cust_name_r,
            ach_id=str(ach_r or ""),
            amount=int(amount_r),
            when=when_r,
            mode=mode_r,
            utr=utr_r,
            ref=ref_r,
            notes=notes_r,
        )
        st.success("Receipt recorded. (Category: Cash Received)")
        st.session_state["force_refresh"] = True
        st.rerun()

st.divider()

# =========================
# Package ledger (confirmed only) (deferred)
# =========================
st.subheader("📦 Package ledger (confirmed only)")
if _defer_guard("Package ledger"):
    pass
else:
    iid_l, cust_l, ach_l = choose_package("Pick a confirmed package to view ledger", key="ledger_pick")
    if iid_l:
        df_pkg = fetch_entries(start=None, end=None, itinerary_id=iid_l)
        if df_pkg.empty:
            st.info("No entries yet for this package.")
        else:
            # Expenses
            exp_tbl = df_pkg[df_pkg["Kind"] == "expense"].copy()
            exp_tbl = exp_tbl[
                ["Date", "Employee", "Category", "Subheader", "Amount (₹)", "Notes"]
            ].sort_values("Date")
            st.markdown("**Expenses**")
            st.dataframe(exp_tbl, use_container_width=True, hide_index=True)

            # Cash received (deduct from total)
            recv_tbl = df_pkg[df_pkg["Kind"] == "cash_received"].copy()
            if not recv_tbl.empty:
                recv_tbl = recv_tbl[
                    ["Date", "Collector", "Mode", "UTR", "Amount (₹)", "Ref", "Notes"]
                ].sort_values("Date")
                st.markdown("**Cash received (deducted from total)**")
                st.dataframe(recv_tbl, use_container_width=True, hide_index=True)

            # Summaries
            with st.expander("Show summaries"):
                if not exp_tbl.empty:
                    by_emp = exp_tbl.groupby("Employee", as_index=False)["Amount (₹)"].sum()
                    by_cat = exp_tbl.groupby("Category", as_index=False)["Amount (₹)"].sum()
                    csum1, csum2 = st.columns(2)
                    with csum1:
                        st.markdown("**By employee**")
                        st.dataframe(by_emp, use_container_width=True, hide_index=True)
                    with csum2:
                        st.markdown("**By category**")
                        st.dataframe(by_cat, use_container_width=True, hide_index=True)

                total_exp = int(exp_tbl["Amount (₹)"].sum()) if not exp_tbl.empty else 0
                total_recv = int(
                    df_pkg.loc[df_pkg["Kind"] == "cash_received", "Amount (₹)"].sum()
                )
                st.info(
                    f"**Totals** – Expenses: ₹ {total_exp:,} | Cash received: ₹ {total_recv:,} | **Net**: ₹ {total_exp - total_recv:,}"
                )

            st.caption(
                "Settlements shown below are global for the employee (not tied to one package)."
            )
            st.dataframe(
                df_pkg[df_pkg["Kind"] == "settlement"][
                    ["Date", "Employee", "Amount (₹)", "Ref", "Notes"]
                ],
                use_container_width=True,
                hide_index=True,
            )

st.divider()

# =========================
# Team balances table (period + all-time) (deferred)
# =========================
st.subheader("👥 Balances")
if _defer_guard("Team balances"):
    pass
else:
    search_emp = st.text_input("Search employee (optional)", key="emp_search")
    emps = (emp_filter if emp_filter else all_employees()) or []
    if search_emp and search_emp.strip():
        s = search_emp.strip().lower()
        emps = [e for e in emps if s in e.lower()]

    rows = []
    for e in emps:
        em = totals_for_employee(e, start, end)
        ea = totals_for_employee(e, None, None)
        rows.append(
            {
                "Employee": e,
                "Paid (period)": em[0],
                "Settled (period)": em[1],
                "Balance (period)": em[2],
                "Paid (all time)": ea[0],
                "Settled (all time)": ea[1],
                "Balance (all time)": ea[2],
            }
        )

    df_bal = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=
            [
                "Employee",
                "Paid (period)",
                "Settled (period)",
                "Balance (period)",
                "Paid (all time)",
                "Settled (all time)",
                "Balance (all time)",
            ]
        )
    )
    df_bal = df_bal.sort_values(["Balance (period)", "Balance (all time)"], ascending=False)
    st.dataframe(df_bal, use_container_width=True, hide_index=True)

st.divider()

# =========================
# My entries (in selected period) (deferred)
# =========================
st.subheader("📜 My entries (in selected period)")
if _defer_guard("My entries table"):
    pass
else:
    df_me = fetch_entries(start, end, employee=user, itinerary_id=None)
    if df_me.empty:
        st.info("No entries in this period.")
    else:
        show_cols = [
            "Kind",
            "Date",
            "Customer",
            "ACH ID",
            "Category",
            "Subheader",
            "Amount (₹)",
            "Mode",
            "UTR",
            "Notes",
            "Ref",
        ]
        st.dataframe(
            df_me[show_cols].sort_values(["Date", "Kind"]),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# =========================
# Admin – Detailed table view + Excel export (range or month)
# =========================
if is_admin:
    st.subheader("📑 Admin – Detailed expense/receipts viewer & export")

    # Quick picker: Date range or Calendar month
    mode_pick = st.radio(
        "View by",
        ["Date range", "Calendar month"],
        horizontal=True,
        key="rad_admin_view_by",
    )

    def month_bounds(yyyy_mm: str) -> Tuple[date, date]:
        y, m = map(int, yyyy_mm.split("-"))
        first = date(y, m, 1)
        if m == 12:
            last = date(y + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            last = date(y, m + 1, 1) - pd.Timedelta(days=1)
        return first, date.fromordinal(last.toordinal())

    c1, c2, c3 = st.columns([1.1, 1.1, 2])
    if mode_pick == "Date range":
        with c1:
            dr_start = st.date_input("From", value=start, key="admin_from")
        with c2:
            dr_end = st.date_input("To", value=end, key="admin_to")
            if dr_end < dr_start:
                dr_end = dr_start
    else:
        today_ = date.today()
        yyyy_mm_default = f"{today_.year}-{today_.month:02d}"
        with c1:
            yyyy_mm = st.text_input(
                "Month (YYYY-MM)", value=yyyy_mm_default, help="Example: 2025-09", key="admin_month"
            )
        try:
            dr_start, dr_end = month_bounds(yyyy_mm)
        except Exception:
            st.warning("Invalid month format; falling back to current selections.")
            dr_start, dr_end = start, end

    with c3:
        reps = st.multiselect(
            "Representatives (optional)", options=all_employees(), default=[], key="admin_reps"
        )

    # Data load (respect defer)
    if _defer_guard("Admin detailed export tables"):
        pass
    else:
        # Pull expenses / settlements / receipts for the selected span & reps
        df_any = fetch_entries_any(start=dr_start, end=dr_end, employees=(reps or None))
        if df_any.empty:
            st.info("No entries for the selected filters.")
        else:
            # Split tables
            df_exp = df_any[df_any["Kind"] == "expense"].copy()
            df_set = df_any[df_any["Kind"] == "settlement"].copy()
            df_recv = df_any[df_any["Kind"] == "cash_received"].copy()

            # Reorder columns for readability
            exp_cols = [
                "Date",
                "Employee",
                "Customer",
                "ACH ID",
                "itinerary_id",
                "Category",
                "Subheader",
                "Amount (₹)",
                "Notes",
            ]
            set_cols = ["Date", "Employee", "Amount (₹)", "Ref", "Notes"]
            recv_cols = [
                "Date",
                "Collector",
                "Customer",
                "ACH ID",
                "itinerary_id",
                "Mode",
                "UTR",
                "Amount (₹)",
                "Ref",
                "Notes",
            ]

            if not df_exp.empty:
                st.markdown("**Expenses (detailed)**")
                st.dataframe(df_exp[exp_cols].sort_values("Date"), use_container_width=True, hide_index=True)
            if not df_recv.empty:
                st.markdown("**Cash received (detailed)**")
                st.dataframe(df_recv[recv_cols].sort_values("Date"), use_container_width=True, hide_index=True)
            if not df_set.empty:
                st.markdown("**Settlements**")
                st.dataframe(df_set[set_cols].sort_values("Date"), use_container_width=True, hide_index=True)

            # Summaries – representative-wise & category-wise and collector-wise
            csum1, csum2 = st.columns(2)
            if not df_exp.empty:
                with csum1:
                    st.markdown("**Summary – by Representative (Expenses)**")
                    by_rep = df_exp.groupby("Employee", as_index=False)["Amount (₹)"].sum()
                    st.dataframe(by_rep, use_container_width=True, hide_index=True)
                with csum2:
                    st.markdown("**Summary – by Category (Expenses)**")
                    by_cat = df_exp.groupby("Category", as_index=False)["Amount (₹)"].sum()
                    st.dataframe(by_cat, use_container_width=True, hide_index=True)
            else:
                by_rep = pd.DataFrame(columns=["Employee", "Amount (₹)"])
                by_cat = pd.DataFrame(columns=["Category", "Amount (₹)"])

            csum3, csum4 = st.columns(2)
            if not df_recv.empty:
                with csum3:
                    st.markdown("**Summary – by Collector (Cash Received)**")
                    by_coll = df_recv.groupby("Collector", as_index=False)["Amount (₹)"].sum()
                    st.dataframe(by_coll, use_container_width=True, hide_index=True)
                with csum4:
                    st.markdown("**Summary – by Mode (Cash Received)**")
                    by_mode = df_recv.groupby("Mode", as_index=False)["Amount (₹)"].sum()
                    st.dataframe(by_mode, use_container_width=True, hide_index=True)
            else:
                by_coll = pd.DataFrame(columns=["Collector", "Amount (₹)"])
                by_mode = pd.DataFrame(columns=["Mode", "Amount (₹)"])

            # Build Excel file in-memory
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                # Sorted tables
                df_exp_sorted = df_exp[exp_cols].sort_values("Date") if not df_exp.empty else pd.DataFrame(columns=exp_cols)
                df_set_sorted = df_set[set_cols].sort_values("Date") if not df_set.empty else pd.DataFrame(columns=set_cols)
                df_recv_sorted = (
                    df_recv[recv_cols].sort_values("Date") if not df_recv.empty else pd.DataFrame(columns=recv_cols)
                )

                df_exp_sorted.to_excel(xw, index=False, sheet_name="Expenses")
                df_recv_sorted.to_excel(xw, index=False, sheet_name="Cash_Received")
                df_set_sorted.to_excel(xw, index=False, sheet_name="Settlements")
                by_rep.to_excel(xw, index=False, sheet_name="Summary_Rep")
                by_cat.to_excel(xw, index=False, sheet_name="Summary_Category")
                by_coll.to_excel(xw, index=False, sheet_name="Summary_Collector")
                by_mode.to_excel(xw, index=False, sheet_name="Summary_Mode")

                # Cover sheet with metadata + net calc
                total_exp = int(df_exp_sorted["Amount (₹)"].sum()) if not df_exp_sorted.empty else 0
                total_recv = int(df_recv_sorted["Amount (₹)"].sum()) if not df_recv_sorted.empty else 0
                meta = pd.DataFrame(
                    {
                        "Field": [
                            "Generated By",
                            "From",
                            "To",
                            "Representatives",
                            "Total Expenses",
                            "Total Cash Received",
                            "Net (Expenses - Cash)",
                        ],
                        "Value": [
                            user,
                            dr_start.isoformat(),
                            dr_end.isoformat(),
                            ", ".join(reps) if reps else "All",
                            f"₹ {total_exp:,}",
                            f"₹ {total_recv:,}",
                            f"₹ {total_exp - total_recv:,}",
                        ],
                    }
                )
                meta.to_excel(xw, index=False, sheet_name="Report_Meta")

            st.download_button(
                label="⬇️ Download Excel",
                data=buf.getvalue(),
                file_name=f"Splitwise_Report_{dr_start.isoformat()}_to_{dr_end.isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="btn_dl_excel",
            )

# Clear force flag at end of successful load cycle
_clear_force_refresh()
