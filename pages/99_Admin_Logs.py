# pages/99_Admin_Logs.py
from __future__ import annotations

from datetime import datetime, timedelta, time as dtime, date
from zoneinfo import ZoneInfo
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(page_title="Admin – User Logs", layout="wide")
st.title("🛡️ Admin – User Activity Logs")

# =========================
# Admin gate (consistent with other pages)
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

from tak_audit import now_ist, read_logs, audit_pageview
audit_pageview(st.session_state.get("user", "Unknown"), page="99_Admin_Logs")

IST = ZoneInfo("Asia/Kolkata")

# =========================
# Helpers
# =========================
def _daterange_to_ist_dt(d1: date, d2: date) -> Tuple[datetime, datetime]:
    start_dt = datetime.combine(d1, dtime.min).astimezone(IST)
    end_dt   = datetime.combine(d2, dtime.max).astimezone(IST)
    # attach tzinfo if naive (fallback)
    if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=IST)
    if end_dt.tzinfo   is None: end_dt   = end_dt.replace(tzinfo=IST)
    return start_dt, end_dt

def _coerce_dt(x) -> Optional[datetime]:
    try:
        return pd.to_datetime(x)
    except Exception:
        return None

# =========================
# Filters (with quick ranges)
# =========================
with st.container():
    today = now_ist().date()
    pr1, pr2, pr3, pr4 = st.columns([1.2, 1.2, 1.8, 1.2])

    with pr1:
        preset = st.selectbox(
            "Quick range",
            ["Last 24h", "Last 7 days", "This month", "Last month", "Custom"],
            index=1
        )

    if preset == "Last 24h":
        start_d, end_d = today - timedelta(days=1), today
    elif preset == "Last 7 days":
        start_d, end_d = today - timedelta(days=7), today
    elif preset == "This month":
        start_d, end_d = today.replace(day=1), today
    elif preset == "Last month":
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        start_d, end_d = last_prev.replace(day=1), last_prev
    else:
        start_d, end_d = today - timedelta(days=7), today

    with pr2:
        start_d = st.date_input("From (IST)", value=start_d, key="from_ist")
    with pr3:
        end_d = st.date_input("To (IST, inclusive)", value=end_d, key="to_ist")
        if end_d < start_d:
            end_d = start_d
    with pr4:
        limit = st.number_input("Max rows", min_value=100, max_value=50000, value=5000, step=500)

# Secondary filters
f1, f2, f3 = st.columns([1.2, 1.2, 2.2])
with f1:
    action_f = st.multiselect("Action", ["login", "page_view", "logout"], default=[])
with f2:
    user_f = st.text_input("Filter by user (optional)")
with f3:
    page_search = st.text_input("Search in page / extra (optional)")

start_dt_ist, end_dt_ist = _daterange_to_ist_dt(start_d, end_d)

# =========================
# Cached fetch
# =========================
@st.cache_data(ttl=90, show_spinner=False)
def load_logs(start_dt_ist: datetime, end_dt_ist: datetime,
              user: Optional[str], actions: Tuple[str, ...], limit: int) -> pd.DataFrame:
    # tak_audit.read_logs expects IST datetimes
    df = read_logs(start_dt_ist, end_dt_ist,
                   user=(user or None),
                   action=(None if not actions else list(actions)),
                   limit=int(limit))
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

df = load_logs(start_dt_ist, end_dt_ist, user_f.strip() or None, tuple(action_f), int(limit)).copy()

# Optional free-text search across 'page' and 'extra'
if (page_search or "").strip() and not df.empty:
    s = page_search.strip().lower()
    for col in ("page", "extra"):
        if col not in df.columns:
            df[col] = ""
    df = df[
        df["page"].astype(str).str.lower().str.contains(s) |
        df["extra"].astype(str).str.lower().str.contains(s)
    ]

# Normalize columns for safety
for c in ("ts_ist", "ts_ist_str", "action", "user", "page", "session_id", "extra"):
    if c not in df.columns:
        df[c] = ""

# Coerce timestamp
if not df.empty:
    df["ts_ist"] = df["ts_ist"].apply(_coerce_dt)
    # maintain readable string (existing ts_ist_str from tak_audit is already IST)
    if "ts_ist_str" not in df or df["ts_ist_str"].isna().all():
        df["ts_ist_str"] = df["ts_ist"].dt.tz_localize(IST, nonexistent="NaT", ambiguous="NaT", errors="ignore").dt.strftime("%Y-%m-%d %H:%M %Z")

# =========================
# KPIs
# =========================
c1, c2, c3, c4 = st.columns(4)
total_rows = 0 if df.empty else len(df)
logins = 0 if df.empty else int((df["action"] == "login").sum())
pageviews = 0 if df.empty else int((df["action"] == "page_view").sum())
logouts = 0 if df.empty else int((df["action"] == "logout").sum())

c1.metric("Total rows", total_rows)
c2.metric("Logins", logins)
c3.metric("Page views", pageviews)
c4.metric("Logouts", logouts)

st.caption(f"Range: **{start_d} → {end_d} (IST)**  •  Rows limited to **{int(limit)}**.")

st.divider()

# =========================
# Summaries
# =========================
if df.empty:
    st.info("No logs for the selected filters.")
    st.stop()

# Last login per user
if "action" in df.columns and "user" in df.columns:
    last_login = (
        df[df["action"] == "login"]
        .groupby("user")["ts_ist_str"]
        .max()
        .reset_index()
        .rename(columns={"ts_ist_str": "Last login (IST)", "user": "User"})
        .sort_values("Last login (IST)", ascending=False)
    )
else:
    last_login = pd.DataFrame(columns=["User", "Last login (IST)"])

s1, s2 = st.columns([1.2, 1.2])
with s1:
    st.subheader("Last login per user")
    st.dataframe(last_login, use_container_width=True, hide_index=True)

with s2:
    st.subheader("Events by page (top 20)")
    by_page = (
        df.groupby("page", dropna=False)["action"].count()
        .rename("Events")
        .reset_index()
        .sort_values("Events", ascending=False)
        .head(20)
    )
    st.dataframe(by_page, use_container_width=True, hide_index=True)

# Events by user (all actions)
st.subheader("Events by user")
by_user = (
    df.groupby("user", dropna=False)["action"].count()
    .rename("Events")
    .reset_index()
    .sort_values("Events", ascending=False)
)
st.dataframe(by_user, use_container_width=True, hide_index=True)

st.divider()

# =========================
# All events (ordered)
# =========================
st.subheader("All events")
show_cols = ["ts_ist_str", "action", "user", "page", "session_id", "extra"]
for c in show_cols:
    if c not in df.columns:
        df[c] = ""
ordered = df.sort_values("ts_ist", ascending=False, na_position="last")
st.dataframe(ordered[show_cols], use_container_width=True, hide_index=True)

# Export
csv = ordered[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, f"user_logs_{start_d}_to_{end_d}.csv", "text/csv", use_container_width=True)

# =========================
# Session explorer (optional)
# =========================
with st.expander("🔍 Explore by session"):
    if "session_id" in df.columns and not df["session_id"].isna().all():
        sess_ids = ordered["session_id"].dropna().astype(str).unique().tolist()
        sel = st.selectbox("Open a session", [""] + sess_ids)
        if sel:
            sess = ordered[ordered["session_id"].astype(str) == sel][show_cols].copy()
            st.dataframe(sess, use_container_width=True, hide_index=True)
    else:
        st.caption("No session IDs in current result set.")
