# pages/99_Admin_Logs.py
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from tak_audit import now_ist, read_logs

st.set_page_config(page_title="Admin – User Logs", layout="wide")
st.title("🛡️ Admin – User Activity Logs")

# simple admin gate (reuse your existing pattern if you prefer)
ADMIN_PASS_DEFAULT = "Arpith&92"
ADMIN_PASS = str(st.secrets.get("admin_pass", ADMIN_PASS_DEFAULT))
with st.sidebar:
    st.markdown("### Admin access")
    p = st.text_input("Enter admin password", type="password")
if (p or "").strip() != ADMIN_PASS.strip():
    st.stop()

IST = ZoneInfo("Asia/Kolkata")
today = now_ist().date()
start_d = st.date_input("From (IST)", value=today - timedelta(days=7))
end_d   = st.date_input("To (IST - inclusive)", value=today)
user_f  = st.text_input("Filter by user (optional)")
action_f= st.selectbox("Action", ["", "login", "page_view"], index=0)
limit   = st.number_input("Max rows", min_value=100, max_value=20000, value=5000, step=100)

start_dt = datetime.combine(start_d, datetime.min.time(), tzinfo=IST)
end_dt   = datetime.combine(end_d,   datetime.max.time(), tzinfo=IST)

df = read_logs(start_dt, end_dt, user=user_f or None, action=(action_f or None), limit=int(limit))

c1, c2, c3 = st.columns(3)
c1.metric("Total rows", len(df))
c2.metric("Logins", int((df["action"]=="login").sum()) if not df.empty else 0)
c3.metric("Page views", int((df["action"]=="page_view").sum()) if not df.empty else 0)

st.divider()

if df.empty:
    st.info("No logs for the selected range.")
    st.stop()

# Quick last login per user
last_login = (
    df[df["action"]=="login"]
    .groupby("user")["ts_ist_str"]
    .max()
    .reset_index()
    .rename(columns={"ts_ist_str":"Last login (IST)"})
)
st.subheader("Last login per user")
st.dataframe(last_login, use_container_width=True, hide_index=True)

st.subheader("All events")
# nice order
show_cols = ["ts_ist_str","action","user","page","session_id","extra"]
for c in show_cols:
    if c not in df.columns: df[c] = ""
st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

# Export
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "user_logs.csv", "text/csv")
