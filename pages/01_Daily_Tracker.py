import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import date

st.set_page_config(page_title="Daily Tracker", layout="wide")
st.title("🧪 Daily Tracker – Smoke Test")

# 1) pymongo import check is implicit by import above
st.success("✅ pymongo import OK")

# 2) secrets + db connect with guard
try:
    MONGO_URI = st.secrets["mongo_uri"]
except Exception as e:
    st.error(f"❌ No mongo_uri in Streamlit Cloud secrets: {e}")
    st.stop()

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    db = client["TAK_DB"]
    updates_col = db["daily_updates"]
    # ping server
    client.admin.command("ping")
    st.success("✅ Connected to MongoDB")
except Exception as e:
    st.error(f"❌ MongoDB connection failed: {e}")
    st.stop()

# 3) tiny form to insert one record
st.subheader("Quick Insert")
with st.form("quick_form"):
    d = st.date_input("Date", date.today())
    client_name = st.text_input("Client Name")
    status = st.selectbox("Status", ["confirmed", "enquiry"])
    pkg = st.number_input("Package Cost", min_value=0, step=100)
    exp = st.number_input("Actual Expenses", min_value=0, step=100)
    by = st.text_input("Updated By")
    ok = st.form_submit_button("Insert")

if ok:
    rec = {
        "date": d.strftime("%Y-%m-%d"),
        "client_name": client_name.strip(),
        "status": status,
        "package_cost": int(pkg),
        "actual_expenses": int(exp),
        "profit_loss": int(pkg) - int(exp),
        "updated_by": by.strip(),
    }
    try:
        updates_col.insert_one(rec)
        st.success("✅ Inserted")
    except Exception as e:
        st.error(f"Insert failed: {e}")

# 4) show last 10 records
st.subheader("Recent (last 10)")
try:
    rows = list(updates_col.find({}, {"_id": 0}).sort("date", -1).limit(10))
    st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame())
except Exception as e:
    st.error(f"Fetch failed: {e}")
