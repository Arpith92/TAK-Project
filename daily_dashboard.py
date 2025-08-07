import streamlit as st
st.title("Debug Loader")
try:
    import pymongo
    st.success("pymongo imported successfully")
except Exception as e:
    st.error(f"Import Error: {e}")

# ✅ Full Streamlit App for Daily Updates, Profit Tracking & Admin Dashboard

import streamlit as st
import pandas as pd
from pymongo import MongoClient
import datetime

# ---- MongoDB Setup ----
MONGO_URI = st.secrets["mongo_uri"]  # store safely in Streamlit Cloud
client = MongoClient(MONGO_URI)
db = client["TAK_DB"]
updates_col = db["daily_updates"]

# ---- Page Config ----
st.set_page_config(page_title="Daily Itinerary Tracker", layout="wide")
st.title("📦 Daily Itinerary Updates Dashboard")

# ---- Navigation ----
page = st.sidebar.selectbox("Choose Page", ["Submit Daily Update", "Analytics Dashboard", "Delete Entry"])

# ---- PAGE 1: Data Entry ----
if page == "Submit Daily Update":
    st.subheader("📝 Submit a Daily Update")
    with st.form("daily_update_form"):
        update_date = st.date_input("Date", datetime.date.today())
        client = st.text_input("Client Name")
        status = st.selectbox("Status", ["confirmed", "enquiry"])
        package_cost = st.number_input("Total Package Cost", min_value=0)
        actual_expenses = st.number_input("Actual Expenses", min_value=0)
        updated_by = st.text_input("Updated By")
        remarks = st.text_area("Remarks (optional)")
        submitted = st.form_submit_button("💾 Save Entry")

    if submitted:
        profit_loss = package_cost - actual_expenses
        record = {
            "date": update_date.strftime("%Y-%m-%d"),
            "client_name": client.strip(),
            "status": status,
            "package_cost": package_cost,
            "actual_expenses": actual_expenses,
            "profit_loss": profit_loss,
            "updated_by": updated_by.strip(),
            "remarks": remarks.strip()
        }
        updates_col.insert_one(record)
        st.success("✅ Entry saved successfully!")

# ---- PAGE 2: Analytics ----
elif page == "Analytics Dashboard":
    st.subheader("📊 Analytics Dashboard")
    data = list(updates_col.find({}, {"_id": 0}))

    if not data:
        st.info("No data found.")
    else:
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Confirmed", df[df.status == "confirmed"].shape[0])
        col2.metric("Total Enquiries", df[df.status == "enquiry"].shape[0])
        col3.metric("Total Profit", f"₹ {df['profit_loss'].sum():,.0f}")

        st.markdown("### 📅 Daily Summary")
        summary = df.groupby("date").agg({
            "client_name": "count",
            "profit_loss": "sum"
        }).rename(columns={"client_name": "Total Packages"})
        st.line_chart(summary)

        st.markdown("### 📋 All Records")
        st.dataframe(df.sort_values("date", ascending=False))

# ---- PAGE 3: Delete Entries ----
elif page == "Delete Entry":
    st.subheader("🗑️ Delete a Record")
    client_to_delete = st.text_input("Enter Client Name to Delete")

    if st.button("Delete Record"):
        if client_to_delete.strip():
            deleted = updates_col.delete_many({"client_name": client_to_delete.strip()})
            st.success(f"✅ Deleted {deleted.deleted_count} record(s) for client: {client_to_delete}")
        else:
            st.warning("⚠️ Please enter a client name to delete.")
