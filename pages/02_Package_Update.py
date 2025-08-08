import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# Optional: pretty calendar view (falls back to a table if not available)
CALENDAR_AVAILABLE = True
try:
    from streamlit_calendar import calendar
except Exception:
    CALENDAR_AVAILABLE = False

# ----------------------------
# MongoDB Setup
# ----------------------------
MONGO_URI = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
db = client["TAK_DB"]

col_itineraries = db["itineraries"]          # created by app.py
col_updates     = db["package_updates"]      # status + booking_date
col_expenses    = db["expenses"]             # vendor costs + totals

# ----------------------------
# Utilities
# ----------------------------
def to_int_money(x):
    """Convert formatted money strings ('50,999', etc.) to int. Returns 0 if invalid."""
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        try:
            return int(round(x))
        except Exception:
            return 0
    s = str(x)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0

def fetch_itineraries_df():
    rows = list(col_itineraries.find({}))
    if not rows:
        return pd.DataFrame()

    for r in rows:
        r["itinerary_id"] = str(r.get("_id"))
        # app.py stores start_date/end_date as strings "YYYY-MM-DD"
        try:
            r["start_date"] = pd.to_datetime(r.get("start_date")).date()
        except Exception:
            r["start_date"] = None
        try:
            r["end_date"] = pd.to_datetime(r.get("end_date")).date()
        except Exception:
            r["end_date"] = None

        # package_cost is saved as formatted string (e.g. "50,999")
        r["package_cost_num"] = to_int_money(r.get("package_cost"))
    return pd.DataFrame(rows)

def fetch_updates_df():
    rows = list(col_updates.find({}, {"_id":0}))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["itinerary_id","status","booking_date"])

def fetch_expenses_df():
    rows = list(col_expenses.find({}, {"_id":0}))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["itinerary_id","total_expenses","profit"])

def upsert_status(itinerary_id, status, booking_date):
    doc = {
        "itinerary_id": itinerary_id,
        "status": status,
        "updated_at": datetime.utcnow()
    }
    if status == "confirmed":
        doc["booking_date"] = booking_date
    else:
        doc["booking_date"] = None
    col_updates.update_one({"itinerary_id": itinerary_id}, {"$set": doc}, upsert=True)

def save_expenses(itinerary_id, client_name, booking_date, package_cost, vendors, notes=""):
    total_expenses = sum(int(v or 0) for _, v in vendors)
    profit = int(package_cost) - int(total_expenses)
    doc = {
        "itinerary_id": itinerary_id,
        "client_name": client_name,
        "booking_date": booking_date,
        "package_cost": int(package_cost),
        "total_expenses": int(total_expenses),
        "profit": int(profit),
        "vendors": [{"name": n, "cost": int(v or 0)} for n, v in vendors],
        "notes": notes,
        "saved_at": datetime.utcnow(),
    }
    col_expenses.update_one({"itinerary_id": itinerary_id}, {"$set": doc}, upsert=True)
    return profit

def has_expense(itinerary_id):
    return col_expenses.count_documents({"itinerary_id": itinerary_id}) > 0

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Package Update", layout="wide")
st.title("📦 Package Update")

df_it = fetch_itineraries_df()
if df_it.empty:
    st.info("No packages found yet. Upload a file in the main app first.")
    st.stop()

df_up = fetch_updates_df()
df_exp = fetch_expenses_df()

# Merge status
df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")

# ----------------------------
# Summary KPIs
# ----------------------------
pending_count = (df["status"] == "pending").sum()
cancelled_count = (df["status"] == "cancelled").sum()

confirmed = df[df["status"] == "confirmed"].copy()
confirmed_ids_with_expense = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
confirmed_expense_pending = confirmed[~confirmed["itinerary_id"].isin(confirmed_ids_with_expense)].shape[0]

k1, k2, k3 = st.columns(3)
k1.metric("🟡 Enquiry status update pending", int(pending_count))
k2.metric("🟠 Confirmed – expense entry pending", int(confirmed_expense_pending))
k3.metric("🔴 Cancelled packages", int(cancelled_count))

st.divider()

# ----------------------------
# 1) Status Update Table (pending / under_discussion)
# ----------------------------
st.subheader("1) Update Status for Pending / Under Discussion")

editable = df[df["status"].isin(["pending", "under_discussion"])].copy()
if editable.empty:
    st.success("No pending or under-discussion packages right now. 🎉")
else:
    cols = [
        "itinerary_id", "client_name", "final_route", "total_pax",
        "start_date", "end_date", "package_cost", "status", "booking_date"
    ]
    for c in ["client_name","final_route","total_pax","package_cost"]:
        if c not in editable.columns:
            editable[c] = ""

    editable = editable.reindex(columns=[c for c in cols if c in editable.columns])

    st.caption("Set **Status** per row. If you choose **confirmed**, also set **Booking date**.")
    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        column_config={
            "status": st.column_config.SelectboxColumn(
                "Status",
                options=["pending","under_discussion","confirmed","cancelled"],
                required=True,
            ),
            "booking_date": st.column_config.DateColumn(
                "Booking date",
                help="Required when status is confirmed",
                format="YYYY-MM-DD",
            ),
        },
    )

    if st.button("💾 Save Status Updates"):
        saved, errors = 0, 0
        for _, r in edited.iterrows():
            itinerary_id = r["itinerary_id"]
            status = r["status"]
            bdate = r.get("booking_date")
            if status == "confirmed":
                if pd.isna(bdate):
                    errors += 1
                    continue
                bdate = pd.to_datetime(bdate).date().isoformat()
            else:
                bdate = None
            try:
                upsert_status(itinerary_id, status, bdate)
                saved += 1
            except Exception:
                errors += 1
        if saved:
            st.success(f"Saved {saved} update(s).")
        if errors:
            st.warning(f"{errors} row(s) skipped (missing/invalid booking date for confirmed).")
        #st.experimental_rerun()
        st.rerun()

st.divider()

# ----------------------------
# 2) Expense Entry for Confirmed Packages
# ----------------------------
st.subheader("2) Enter Expenses for Confirmed Packages")

# recompute after save
df_up = fetch_updates_df()
df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")
confirmed = df[df["status"] == "confirmed"].copy()

if confirmed.empty:
    st.info("No confirmed packages yet.")
else:
    have_expense = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
    confirmed["expense_entered"] = confirmed["itinerary_id"].isin(have_expense)

    left, right = st.columns([2,1])
    with left:
        st.dataframe(
            confirmed[["itinerary_id","client_name","final_route","total_pax","package_cost","booking_date","expense_entered"]].sort_values("booking_date"),
            use_container_width=True
        )
    with right:
        st.markdown("**Select a confirmed package to add/edit expenses:**")
        options = confirmed["itinerary_id"] + " | " + confirmed["client_name"] + " | " + confirmed["booking_date"].fillna("").astype(str)
        sel = st.selectbox("Choose package", options.tolist() if not options.empty else [])
        chosen_id = sel.split(" | ")[0] if sel else None

    if chosen_id:
        row = confirmed[confirmed["itinerary_id"] == chosen_id].iloc[0]
        client_name = row.get("client_name","")
        booking_date = row.get("booking_date","")
        base_cost = to_int_money(row.get("package_cost") or row.get("package_cost_num"))

        st.markdown(f"**Client:** {client_name}  \n**Booking date:** {booking_date}  \n**Package cost (₹):** {base_cost:,}")
        st.markdown("#### Expense Inputs")

        with st.form("expense_form"):
            auto_vendor = st.text_input("Auto Vendor Name")
            auto_cost   = st.number_input("Auto Cost (₹)", min_value=0, step=100)

            c1, c2 = st.columns(2)
            with c1:
                car_vendor_1 = st.text_input("Car vendor Name-1")
                car_cost_1   = st.number_input("Car Cost-1 (₹)", min_value=0, step=100)
                hotel_vendor_1 = st.text_input("Hotel vendor-1")
                hotel_cost_1   = st.number_input("Hotel vendor-1 cost (₹)", min_value=0, step=100)
                hotel_vendor_3 = st.text_input("Hotel vendor-3")
                hotel_cost_3   = st.number_input("Hotel vendor-3 cost (₹)", min_value=0, step=100)
                hotel_vendor_5 = st.text_input("Hotel vendor-5")
                hotel_cost_5   = st.number_input("Hotel vendor-5 cost (₹)", min_value=0, step=100)
            with c2:
                car_vendor_2 = st.text_input("Car vendor Name-2")
                car_cost_2   = st.number_input("Car Cost-2 (₹)", min_value=0, step=100)
                hotel_vendor_2 = st.text_input("Hotel vendor-2")
                hotel_cost_2   = st.number_input("Hotel vendor-2 cost (₹)", min_value=0, step=100)
                hotel_vendor_4 = st.text_input("Hotel vendor-4")
                hotel_cost_4   = st.number_input("Hotel vendor-4 cost (₹)", min_value=0, step=100)
                other_pooja_vendor = st.text_input("Other Pooja vendor")
                other_pooja_cost   = st.number_input("Other Pooja vendor cost (₹)", min_value=0, step=100)

            bhas_vendor = st.text_input("Bhasmarathi vendor")
            bhas_cost   = st.number_input("Bhasmarathi Cost (₹)", min_value=0, step=100)

            photo_cost  = st.number_input("Photo frame cost (₹)", min_value=0, step=100)
            other_exp   = st.number_input("Any other expense (₹)", min_value=0, step=100)
            notes       = st.text_area("Notes (optional)")

            submit = st.form_submit_button("💾 Save Expenses")

        if submit:
            vendors = [
                ("Auto", auto_cost),
                (f"Car-1 | {car_vendor_1}", car_cost_1),
                (f"Car-2 | {car_vendor_2}", car_cost_2),
                (f"Hotel-1 | {hotel_vendor_1}", hotel_cost_1),
                (f"Hotel-2 | {hotel_vendor_2}", hotel_cost_2),
                (f"Hotel-3 | {hotel_vendor_3}", hotel_cost_3),
                (f"Hotel-4 | {hotel_vendor_4}", hotel_cost_4),
                (f"Hotel-5 | {hotel_vendor_5}", hotel_cost_5),
                (f"Bhasmarathi | {bhas_vendor}", bhas_cost),
                (f"Other Pooja | {other_pooja_vendor}", other_pooja_cost),
                ("Photo frame", photo_cost),
                ("Other", other_exp),
            ]
            profit = save_expenses(chosen_id, client_name, booking_date, base_cost, vendors, notes)
            st.success(f"Expenses saved. 💰 Profit: ₹ {profit:,}")
            #st.experimental_rerun()
            st.rerun()

st.divider()

# ----------------------------
# 3) Calendar – Confirmed packages
# ----------------------------
st.subheader("3) Calendar – Confirmed Packages")

confirmed = df[df["status"] == "confirmed"].copy()
if confirmed.empty:
    st.info("No confirmed packages to show on calendar.")
else:
    events = []
    for _, r in confirmed.iterrows():
        if pd.isna(r.get("booking_date")):
            continue
        title = f"{r.get('client_name','')}_{r.get('total_pax','')}pax"
        start = pd.to_datetime(r["booking_date"]).strftime("%Y-%m-%d")
        events.append({"title": title, "start": start})

    if CALENDAR_AVAILABLE:
        st.caption("Interactive calendar (month view).")
        calendar(
            options={"initialView": "dayGridMonth", "height": 600, "events": events},
            key="pkg_cal",
        )
    else:
        st.caption("Calendar component not installed. Showing a simple list instead.")
        display = pd.DataFrame(events).rename(columns={"title":"Package", "start":"Date"})
        st.dataframe(display.sort_values("Date"), use_container_width=True)

