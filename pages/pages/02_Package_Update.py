import streamlit as st
import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, date

# Optional calendar component
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

col_itineraries = db["itineraries"]          # from app.py
col_updates     = db["package_updates"]      # status + booking_date
col_expenses    = db["expenses"]             # vendor costs + totals

# ----------------------------
# Utils
# ----------------------------
def to_int_safe(x):
    """Convert money strings like '50,999' to int. Returns 0 on failure."""
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

def str_id(x):
    return str(x) if x is not None else None

def get_itineraries_df():
    rows = list(col_itineraries.find({}))
    if not rows:
        return pd.DataFrame()
    # normalize
    for r in rows:
        r["itinerary_id"] = str_id(r.get("_id"))
        # parse dates
        if isinstance(r.get("start_date"), str):
            try:
                r["start_date"] = pd.to_datetime(r["start_date"]).date()
            except Exception:
                pass
        if isinstance(r.get("end_date"), str):
            try:
                r["end_date"] = pd.to_datetime(r["end_date"]).date()
            except Exception:
                pass
        # numeric package cost
        r["package_cost_num"] = to_int_safe(r.get("package_cost"))
    df = pd.DataFrame(rows)
    return df

def get_updates_df():
    rows = list(col_updates.find({}, {"_id":0}))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["itinerary_id","status","booking_date"])

def get_expenses_df():
    rows = list(col_expenses.find({}, {"_id":0}))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["itinerary_id","total_expenses","profit"])

def upsert_update(itinerary_id, status, booking_date):
    doc = {
        "itinerary_id": itinerary_id,
        "status": status,
        "updated_at": datetime.utcnow()
    }
    if status == "confirmed":
        doc["booking_date"] = booking_date
    else:
        doc["booking_date"] = None
    col_updates.update_one(
        {"itinerary_id": itinerary_id},
        {"$set": doc},
        upsert=True
    )

def has_expense_record(itinerary_id):
    return col_expenses.count_documents({"itinerary_id": itinerary_id}) > 0

def save_expenses(itinerary_id, client_name, booking_date, base_package_cost, vendors, notes=""):
    total_expenses = sum(v for _,v in vendors if v is not None)
    profit = base_package_cost - total_expenses
    doc = {
        "itinerary_id": itinerary_id,
        "client_name": client_name,
        "booking_date": booking_date,
        "package_cost": int(base_package_cost),
        "total_expenses": int(total_expenses),
        "profit": int(profit),
        "vendors": [{"name": n, "cost": int(v or 0)} for n,v in vendors],
        "notes": notes,
        "saved_at": datetime.utcnow()
    }
    col_expenses.update_one({"itinerary_id": itinerary_id}, {"$set": doc}, upsert=True)
    return profit

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Package Update", layout="wide")
st.title("📦 Package Update")

# Load data
df_it = get_itineraries_df()
df_up = get_updates_df()
df_exp = get_expenses_df()

if df_it.empty:
    st.info("No itineraries found yet. Generate packages from the main app first.")
    st.stop()

# Merge: itineraries + status
df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")  # pending = no status yet

# ----------------------------
# Summary KPIs
# ----------------------------
pending_count = (df["status"] == "pending").sum()
cancelled_count = (df["status"] == "cancelled").sum()

# Confirmed with/without expenses
confirmed_df = df[df["status"] == "confirmed"].copy()
if not df_exp.empty:
    confirmed_with_expense_ids = set(df_exp["itinerary_id"])
else:
    confirmed_with_expense_ids = set()
confirmed_expense_pending = confirmed_df[~confirmed_df["itinerary_id"].isin(confirmed_with_expense_ids)].shape[0]

c1, c2, c3 = st.columns(3)
c1.metric("🟡 Enquiry status update pending", int(pending_count))
c2.metric("🟠 Confirmed – expense entry pending", int(confirmed_expense_pending))
c3.metric("🔴 Cancelled packages", int(cancelled_count))

st.divider()

# ----------------------------
# Section 1: Status Update Table
# ----------------------------
st.subheader("1) Update Status for New/Under Discussion Packages")

editable_df = df[
    (df["status"].isin(["pending","under_discussion"]))
].copy()

if editable_df.empty:
    st.success("No pending/under-discussion packages right now. 🎉")
else:
    view_cols = [
        "itinerary_id", "client_name", "total_pax", "final_route",
        "start_date", "end_date", "package_cost", "status", "booking_date"
    ]
    for c in ["package_cost", "client_name", "total_pax", "final_route"]:
        if c not in editable_df.columns:
            editable_df[c] = ""

    editable_df = editable_df.reindex(columns=[c for c in view_cols if c in editable_df.columns])

    st.caption("Tip: set **Status** per row. If you choose **confirmed**, also set **Booking date**.")
    edited = st.data_editor(
        editable_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "status": st.column_config.SelectboxColumn(
                "Status",
                options=["pending","under_discussion","confirmed","cancelled"],
                help="Set the current status for the package",
                required=True
            ),
            "booking_date": st.column_config.DateColumn(
                "Booking date",
                help="Required if status is confirmed",
                format="YYYY-MM-DD"
            ),
            "package_cost": st.column_config.TextColumn("Package Cost"),
        },
        hide_index=True
    )

    if st.button("💾 Save Status Updates"):
        errors = 0
        saved = 0
        for _, r in edited.iterrows():
            itinerary_id = r.get("itinerary_id")
            status = r.get("status")
            bdate = r.get("booking_date")

            if status == "confirmed":
                if pd.isna(bdate):
                    errors += 1
                    continue
                try:
                    bdate = pd.to_datetime(bdate).date().isoformat()
                except Exception:
                    errors += 1
                    continue
            else:
                bdate = None

            try:
                upsert_update(itinerary_id, status, bdate)
                saved += 1
            except Exception:
                errors += 1
        if saved:
            st.success(f"Saved {saved} update(s).")
        if errors:
            st.warning(f"{errors} row(s) skipped (missing/invalid booking date for confirmed).")
        st.experimental_rerun()

st.divider()

# ----------------------------
# Section 2: Expense Entry for Confirmed Packages
# ----------------------------
st.subheader("2) Enter Expenses for Confirmed Packages")

# Recompute after possible updates
df_up = get_updates_df()
df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")
confirmed_df = df[df["status"] == "confirmed"].copy()

if confirmed_df.empty:
    st.info("No confirmed packages yet.")
else:
    # Mark which ones already have expense records
    if not df_exp.empty:
        have_expense = set(df_exp["itinerary_id"])
    else:
        have_expense = set()

    confirmed_df["expense_entered"] = confirmed_df["itinerary_id"].isin(have_expense)

    left, right = st.columns([2,1])
    with left:
        st.markdown("**Confirmed packages:**")
        st.dataframe(
            confirmed_df[["itinerary_id","client_name","total_pax","package_cost","booking_date","expense_entered"]]
            .sort_values("booking_date", ascending=True),
            use_container_width=True
        )
    with right:
        st.markdown("**Select a confirmed package to add/edit expenses:**")
        options = confirmed_df["itinerary_id"] + " | " + confirmed_df["client_name"] + " | " + confirmed_df["booking_date"].fillna("").astype(str)
        selection = st.selectbox("Choose package", options.tolist() if not options.empty else [])
        chosen_id = None
        if selection:
            chosen_id = selection.split(" | ")[0]

    if chosen_id:
        pkg_row = confirmed_df[confirmed_df["itinerary_id"]==chosen_id].iloc[0]
        client_name = pkg_row.get("client_name","")
        booking_date = pkg_row.get("booking_date","")
        base_cost = to_int_safe(pkg_row.get("package_cost") or pkg_row.get("package_cost_num"))

        st.markdown(f"**Client:** {client_name}  \n**Booking date:** {booking_date}  \n**Package cost (₹):** {base_cost:,}")

        st.markdown("#### Expense Inputs")
        with st.form("expense_form"):
            # you can customize labels freely; kept your list but grouped
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

            submit_exp = st.form_submit_button("💾 Save Expenses")

        if submit_exp:
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
                ("Other", other_exp)
            ]
            profit = save_expenses(
                chosen_id,
                client_name,
                booking_date,
                base_cost,
                vendors,
                notes
            )
            st.success(f"Expenses saved. 💰 Profit: ₹ {profit:,}")
            st.experimental_rerun()

st.divider()

# ----------------------------
# Section 3: Calendar of Confirmed Packages
# ----------------------------
st.subheader("3) Calendar – Confirmed Packages")

confirmed_df = df[df["status"]=="confirmed"].copy()
if confirmed_df.empty:
    st.info("No confirmed packages to show on calendar.")
else:
    # create events from booking_date
    evts = []
    for _, r in confirmed_df.iterrows():
        if pd.isna(r.get("booking_date")):
            continue
        title = f"{r.get('client_name','')}_{r.get('total_pax','')}pax"
        start = pd.to_datetime(r["booking_date"]).strftime("%Y-%m-%d")
        evts.append({"title": title, "start": start})

    if CALENDAR_AVAILABLE:
        st.caption("Interactive calendar (month view).")
        calendar(
            options={
                "initialView": "dayGridMonth",
                "height": 600,
                "events": evts
            },
            key="pkg_cal",
        )
    else:
        st.caption("Calendar component not installed. Showing a date list instead.")
        disp = pd.DataFrame(evts).rename(columns={"title":"Package", "start":"Date"})
        st.dataframe(disp.sort_values("Date"), use_container_width=True)
