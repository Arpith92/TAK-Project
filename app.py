import streamlit as st
import pandas as pd
import io
import requests
import math
import locale

# ----------------- PART 1: File Load & Setup -------------------
# GitHub raw URLs for static files
CODE_FILE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Code.xlsx"
BHASMARATHI_TYPE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Bhasmarathi_Type.xlsx"
STAY_CITY_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Stay_City.xlsx"

# Function to read Excel from GitHub URL
def read_excel_from_url(url, sheet_name=None):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_excel(io.BytesIO(response.content), sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error reading file from {url}: {e}")
        return None

# App UI
st.title("TAK Project Input Loader")

uploaded_file = st.file_uploader("Upload date-based Excel file", type=["xlsx"])
client_name = st.text_input("Enter the client name").strip()

if uploaded_file and client_name:
    try:
        input_data = pd.ExcelFile(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()

    if client_name not in input_data.sheet_names:
        st.error(f"Sheet '{client_name}' not found.")
        st.info(f"Available sheets: {input_data.sheet_names}")
        st.stop()

    client_data = input_data.parse(sheet_name=client_name)

    # Load static data from GitHub
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
    bhasmarathi_type_df = read_excel_from_url(BHASMARATHI_TYPE_URL)

    # ----------------- PART 2: Match Codes -------------------
    itinerary = []
    for _, row in client_data.iterrows():
        code = row.get('Code', None)
        if code is None:
            itinerary.append({
                'Date': row.get('Date', 'N/A'),
                'Time': row.get('Time', 'N/A'),
                'Description': "No code provided in row"
            })
            continue

        particulars = code_df.loc[code_df['Code'] == code, 'Particulars'].values
        if particulars.size > 0:
            description = particulars[0]
        else:
            description = f"No description found for code {code}"

        itinerary.append({
            'Date': row.get('Date', 'N/A'),
            'Time': row.get('Time', 'N/A'),
            'Description': description
        })

    # ----------------- PART 3: Summary and Cost -------------------
    try:
        locale.setlocale(locale.LC_ALL, 'en_IN')
    except locale.Error:
        st.warning("Indian locale setting not supported. Fallback formatting will be used.")

    start_date = pd.to_datetime(client_data['Date'].min())
    end_date = pd.to_datetime(client_data['Date'].max())
    total_days = (end_date - start_date).days + 1
    total_nights = total_days - 1
    total_pax = int(client_data['Total Pax'].iloc[0])
    night_text = "Night" if total_nights == 1 else "Nights"
    person_text = "Person" if total_pax == 1 else "Persons"

    # Route generation
    route_parts = []
    for code in client_data['Code']:
        matched_routes = code_df.loc[code_df['Code'] == code, 'Route']
        if not matched_routes.empty:
            route_parts.append(matched_routes.iloc[0])

    route_list = [route_parts[0]] if route_parts else []
    for part in route_parts[1:]:
        if part != route_list[-1]:
            route_list.append(part)
    final_route = "-".join(route_list)

    def calculate_package_cost(df):
        car = df['Car Cost'].sum()
        hotel = df['Hotel Cost'].sum()
        bhasma = df['Bhasmarathi Cost'].sum()
        return math.ceil((car + hotel + bhasma) / 1000) * 1000 - 1

    total_package_cost = calculate_package_cost(client_data)
    try:
        formatted_cost = int(locale.format_string("%d", total_package_cost, grouping=True).replace(",", ""))
        formatted_cost1 = f"{formatted_cost:,}".replace(",", "X").replace("X", ",", 1)
    except:
        formatted_cost1 = f"{total_package_cost:,}"

    # ----------------- PART 4: Additional Details -------------------
    car_types_str = '-'.join(client_data['Car Type'].dropna().unique())
    hotel_types_str = '-'.join(client_data['Hotel Type'].dropna().unique())

    bhasmarathi_descriptions = []
    if bhasmarathi_type_df is not None:
        for bhas_type in client_data['Bhasmarathi Type'].dropna().unique():
            match = bhasmarathi_type_df.loc[bhasmarathi_type_df['Bhasmarathi Type'] == bhas_type, 'Description']
            if not match.empty:
                bhasmarathi_descriptions.append(match.iloc[0])
    bhasmarathi_desc_str = '-'.join(bhasmarathi_descriptions)
    details_line = f"({car_types_str},{hotel_types_str},{bhasmarathi_desc_str})"

    greeting = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
    plan = f"*Plan:- {total_days} Days and {total_nights} {night_text} {final_route} for {total_pax} {person_text}*"

    # ----------------- PART 5: Build Full Itinerary -------------------
    itinerary_message = greeting + plan + "\n\n*Itinerary:*\n"
    grouped_itinerary = {}

    for entry in itinerary:
        if entry['Date'] != 'N/A' and pd.notna(entry['Date']):
            date = pd.to_datetime(entry['Date']).strftime('%d-%b-%Y')
            if date not in grouped_itinerary:
                grouped_itinerary[date] = []
            grouped_itinerary[date].append(f"{entry['Time']}: {entry['Description']}")

    day_number = 1
    first_day = True
    for date, events in grouped_itinerary.items():
        itinerary_message += f"\n*Day {day_number}: {date}*\n"
        for event in events:
            if first_day:
                itinerary_message += f"{event}\n"
                first_day = False
            else:
                itinerary_message += f"{event[5:]}\n"
        day_number += 1

    itinerary_message += f"\n*Package cost: ₹ {formatted_cost1}/-*\n{details_line}"

    # ----------------- DISPLAY OUTPUT -------------------
    st.subheader("Trip Summary")
    st.markdown(f"""
    - **Total Days:** {total_days}
    - **Total Nights:** {total_nights} {night_text}
    - **Total Pax:** {total_pax} {person_text}
    - **Route:** `{final_route}`
    - **Package Cost:** ₹ {formatted_cost1}
    """)

    st.subheader("Final Message Preview")
    st.text_area("Client Message", value=greeting + plan + f"\n\nDetails: {details_line}", height=200)

    st.subheader("Full Itinerary")
    st.text_area("Formatted Itinerary", itinerary_message, height=400)
