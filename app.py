import streamlit as st
import pandas as pd
import io
import requests
import math
import locale

# GitHub raw URLs for static files
CODE_FILE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Code.xlsx"
BHASMARATHI_TYPE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Bhasmarathi_Type.xlsx"
STAY_CITY_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Stay_City.xlsx"

# Function to read Excel from URL
def read_excel_from_url(url, sheet_name=None):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_excel(io.BytesIO(response.content), sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error reading file from {url}: {e}")
        return None

# App UI
st.title("TAK Project Itinerary Generator")

# Upload date-based Excel file
uploaded_file = st.file_uploader("Upload date-based Excel file", type=["xlsx"])

# Enter client name
client_name = st.text_input("Enter the client name").strip()

if uploaded_file and client_name:
    try:
        input_data = pd.ExcelFile(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()

    if client_name not in input_data.sheet_names:
        st.error(f"Sheet '{client_name}' not found in uploaded file.")
        st.info(f"Available sheets: {input_data.sheet_names}")
        st.stop()

    client_data = input_data.parse(sheet_name=client_name)
    st.success(f"'{client_name}' sheet found. Proceeding with processing...")

    # Load static Excel files
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
    bhasmarathi_type_df = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet_name="Bhasmarathi_Type")

    if bhasmarathi_type_df is not None:
        bhasmarathi_type_df.columns = bhasmarathi_type_df.columns.str.strip()
        st.subheader("Bhasmarathi Type Preview")
        st.dataframe(bhasmarathi_type_df.head())

    if stay_city_df is not None:
        st.subheader("Stay City Preview")
        st.dataframe(stay_city_df.head())

    if code_df is not None:
        st.subheader("Code File Preview")
        st.dataframe(code_df.head())

    # Match codes and generate itinerary
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

    # Part 3: Calculate totals and route
    start_date = pd.to_datetime(client_data['Date'].min())
    end_date = pd.to_datetime(client_data['Date'].max())
    total_days = (end_date - start_date).days + 1
    total_nights = total_days - 1
    total_pax = int(client_data['Total Pax'].iloc[0])
    night_text = "Night" if total_nights == 1 else "Nights"
    person_text = "Person" if total_pax == 1 else "Persons"

    # Generate route from codes
    route_parts = []
    for code in client_data['Code']:
        matched_routes = code_df.loc[code_df['Code'] == code, 'Route']
        if not matched_routes.empty:
            route_parts.append(matched_routes.iloc[0])
    route = '-'.join(route_parts).replace(' -', '-').replace('- ', '-')
    route_list = route.split('-')
    final_route = '-'.join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i - 1]])

    # Calculate package cost
    def calculate_package_cost(df):
        car_cost = df['Car Cost'].sum()
        hotel_cost = df['Hotel Cost'].sum()
        bhasmarathi_cost = df['Bhasmarathi Cost'].sum()
        total = car_cost + hotel_cost + bhasmarathi_cost
        return math.ceil(total / 1000) * 1000 - 1

    try:
        locale.setlocale(locale.LC_ALL, 'en_IN')
        use_locale = True
    except locale.Error:
        use_locale = False

    total_package_cost = calculate_package_cost(client_data)
    if use_locale:
        formatted_cost = locale.format_string("%d", total_package_cost, grouping=True)
    else:
        formatted_cost = f"{total_package_cost:,}"

    formatted_cost1 = formatted_cost.replace(",", "X").replace("X", ",", 1)

    # Part 4: Extract and match types
    car_types = client_data['Car Type'].dropna().unique()
    car_types_str = '-'.join(car_types)

    hotel_types = client_data['Hotel Type'].dropna().unique()
    hotel_types_str = '-'.join(hotel_types)

    bhasmarathi_types = client_data['Bhasmarathi Type'].dropna().unique()
    bhasmarathi_descriptions = []

    for bhas_type in bhasmarathi_types:
        match = bhasmarathi_type_df.loc[bhasmarathi_type_df['Bhasmarathi Type'] == bhas_type, 'Description']
        if not match.empty:
            bhasmarathi_descriptions.append(match.iloc[0])

    bhasmarathi_desc_str = '-'.join(bhasmarathi_descriptions)
    details_line = f"({car_types_str},{hotel_types_str},{bhasmarathi_desc_str})"

    greeting = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
    plan = f"*Plan:- {total_days}Days and {total_nights}{night_text} {final_route} for {total_pax} {person_text}*"

    # Part 5: Build final itinerary message
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
        itinerary_message += f"\n*Day{day_number}:{date}*\n"
        for event in events:
            itinerary_message += f"{event if first_day else event[5:]}\n"
            first_day = False
        day_number += 1

    itinerary_message += f"\n*Package cost: {formatted_cost1}/-*\n{details_line}"

    # Display final message
    st.subheader("Generated Itinerary Message")
    st.text_area("Preview", itinerary_message, height=400)
