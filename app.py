import streamlit as st
import pandas as pd
import math
import locale
import numpy as np
from io import BytesIO

st.title("TravelAajKal Itinerary Generator")

# File uploads
input_excel = st.file_uploader("Upload Client Input Excel File", type=["xlsx"])
code_file = st.file_uploader("Upload Code.xlsx File", type=["xlsx"])
bhasmarathi_file = st.file_uploader("Upload Bhasmarathi_Type.xlsx File", type=["xlsx"])
stay_city_file = st.file_uploader("Upload Stay_City.xlsx File", type=["xlsx"])

date = st.text_input("Enter the date (dd-mmm-yyyy)")
client_name = st.text_input("Enter the client name")

if st.button("Generate Itinerary"):

    if not input_excel or not code_file or not bhasmarathi_file or not stay_city_file:
        st.error("Please upload all required files.")
        st.stop()

    try:
        stay_city_data = pd.read_excel(stay_city_file, sheet_name="Stay_City")
    except Exception as e:
        st.error(f"Error loading Stay_City file: {e}")
        st.stop()

    try:
        input_data = pd.ExcelFile(input_excel)
    except Exception as e:
        st.error(f"Error reading input file: {e}")
        st.stop()

    if client_name not in input_data.sheet_names:
        st.error(f"Sheet named '{client_name}' not found. Available sheets: {input_data.sheet_names}")
        st.stop()

    client_data = input_data.parse(sheet_name=client_name)

    try:
        code_data = pd.read_excel(code_file, sheet_name='Code')
    except Exception as e:
        st.error(f"Error loading code file: {e}")
        st.stop()

    try:
        bhasmarathi_data = pd.read_excel(bhasmarathi_file)
    except Exception as e:
        st.error(f"Error loading Bhasmarathi_Type file: {e}")
        st.stop()

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

        particulars = code_data.loc[code_data['Code'] == code, 'Particulars'].values
        if particulars.size > 0:
            description = particulars[0]
        else:
            description = f"No description found for code {code}"

        itinerary.append({
            'Date': row.get('Date', 'N/A'),
            'Time': row.get('Time', 'N/A'),
            'Description': description
        })

    start_date = pd.to_datetime(client_data['Date'].min())
    end_date = pd.to_datetime(client_data['Date'].max())
    total_days = (end_date - start_date).days + 1
    total_nights = total_days - 1
    total_pax = int(client_data['Total Pax'].iloc[0])
    night_text = "Night" if total_nights == 1 else "Nights"
    person_text = "Person" if total_pax == 1 else "Persons"

    route_parts = []
    for code in client_data['Code']:
        matched_routes = code_data.loc[code_data['Code'] == code, 'Route']
        if not matched_routes.empty:
            route_parts.append(matched_routes.iloc[0])

    route = '-'.join(route_parts).replace(' -', '-').replace('- ', '-')
    route_list = route.split('-')
    final_route = '-'.join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i - 1]])

    def calculate_package_cost(df):
        car_cost = df['Car Cost'].sum()
        hotel_cost = df['Hotel Cost'].sum()
        bhasmarathi_cost = df['Bhasmarathi Cost'].sum()
        total_cost = car_cost + hotel_cost + bhasmarathi_cost
        return math.ceil(total_cost / 1000) * 1000 - 1

    locale.setlocale(locale.LC_ALL, 'en_IN')
    total_package_cost = calculate_package_cost(client_data)
    formatted_cost = int(locale.format_string("%d", total_package_cost, grouping=True).replace(",", ""))
    formatted_cost1 = f"{formatted_cost:,}".replace(",", "X").replace("X", ",", 1)

    car_types = client_data['Car Type'].dropna().unique()
    hotel_types = client_data['Hotel Type'].dropna().unique()
    bhasmarathi_types = client_data['Bhasmarathi Type'].dropna().unique()

    car_types_str = '-'.join(car_types)
    hotel_types_str = '-'.join(hotel_types)

    bhasmarathi_descriptions = []
    for bhas_type in bhasmarathi_types:
        match = bhasmarathi_data.loc[bhasmarathi_data['Bhasmarathi Type'] == bhas_type, 'Description']
        if not match.empty:
            bhasmarathi_descriptions.append(match.iloc[0])
    bhasmarathi_desc_str = '-'.join(bhasmarathi_descriptions)

    details_line = f"({car_types_str},{hotel_types_str},{bhasmarathi_desc_str})"
    greeting = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
    plan = f"*Plan:- {total_days}Days and {total_nights}{night_text} {final_route} for {total_pax} {person_text}*"
    itinerary_message = greeting + plan + "\n\n*Itinerary:*\n"

    grouped_itinerary = {}
    for entry in itinerary:
        if entry['Date'] != 'N/A' and pd.notna(entry['Date']):
            date_str = pd.to_datetime(entry['Date']).strftime('%d-%b-%Y')
            if date_str not in grouped_itinerary:
                grouped_itinerary[date_str] = []
            grouped_itinerary[date_str].append(f"{entry['Time']}: {entry['Description']}")

    day_number = 1
    first_day = True
    for date, events in grouped_itinerary.items():
        itinerary_message += f"\n*Day{day_number}:{date}*\n"
        for event in events:
            if first_day:
                itinerary_message += f"{event}\n"
                first_day = False
            else:
                itinerary_message += f"{event[5:]}\n"
        day_number += 1

    itinerary_message += f"\n*Package cost: {formatted_cost1}/-*\n{details_line}"
    st.text_area("Generated Itinerary", itinerary_message, height=600)
