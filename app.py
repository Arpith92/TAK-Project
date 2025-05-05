import os
import pandas as pd
import math
import locale
import numpy as np
import streamlit as st

st.set_page_config(page_title="TAK Itinerary Generator", layout="wide")
st.title("🧳 TravelAajKal Itinerary Generator")

# Set default paths
default_input_folder = r"C:\Users\Arpith Shetty\Desktop\TAK Project\Ver_1\Input"
code_file_path = r"C:\Users\Arpith Shetty\Desktop\TAK Project\Ver_1\code.xlsx"
bhasmarathi_type_path = r"C:\Users\Arpith Shetty\Desktop\TAK Project\Ver_1\Bhasmarathi_Type.xlsx"
stay_city_path = r"C:\Users\Arpith Shetty\Desktop\TAK Project\Ver_1\Stay_City.xlsx"

# Load Stay_City data
try:
    stay_city_data = pd.read_excel(stay_city_path, sheet_name="Stay_City")
except Exception as e:
    st.error(f"Error loading Stay_City file: {e}")
    st.stop()

# Input Section
date = st.text_input("Enter the Date (dd-mmm-yyyy)", value="10-Dec-2024")
client_name = st.text_input("Enter Client Name", value="Himangani")

# Check if input file exists
input_file_path = os.path.join(default_input_folder, f"{date}.xlsx")
if not os.path.exists(input_file_path):
    st.error(f"Input file not found at: {input_file_path}")
    st.stop()

# Load Excel File
try:
    input_data = pd.ExcelFile(input_file_path)
except Exception as e:
    st.error(f"Error loading file {input_file_path}: {e}")
    st.stop()

if client_name not in input_data.sheet_names:
    st.error(f"Sheet named '{client_name}' not found. Available sheets: {input_data.sheet_names}")
    st.stop()

client_data = input_data.parse(sheet_name=client_name)

# Load Code Data
if not os.path.exists(code_file_path):
    st.error(f"Code file not found at: {code_file_path}")
    st.stop()

try:
    code_data = pd.read_excel(code_file_path, sheet_name="Code")
except Exception as e:
    st.error(f"Error loading code file: {e}")
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

# Date calculations
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
    matched_routes = code_data.loc[code_data['Code'] == code, 'Route']
    if not matched_routes.empty:
        route_parts.append(matched_routes.iloc[0])

route = '-'.join(route_parts).replace(' -', '-').replace('- ', '-')
route_list = route.split('-')
final_route = '-'.join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i - 1]])

# Package cost calculation
def calculate_package_cost(input_data):
    car_cost = input_data['Car Cost'].sum()
    hotel_cost = input_data['Hotel Cost'].sum()
    bhasmarathi_cost = input_data['Bhasmarathi Cost'].sum()
    total_cost = car_cost + hotel_cost + bhasmarathi_cost
    return math.ceil(total_cost / 1000) * 1000 - 1

locale.setlocale(locale.LC_ALL, 'en_IN')
total_package_cost = calculate_package_cost(client_data)
formatted_cost = int(locale.format_string("%d", total_package_cost, grouping=True).replace(",", ""))
formatted_cost1 = (f"{formatted_cost:,}".replace(",", "X").replace("X", ",", 1))

# Car, Hotel, Bhasmarathi Types
car_types = client_data['Car Type'].dropna().unique()
hotel_types = client_data['Hotel Type'].dropna().unique()

# Load Bhasmarathi descriptions
if not os.path.exists(bhasmarathi_type_path):
    st.error(f"File not found: {bhasmarathi_type_path}")
    st.stop()

try:
    bhasmarathi_data = pd.read_excel(bhasmarathi_type_path)
except Exception as e:
    st.error(f"Error loading Bhasmarathi_Type file: {e}")
    st.stop()

bhasmarathi_types = client_data['Bhasmarathi Type'].dropna().unique()
bhasmarathi_descriptions = []

for bhas_type in bhasmarathi_types:
    match = bhasmarathi_data.loc[bhasmarathi_data['Bhasmarathi Type'] == bhas_type, 'Description']
    if not match.empty:
        bhasmarathi_descriptions.append(match.iloc[0])

car_types_str = '-'.join(car_types)
hotel_types_str = '-'.join(hotel_types)
bhasmarathi_desc_str = '-'.join(bhasmarathi_descriptions)
details_line = f"({car_types_str},{hotel_types_str},{bhasmarathi_desc_str})"

# Build itinerary message
greeting = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
plan = f"*Plan:- {total_days}Days and {total_nights}{night_text} {final_route} for {total_pax} {person_text}*"
itinerary_message = greeting + plan + "\n\n*Itinerary:*\n"

# Group by date and add events
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
        if first_day:
            itinerary_message += f"{event}\n"
            first_day = False
        else:
            itinerary_message += f"{event[5:]}\n"
    day_number += 1

# Package cost and details
itinerary_message += f"\n*Package cost: {formatted_cost1}/-*\n{details_line}"

# Show output
st.markdown("### 📝 Final Itinerary Message:")
st.text_area("Generated Message", itinerary_message, height=600)

# Inclusions
inclusions = []
if not client_data['Car Type'].dropna().empty:
    inclusions.extend([
        f"Entire travel as per itinerary by {car_types_str}.",
        "Toll, parking, and driver bata are included.",
        "Airport/ Railway station pickup and drop."
    ])

st.markdown("### ✅ Inclusions")
for item in inclusions:
    st.markdown(f"- {item}")
