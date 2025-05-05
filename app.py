import streamlit as st
import pandas as pd
import os
import io
import requests

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

# Upload date-based Excel file
uploaded_file = st.file_uploader("Upload date-based Excel file", type=["xlsx"])

# Enter client name
client_name = st.text_input("Enter the client name").strip()

# Process once both inputs are available
if uploaded_file and client_name:
    # Read uploaded Excel
    try:
        input_data = pd.ExcelFile(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()

    if client_name not in input_data.sheet_names:
        st.error(f"Sheet '{client_name}' not found in uploaded file.")
        st.info(f"Available sheets: {input_data.sheet_names}")
        st.stop()

    st.success(f"'{client_name}' sheet found. Proceeding with further processing...")

    # Load static Excel files from GitHub
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
    code_df = read_excel_from_url(CODE_FILE_URL)
    bhasmarathi_type_df = read_excel_from_url(BHASMARATHI_TYPE_URL)

    if stay_city_df is not None:
        st.subheader("Stay City Data Preview")
        st.dataframe(stay_city_df.head())

    if code_df is not None:
        st.subheader("Code File Preview")
        st.dataframe(code_df.head())

    if bhasmarathi_type_df is not None:
        st.subheader("Bhasmarathi Type Preview")
        st.dataframe(bhasmarathi_type_df.head())

# Load the client sheet
client_data = input_data.parse(sheet_name=client_name)

# Ensure code_df is loaded and has the expected format
if code_df is None or 'Code' not in code_df.columns or 'Particulars' not in code_df.columns:
    st.error("Code file is either not loaded properly or missing required columns ('Code', 'Particulars').")
    st.stop()

# Display client data preview
st.subheader(f"{client_name} Sheet Preview")
st.dataframe(client_data.head())

# Match codes and generate itinerary
itinerary = []
for _, row in client_data.iterrows():
    code = row.get('Code', None)
    date = row.get('Date', 'N/A')
    time = row.get('Time', 'N/A')

    if pd.isna(code):
        itinerary.append({
            'Date': date,
            'Time': time,
            'Description': "No code provided in row"
        })
        continue

    particulars = code_df.loc[code_df['Code'] == code, 'Particulars'].values
    if particulars.size > 0:
        description = particulars[0]
    else:
        description = f"No description found for code {code}"

    itinerary.append({
        'Date': date,
        'Time': time,
        'Description': description
    })

# Convert itinerary to DataFrame
itinerary_df = pd.DataFrame(itinerary)

# Display the generated itinerary
st.subheader("Generated Itinerary")
st.dataframe(itinerary_df)

# Set locale for Indian number formatting
try:
    locale.setlocale(locale.LC_ALL, 'en_IN')
except locale.Error:
    st.warning("Indian locale setting not supported on this system. Falling back to default formatting.")

# --- Duration calculation ---
start_date = pd.to_datetime(client_data['Date'].min())
end_date = pd.to_datetime(client_data['Date'].max())
total_days = (end_date - start_date).days + 1
total_nights = total_days - 1

# --- Pax handling ---
total_pax = int(client_data['Total Pax'].iloc[0])
night_text = "Night" if total_nights == 1 else "Nights"
person_text = "Person" if total_pax == 1 else "Persons"

# --- Route generation ---
route_parts = []
for code in client_data['Code']:
    matched_routes = code_df.loc[code_df['Code'] == code, 'Route']
    if not matched_routes.empty:
        route_parts.append(matched_routes.iloc[0])

# Remove duplicates and clean formatting
route_list = [route_parts[0]] if route_parts else []
for part in route_parts[1:]:
    if part != route_list[-1]:
        route_list.append(part)

final_route = "-".join(route_list)

# --- Cost calculation function ---
def calculate_package_cost(df):
    car_cost = df['Car Cost'].sum()
    hotel_cost = df['Hotel Cost'].sum()
    bhasmarathi_cost = df['Bhasmarathi Cost'].sum()
    total = car_cost + hotel_cost + bhasmarathi_cost
    return math.ceil(total / 1000) * 1000 - 1

# Calculate and format cost
total_package_cost = calculate_package_cost(client_data)

try:
    formatted_cost = int(locale.format_string("%d", total_package_cost, grouping=True).replace(",", ""))
    formatted_cost_display = f"{formatted_cost:,}".replace(",", "X").replace("X", ",", 1)
except:
    formatted_cost_display = f"{total_package_cost:,}"  # fallback formatting

# --- Display results ---
st.subheader("Trip Summary")
st.markdown(f"""
- **Total Days:** {total_days}
- **Total Nights:** {total_nights} {night_text}
- **Total Pax:** {total_pax} {person_text}
- **Route:** `{final_route}`
- **Package Cost:** ₹ {formatted_cost_display}
""")
