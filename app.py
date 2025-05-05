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
