import streamlit as st
import pandas as pd
import io
import requests
import math
import locale
import pyperclip

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

# ⛑️ Stop the script until both inputs are provided
if not uploaded_file or not client_name:
    st.info("⬆️ Upload the Excel and enter the client name to continue.")
    st.stop()

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

    #if bhasmarathi_type_df is not None:
     #   bhasmarathi_type_df.columns = bhasmarathi_type_df.columns.str.strip()
        #st.subheader("Bhasmarathi Type Preview")
        #st.dataframe(bhasmarathi_type_df.head())

    #if stay_city_df is not None:
     #   st.subheader("Stay City Preview")
      #  st.dataframe(stay_city_df.head())

    #if code_df is not None:
        #st.subheader("Code File Preview")
       # st.dataframe(code_df.head())

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
    #st.subheader("Generated Itinerary Message")
    #st.text_area("Preview", itinerary_message, height=400)

# 6. Initialize inclusions list
inclusions = []

# 1. If Car Type has value
if not client_data['Car Type'].dropna().empty:
    inclusions.append(f"Entire travel as per itinerary by {car_types_str}.")
    inclusions.append("Toll, parking, and driver bata are included.")
    inclusions.append("Airport/ Railway station pickup and drop.")

# 2. If Bhasmarathi Type has value
if not client_data['Bhasmarathi Type'].dropna().empty:
    inclusions.append(f"{bhasmarathi_desc_str} for {total_pax} {person_text}.")
    inclusions.append("Bhasm-Aarti pickup and drop.")

# 3. Hotel stay
# Check if default room type is available in client data
if "Room Type" in client_data.columns:
    default_room_configuration = client_data["Room Type"].iloc[0]  # Default value from client data

# Iterate through each row in client_data
if "Stay City" in client_data.columns and "Room Type" in client_data.columns:
    city_nights = {}
    for i in range(len(client_data)):
        stay_city = client_data["Stay City"].iloc[i]
        room_type = client_data["Room Type"].iloc[i]

        # Skip rows with NaN values in Stay City
        if pd.isna(stay_city):
            continue
        stay_city = stay_city.strip()  # Clean any extra spaces

        # Compare current Stay City with previous row and count nights
        if i > 0 and client_data["Stay City"].iloc[i] == client_data["Stay City"].iloc[i - 1]:
            city_nights[stay_city] += 1  # Increment nights for the same city
        else:
            city_nights[stay_city] = 1  # Start counting nights for a new city

    # Initialize total night counter
    total_used_nights = 0

    # Build inclusions dynamically
    for i in range(len(client_data)):
        stay_city = client_data["Stay City"].iloc[i]
        room_type = client_data["Room Type"].iloc[i]

        # Skip rows with NaN values in Stay City
        if pd.isna(stay_city):
            continue
        stay_city = stay_city.strip()  # Clean any extra spaces

        # Get city name and check if total nights constraint is met
        matching_row = stay_city_df[stay_city_df["Stay City"] == stay_city]
        if not matching_row.empty:
            city_name = matching_row["City"].iloc[0]

            # Check total nights constraint
            if total_used_nights + city_nights[stay_city] <= total_nights:
                inclusions.append(
                    f"{city_nights[stay_city]}Night stay in {city_name} with {room_type} in {hotel_types_str}."
                )
                total_used_nights += city_nights[stay_city]
            else:
                break  # Stop if the total nights exceed the allowed limit                

# 4. If Hotel Type has value
if not client_data['Hotel Type'].dropna().empty:
    inclusions.append("Standard check-in at 12:00 PM and check-out at 09:00 AM.")
    inclusions.append("Early check-in and late check-out are subject to room availability.")
    # Optionally, uncomment this if you want to add conditions for specific hotel types
    # if not client_data.loc[client_data['Hotel Type'] != 'Standard AC Hotel room only'].empty:
    #     inclusions.append("Breakfast included.")

# Combine inclusions into a formatted list
inclusions_section = "*Inclusions:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(inclusions)])

# Combine with the itinerary message
final_message = itinerary_message + "\n\n" + inclusions_section

# Display final message in the Streamlit app
#st.subheader("Final Itinerary Message with Inclusions")
#st.text_area("Preview", final_message, height=400)

# 7. Initialize exclusions list
exclusions = []

# 1. Bhasmarathi pick-up and drop (if Bhasmarathi Type not blank)
#if not client_data['Bhasmarathi Type'].dropna().empty:
 #   exclusions.append("Bhasmarathi pick-up and drop.")

# 2. Meals or beverages
exclusions.append("Any meals or beverages not specified in the itinerary are not included. (e.g., Breakfast, lunch, dinner, snacks, personal beverages).")

# 3. Entry fees (if Car Type is not blank)
if not client_data['Car Type'].dropna().empty:
    exclusions.append("Entry fees for any tourist attractions, temples, or monuments not specified in the inclusions.")

# 4. Travel insurance
exclusions.append("Travel insurance.")

# 5. Personal expenses (if Car Type is not blank)
if not client_data['Car Type'].dropna().empty:
    exclusions.append("Expenses related to personal shopping, tips, or gratuities.")

# 6. Early check-in or late check-out charges (if Hotel Type is not blank)
if not client_data['Hotel Type'].dropna().empty:
    exclusions.append("Any additional charges for early check-in or late check-out if rooms are not available.")

# 7. Costs due to unforeseen events (if Car Type is not blank)
if not client_data['Car Type'].dropna().empty:
    exclusions.append("Costs arising due to natural events, unforeseen roadblocks, or personal travel changes.")

# 8. Charges for additional sightseeing spots (if Car Type is not blank)
if not client_data['Car Type'].dropna().empty:
    exclusions.append("Additional charges for any sightseeing spots not listed in the itinerary.")

# Combine exclusions into a formatted list
exclusions_section = "*Exclusions:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(exclusions)])

# Initialize important notes list
important_notes = []

# 1. Additional charges for tourist attractions (Car Type is not blank)
if not client_data['Car Type'].dropna().empty:
    important_notes.append("Any tourist attractions not mentioned in the itinerary will incur additional charges.")

# 2. Visits to tourist spots (Car Type is not blank)
if not client_data['Car Type'].dropna().empty:
    important_notes.append("Visits to tourist spots or temples are subject to traffic conditions and temple management restrictions. If any tourist spot or temple is closed on the specific day of travel due to unforeseen circumstances, TravelaajKal will not be responsible, and no refunds will be provided.")

# 3. Bhasma Aarti ticket details (Bhasmarathi Type is not blank)
if not client_data['Bhasmarathi Type'].dropna().empty:
    important_notes.append("For Bhasm-Aarti, we will provide tickets, but timely arrival at the temple and seating arrangements are beyond our control.")

# 4. Hotel entry rules (Hotel Type is not blank)
#if not client_data['Hotel Type'].dropna().empty:
 #   important_notes.append("Entry to the hotel is subject to the hotel's rules and regulations. A valid ID proof (Indian citizenship) is required. Only married couples are allowed entry.")

# 5. Bhasma Aarti ticket cancellation policy (Bhasmarathi Type is not blank)
if not client_data['Bhasmarathi Type'].dropna().empty:
    important_notes.append("We only facilitate the booking of Bhasm-Aarti tickets. The ticket cost will be charged at actuals, as mentioned on the ticket.")

# 6. Bhasma Aarti ticket cancellation policy (Bhasmarathi Type is not blank)
if not client_data['Bhasmarathi Type'].dropna().empty:
    important_notes.append("No commitment can be made regarding ticket availability. Bhasm-Aarti tickets are subject to availability and may be canceled at any time based on the decisions of the temple management committee. In case of an unconfirmed ticket, the ticket cost will be refunded.")

# 7. Hotel entry rules (Hotel Type is not blank)
if not client_data['Hotel Type'].dropna().empty:
    important_notes.append("Entry to the hotel is subject to the hotel's rules and regulations. A valid ID proof (Indian citizenship) is required. Only married couples are allowed entry.")

# 8. Hotel entry rules (Hotel Type is not blank) with child rule
if not client_data['Hotel Type'].dropna().empty:
    important_notes.append("Children above 9 years will be considered as adults. Children under 9 years must share the same bed with parents. If an extra bed is required, additional charges will apply.")

# Combine important notes into a formatted list
important_notes_section = "\n*Important Notes:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(important_notes)])

# Combine exclusions and important notes with the itinerary message
final_message_with_exclusions_and_notes = final_message + "\n\n" + exclusions_section + "\n\n" + important_notes_section

# Display final message with exclusions and important notes in the Streamlit app
#st.subheader("Final Itinerary Message with Exclusions and Important Notes")
#st.text_area("Preview", final_message_with_exclusions_and_notes, height=500)

# Initialize Cancellation Policy
cancellation_policy = """
*Cancellation Policy:-*
1. 30+ days before travel → 20% of the advance amount will be deducted.
2. 15-29 days before travel → 50% of the advance amount will be deducted.
3. Less than 15 days before travel → No refund on the advance amount.
4. No refund for no-shows, last-minute cancellations, or early departures.
5. One-time rescheduling is allowed if requested at least 15 days before the travel date, subject to availability.
"""

# Payment terms
payment_terms = """*Payment Terms:-*
50% advance and remaining 50% after arrival at Ujjain.
"""

# Add booking confirmation message and company account details
booking_confirmation = """For booking confirmation, please make the advance payment to the company's current account provided below.

*Company Account details:-*
Account Name: ACHALA HOLIDAYS PVT LTD
Bank: Axis Bank
Account No: 923020071937652
IFSC Code: UTIB0000329
MICR Code: 452211003
Branch Address: Ground Floor, 77, Dewas Road, Ujjain, Madhya Pradesh 456010

Regards,
Team TravelAajKal™️
Reg. Achala Holidays Pvt Limited
Visit :- www.travelaajkal.com
Follow us :- https://www.instagram.com/travelaaj_kal/

*Great news! ACHALA HOLIDAYS PVT LTD is now a DPIIT-recognized Startup by the Government of India.*
*Thank you for your support as we continue to redefine travel.*
*Travel Aaj aur Kal with us!*

TravelAajKal® is a registered trademark of Achala Holidays Pvt Ltd.
"""

# Combine everything into a final output string
final_output = f"""
{final_message}

{exclusions_section}

{important_notes_section}

{cancellation_policy}

{payment_terms}

{booking_confirmation}
"""

# Display the final output in the Streamlit app
st.subheader("Final Itinerary Details")
st.text_area("Preview", final_output, height=800)


# Provide a download button (works reliably for copy purposes too)
st.download_button(
    label="📋 Copy / Download Itinerary",
    data=final_output,
    file_name="itinerary.txt",
    mime="text/plain"
)

# 👇 Add MongoDB saving block here
from pymongo import MongoClient
import datetime

MONGO_URI = "mongodb+srv://TAK_USER:Arpith%2692@cluster0.ewncl10.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["TAK_DB"]
collection = db["itineraries"]

record = {
    "client_name": client_name,
    "upload_date": datetime.datetime.utcnow(),
    "start_date": str(start_date.date()),
    "end_date": str(end_date.date()),
    "total_days": total_days,
    "total_pax": total_pax,
    "final_route": final_route,
    "car_types": car_types_str,
    "hotel_types": hotel_types_str,
    "bhasmarathi_types": bhasmarathi_desc_str,
    "package_cost": formatted_cost1,
    "itinerary_text": final_output
}

try:
    collection.insert_one(record)
    st.success("✅ Itinerary saved to MongoDB")
except Exception as e:
    st.error(f"❌ Failed to save: {e}")
