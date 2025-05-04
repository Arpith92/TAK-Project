import os
import pandas as pd
import math
import locale
import streamlit as st

# Input fields
date = st.text_input("Enter the date (dd-mmm-yyyy)")
client_name = st.text_input("Enter the client name")

# Define raw GitHub URLs
code_file_url = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Code.xlsx"
bhasmarathi_type_url = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Bhasmarathi_Type.xlsx"

if date and client_name:
    input_file_url = f"https://raw.githubusercontent.com/Arpith92/TAK-Project/main/{date}.xlsx"
    try:
        input_data = pd.ExcelFile(input_file_url)
        if client_name not in input_data.sheet_names:
            st.error(f"Sheet '{client_name}' not found in {date}.xlsx")
            st.write("Available sheets:", input_data.sheet_names)
        else:
            client_data = input_data.parse(sheet_name=client_name)
            st.success("Client data loaded successfully!")
            st.dataframe(client_data)

            # Load Code.xlsx
            try:
                code_data = pd.read_excel(code_file_url, sheet_name="Code")
                st.success("Code file loaded successfully!")

                # Generate Itinerary
                itinerary = []
                for _, row in client_data.iterrows():
                    code = row.get('Code', None)
                    if pd.isna(code):
                        itinerary.append({
                            'Date': row.get('Date', 'N/A'),
                            'Time': row.get('Time', 'N/A'),
                            'Description': "No code provided"
                        })
                        continue
                    particulars = code_data.loc[code_data['Code'] == code, 'Particulars'].values
                    description = particulars[0] if len(particulars) > 0 else f"No description for code {code}"
                    itinerary.append({
                        'Date': row.get('Date', 'N/A'),
                        'Time': row.get('Time', 'N/A'),
                        'Description': description
                    })

                itinerary_df = pd.DataFrame(itinerary)
                st.subheader("Generated Itinerary")
                st.dataframe(itinerary_df)

                # Calculate days/nights
                start_date = pd.to_datetime(client_data['Date'].min())
                end_date = pd.to_datetime(client_data['Date'].max())
                total_days = (end_date - start_date).days + 1
                total_nights = total_days - 1
                total_pax = int(client_data['Total Pax'].iloc[0])
                night_text = "Night" if total_nights == 1 else "Nights"
                person_text = "Person" if total_pax == 1 else "Persons"

                # Generate route
                route_parts = []
                for code in client_data['Code']:
                    matched = code_data.loc[code_data['Code'] == code, 'Route']
                    if not matched.empty:
                        route_parts.append(matched.iloc[0])
                route = '-'.join(route_parts).replace(' -', '-').replace('- ', '-')
                route_list = route.split('-')
                final_route = '-'.join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i - 1]])

                # Cost calculation
                def calculate_package_cost(df):
                    cost = df['Car Cost'].sum() + df['Hotel Cost'].sum() + df['Bhasmarathi Cost'].sum()
                    return math.ceil(cost / 1000) * 1000 - 1

                locale.setlocale(locale.LC_ALL, 'en_IN')
                total_package_cost = calculate_package_cost(client_data)
                formatted_cost = locale.format_string("%d", total_package_cost, grouping=True)

                # Vehicle/Hotel/Bhasmarathi details
                car_types = '-'.join(client_data['Car Type'].dropna().unique())
                hotel_types = '-'.join(client_data['Hotel Type'].dropna().unique())

                try:
                    bhas_data = pd.read_excel(bhasmarathi_type_url)
                    bhasmarathi_types = client_data['Bhasmarathi Type'].dropna().unique()
                    bhasmarathi_descriptions = [
                        bhas_data.loc[bhas_data['Bhasmarathi Type'] == btype, 'Description'].values[0]
                        for btype in bhasmarathi_types if not bhas_data.loc[bhas_data['Bhasmarathi Type'] == btype].empty
                    ]
                    bhasmarathi_desc_str = '-'.join(bhasmarathi_descriptions)
                except Exception as e:
                    st.error(f"Error loading Bhasmarathi_Type: {e}")
                    bhasmarathi_desc_str = ""

                details_line = f"({car_types},{hotel_types},{bhasmarathi_desc_str})"

                # Summary
                st.markdown("---")
                st.subheader("Itinerary Summary")
                st.markdown(f"**Client:** {client_name}")
                st.markdown(f"**Dates:** {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}")
                st.markdown(f"**Days:** {total_days}, **{total_nights} {night_text}**, **{total_pax} {person_text}")
                st.markdown(f"**Route:** {final_route}")
                st.markdown(f"**Package Cost:** ₹ {formatted_cost}")
                st.markdown(f"**Details:** {details_line}")

            except Exception as e:
                st.error(f"Error loading Code.xlsx: {e}")

    except Exception as e:
        st.error(f"Error loading input Excel: {e}")
else:
    st.info("Please enter both the date and client name to continue.")

# Generate the itinerary message
greeting = f"**Greetings from TravelAajkal**  \n\n**Client Name:** {client_name}  \n"
plan = f"**Plan:** {total_days} Days and {total_nights} {night_text} | Route: {final_route} | {total_pax} {person_text}  \n\n"
itinerary_message = greeting + plan + "**Itinerary:**  \n"

# Group itinerary entries by date
grouped_itinerary = {}
for entry in itinerary:
    entry_date = entry['Date']
    if entry_date != 'N/A' and pd.notna(entry_date):
        formatted_date = pd.to_datetime(entry_date).strftime('%d-%b-%Y')
        if formatted_date not in grouped_itinerary:
            grouped_itinerary[formatted_date] = []
        time = entry['Time'] if pd.notna(entry['Time']) else ""
        grouped_itinerary[formatted_date].append(f"{time} - {entry['Description']}".strip())

# Build formatted itinerary
day_number = 1
for date, events in grouped_itinerary.items():
    itinerary_message += f"\n**Day {day_number}: {date}**  \n"
    for event in events:
        itinerary_message += f"- {event}  \n"
    day_number += 1

# Add the total package cost at the end
itinerary_message += f"\n*Package cost: {formatted_cost1}/-*\n{details_line}"

# Initialize inclusions list
inclusions = []

# 1. If Car Type has value
if not client_data['Car Type'].dropna().empty:
    inclusions.append(f"Entire travel as per itinerary by {car_types_str}.")
    inclusions.append("Toll, parking, and driver bata are included.")
    inclusions.append("Airport/ Railway station pickup and drop.")

# 2. If Bhasmarathi Type has value
if not client_data['Bhasmarathi Type'].dropna().empty:
    #total_pax = (client_data['Total Pax'].iloc[0])  # Assuming 'Total Pax' column exists
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
        matching_row = stay_city_data[stay_city_data["Stay City"] == stay_city]
        if not matching_row.empty:
            city_name = matching_row["City"].iloc[0]

            # Check total nights constraint
            if total_used_nights + city_nights[stay_city] <= total_nights:
                inclusions.append(
                    f"{city_nights[stay_city]}Night {city_name} stay in {room_type} in {hotel_types_str}."
                )
                total_used_nights += city_nights[stay_city]
            else:
                break  # Stop if the total nights exceed the allowed limit                
# 4. If Hotel Type has value
if not client_data['Hotel Type'].dropna().empty:
    inclusions.append("Standard check-in at 12:00 PM and check-out at 09:00 AM.")
    inclusions.append("Early check-in and late check-out are subject to room availability.")
  #  if not client_data.loc[client_data['Hotel Type'] != 'Standard AC Hotel room only'].empty:
       # inclusions.append("Breakfast included.")

# Combine inclusions into a formatted list
inclusions_section = "*Inclusions:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(inclusions)])

# Combine with the itinerary message
final_message = itinerary_message + "\n\n" + inclusions_section

# Initialize exclusions list
exclusions = []

# 1. Bhasmarathi pick-up and drop (if Bhasmarathi Type not blank)
#if not client_data['Bhasmarathi Type'].dropna().empty:
 #   exclusions.append("Bhasmarathi pick-up and drop.")

# 2. Meals or beverages
exclusions.append("Any meals or beverages not specified in the itinerary are not included.(e.g.Breakfast,lunch, dinner snacks, personal beverages).")

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
exclusions_section = "\n*Exclusions:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(exclusions)])

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

# 5.Bhasma Aarti ticket cancellation policy (Bhasmarathi Type is not blank)
if not client_data['Bhasmarathi Type'].dropna().empty:
    important_notes.append("We only facilitate the booking of Bhasm-Aarti tickets. The ticket cost will be charged at actuals, as mentioned on the ticket.")

#6.Bhasma Aarti ticket cancellation policy (Bhasmarathi Type is not blank)
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

# Initialize Cancellation Policy
Cancellation_Policy = """
*Cancellation Policy:-*
1. 30+ days before travel → 20% of the advance amount will be deducted.
2. 15-29 days before travel → 50% of the advance amount will be deducted.
3. Less than 15 days before travel → No refund on the advance amount.
4. No refund for no-shows, last-minute cancellations, or early departures.
5. One-time rescheduling is allowed if requested at least 15 days before the travel date, subject to availability.
"""

#Payment terms
Payment_terms = """*Payment Terms:-*
50% advance and reamining 50% after arrival at Ujjain.
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

# Print the final output
st.markdown(final_message)

# Print or add the exclusions section to your final output
st.markdown(exclusions_section)

# Print or add the important notes section to your final output
st.markdown(important_notes_section)

st.markdown(Cancellation_Policy)

st.markdown(Payment_terms)

# Print or append this section to your f4inal output
st.markdown(booking_confirmation)

st.markdown("---")
st.subheader("Itinerary Summary")

# Display the route and trip info
st.markdown(f"**Client Name:** {client_name}")
st.markdown(f"**Date Range:** {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}")
st.markdown(f"**Total Days:** {total_days}, **{total_nights} {night_text}**, **{total_pax} {person_text}**")
st.markdown(f"**Route:** {final_route}")
st.markdown(f"**Package Cost:** ₹ {formatted_cost1}")
st.markdown(f"**Details:** {details_line}")

# Convert itinerary to HTML for printing
itinerary_df = pd.DataFrame(itinerary)
itinerary_html = itinerary_df.to_html(index=False)

# Custom HTML and CSS for printing
print_html = f"""
    <style>
        @media print {{
            body {{
                font-family: Arial, sans-serif;
            }}
            .no-print {{
                display: none;
            }}
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>

    <div>
        <h2>Travel Itinerary</h2>
        <p><strong>Client:</strong> {client_name}</p>
        <p><strong>Dates:</strong> {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}</p>
        <p><strong>Route:</strong> {final_route}</p>
        <p><strong>Cost:</strong> ₹ {formatted_cost1}</p>
        <p><strong>Details:</strong> {details_line}</p>
        <br/>
        {itinerary_html}
    </div>

    <div class="no-print">
        <button onclick="window.print()">🖨️ Print Itinerary</button>
    </div>
"""

# Render the printable itinerary
st.components.v1.html(print_html, height=800, scrolling=True)

