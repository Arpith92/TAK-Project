import os
import pandas as pd
import math
import locale
import streamlit as st

# Input fields
#date = st.text_input("Enter the date (dd-mmm-yyyy)")
#client_name = st.text_input("Enter the client name")

date= 01-April-2025
client_name=Saurav Saini

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
            car_types_str = car_types
            hotel_types_str = hotel_types

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

            itinerary_message += f"\n*Package cost: ₹{formatted_cost}/-*\n{details_line}"

            # ---------------- Inclusions ----------------
            inclusions = []

            if not client_data['Car Type'].dropna().empty:
                inclusions.append(f"Entire travel as per itinerary by {car_types_str}.")
                inclusions.append("Toll, parking, and driver bata are included.")
                inclusions.append("Airport/ Railway station pickup and drop.")

            if not client_data['Bhasmarathi Type'].dropna().empty:
                inclusions.append(f"{bhasmarathi_desc_str} for {total_pax} {person_text}.")
                inclusions.append("Bhasm-Aarti pickup and drop.")

            if "Room Type" in client_data.columns and "Stay City" in client_data.columns:
                city_nights = {}
                for i in range(len(client_data)):
                    stay_city = client_data["Stay City"].iloc[i]
                    if pd.isna(stay_city):
                        continue
                    stay_city = stay_city.strip()
                    if stay_city in city_nights:
                        city_nights[stay_city] += 1
                    else:
                        city_nights[stay_city] = 1

                for city, nights in city_nights.items():
                    room_type = client_data.loc[client_data["Stay City"] == city, "Room Type"].iloc[0]
                    inclusions.append(f"{nights} Night {city} stay in {room_type} in {hotel_types_str}.")

            if not client_data['Hotel Type'].dropna().empty:
                inclusions.append("Standard check-in at 12:00 PM and check-out at 09:00 AM.")
                inclusions.append("Early check-in and late check-out are subject to room availability.")

            inclusions_section = "*Inclusions:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(inclusions)])

            # Final message
            final_message = itinerary_message + "\n\n" + inclusions_section
            st.markdown("---")
            st.subheader("Final Itinerary Message")
            st.markdown(final_message)

    except Exception as e:
        st.error(f"Error loading input Excel: {e}")
else:
    st.info("Please enter both the date and client name to continue.")
