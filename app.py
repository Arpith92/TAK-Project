import streamlit as st
import pandas as pd
import os
import locale
from datetime import timedelta
from docx import Document
from docx.shared import Pt

# --- Handle Locale Setting ---
try:
    locale.setlocale(locale.LC_ALL, '')
except locale.Error:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

st.title("TravelaajKal Itinerary Generator")

# --- File Uploads ---
client_file = st.file_uploader("Upload Client Excel File", type=["xlsx"])
cost_file = st.file_uploader("Upload Cost Excel File", type=["xlsx"])

# --- Generate Itinerary Button ---
if st.button("Generate Itineraries"):
    if not client_file or not cost_file:
        st.error("Please upload both Client and Cost Excel files.")
    else:
        try:
            client_df = pd.read_excel(client_file, engine='openpyxl')
            cost_car = pd.read_excel(cost_file, sheet_name='Car', engine='openpyxl')
            cost_bhasma = pd.read_excel(cost_file, sheet_name='Bhasmarathi', engine='openpyxl')
            cost_hotel = pd.read_excel(cost_file, sheet_name='Hotel', engine='openpyxl')
            st.success("Files loaded successfully!")

            # Loop through each client
            for idx, row in client_df.iterrows():
                name = row['Name']
                start_date = pd.to_datetime(row['Start Date'])
                end_date = pd.to_datetime(row['End Date'])
                persons = row['Total Persons']
                pickup = row['Pickup Point']
                drop = row['Drop Point']
                destinations = row['Destinations'].split(',')

                # --- Cost Calculations ---
                car_cost = cost_car['Cost'].sum()

                if persons > 9:
                    bhasma_cost = cost_bhasma['Package Cost'].iloc[0] * persons
                else:
                    bhasma_cost = 0

                hotel_cost = cost_hotel['Hotel Cost'].sum()
                total_cost = car_cost + bhasma_cost + hotel_cost

                # --- Itinerary Creation ---
                doc = Document()
                doc.add_heading(f"{name}'s Custom Itinerary", 0)

                doc.add_paragraph(f"Pickup Point: {pickup}")
                doc.add_paragraph(f"Drop Point: {drop}")
                doc.add_paragraph(f"Total Persons: {persons}")
                doc.add_paragraph(f"Travel Dates: {start_date.date()} to {end_date.date()}")

                doc.add_heading("Day-wise Plan:", level=1)
                for i, dest in enumerate(destinations):
                    day = start_date + timedelta(days=i)
                    doc.add_paragraph(f"Day {i+1} - {day.strftime('%d-%b-%Y')}: Visit {dest.strip()}")

                doc.add_heading("Inclusions", level=2)
                doc.add_paragraph("- AC Sedan/Tempo Traveller")
                doc.add_paragraph("- Accommodation with Breakfast")
                doc.add_paragraph("- Toll, Parking, Driver Allowance")

                doc.add_heading("Estimated Total Package Cost", level=2)
                doc.add_paragraph(f"₹ {total_cost:,}")

                doc.add_heading("Payment Terms", level=2)
                doc.add_paragraph("50% advance to confirm booking. Remaining 50% before trip start.")

                doc.add_heading("Important Notes", level=2)
                doc.add_paragraph("TravelaajKal team will be in touch 24/7 during the tour.")

                # Save file
                filename = f"{name.replace(' ', '_')}_Itinerary.docx"
                output_path = os.path.join("itineraries", filename)
                os.makedirs("itineraries", exist_ok=True)
                doc.save(output_path)
                st.success(f"Itinerary generated for {name} ✅")

        except Exception as e:
            st.error(f"Error loading input Excel: {e}")
