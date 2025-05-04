import os
import pandas as pd
import math
import locale
import numpy as np

# File paths
input_folder = r"C:\Users\Arpith Shetty\Desktop\TAK Project\Ver_1\Input"
code_file_path = r"C:\Users\Arpith Shetty\Desktop\TAK Project\Ver_1\code.xlsx"
bhasmarathi_type_path = r"C:\Users\Arpith Shetty\Desktop\TAK Project\Ver_1\Bhasmarathi_Type.xlsx"

# Load Stay_City data
stay_city_data = r"C:\Users\Arpith Shetty\Desktop\TAK Project\Ver_1\Stay_City.xlsx"
try:
    stay_city_data = pd.read_excel(stay_city_data, sheet_name="Stay_City")
except Exception as e:
    print(f"Error loading Stay_City file: {e}")
    exit()

# Input date and client name
date = input("Enter the date (dd-mmm-yyyy): ").strip()
client_name = input("Enter the client name: ").strip()

#df['Time'] = df['Time'].fillna("")

# Input date and client name
#date = "10-Dec-2024"
#client_name = "Himangani"

# File selection
input_file = os.path.join(input_folder, f"{date}.xlsx")
if not os.path.exists(input_file):
    print(f"Error: File {input_file} does not exist.")
    exit()

# Load input file and search for client sheet
try:
    input_data = pd.ExcelFile(input_file)
except Exception as e:
    print(f"Error loading file {input_file}: {e}")
    exit()

if client_name not in input_data.sheet_names:
    print(f"Error: Sheet named '{client_name}' not found in file {input_file}.")
    print("Available sheets:", input_data.sheet_names)
    exit()

# Load the client sheet
client_data = input_data.parse(sheet_name=client_name)

# Load the code file
if not os.path.exists(code_file_path):
    print(f"Error: File {code_file_path} does not exist.")
    exit()

try:
    # Replace 'Sheet1' with the actual sheet name in code.xlsx
    code_data = pd.read_excel(code_file_path, sheet_name='Code')
except Exception as e:
    print(f"Error loading code file: {e}")
    exit()

#client_data = input_data.parse(sheet_name=client_name)
#print("Available sheets:", input_data.sheet_names)

if client_name not in input_data.sheet_names:
    print(f"Error: Sheet named '{client_name}' not found in file {input_file}.")
    #print("Available sheets:", input_data.sheet_names)
    exit()

# Load the client sheet
#client_data = input_data.parse(sheet_name=client_name)

# Load the code file
if not os.path.exists(code_file_path):
    print(f"Error: File {code_file_path} does not exist.")
    exit()

try:
    # Replace 'Sheet1' with the actual sheet name in code.xlsx
    code_data = pd.read_excel(code_file_path, sheet_name='Code')
except Exception as e:
    print(f"Error loading code file: {e}")
    exit()

# Ensure code_data is defined
if 'code_data' not in locals():
    print("Error: code_data not defined. Check the code file and sheet name.")
    exit()

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
        itinerary.append({
            'Date': row.get('Date', 'N/A'),
            'Time': row.get('Time', 'N/A'),
            'Description': description
        })
    else:
        itinerary.append({
            'Date': row.get('Date', 'N/A'),
            'Time': row.get('Time', 'N/A'),
            'Description': f"No description found for code {code}"
        })

# Calculate total days and nights
start_date = pd.to_datetime(client_data['Date'].min())
end_date = pd.to_datetime(client_data['Date'].max())
total_days = (end_date - start_date).days + 1
total_nights = total_days - 1

# Get total pax
total_pax = int(client_data['Total Pax'].iloc[0])

# Determine the correct singular or plural for nights
night_text = "Night" if total_nights == 1 else "Nights"

# Determine the correct singular or plural for nights
person_text = "Person" if {total_pax} == 1 else "Persons"

# Generate route by matching codes
route_parts = []
for code in client_data['Code']:
    matched_routes = code_data.loc[code_data['Code'] == code, 'Route']
    if not matched_routes.empty:
        route_parts.append(matched_routes.iloc[0])

# Join the routes with a separator and ensure no unnecessary spaces
route = '-'.join(route_parts).replace(' -', '-').replace('- ', '-')

# Remove consecutive duplicate city names
route_list = route.split('-')
final_route = '-'.join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i - 1]])

# Join cleaned route list with "-"
route = "-".join(route_list)

# Calculate total package cost (from single sheet with all relevant columns)
def calculate_package_cost(input_data):
    # Sum the relevant costs from the columns in the same sheet
    car_cost = input_data['Car Cost'].sum()
    hotel_cost = input_data['Hotel Cost'].sum()
    bhasmarathi_cost = input_data['Bhasmarathi Cost'].sum()
    
    total_cost = car_cost + hotel_cost + bhasmarathi_cost
    # Apply ceiling to the sum, subtract 1 as per your formula
    total_package_cost = math.ceil(total_cost / 1000) * 1000 - 1
    return total_package_cost

locale.setlocale(locale.LC_ALL, 'en_IN')
total_package_cost = calculate_package_cost(client_data)
formatted_cost = int(locale.format_string("%d", total_package_cost, grouping=True).replace(",", ""))
formatted_cost1 =(f"{formatted_cost:,}".replace(",", "X").replace("X", ",", 1))

# Retrieve package cost
#total_package_cost = calculate_package_cost(client_data)

# Extract car types, hotel types, and Bhasmarathi descriptions
car_types = client_data['Car Type'].dropna().unique()
car_types_str = '-'.join(car_types)

hotel_types = client_data['Hotel Type'].dropna().unique()
hotel_types_str = '-'.join(hotel_types)

# Load Bhasmarathi type mapping
if not os.path.exists(bhasmarathi_type_path):
    print(f"Error: File {bhasmarathi_type_path} does not exist.")
    exit()

try:
    bhasmarathi_data = pd.read_excel(bhasmarathi_type_path)
except Exception as e:
    print(f"Error loading Bhasmarathi_Type file: {e}")
    exit()

bhasmarathi_types = client_data['Bhasmarathi Type'].dropna().unique()
bhasmarathi_descriptions = []

for bhas_type in bhasmarathi_types:
    match = bhasmarathi_data.loc[bhasmarathi_data['Bhasmarathi Type'] == bhas_type, 'Description']
    if not match.empty:
        bhasmarathi_descriptions.append(match.iloc[0])

bhasmarathi_desc_str = '-'.join(bhasmarathi_descriptions)

# Combine into final line
details_line = f"({car_types_str},{hotel_types_str},{bhasmarathi_desc_str})"

# Generate the itinerary message
greeting = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
plan = f"*Plan:- {total_days}Days and {total_nights}{night_text} {final_route} for {total_pax} {person_text}*"

# Start building the itinerary
itinerary_message = greeting + plan + "\n\n*Itinerary:*\n"

# Group by date and add events to the itinerary
grouped_itinerary = {}

for entry in itinerary:
    if entry['Date'] != 'N/A' and pd.notna(entry['Date']):
        date = pd.to_datetime(entry['Date']).strftime('%d-%b-%Y')  # Format the date
        if date not in grouped_itinerary:
            grouped_itinerary[date] = []
        grouped_itinerary[date].append(f"{entry['Time']}: {entry['Description']}")

# Iterate over the grouped itinerary to format it
day_number = 1  # Initialize the day number for the first day
first_day = True  # Flag to track the first day for adding time
for date, events in grouped_itinerary.items():
    itinerary_message += f"\n*Day{day_number}:{date}*\n"  # Add day number
    for event in events:
        if first_day:
            itinerary_message += f"{event}\n"
            first_day = False
        else:
            itinerary_message += f"{event[5:]}\n"  # Remove time for other days
    
    day_number += 1  # Increment the day number for the next iteration

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
print(final_message)

# Print or add the exclusions section to your final output
print(exclusions_section)

# Print or add the important notes section to your final output
print(important_notes_section)

print(Cancellation_Policy)

print(Payment_terms)

# Print or append this section to your f4inal output
print(booking_confirmation)
