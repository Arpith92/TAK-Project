import pandas as pd

# Read Excel from GitHub (use raw link)
excel_url = 'https://raw.githubusercontent.com/Arpith92/TAK_Project_Test2/main/Code.xlsx'
df = pd.read_excel(excel_url, sheet_name=0)

# Input: Total number of days
total_days = int(input("Enter total number of travel days: "))

# Store day-wise particulars
daily_particulars = {}

for day in range(1, total_days + 1):
    code_input = input(f"Enter travel code for Day {day} (e.g., ip-id): ").strip()
    
    # VLOOKUP simulation
    match_row = df[df['Code'].str.lower() == code_input.lower()]
    
    if not match_row.empty:
        particulars = match_row.iloc[0]['Particulars']
        daily_particulars[f'Day {day}'] = particulars
    else:
        daily_particulars[f'Day {day}'] = 'Code not found in database.'

# Output result
for day, detail in daily_particulars.items():
    print(f"{day}: {detail}")
