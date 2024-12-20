import pandas as pd
from datetime import datetime

# Read the CSV file
df = pd.read_csv('stage1_test.csv')

# Check for NaN or inf values in the 'season' column
# Option 1: Coerce non-numeric values to NaN, then fill NaN with 2024
df['season'] = pd.to_numeric(df['season'], errors='coerce')  # Convert non-numeric to NaN
df['season'] = df['season'].fillna(2019).astype(int)  # Fill NaN with 2024 and convert to int

# Extract year from the 'season' column and create a date assuming '20xx-07-01'
df['year'] = df['season']
df['date'] = pd.to_datetime(df['year'].astype(str) + '-07-01')

# Define the reference date (2016-01-01)
reference_date = datetime(2016, 1, 1)

# Convert the 'date' to numeric format (days since reference_date)
df['date_numeric'] = (df['date'] - reference_date).dt.days

# Standardize the date (divide by 2000)
df['date_standardized'] = df['date_numeric'] / 2000

# Drop the 'date' and 'date_numeric' columns as they are no longer needed
df.drop(columns=['date', 'date_numeric', 'year'], inplace=True)

# Sort the DataFrame by 'date_standardized'
df = df.sort_values(by='date_standardized')

# Save the new DataFrame to a new CSV file
df.to_csv('stage1_dateFormatted_test.csv', index=False)
