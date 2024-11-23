import pandas as pd
from datetime import datetime

# Read the CSV file into a DataFrame
df = pd.read_csv('train_data_all.csv')

# Convert 'date' column to datetime format (assuming it's in 'Y/M/D' format)
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

# Normalize the 'date' column by calculating the number of days since a reference date (e.g., '2000-01-01')
reference_date = datetime(2000, 1, 1)
df['date_numeric'] = (df['date'] - reference_date).dt.days

# Standardize the 'data_numeric' column (mean=0, std=1)
mean = df['date_numeric'].mean()
std = df['date_numeric'].std()
df['date_standardized'] = (df['date_numeric'] - mean) / std

# Drop the original 'date' and 'date_numeric' columns if you no longer want them
df.drop(columns=['date', 'date_numeric'], inplace=True)

# Save the new DataFrame to a new CSV file
df.to_csv('train_date_standardized.csv', index=False)

print("Standardized CSV file saved as 'train_data_standardized.csv'")
