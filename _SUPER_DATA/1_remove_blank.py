import pandas as pd

# Read the CSV file
df = pd.read_csv('stage1_train.csv')

# Define a function to count blanks in each row
# We'll use 'isna()' to check for NaN values (blanks)
df['blank_count'] = df.isna().sum(axis=1)

# Drop rows where blank_count is greater than 40
df_cleaned = df[df['blank_count'] <= 40]

# Drop the 'blank_count' column as it's no longer needed
df_cleaned = df_cleaned.drop(columns=['blank_count'])

# Output the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('stage1_wash_train.csv', index=False)
