import pandas as pd

# Read the CSV file
df = pd.read_csv('stage2_train.csv')

# Define a function to count blanks in each row
# We'll use 'isna()' to check for NaN values (blanks)
df['blank_count'] = df.isna().sum(axis=1)

# Calculate the threshold: more than (number of columns) / 5 blanks
threshold = df.shape[1] / 5

# Drop rows where blank_count is greater than the threshold
df_cleaned = df[df['blank_count'] <= threshold]

# Drop the 'blank_count' column as it's no longer needed
df_cleaned = df_cleaned.drop(columns=['blank_count'])

# Output the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('stage2_wash_train.csv', index=False)

