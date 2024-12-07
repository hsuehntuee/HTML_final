import pandas as pd

# Read the CSV file into a DataFrame
df_test = pd.read_csv('same_season_test_data.csv')

# Normalize the 'season' column (mean=0, std=1)
season_mean = df_test['season'].mean()
season_std = df_test['season'].std()
df_test['date_standardized'] = (df_test['season'] - season_mean) / season_std

# Drop the original 'season' column
df_test.drop(columns=['season'], inplace=True)

# Save the new DataFrame to a new CSV file
df_test.to_csv('kaggle_test.csv', index=False)

print("Normalized CSV file saved as 'date_standardized'")
