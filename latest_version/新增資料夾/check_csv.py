import pandas as pd

# Load the CSV files into pandas DataFrames
df1 = pd.read_csv('result_team_DNN.csv')  # Replace with the path to your first CSV
df2 = pd.read_csv('result4.csv')  # Replace with the path to your second CSV

# Merge both DataFrames on 'id' to compare the 'home_team_win' values
merged_df = pd.merge(df1, df2, on='id', suffixes=('_file1', '_file2'), how='outer', indicator=True)

# Find rows where 'home_team_win' differs between the two files
differences = merged_df[merged_df['home_team_win_file1'] != merged_df['home_team_win_file2']]

# Find rows that are only present in one of the files
only_in_file1 = merged_df[merged_df['_merge'] == 'left_only']
only_in_file2 = merged_df[merged_df['_merge'] == 'right_only']

# Print the differences
print(f"Differences between the two CSV files:\n")
print(differences)

# Optionally, you can save the differences to a new CSV file
differences.to_csv('differences.csv', index=False)

# If you want to see rows that are unique to each file:
print(f"\nRows only in file 1:\n")
print(only_in_file1)

print(f"\nRows only in file 2:\n")
print(only_in_file2)