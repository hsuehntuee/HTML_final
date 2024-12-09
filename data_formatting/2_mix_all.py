import pandas as pd

# Read the CSV files
df1 = pd.read_csv('stage1_wash_train.csv')
df2 = pd.read_csv('stage2_wash_train.csv')

# Identify common columns between the two datasets
common_columns = df1.columns.intersection(df2.columns)

# Retain only the common columns in both dataframes
df1 = df1[common_columns]
df2 = df2[common_columns]

# Combine the datasets
df_combined = pd.concat([df1, df2])

# Shuffle the combined dataset (optional)
# df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset to a new CSV
df_combined.to_csv('stage12_wash_train.csv', index=False)


