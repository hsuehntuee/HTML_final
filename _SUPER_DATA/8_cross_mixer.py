import pandas as pd
import numpy as np

# Load both datasets
df_train = pd.read_csv('stage_train.csv')
df_train_reverse = pd.read_csv('stage_train_reverse.csv')

# Ensure both datasets have the same number of rows
assert len(df_train) == len(df_train_reverse), "The two CSVs must have the same number of rows"

# Number of rows in each CSV
n_rows = len(df_train)

# Create an array of indices
indices = np.arange(n_rows)

# Randomly select 50% of the indices for df_train (from stage_train.csv)
train_indices = np.random.choice(indices, size=n_rows // 2, replace=False)

# The remaining indices will be used for df_train_reverse (from stage_train_reverse.csv)
reverse_indices = np.setdiff1d(indices, train_indices)

# Select rows from both DataFrames using the indices
df_train_selected = df_train.iloc[train_indices].reset_index(drop=True)
df_train_reverse_selected = df_train_reverse.iloc[reverse_indices].reset_index(drop=True)

# Concatenate the selected rows from both datasets
df_mixed = pd.concat([df_train_selected, df_train_reverse_selected], ignore_index=True)

# Save the mixed DataFrame to a new CSV file
df_mixed.to_csv('stage_train_mixed_exclusive.csv', index=False)

print("Mixing completed and saved to 'stage_train_mixed_exclusive.csv'")
