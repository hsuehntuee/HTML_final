import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('stage_train.csv')

# Get all columns that start with "home_" and "away_"
home_columns = [col for col in df.columns if col.startswith('home_') and col != 'home_team_win']
away_columns = [col for col in df.columns if col.startswith('away_')]

# Swap the "home_" and "away_" columns
for home_col, away_col in zip(home_columns, away_columns):
    # Swap values between home_ and away_ columns
    df[home_col], df[away_col] = df[away_col], df[home_col]

# Flip the "home_team_win" column (True <-> False)
df['home_team_win'] = ~df['home_team_win']

# Save the result to a new CSV file
df.to_csv('stage_train_reverse.csv', index=False)

print("done")
