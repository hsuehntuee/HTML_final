import pandas as pd

# Load the dataset
df = pd.read_csv('kaggle_train.csv')

# Identify home and away columns that match by suffix
home_columns = [col for col in df.columns if col.startswith('home_')]
away_columns = [col for col in df.columns if col.startswith('away_')]

# Ensure that home and away columns match by suffix
for home_col in home_columns:
    # Find the corresponding away column by matching the suffix
    suffix = home_col[5:]  # Strip the 'home_' prefix
    away_col = 'away_' + suffix
    
    if away_col in df.columns:
        # Swap the home and away columns using a temporary column to avoid overwriting
        temp = df[home_col].copy()  # Create a temporary copy of the home column
        df[home_col] = df[away_col]  # Assign the away column to the home column
        df[away_col] = temp  # Assign the temporary copy (original home column) to the away column

# Ensure 'home_team_win' is boolean and flip the column (True <-> False)
df['home_team_win'] = df['home_team_win'].apply(lambda x: x == 'True' if isinstance(x, str) else x)  # Convert to boolean
df['home_team_win'] = ~df['home_team_win']  # Flip the boolean values using bitwise NOT
# Output the result as a new CSV
df.to_csv('flipped_kaggle_train.csv', index=False)
