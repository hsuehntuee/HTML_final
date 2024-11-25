import pandas as pd

# Load the original dataset and the flipped dataset
df_original = pd.read_csv('kaggle_train.csv')
df_flipped = pd.read_csv('flipped_kaggle_train.csv')

# Randomly sample 50% from the original dataset
df_original_sampled = df_original.sample(frac=0.4, random_state=42)

# Get the IDs from the sampled original dataset
sampled_ids = df_original_sampled['id'].tolist()
# Filter the flipped dataset to exclude the sampled IDs from the original dataset
df_flipped_filtered = df_flipped[~df_flipped['id'].isin(sampled_ids)]


# Combine the sampled datasets
df_combined = pd.concat([df_original_sampled, df_flipped_filtered])

# Shuffle the combined dataset to ensure randomness
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset to a new CSV
df_combined.to_csv('4_10_balanced_train_data.csv', index=False)

print(f"Combined dataset saved as 'balanced_train_data.csv' with {len(df_combined)} rows.")
