import pandas as pd

df1 = pd.read_csv('kaggle_train.csv')
df2 = pd.read_csv('flipped_kaggle_train.csv')


# Combine the sampled datasets
df_combined = pd.concat([df1, df2])

# Shuffle the combined dataset to ensure randomness
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset to a new CSV
df_combined.to_csv('mix_all.csv', index=False)

