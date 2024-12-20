import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('stage1_train.csv')

# Fill all missing values in the entire DataFrame with -100
df = df.fillna(-100)

# Save the result to a new CSV file
df.to_csv('stage1_train.csv', index=False)

print("done")
