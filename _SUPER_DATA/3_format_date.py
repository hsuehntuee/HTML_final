import pandas as pd
from datetime import datetime


df = pd.read_csv('stage2_test.csv')


df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')


reference_date = datetime(2016, 1, 1)
df['date_numeric'] = (df['date'] - reference_date).dt.days

df['date_standardized'] = (df['date_numeric']) / 2000

df.drop(columns=['date', 'date_numeric'], inplace=True)


df = df.sort_values(by='date_standardized')



# Save the new DataFrame to a new CSV file
df.to_csv('stage2_dateFormatted_test.csv', index=False)

