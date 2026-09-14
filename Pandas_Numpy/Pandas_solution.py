import pandas as pd
import numpy as np

# Load the dataset

df = pd.read_csv('sales_data.csv')


# 1. Get a summary of the DataFrame

print(df.info())

# 2. Get a summary of the numeric columns
print(df.describe())


# 1. Access the first 5 rows
print(df.head())

# 2. Access a specific column (Product or Price)
print(df['Product'].unique())

# 3. Access rows where Sales > 500
print(df[df['Sales'] > 500])


# 1. Filter data for Region = East
print(df[df['Region'] == 'East'])

# 2. Filter data where Price > 100 and Sales < 1000
print(df[(df['Price'] > 100) & (df['Sales'] < 1000)])


# 1. Check for missing values
print(df.isnull().sum())

# 2. Fill missing values in the Sales column with the median
df['Sales'] = df['Sales'].fillna(df['Sales'].median())

# 3. Drop rows where Product is missing
df.dropna(subset=['Product'], inplace=True)


# 1. Add a new column Discounted_Price (10% off)
df['Discounted_Price'] = df['Price'] * 0.9

# 2. Display the first 5 rows to confirm
print(df.head())


# 1. Drop the Discounted_Price column
df.drop(columns=['Discounted_Price'], inplace=True)

# 2. Confirm the column has been dropped
print(df.columns)


# 1. Create a new column Profit (Sales - Cost)
df['Profit'] = df['Sales'] - df['Cost']

# 2. Create a new column Log_Profit using np.log()
df['Log_Profit'] = np.log(df['Profit'])

# Display the first 5 rows to confirm
print(df.head())
