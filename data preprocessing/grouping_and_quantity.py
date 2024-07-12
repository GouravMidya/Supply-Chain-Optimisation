# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:13:19 2024

@author: goura
"""
#%%
import pandas as pd

# If you don't have a 'Quantity' column, create one (assuming each row is one item)
if 'Quantity' not in expanded_df.columns:
    expanded_df['Quantity'] = 1

# Convert 'datetime' to date
expanded_df['date'] = expanded_df['datetime'].dt.date

# Group by date, time_group, Standardized_Item, and Size, then sum the quantities
grouped_df = expanded_df.groupby(['date', 'time_group', 'Standardized_Item', 'Size'])['Quantity'].sum().reset_index()

# Sort the resulting dataframe
grouped_df = grouped_df.sort_values(['date', 'time_group', 'Standardized_Item', 'Size'])

# Display the first few rows of the result
print(grouped_df.head(10))

# Display some summary statistics
print(grouped_df.describe())

# Check the total number of unique combinations
print(f"Total unique combinations: {len(grouped_df)}")

# Check the number of unique items
print(f"Number of unique standardized items: {grouped_df['Standardized_Item'].nunique()}")

# Check the number of unique sizes
print(f"Number of unique sizes: {grouped_df['Size'].nunique()}")