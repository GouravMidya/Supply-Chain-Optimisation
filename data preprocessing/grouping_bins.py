# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:12:44 2024

@author: goura
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
#%%
file_path="C:/Users/goura/Documents/Supply Chain Optimisation/persiana order data/Orders_Airoli_2023_Q2.xlsx"
df = pd.read_excel(file_path,header=4)
df_copy = df.copy()

#%%
# Enable tqdm for pandas operations
tqdm.pandas(desc="Converting to datetime")

# Define a function to convert to datetime
def convert_to_datetime(x):
    return pd.to_datetime(x, format='mixed', errors='coerce')

# Apply the conversion function with a progress bar
expanded_df['datetime'] = expanded_df['Created'].progress_apply(convert_to_datetime)


#%%

# Enable tqdm for pandas operations
tqdm.pandas(desc="Converting to datetime")

# Convert 'Created' column to datetime, handling null values
expanded_df['datetime'] = pd.to_datetime(expanded_df['Created'], format='mixed', errors='coerce')

# Create a function for time grouping that handles null values
def get_time_group(dt):
    if pd.isnull(dt):
        return 'Unknown'
    hour = dt.hour
    if 11 <= hour < 15:
        return '11:00-15:00'
    elif 15 <= hour < 19:
        return '15:00-19:00'
    elif 19 <= hour < 23:
        return '19:00-23:00'
    else:
        return '23:00-03:00'

# Apply the time grouping function with a progress bar
expanded_df['time_group'] = expanded_df['datetime'].progress_apply(get_time_group)

# Print some information to verify the conversion
print(expanded_df['datetime'].dtype)
print(expanded_df['time_group'].value_counts(dropna=False))
print(expanded_df[df['time_group'] == 'Unknown'].shape[0], "rows have unknown time group")

#%%

