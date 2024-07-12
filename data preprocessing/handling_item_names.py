# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 07:38:19 2024

@author: goura
"""
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import process
from tqdm import tqdm
#%%
file_path="C:/Users/goura/Documents/Supply Chain Optimisation/persiana order data/Orders_Airoli_2023_Q2.xlsx"
df = pd.read_excel(file_path,header=4)

#%%
df_copy = df.copy()

#%% Splitting multiple items in same cell to 1 item per cell

def split_items(row):
    if pd.isna(row['Items']):
        # Return a DataFrame with a single row, setting 'Item' to NaN or a placeholder
        return pd.DataFrame([row.drop('Items').tolist() + [np.nan]], 
                            columns=row.drop('Items').index.tolist() + ['Item'])
    else:
        items = str(row['Items']).split(',')
        return pd.DataFrame([row.drop('Items').tolist() + [item.strip()] for item in items], 
                            columns=row.drop('Items').index.tolist() + ['Item'])


# Enable tqdm for pandas operations
tqdm.pandas(desc="Splitting Items")
# Apply the function to each row with progress bar and concatenate the results
expanded_df = pd.concat(df_copy.progress_apply(split_items, axis=1).tolist(), ignore_index=True)

# Reset the index if needed
expanded_df = expanded_df.reset_index(drop=True)

#%%

def extract_main_item(item_name):
    if pd.isna(item_name):
        return "Unknown Item"
    return re.split(r'\s*[\(\-]', str(item_name))[0].strip()

def standardize_item(item_name, unique_main_items, threshold=80):
    if pd.isna(item_name):
        return "Unknown Item"
    main_item = extract_main_item(str(item_name))
    match = process.extractOne(main_item, unique_main_items)
    if match[1] >= threshold:
        return match[0]
    return main_item
#%%
# Get unique main items once to avoid recalculating for each row
unique_main_items = df['Item'].dropna().apply(extract_main_item).unique()


#%%
# Apply the standardization with progress bar
tqdm.pandas(desc="Standardizing Items")
expanded_df['Standardized_Item'] = expanded_df['Item'].progress_apply(lambda x: standardize_item(x, unique_main_items))
#%%
# Extract size information, handling null values
tqdm.pandas(desc="Extracting Sizes")
expanded_df['Size'] = expanded_df['Item'].progress_apply(lambda x: re.search(r'\((.*?)\)', str(x)).group(1) if pd.notna(x) and '(' in str(x) else None)


#%%
# Manual corrections for common items
manual_corrections = {
    'Caramel Custerd': 'Caramel Custard',
    # Add more corrections here
}

expanded_df['Standardized_Item'] = expanded_df['Standardized_Item'].replace(manual_corrections)


#%%

unique_items_before = expanded_df['Item'].value_counts()

unique_items_after = expanded_df['Standardized_Item'].value_counts()

#%%
