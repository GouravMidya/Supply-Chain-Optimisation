# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:12:35 2024

@author: goura
"""
#%%
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import re
from fuzzywuzzy import process
from tqdm import tqdm

#%% Specify the folder path containing the Excel files
folder_path = 'C:/Users/goura/Documents/Supply Chain Optimisation/small dataset'

# Initialize an empty list to store individual DataFrames
dfs = []
#%% Iterate through all Excel files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        file_path = os.path.join(folder_path, filename)
        
        # Read the Excel file, specifying the header row
        df = pd.read_excel(file_path, header=4)  # 0-based index, so 3 means 4th row
        
        # Extract the location from the filename
        location_match = re.search(r'Orders_([A-Za-z]+)_', filename)
        if location_match:
            location = location_match.group(1)
        else:
            location = 'Unknown'  # Default value if no location found
        
        # Add the location column to the DataFrame
        df['Location'] = location
        
        # Append the DataFrame to the list
        dfs.append(df)
#%% Concatenate all DataFrames in the list
combined_df = pd.concat(dfs, ignore_index=True)

#%% data exploration

combined_df.info()

#%%

combined_df.is_null().sum()

#%%

combined_df['order_date'] = pd.DatetimeIndex(combined_df['Created']).date
combined_df['order_date'] = pd.DatetimeIndex(combined_df['order_date'])
combined_df['order_time'] = pd.DatetimeIndex(combined_df['Created']).time

#%%
combined_df['order_time']=combined_df['order_time'].astype('string')
combined_df[['Hour','Minute', 'Second']]= combined_df['order_time'].str.split(":",expand=True)

#%%
combined_df["Hour"].value_counts()

#%%
# Create a list of hours in order
hour_order = list(range(11,24))  # This creates a list from 0 to 23

sns.countplot(data=combined_df,x="Hour",palette="plasma", order=hour_order)
plt.xticks(rotation=90)
plt.xlabel("Hour",fontsize=10,color="purple")
plt.ylabel("Frequency",fontsize=10,color="purple")
plt.title("Order by hour",color="purple")
plt.show()

#%%
combined_df['order_dayOfWeek'] = combined_df['order_date'].dt.day_name()
combined_df['order_dayOfWeek'].value_counts()

#%% Chart of distribution of order based on day of week

# Define the order of days
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create the countplot with ordered days
sns.countplot(data=combined_df, x="order_dayOfWeek", palette="viridis", order=day_order)

plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees for better readability
plt.xlabel("Order by day of week", fontsize=10, color="green")
plt.ylabel("Frequency", fontsize=10, color="green")
plt.title("DAYS OF WEEK", color="green")
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()

#%%

combined_df['order_month'] =pd.DatetimeIndex (combined_df['order_date']).month
combined_df.loc[(combined_df['order_month'] ==1), 'order_month'] = 'January'
combined_df.loc[(combined_df['order_month'] ==2), 'order_month'] = 'February'
combined_df.loc[(combined_df['order_month'] ==3), 'order_month'] = 'March'
combined_df.loc[(combined_df['order_month'] ==4), 'order_month'] = 'April'
combined_df.loc[(combined_df['order_month'] ==5), 'order_month'] = 'May'
combined_df.loc[(combined_df['order_month'] ==6), 'order_month'] = 'June'
combined_df.loc[(combined_df['order_month'] ==7), 'order_month'] = 'July'
combined_df.loc[(combined_df['order_month'] ==8), 'order_month'] = 'August'
combined_df.loc[(combined_df['order_month'] ==9), 'order_month'] = 'September'
combined_df.loc[(combined_df['order_month'] ==10), 'order_month'] = 'October'
combined_df.loc[(combined_df['order_month'] ==11), 'order_month'] = 'November'
combined_df.loc[(combined_df['order_month'] ==12), 'order_month'] = 'December'
combined_df['order_month'].value_counts()

#%% Plotting order quantity month wise

# Define the order of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

# Create the countplot with ordered months
sns.countplot(data=combined_df, x="order_month", palette="CMRmap", order=month_order)

plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees for better readability
plt.xlabel("Months", fontsize=10, color="black")
plt.ylabel("Frequency", fontsize=10, color="black")
plt.title("MONTHS", color="black")
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()

#%%

combined_df_copy = combined_df.copy()

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

# Apply the function to each row and concatenate the results
expanded_df = pd.concat(combined_df_copy.apply(split_items, axis=1).tolist(), ignore_index=True)

# Reset the index if needed
expanded_df = expanded_df.reset_index(drop=True)

#%%

unique_items = expanded_df['Item'].unique()

#%%

# Specify the file path where you want to save the Excel file
output_path = 'expanded_order_data.xlsx'

# Save the DataFrame to Excel
expanded_df.to_excel(output_path, index=False, engine='openpyxl')

print(f"DataFrame has been saved to {output_path}")


#%%

expanded_df.columns

#%%

print(unique_items[0:50])

#%%

item_val_count = expanded_df['Item'].value_counts()

#%%

print(item_val_count[0:50])

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
unique_main_items = expanded_df['Item'].dropna().apply(extract_main_item).unique()

#%%
# Apply the standardization with progress bar
tqdm.pandas(desc="Standardizing Items")
expanded_df['Standardized_Item'] = expanded_df['Item'].progress_apply(lambda x: standardize_item(x, unique_main_items))
#%%
# Extract size information, handling null values
tqdm.pandas(desc="Extracting Sizes")
expanded_df['Size'] = expanded_df['Item'].progress_apply(lambda x: re.search(r'\((.*?)\)', str(x)).group(1) if pd.notna(x) and '(' in str(x) else None)

