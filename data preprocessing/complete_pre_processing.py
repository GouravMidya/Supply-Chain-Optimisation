# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:05:12 2024

@author: gourav
"""

#%% Imports

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import re
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import os
from scipy.stats import zscore

#%%Functions decleration
#Importing Data from a folder
def combine_excel_files(folder_path):
    # Initialize an empty list to store individual DataFrames
    dfs = []
    
    # Get the list of Excel files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]
    
    # Iterate through all Excel files in the folder with a progress bar
    for filename in tqdm(excel_files, desc="Processing Excel files"):
        file_path = os.path.join(folder_path, filename)
        
        # Read the Excel file, specifying the header row
        df = pd.read_excel(file_path, header=4)  # 0-based index, so 4 means 5th row
        
        # Append the DataFrame to the list
        dfs.append(df)
    
    # Concatenate all DataFrames in the list
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

#Splitting multiple items in same cell to 1 item per cell
def split_items(row):
    if pd.isna(row['Items']):
        # Return a DataFrame with a single row, setting 'Item' to NaN or a placeholder
        return pd.DataFrame([row.drop('Items').tolist() + [np.nan]], 
                            columns=row.drop('Items').index.tolist() + ['Item'])
    else:
        items = str(row['Items']).split(',')
        return pd.DataFrame([row.drop('Items').tolist() + [item.strip()] for item in items], 
                            columns=row.drop('Items').index.tolist() + ['Item'])

# Create a cache
item_cache = {}

#Split the Item name apart from the Serving Sizes and extras
def extract_main_item(item_name):
    if pd.isna(item_name):
        return "Unknown Item"
    return re.split(r'\s*[\(\-]', str(item_name))[0].strip()

#Finding similar names and clubbing them together
def standardize_items(items, unique_main_items, threshold=80, batch_size=1000):
    def standardize_batch(batch):
        results = []
        for item in batch:
            if pd.isna(item):
                results.append("Unknown Item")
            elif item in item_cache:
                results.append(item_cache[item])
            else:
                main_item = extract_main_item(str(item))
                match = process.extractOne(main_item, unique_main_items)
                if match[1] >= threshold:
                    result = match[0]
                else:
                    result = main_item
                item_cache[item] = result
                results.append(result)
        return results

    results = []
    for i in tqdm(range(0, len(items), batch_size), desc="Standardizing Items"):
        batch = items[i:i+batch_size]
        results.extend(standardize_batch(batch))
    
    return results

# Define a function to convert to datetime
def convert_to_datetime(x):
    return pd.to_datetime(x, format='mixed', errors='coerce')

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

def categorize_size(size):
    if pd.isna(size):
        return 'Unknown'
    size = str(size).lower()
    if 'small' in size or 's' == size:
        return 'Small'
    elif 'medium' in size or 'm' == size:
        return 'Medium'
    elif 'large' in size or 'l' == size:
        return 'Large'
    else:
        return 'Other'

def get_indian_season(month):
    if 3 <= month <= 5:
        return 'Summer'
    elif 6 <= month <= 9:
        return 'Monsoon'
    elif 10 <= month <= 11:
        return 'Post-Monsoon'
    else:  # 12, 1, 2
        return 'Winter'

#%% Preprocessing
# Importing and combining all excel files
folder_path = 'C:/Users/goura/Documents/Supply Chain Optimisation/persiana order data'
combined_df = combine_excel_files(folder_path)

#%%
#Selecting Required Columns as features
# List the columns you want to keep
columns_to_keep = [
    'Items', 
    'Created'
]

# Create a new DataFrame with only the selected columns
filtered_df = combined_df[columns_to_keep].dropna(subset=['Items'])

filtered_df.info()

#%%Explode the items
tqdm.pandas(desc="Splitting Items")
# Apply the function to each row with progress bar and concatenate the results
expanded_df = pd.concat(filtered_df.progress_apply(split_items, axis=1).tolist(), ignore_index=True)
# Reset the index if needed
#expanded_df = expanded_df.reset_index(drop=True)

#%% Normalizing the items name and seperating the serving Sizes
# Get unique main items once to avoid recalculating for each row
unique_main_items = expanded_df['Item'].apply(extract_main_item).unique()

# Apply the standardization
expanded_df['Standardized_Item'] = standardize_items(expanded_df['Item'].values, unique_main_items)


#%% Extract size information, handling null values
tqdm.pandas(desc="Extracting Sizes")
expanded_df['Size'] = expanded_df['Item'].progress_apply(lambda x: re.search(r'\((.*?)\)', str(x)).group(1) if pd.notna(x) and '(' in str(x) else None)

#%%

data = expanded_df['Standardized_Item'].value_counts()

# Set up the figure size
plt.figure(figsize=(12, 15))

# Create the horizontal bar plot for top 20 categories
sns.barplot(x=data.values[:30], y=data.index[:30], palette='viridis')

# Customize the plot
plt.title('Top 20 Standardized Items', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Standardized Item', fontsize=12)

# Add value labels to the end of each bar
for i, v in enumerate(data.values[:30]):
    plt.text(v + 0.5, i, str(v), va='center')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Print some additional information
print(f"Total number of unique categories: {len(data)}")
print(f"Count range: {data.min()} to {data.max()}")
#%%
# Manual corrections for common items
manual_corrections = {
    'Caramel Custerd': 'Caramel Custard',
    # Add more corrections here
}

expanded_df['Standardized_Item'] = expanded_df['Standardized_Item'].replace(manual_corrections)


#%%
# Convert 'Created' to datetime
expanded_df['Created'] = pd.to_datetime(expanded_df['Created'])

# Apply the time grouping function
expanded_df['TimeGroup'] = expanded_df['Created'].apply(get_time_group)

# Extract date
expanded_df['Date'] = expanded_df['Created'].dt.date

# Ensure 'Quantity' column exists
if 'Quantity' not in expanded_df.columns:
    expanded_df['Quantity'] = 1

# Handle NaN values in the 'Size' column
expanded_df['Size'] = expanded_df['Size'].fillna('Unknown')

# Group by date, time_group, Standardized_Item, and Size, then sum the quantities
grouped_df = expanded_df.groupby(['Date', 'TimeGroup', 'Standardized_Item', 'Size'])['Quantity'].sum().reset_index()

# Sort the resulting dataframe
grouped_df = grouped_df.sort_values(['Date', 'TimeGroup', 'Standardized_Item', 'Size'])

# Convert 'Date' to datetime
grouped_df['Date'] = pd.to_datetime(grouped_df['Date'])
#%% Now continue with the rest of the feature engineering on the grouped_df

# 1. Date-related features
grouped_df['Year'] = pd.to_datetime(grouped_df['Date']).dt.year
grouped_df['Month'] = pd.to_datetime(grouped_df['Date']).dt.month
grouped_df['Day'] = pd.to_datetime(grouped_df['Date']).dt.day
grouped_df['DayOfWeek'] = pd.to_datetime(grouped_df['Date']).dt.dayofweek
grouped_df['WeekOfYear'] = pd.to_datetime(grouped_df['Date']).dt.isocalendar().week
grouped_df['Quarter'] = pd.to_datetime(grouped_df['Date']).dt.quarter
grouped_df['IsWeekend'] = pd.to_datetime(grouped_df['Date']).dt.dayofweek.isin([5, 6]).astype(int)

#%% 2. Time-based features
grouped_df['DaysSinceFirstSale'] = (grouped_df['Date'] - grouped_df['Date'].min()).dt.days
grouped_df['DaysSinceLastSale'] = grouped_df.groupby(['Standardized_Item', 'Size'])['Date'].diff().dt.days

#%% Rolling features
def rolling_sum(group, window):
    return group.rolling(window=window, min_periods=1).sum()

grouped_df['RollingCount7Day'] = grouped_df.groupby(['Standardized_Item', 'Size'])['Quantity'].transform(lambda x: rolling_sum(x, 7))
grouped_df['RollingCount30Day'] = grouped_df.groupby(['Standardized_Item', 'Size'])['Quantity'].transform(lambda x: rolling_sum(x, 30))
grouped_df['RollingCount90Day'] = grouped_df.groupby(['Standardized_Item', 'Size'])['Quantity'].transform(lambda x: rolling_sum(x, 90))
#%% 3. Lag features
grouped_df['PrevDayQuantity'] = grouped_df.groupby(['Standardized_Item', 'Size', 'TimeGroup'])['Quantity'].shift(1)
grouped_df['PrevWeekQuantity'] = grouped_df.groupby(['Standardized_Item', 'Size', 'TimeGroup'])['Quantity'].shift(7)
grouped_df['PrevMonthQuantity'] = grouped_df.groupby(['Standardized_Item', 'Size', 'TimeGroup'])['Quantity'].shift(30)

#%% 4. Item-related features
grouped_df['ItemPopularity'] = grouped_df.groupby(['Standardized_Item', 'Size'])['Quantity'].transform('sum')

#%% 5. Size-related features

grouped_df['SizeCategory'] = grouped_df['Size'].apply(categorize_size)
grouped_df['IsStandardSize'] = grouped_df['SizeCategory'].isin(['Small', 'Medium', 'Large']).astype(int)

#%% 6. Seasonality features
grouped_df['Season'] = grouped_df['Month'].apply(get_indian_season)
#%% 7. Trend features
grouped_df['CumulativeSales'] = grouped_df.groupby(['Standardized_Item', 'Size']).cumcount() + 1

#%% Exponential moving average
grouped_df['ExpMovingAverage'] = grouped_df.groupby(['Standardized_Item', 'Size'])['Quantity'].transform(lambda x: x.ewm(span=30).mean())

#%% 9. Interaction features
grouped_df['ItemMonth'] = grouped_df['Standardized_Item'] + '_' + grouped_df['Month'].astype(str)
grouped_df['ItemDayOfWeek'] = grouped_df['Standardized_Item'] + '_' + grouped_df['DayOfWeek'].astype(str)
grouped_df['SizeSeason'] = grouped_df['Size'] + '_' + grouped_df['Season'].astype(str)
grouped_df['ItemTimeGroup'] = grouped_df['Standardized_Item'] + '_' + grouped_df['TimeGroup']

#%% 10. Encoding features

# Frequency Encoding
for col in ['Standardized_Item', 'Size']:
    frequency = grouped_df[col].value_counts(normalize=True)
    grouped_df[f'{col}_Frequency'] = grouped_df[col].map(frequency)

# Target Encoding (we already have this for 'Standardized_Item')
def target_encode(df, column, target):
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = 1 / (1 + np.exp(-(counts - 10) / 10))
    return smooth * means + (1 - smooth) * global_mean

grouped_df['ItemTargetEncode'] = grouped_df['Standardized_Item'].map(target_encode(grouped_df, 'Standardized_Item', 'Quantity'))

# One-hot encoding only for low-cardinality categorical variables
grouped_df = pd.get_dummies(grouped_df, columns=['Season', 'TimeGroup'], 
                    prefix=['Season', 'TimeGroup'])

# ... (rest of the code remains the same)
# Handle missing values
grouped_df = grouped_df.fillna(method='ffill').fillna(method='bfill')

# Normalize numerical features
numerical_columns = grouped_df.select_dtypes(include=[np.number]).columns
grouped_df[numerical_columns] = grouped_df[numerical_columns].apply(zscore)

print(grouped_df.columns)