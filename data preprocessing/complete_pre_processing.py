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

def standardize_items(df):
    # Create a dictionary to store Standardized_Items, sizes, categories, and special requests
    standard_dict = {}
    
    # Regular expressions for extracting size information
    size_patterns = {
        'Half Kilo': r'\bHalf Kilo\b|\bHalf Kg\b|\(Serves 3-4\)|\(Serves 3 - 4\)|\s-\s*Half\s*Kilo(?:\s*-\s*Serves\s*3-4)?\b',
        'Kilo': r'\bKilo\b|\(Serves 5-6\)|\(Serves 5 -6\)|\s-\s*Kilo(?:\s*-\s*Serves\s*5-6)?\b',
        'Small': r'\bSmall\b|\s-\s*Small(?:\s*-\s*Serves\s*\d+(?:-\d+)?)?\b',
        'Regular': r'\bRegular\b|\(Serves 1\)|\(Serves - 1\)|\s-\s*Regular(?:\s*-\s*Serves\s*1)?\b',
        'Medium': r'\bMedium\b|\(Serves 1-2\)|\(Serves 1 -2\)|\s-\s*Medium\s*-\s*Serves\s*1-2\b',
        'Large': r'\bLarge\b|\(Serves 2-3\)|\(Serves 2 -3\)|\s-\s*Large\s*-\s*Serves\s*2-3\b',
        'Half': r'\bHalf\b|\(Half\)|\[500 Ml\]|\s-\s*Half(?:\s*-\s*500\s*Ml)?\b',
        'Full': r'\bFull\b|\[650 Ml\]|\s-\s*Full(?:\s*-\s*650\s*Ml)?\b'
    }
    
    def extract_size(name):
        for size, pattern in size_patterns.items():
            if re.search(pattern, name, re.IGNORECASE):
                return size
        return 'Regular'  # Default size if not specified
    
    # Function to determine category
    def determine_category(name):
        if 'combo' in name.lower() or 'meal' in name.lower():
            return 'Combo Meals'
        elif any(word in name.lower() for word in ['biryani', 'rice', 'pulao']):
            return 'Biryani & Rice'
        elif any(word in name.lower() for word in ['chicken', 'mutton', 'prawns', 'egg', 'murg', 'gosht', 'keema']):
            return 'Non-Veg Main Course'
        elif any(word in name.lower() for word in ['paneer', 'veg', 'gobi', 'aloo', 'dal']):
            return 'Veg Main Course'
        elif any(word in name.lower() for word in ['naan', 'roti', 'bread', 'paratha']):
            return 'Breads'
        elif any(word in name.lower() for word in ['soup', 'salad', 'kebab', 'tikka', 'starter']):
            return 'Starters'
        elif any(word in name.lower() for word in ['water', 'drink', 'soda', 'chaas', 'buttermilk', 'lassi']):
            return 'Beverages'
        elif any(word in name.lower() for word in ['pudding', 'caramel', 'custard']):
            return 'Desserts'
        else:
            return 'Others'
    
    # Function to extract special requests
    def extract_special_requests(name):
        special_requests = []
        patterns = [
            r'Medium Spicy',
            r'Less Spicy',
            r'Spicy',
            r'Boneless',
            r'Red',
            r'White',
            r'Chef Special',
            r'Extra Cheese',
            r'Extra Sauce',
            r'Off'
        ]
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                special_requests.append(pattern)
        return ', '.join(special_requests) if special_requests else None

    # Function to clean and standardize item names
    def clean_name(item):
        # Remove size information and special requests
        clean = re.sub(r'\(.*?\)|\[.*?\]|\s-\s*(?:Small|Medium|Large|Regular|Half Kilo|Half Kg|Kilo|Half|Full)(?:\s*-\s*Serves\s*\d+(?:-\d+)?)?', '', item)
        for request in (extract_special_requests(item).split(', ') if extract_special_requests(item) else []):
            clean = clean.replace(request, '')
        
        # Remove any remaining parentheses and brackets
        clean = re.sub(r'[(){}\[\]]', '', clean)
        clean = re.sub(r'\s*-\s*', ' ', clean)
        
        # Remove extra spaces and dashes
        clean = re.sub(r'\s+', ' ', clean)
        clean = clean.strip()
        
        # Convert to title case for consistency
        clean = clean.title()
        
        return clean

    # First pass: Clean all names
    cleaned_items = {item: clean_name(item) for item in df['Item'].unique()}

    # Second pass: Apply fuzzy matching
    unique_cleaned_items = list(set(cleaned_items.values()))
    for item in tqdm(df['Item'].unique(), desc="Applying fuzzy matching"):
        cleaned_item = cleaned_items[item]
        best_match = max(unique_cleaned_items, key=lambda x: fuzz.ratio(cleaned_item, x))
        if fuzz.ratio(cleaned_item, best_match) >= 80:  # Adjust this threshold as needed
            standard_dict[item] = {
                'Standardized_Item': best_match,
                'Size': extract_size(item),
                'Category': determine_category(best_match),
                'Special Requests': extract_special_requests(item)
            }
        else:
            standard_dict[item] = {
                'Standardized_Item': cleaned_item,
                'Size': extract_size(item),
                'Category': determine_category(cleaned_item),
                'Special Requests': extract_special_requests(item)
            }
    
    # Apply standardization to the DataFrame
    df['Standardized_Item'] = df['Item'].map(lambda x: standard_dict[x]['Standardized_Item'])
    df['Size'] = df['Item'].map(lambda x: standard_dict[x]['Size'])
    df['Category'] = df['Item'].map(lambda x: standard_dict[x]['Category'])
    df['Special Requests'] = df['Item'].map(lambda x: standard_dict[x]['Special Requests'])
    
    # Manual corrections for common items
    manual_corrections = {
        'Caramel Custerd': 'Caramel Custard',
        'Prawns Hyderabdi Dum Biryani': 'Prawns Hyderabadi Dum Biryani'
        # Add more corrections here
    }
    
    df['Standardized_Item'] = df['Standardized_Item'].replace(manual_corrections)
    
    return df

def process_data(df):
    df = standardize_items(df)
    return df

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
#expanded_df = expanded_df.reset_index(drop=True)'

#%% Normalizing the items name and seperating the serving Sizes as well as Category 

expanded_df = process_data(expanded_df)

#%% Data Visualization
data = expanded_df['Standardized_Item'].value_counts()

# Set up the figure size
plt.figure(figsize=(12, 15))

# Create the horizontal bar plot for top 20 categories
sns.barplot(x=data.values[:30], y=data.index[:30], palette='viridis')

# Customize the plot
plt.title('Top 30 Standardized Items', fontsize=16)
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
# Convert 'Created' to datetime
expanded_df['Created'] = pd.to_datetime(expanded_df['Created'])

# Apply the time grouping function
expanded_df['TimeGroup'] = expanded_df['Created'].apply(get_time_group)

# Extract date
expanded_df['Date'] = expanded_df['Created'].dt.date

# Ensure 'Quantity' column exists
if 'Quantity' not in expanded_df.columns:
    expanded_df['Quantity'] = 1
#%%
# Group by date, time_group, Standardized_Item, Size, and Category, then sum the quantities
grouped_df = expanded_df.groupby(['Date', 'TimeGroup', 'Standardized_Item', 'Size', 'Category'])['Quantity'].sum().reset_index()

# Sort the resulting dataframe
grouped_df = grouped_df.sort_values(['Date', 'TimeGroup', 'Standardized_Item', 'Size', 'Category'])

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

#grouped_df['IsStandardSize'] = grouped_df['SizeCategory'].isin(['Small', 'Medium', 'Large']).astype(int)

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

# Create a copy of grouped_df for scaling
scaled_df = grouped_df.copy()

# Frequency Encoding
for col in ['Standardized_Item', 'Size']:
    frequency = grouped_df[col].value_counts(normalize=True)
    scaled_df[f'{col}_Frequency'] = scaled_df[col].map(frequency)

# Target Encoding (we already have this for 'Standardized_Item')
def target_encode(df, column, target):
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = 1 / (1 + np.exp(-(counts - 10) / 10))
    return smooth * means + (1 - smooth) * global_mean

scaled_df['ItemTargetEncode'] = scaled_df['Standardized_Item'].map(target_encode(scaled_df, 'Standardized_Item', 'Quantity'))

# One-hot encoding only for low-cardinality categorical variables
scaled_df = pd.get_dummies(scaled_df, columns=['Season', 'TimeGroup'], 
                    prefix=['Season', 'TimeGroup'])

# Handle missing values
scaled_df = scaled_df.fillna(method='ffill').fillna(method='bfill')

# Normalize numerical features
numerical_columns = scaled_df.select_dtypes(include=[np.number]).columns
scaled_df[numerical_columns] = scaled_df[numerical_columns].apply(zscore)

print("Columns in scaled_df:")
print(scaled_df.columns)

print("\nColumns in grouped_df:")
print(grouped_df.columns)