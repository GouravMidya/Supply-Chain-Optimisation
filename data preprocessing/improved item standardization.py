# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 19:50:14 2024

@author: goura
"""
main_df = expanded_df['Item'].to_frame()

#%%
import pandas as pd
import re
from fuzzywuzzy import fuzz

def standardize_items(df):
    # Create a dictionary to store standardized names, sizes, categories, and special requests
    standard_dict = {}
    
    # Regular expressions for extracting size information
    size_patterns = {
        'Small': r'\bSmall\b|\s-\s*Small(?:\s*-\s*Serves\s*\d+(?:-\d+)?)?\b',
        'Regular': r'\bRegular\b|\(Serves 1\)|\(Serves - 1\)|\s-\s*Regular(?:\s*-\s*Serves\s*1)?\b',
        'Medium': r'\bMedium\b|\(Serves 1-2\)|\(Serves 1 -2\)|\s-\s*Medium\s*-\s*Serves\s*1-2\b',
        'Large': r'\bLarge\b|\(Serves 2-3\)|\(Serves 2 -3\)|\s-\s*Large\s*-\s*Serves\s*2-3\b',
        'Half': r'\bHalf\b|\(Half\)|\[500 Ml\]|\s-\s*Half(?:\s*-\s*500\s*Ml)?\b',
        'Full': r'\bFull\b|\[650 Ml\]|\s-\s*Full(?:\s*-\s*650\s*Ml)?\b',
        'Half Kilo': r'\bHalf Kilo\b|\bHalf Kg\b|\(Serves 3-4\)|\(Serves 3 - 4\)|\s-\s*Half\s*Kilo(?:\s*-\s*Serves\s*3-4)?\b',
        'Kilo': r'\bKilo\b|\(Serves 5-6\)|\(Serves 5 -6\)|\s-\s*Kilo(?:\s*-\s*Serves\s*5-6)?\b'
    }
    
    # Function to extract size from item name
    def extract_size(name):
        for size, pattern in size_patterns.items():
            if re.search(pattern, name, re.IGNORECASE):
                return size
        return 'Regular'  # Default size if not specified
    
    # Function to determine category
    def determine_category(name):
        if 'combo' in name.lower() or 'meal' in name.lower():
            return 'Combo Meals'
        elif any(word in name.lower() for word in ['biryani', 'rice']):
            return 'Biryani & Rice'
        elif any(word in name.lower() for word in ['chicken', 'mutton', 'prawns', 'egg', 'murg']):
            return 'Non-Veg Main Course'
        elif any(word in name.lower() for word in ['paneer', 'veg', 'gobi']):
            return 'Veg Main Course'
        elif any(word in name.lower() for word in ['naan', 'roti', 'bread']):
            return 'Breads'
        elif any(word in name.lower() for word in ['soup', 'salad']):
            return 'Starters'
        elif any(word in name.lower() for word in ['water', 'drink', 'soda', 'chaas', 'buttermilk']):
            return 'Beverages'
        elif any(word in name.lower() for word in ['pudding', 'caramel', 'custard']):
            return 'Deserts'
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
            r'Chef Special'
        ]
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                special_requests.append(pattern)
        return ', '.join(special_requests) if special_requests else None

    # Iterate through the items and standardize them
    for item in df['Item'].unique():
        # Extract special requests
        special_requests = extract_special_requests(item)
        
        # Extract size
        size = extract_size(item)
        
        # Remove size information and special requests from the name
        clean_name = re.sub(r'\(.*?\)|\[.*?\]|\s-\s*(?:Small|Medium|Large|Regular|Half|Full|Half Kilo|Kilo)(?:\s*-\s*Serves\s*\d+(?:-\d+)?)?', '', item)
        for request in (special_requests.split(', ') if special_requests else []):
            clean_name = clean_name.replace(request, '')
        
        # Remove any remaining parentheses and brackets
        clean_name = re.sub(r'[(){}\[\]]', '', clean_name)
        clean_name = re.sub(r'\s*-\s*', ' ', clean_name)
        
        # Remove extra spaces and dashes
        clean_name = re.sub(r'\s+', ' ', clean_name)
        clean_name = clean_name.strip()
        
        # Determine category
        category = determine_category(clean_name)
        
        # Store the standardized information
        standard_dict[item] = {
            'Standardized Name': clean_name,
            'Size': size,
            'Category': category,
            'Special Requests': special_requests
        }
    
    # Apply standardization to the DataFrame
    df['Standardized Name'] = df['Item'].map(lambda x: standard_dict[x]['Standardized Name'])
    df['Size'] = df['Item'].map(lambda x: standard_dict[x]['Size'])
    df['Category'] = df['Item'].map(lambda x: standard_dict[x]['Category'])
    df['Special Requests'] = df['Item'].map(lambda x: standard_dict[x]['Special Requests'])
    
    return df

# Function to find similar items using fuzzy matching
def find_similar_items(df, threshold=80):
    unique_items = df['Standardized Name'].unique()
    similar_items = {}
    
    for item in unique_items:
        matches = []
        for other_item in unique_items:
            if item != other_item:
                similarity = fuzz.ratio(item.lower(), other_item.lower())
                if similarity >= threshold:
                    matches.append((other_item, similarity))
        
        if matches:
            similar_items[item] = sorted(matches, key=lambda x: x[1], reverse=True)
    
    return similar_items

# Function to correct typos based on item frequency and similarity
def correct_typos(df, min_count=5, similarity_threshold=90):
    item_counts = df['Standardized Name'].value_counts()
    low_count_items = item_counts[item_counts < min_count].index
    
    similar_items = find_similar_items(df[df['Standardized Name'].isin(low_count_items)])
    
    corrections = {}
    for item, matches in similar_items.items():
        if matches and matches[0][1] >= similarity_threshold:
            corrections[item] = matches[0][0]
    
    # Apply corrections
    df['Standardized Name'] = df['Standardized Name'].replace(corrections)
    
    return df

# Main processing function
def process_data(df):
    df = standardize_items(df)
    df = correct_typos(df)
    return df

# Assuming your main DataFrame is named 'main_df'
main_df = process_data(main_df)

#%%
data = main_df['Standardized Name'].value_counts()

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
