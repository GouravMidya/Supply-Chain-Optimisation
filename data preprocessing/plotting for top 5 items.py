# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:21:41 2024

@author: goura
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

# Set Plotly to open in browser
pio.renderers.default = "browser"

# Assuming your DataFrame is named 'grouped_df'
# If not, replace 'grouped_df' with your actual DataFrame name

# Get top 5 items by total quantity
top_5_items = grouped_df.groupby('Standardized_Item')['Quantity'].sum().nlargest(5).index

# 1. Time Series Plot for Top 5 Items
plt.figure(figsize=(15, 10))
for item in top_5_items:
    item_data = grouped_df[grouped_df['Standardized_Item'] == item]
    item_data.groupby('Date')['Quantity'].sum().plot(label=item)
plt.title('Daily Total Quantity for Top 5 Items')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.legend()
plt.show()

# 2. Box Plot of Quantity for Top 5 Items
plt.figure(figsize=(12, 6))
sns.boxplot(x='Standardized_Item', y='Quantity', data=grouped_df[grouped_df['Standardized_Item'].isin(top_5_items)])
plt.title('Distribution of Quantity for Top 5 Items')
plt.xticks(rotation=45)
plt.show()

# 3. Bar Plot of Total Quantity for Top 5 Items
plt.figure(figsize=(12, 6))
grouped_df[grouped_df['Standardized_Item'].isin(top_5_items)].groupby('Standardized_Item')['Quantity'].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Total Quantity for Top 5 Items')
plt.xlabel('Standardized Item')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45)
plt.show()

# 4. Heatmap of Average Quantity by Month and Day of Week for each Top 5 Item
for item in top_5_items:
    item_data = grouped_df[grouped_df['Standardized_Item'] == item]
    pivot_df = item_data.pivot_table(values='Quantity', index='Month', columns='DayOfWeek', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlOrRd')
    plt.title(f'Average Quantity by Month and Day of Week for {item}')
    plt.show()

# 5. Interactive Time Series with Plotly (modified to include top 5 items)
fig = px.line(grouped_df[grouped_df['Standardized_Item'].isin(top_5_items)].groupby(['Date', 'Standardized_Item'])['Quantity'].sum().reset_index(), 
              x='Date', y='Quantity', color='Standardized_Item', 
              title='Interactive Daily Total Quantity for Top 5 Items')
fig.show()

# 6. Bubble Chart: Quantity, ItemPopularity, and ExpMovingAverage (filtered for top 5 items)
fig = px.scatter(grouped_df[grouped_df['Standardized_Item'].isin(top_5_items)], 
                 x='ItemPopularity', y='Quantity', 
                 size='ExpMovingAverage', color='Standardized_Item', 
                 hover_name='Standardized_Item', log_x=True, size_max=60,
                 title='Quantity vs Item Popularity for Top 5 Items, sized by Exp Moving Average')
fig.show()

# 7. Alternative to Parallel Coordinates Plot using Matplotlib
from pandas.plotting import parallel_coordinates

plt.figure(figsize=(15, 10))
parallel_coordinates(grouped_df[grouped_df['Standardized_Item'].isin(top_5_items)].sample(1000), 
                     'Standardized_Item', 
                     cols=['Quantity', 'ItemPopularity', 'DaysSinceLastSale', 'RollingCount7Day', 'ExpMovingAverage'])
plt.title('Parallel Coordinates Plot for Top 5 Items')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()