# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:40:21 2024

@author: goura
"""

expanded_df['Standardized_Item'].value_counts()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assuming your DataFrame is named 'grouped_df'
# If not, replace 'grouped_df' with your actual DataFrame name

# Filter data for the two biryani items
biryani_items = ['Chicken Dum Biryani', 'Chicken Hyderabadi Dum Biryani']
biryani_df = grouped_df[grouped_df['Standardized_Item'].isin(biryani_items)]

# 1. Time Series Plot with Moving Average
plt.figure(figsize=(15, 8))
for item in biryani_items:
    item_data = biryani_df[biryani_df['Standardized_Item'] == item]
    daily_qty = item_data.groupby('Date')['Quantity'].sum()
    plt.plot(daily_qty.index, daily_qty.values, alpha=0.3, label=f'{item} (Raw)')
    # 7-day moving average
    ma7 = daily_qty.rolling(window=7).mean()
    plt.plot(ma7.index, ma7.values, linewidth=2, label=f'{item} (7-day MA)')

plt.title('Daily Quantity with 7-day Moving Average', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Quantity', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Smoothed Time Series using Savitzky-Golay filter
plt.figure(figsize=(15, 8))
for item in biryani_items:
    item_data = biryani_df[biryani_df['Standardized_Item'] == item]
    daily_qty = item_data.groupby('Date')['Quantity'].sum()
    # Apply Savitzky-Golay filter
    smooth_qty = savgol_filter(daily_qty.values, window_length=31, polyorder=3)
    plt.plot(daily_qty.index, smooth_qty, linewidth=2, label=item)

plt.title('Smoothed Daily Quantity (Savitzky-Golay Filter)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Smoothed Quantity', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Monthly Aggregated Bar Plot
monthly_data = biryani_df.groupby(['Standardized_Item', pd.Grouper(key='Date', freq='M')])['Quantity'].sum().unstack(level=0)
monthly_data.plot(kind='bar', figsize=(15, 8))
plt.title('Monthly Total Quantity for Biryani Items', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Quantity', fontsize=12)
plt.legend(title='Item', title_fontsize='12', fontsize='10', loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Heatmap of Daily Sales
pivot_df = biryani_df.pivot_table(values='Quantity', index=biryani_df['Date'].dt.dayofweek, 
                                  columns=biryani_df['Date'].dt.month, aggfunc='mean')
pivot_df.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
pivot_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlOrRd')
plt.title('Average Daily Sales by Month and Day of Week', fontsize=16)
plt.tight_layout()
plt.show()

# 5. Interactive Time Series with Plotly
fig = go.Figure()

for item in biryani_items:
    item_data = biryani_df[biryani_df['Standardized_Item'] == item]
    daily_qty = item_data.groupby('Date')['Quantity'].sum()
    # 7-day moving average
    ma7 = daily_qty.rolling(window=7).mean()
    
    fig.add_trace(go.Scatter(x=daily_qty.index, y=daily_qty.values, 
                             mode='lines', name=f'{item} (Raw)', 
                             line=dict(width=0.5, color='gray')))
    fig.add_trace(go.Scatter(x=ma7.index, y=ma7.values, 
                             mode='lines', name=f'{item} (7-day MA)', 
                             line=dict(width=2)))

fig.update_layout(title='Interactive Daily Quantity with 7-day Moving Average',
                  xaxis_title='Date',
                  yaxis_title='Quantity',
                  legend_title='Item',
                  hovermode="x unified")
fig.show()

# 6. Seasonal Decomposition Plot
from statsmodels.tsa.seasonal import seasonal_decompose

plt.figure(figsize=(15, 12))
for i, item in enumerate(biryani_items, 1):
    item_data = biryani_df[biryani_df['Standardized_Item'] == item]
    daily_qty = item_data.groupby('Date')['Quantity'].sum()
    
    # Resample to fill missing dates
    daily_qty = daily_qty.resample('D').sum().fillna(0)
    
    result = seasonal_decompose(daily_qty, model='additive', period=7)
    
    plt.subplot(2, 1, i)
    result.trend.plot(label='Trend')
    plt.title(f'Trend and Seasonality for {item}', fontsize=14)
    plt.legend()
    plt.tight_layout()

plt.show()