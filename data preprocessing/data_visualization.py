# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:50:38 2024

@author: gourav
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

#%%
# 1. Time Series Plot of Daily Total Quantity
plt.figure(figsize=(30, 6))
grouped_df.groupby('Date')['Quantity'].sum().plot()
plt.title('Daily Total Quantity Over Time')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.show()
#%%
# 2. Box Plot of Quantity by Day of Week
plt.figure(figsize=(10, 6))
sns.boxplot(x='DayOfWeek', y='Quantity', data=grouped_df)
plt.title('Distribution of Quantity by Day of Week')
plt.show()
#%%
# 3. Heatmap of Average Quantity by Month and Day of Week
pivot_df = grouped_df.pivot_table(values='Quantity', index='Month', columns='DayOfWeek', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, cmap='YlOrRd')
plt.title('Average Quantity by Month and Day of Week')
plt.show()
#%%
# 4. Bar Plot of Total Quantity by Category
plt.figure(figsize=(12, 6))
grouped_df.groupby('Category')['Quantity'].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Total Quantity by Category')
plt.xlabel('Category')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45)
plt.show()
#%%
# 5. Scatter Plot of Quantity vs. DaysSinceLastSale
plt.figure(figsize=(10, 6))
plt.scatter(grouped_df['DaysSinceLastSale'], grouped_df['Quantity'], alpha=0.5)
plt.title('Quantity vs. Days Since Last Sale')
plt.xlabel('Days Since Last Sale')
plt.ylabel('Quantity')
plt.show()
#%%
# 6. Line Plot of Rolling Averages
plt.figure(figsize=(30, 6))
grouped_df.groupby('Date')['Quantity'].sum().rolling(window=7).mean().plot(label='7-day MA')
grouped_df.groupby('Date')['Quantity'].sum().rolling(window=30).mean().plot(label='30-day MA')
grouped_df.groupby('Date')['Quantity'].sum().rolling(window=90).mean().plot(label='90-day MA')
plt.title('Rolling Averages of Daily Total Quantity')
plt.xlabel('Date')
plt.ylabel('Average Quantity')
plt.legend()
plt.show()
#%%
# 7. Stacked Area Chart of Quantity by Category Over Time
category_time_series = grouped_df.groupby(['Date', 'Category'])['Quantity'].sum().unstack()
category_time_series.plot(kind='area', stacked=True, figsize=(30, 8))
plt.title('Quantity by Category Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
#%%
# 8. Correlation Heatmap
numeric_cols = grouped_df.select_dtypes(include=[np.number]).columns
correlation_matrix = grouped_df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()
#%%
# 9. Seasonal Decomposition Plot
from statsmodels.tsa.seasonal import seasonal_decompose

# Aggregate data to daily level
daily_data = grouped_df.groupby('Date')['Quantity'].sum().resample('D').sum()

# Perform seasonal decomposition
result = seasonal_decompose(daily_data, model='additive', period=7)  # Assuming weekly seasonality

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(60, 20))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
#%%
# 10. Interactive Time Series with Plotly
fig = px.line(grouped_df.groupby('Date')['Quantity'].sum().reset_index(), 
              x='Date', y='Quantity', title='Interactive Daily Total Quantity Over Time')
fig.show()
#%%
# 11. Bubble Chart: Quantity, ItemPopularity, and ExpMovingAverage
fig = px.scatter(grouped_df, x='ItemPopularity', y='Quantity', 
                 size='ExpMovingAverage', color='Category', 
                 hover_name='Standardized_Item', log_x=True, size_max=60,
                 title='Quantity vs Item Popularity, sized by Exp Moving Average')
fig.show()
#%%
# 12. Parallel Coordinates Plot
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
