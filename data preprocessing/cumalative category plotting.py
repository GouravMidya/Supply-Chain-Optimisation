# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:35:50 2024

@author: goura
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your DataFrame is named 'grouped_df'
# If not, replace 'grouped_df' with your actual DataFrame name

# Sort the DataFrame by Date
grouped_df = grouped_df.sort_values('Date')

# Group by Date and Category, sum the Quantity, and calculate cumulative sum
cumulative_df = grouped_df.groupby(['Date', 'Category'])['Quantity'].sum().unstack().fillna(0).cumsum()

# Create the plot
plt.figure(figsize=(15, 10))
for category in cumulative_df.columns:
    plt.plot(cumulative_df.index, cumulative_df[category], label=category, linewidth=2)

plt.title('Cumulative Quantity by Category Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Quantity', fontsize=12)
plt.legend(title='Category', title_fontsize='13', fontsize='11', loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add commas to y-axis labels for better readability of large numbers
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.show()

# If you prefer to use Seaborn for a slightly different aesthetic:
plt.figure(figsize=(15, 10))
sns.lineplot(data=cumulative_df)

plt.title('Cumulative Quantity by Category Over Time (Seaborn)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Quantity', fontsize=12)
plt.legend(title='Category', title_fontsize='13', fontsize='11', loc='center left', bbox_to_anchor=(1, 0.5))

plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.show()