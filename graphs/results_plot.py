# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:41:15 2024

@author: goura
"""

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'results' is your dictionary with model names as keys and (MAE, RMSE) tuples as values

# Prepare data for plotting
models = list(results.keys())
mae_values = [results[model][0] for model in models]
rmse_values = [results[model][1] for model in models]

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Set the width of each bar and the positions of the bars
width = 0.35
x = np.arange(len(models))

# Create the bars
ax.bar(x - width/2, mae_values, width, label='MAE', color='skyblue')
ax.bar(x + width/2, rmse_values, width, label='RMSE', color='lightcoral')

# Customize the plot
ax.set_ylabel('Error')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

# Add value labels on top of each bar
for i, v in enumerate(mae_values):
    ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
for i, v in enumerate(rmse_values):
    ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()