# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:23:00 2024

@author: goura
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the data
# Assuming your DataFrame is named df
df = pd.read_csv('extracted_student_info.csv')

# Replace NaN values with 0
df.fillna(0, inplace=True)

# Drop rows where all sem results are 0
df = df.loc[~(df[['sem_3', 'sem_4', 'sem_5', 'sem_6']] == 0).all(axis=1)]

# Select features and target
X = df[['sem_3', 'sem_4', 'sem_5']]
y = df['sem_6']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = XGBRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Root Mean Squared Error: {rmse:.2f}')


#%%
single_row = pd.DataFrame({
    'sem_3': 7.3,
    'sem_4': 7.71,
    'sem_5': 7.09
})
#%%
single_row = df[82:83][['sem_3','sem_4','sem_5']]
#%%
# Predict using the single row
single_prediction = model.predict(single_row)
print(f'Predicted value for sem_6: {single_prediction[0]:.2f}')

#%%
# Create a DataFrame to show y_test alongside y_pred
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Optionally, you can reset the index to match them one by one
comparison_df.reset_index(drop=True, inplace=True)

# Display the comparison
print(comparison_df)