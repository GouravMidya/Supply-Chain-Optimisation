# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:01:35 2024

@author: goura
"""
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

#%%
# Sort the dataframe by date
grouped_df = grouped_df.sort_values('Date')

# Determine the split point (e.g., use the last 20% of the data for testing)
split_point = int(len(grouped_df) * 0.8)
train_df = grouped_df.iloc[:split_point]
test_df = grouped_df.iloc[split_point:]

print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")

# Define features and target
features = ['Date', 'Standardized_Item', 'Size', 'Year', 'Month', 'Day',
       'DayOfWeek', 'WeekOfYear', 'Quarter', 'IsWeekend', 'DaysSinceFirstSale',
       'DaysSinceLastSale', 'RollingCount7Day', 'RollingCount30Day',
       'RollingCount90Day', 'PrevDayQuantity', 'PrevWeekQuantity',
       'PrevMonthQuantity', 'ItemPopularity', 'CumulativeSales',
       'ExpMovingAverage', 'ItemMonth', 'ItemDayOfWeek', 'SizeSeason',
       'ItemTimeGroup', 'Standardized_Item_Frequency', 'Size_Frequency',
       'ItemTargetEncode', 'Season_Monsoon', 'Season_Post-Monsoon',
       'Season_Summer', 'Season_Winter', 'TimeGroup_11:00-15:00',
       'TimeGroup_15:00-19:00', 'TimeGroup_19:00-23:00',
       'TimeGroup_23:00-03:00']

target = 'Quantity'

# Prepare the data
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, rf_predictions)
mae = mean_absolute_error(y_test, rf_predictions)
r2 = r2_score(y_test, rf_predictions)

print("Random Forest Results:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
xgb_predictions = xgb_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, xgb_predictions)
mae = mean_absolute_error(y_test, xgb_predictions)
r2 = r2_score(y_test, xgb_predictions)

print("\nXGBoost Results:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))