# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:23:34 2024

@author: goura
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
#%%
# Preprocessing
def preprocess_data(df):
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Encode categorical variables
    le_season = LabelEncoder()
    le_size = LabelEncoder()
    le_timegroup = LabelEncoder()

    df['Season_encoded'] = le_season.fit_transform(df['Season'])
    df['Size_encoded'] = le_size.fit_transform(df['Size'])
    df['TimeGroup_encoded'] = le_timegroup.fit_transform(df['TimeGroup'])

    # Select features
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'IsWeekend',
                'DaysSinceLastSale', 'RollingCount7Day', 'RollingCount30Day',
                'RollingCount90Day', 'PrevDayQuantity', 'PrevWeekQuantity', 'PrevMonthQuantity',
                'ItemPopularity', 'ExpMovingAverage', 'Season_encoded', 'Size_encoded', 'TimeGroup_encoded']

    X = df[features]
    y = df['Quantity']

    # Handle missing values
    X = X.fillna(X.mean())

    return X, y, le_season, le_size, le_timegroup

# Split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Evaluate model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

# Create sequences for LSTM/GRU
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Prediction function
def predict_quantity(model, date, time_group, size, df, features, scaler, le_season, le_size, le_timegroup):
    input_date = pd.to_datetime(date)
    
    # Find the closest past date in the dataset
    closest_past_date = df[df['Date'] <= input_date]['Date'].max()
    
    # Get the row corresponding to the closest past date and matching time_group and size
    matching_rows = df[(df['Date'] == closest_past_date) & 
                       (df['TimeGroup'] == time_group) & 
                       (df['Size'] == size)]
    
    if matching_rows.empty:
        # If no exact match, find the closest match
        matching_rows = df[(df['Date'] == closest_past_date) & 
                           (df['TimeGroup'] == time_group)]
        if matching_rows.empty:
            matching_rows = df[df['Date'] == closest_past_date]
    
    input_row = matching_rows.iloc[0]
    
    # Create input features
    input_features = input_row[features].to_dict()
    
    # Update date-related features
    input_features['Year'] = input_date.year
    input_features['Month'] = input_date.month
    input_features['Day'] = input_date.day
    input_features['DayOfWeek'] = input_date.dayofweek
    input_features['WeekOfYear'] = input_date.isocalendar()[1]
    input_features['Quarter'] = (input_date.month - 1) // 3 + 1
    input_features['IsWeekend'] = 1 if input_date.dayofweek >= 5 else 0
    
    # Encode TimeGroup and Size
    input_features['TimeGroup_encoded'] = le_timegroup.transform([time_group])[0]
    input_features['Size_encoded'] = le_size.transform([size])[0]
    
    # Create input DataFrame
    input_data = pd.DataFrame([input_features])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    if isinstance(model, (Sequential)):  # For LSTM/GRU models
        input_data_scaled = input_data_scaled.reshape((1, 1, input_data_scaled.shape[1]))
    
    prediction = model.predict(input_data_scaled)
    
    return prediction[0]
#%%
# Main execution
X, y, le_season, le_size, le_timegroup = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
#%%
# Dictionary to store models and results
models = {}
results = {}
#%%
# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
models['Linear Regression'] = lr
results['Linear Regression'] = evaluate_model(y_test, y_pred_lr, "Linear Regression")
#%%
# 2. Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
models['Decision Tree'] = dt
results['Decision Tree'] = evaluate_model(y_test, y_pred_dt, "Decision Tree")
#%%
# 3. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
models['Random Forest'] = rf
results['Random Forest'] = evaluate_model(y_test, y_pred_rf, "Random Forest")
#%%
# 4. XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
models['XGBoost'] = xgb
results['XGBoost'] = evaluate_model(y_test, y_pred_xgb, "XGBoost")
#%%
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# 5. LSTM
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps=1)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps=1)
#%%
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
y_pred_lstm = model_lstm.predict(X_test_seq)
models['LSTM'] = model_lstm
results['LSTM'] = evaluate_model(y_test_seq, y_pred_lstm.flatten(), "LSTM")
#%%
# 6. GRU
model_gru = Sequential([
    GRU(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(1)
])
model_gru.compile(optimizer='adam', loss='mse')
model_gru.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
y_pred_gru = model_gru.predict(X_test_seq)
models['GRU'] = model_gru
results['GRU'] = evaluate_model(y_test_seq, y_pred_gru.flatten(), "GRU")
#%%
# Plotting results
plt.figure(figsize=(12, 6))
for model, (mae, rmse) in results.items():
    plt.bar(model, mae, alpha=0.8, label='MAE')
    plt.bar(model, rmse, alpha=0.5, label='RMSE')
plt.title('Model Comparison')
plt.ylabel('Error')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print best performing model
best_model = min(results, key=lambda x: results[x][0])  # Based on MAE
print(f"\nBest performing model: {best_model}")
print(f"MAE: {results[best_model][0]:.4f}, RMSE: {results[best_model][1]:.4f}")
#%%
# Example of using the predict_quantity function
date = '2023-09-06'
time_group = '19:00-23:00'
size = 'Regular'

for model_name, model in models.items():
    prediction = predict_quantity(model, date, time_group, size, df, X.columns, scaler, le_season, le_size, le_timegroup)
    print(f"{model_name} prediction for {date}, {time_group}, {size}: {prediction}")