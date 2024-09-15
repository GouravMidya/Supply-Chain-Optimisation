# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:18:05 2024

@author: goura
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#%%
# Assuming df is your DataFrame and features, target are defined

# Function to create sequences for LSTM/GRU
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Function to evaluate model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

# Encode categorical variables
le_season = LabelEncoder()
le_size = LabelEncoder()
le_timegroup = LabelEncoder()

df['Season_encoded'] = le_season.fit_transform(df['Season'])
df['Size_encoded'] = le_size.fit_transform(df['Size'])
df['TimeGroup_encoded'] = le_timegroup.fit_transform(df['TimeGroup'])

features = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'IsWeekend',
            'DaysSinceLastSale', 'RollingCount7Day', 'RollingCount30Day',
            'RollingCount90Day', 'PrevDayQuantity', 'PrevWeekQuantity', 'PrevMonthQuantity',
            'ItemPopularity', 'ExpMovingAverage', 'Season_encoded', 'Size_encoded', 'TimeGroup_encoded']

# Prepare data
X = df[features]
y = df['Quantity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}
#%%
# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Linear Regression'] = evaluate_model(y_test, y_pred_lr, "Linear Regression")
#%%
# 2. Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
results['Decision Tree'] = evaluate_model(y_test, y_pred_dt, "Decision Tree")

# 3. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
results['Random Forest'] = evaluate_model(y_test, y_pred_rf, "Random Forest")

# 4. XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
results['XGBoost'] = evaluate_model(y_test, y_pred_xgb, "XGBoost")

# 5. LSTM
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps=10)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps=10)

model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
y_pred_lstm = model_lstm.predict(X_test_seq)
results['LSTM'] = evaluate_model(y_test_seq, y_pred_lstm, "LSTM")

# 6. GRU
model_gru = Sequential([
    GRU(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(1)
])
model_gru.compile(optimizer='adam', loss='mse')
model_gru.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
y_pred_gru = model_gru.predict(X_test_seq)
results['GRU'] = evaluate_model(y_test_seq, y_pred_gru, "GRU")

# Comparative Analysis
#%%
# 1. Comparing model performance across different forecasting horizons
tscv = TimeSeriesSplit(n_splits=3)
horizons = [1, 7, 30]  # 1 day, 1 week, 1 month

for horizon in horizons:
    print(f"\nForecasting Horizon: {horizon} days")
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Use Random Forest as an example
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test[:horizon])
        
        mae = mean_absolute_error(y_test[:horizon], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[:horizon], y_pred))
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# 2. Analyzing the impact of data stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

print("\nStationarity Test for 'Quantity':")
check_stationarity(df['Quantity'])

# 3. Plotting results
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