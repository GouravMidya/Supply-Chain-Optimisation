# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:07:29 2024

@author: goura
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
import time
#%%
# Assume scaled_df is your DataFrame
print("Preparing data...")

# Ensure Date is in datetime format and sort the dataframe
scaled_df['Date'] = pd.to_datetime(scaled_df['Date'])
scaled_df = scaled_df.sort_values('Date')

# Encode categorical variables
le_dict = {}
for col in tqdm(scaled_df.select_dtypes(include=['object']).columns, desc="Encoding categorical variables"):
    le = LabelEncoder()
    scaled_df[col] = le.fit_transform(scaled_df[col])
    le_dict[col] = le

# Prepare features and target
X = scaled_df.drop(['Date', 'Quantity'], axis=1)
y = scaled_df['Quantity']

# Split the data, ensuring that we're not using future data to predict past
train_size = int(len(scaled_df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
#%%
# Function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

# Function to train model with progress bar and hyperparameter tuning
def train_model_with_tuning(model, param_grid, X_train, y_train, model_name):
    print(f"\nTraining and tuning {model_name}...")
    start_time = time.time()
    
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, 
                                       n_iter=20, cv=tscv, verbose=1, n_jobs=-1, 
                                       scoring='neg_mean_squared_error')
    
    random_search.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"{model_name} training and tuning completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_
#%%
# Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gb_model = train_model_with_tuning(GradientBoostingRegressor(random_state=42), gb_param_grid, X_train, y_train, "Gradient Boosting")
#%%
# XGBoost
xgb_param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb_model = train_model_with_tuning(XGBRegressor(random_state=42), xgb_param_grid, X_train, y_train, "XGBoost")
#%%
# LightGBM
lgbm_param_grid = {
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
lgbm_model = train_model_with_tuning(LGBMRegressor(random_state=42), lgbm_param_grid, X_train, y_train, "LightGBM")
#%%
# Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = train_model_with_tuning(RandomForestRegressor(random_state=42), rf_param_grid, X_train, y_train, "Random Forest")
#%%
# Evaluate individual models
models = [xgb_model] # rf_model, gb_model, lgbm_model,
model_names = ["XGBoost"] #"LightGBM","Random Forest", "Gradient Boosting", 

for model, name in zip(models, model_names):
    pred = model.predict(X_test)
    evaluate_model(y_test, pred, name)
#%%
# Ensemble prediction
print("\nEnsemble Model Evaluation")
ensemble_pred = np.mean([model.predict(X_test) for model in models], axis=0)
evaluate_model(y_test, ensemble_pred, "Ensemble")

# Feature importance (using Random Forest as an example)
print("\nCalculating feature importance...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(feature_importance.head(10))
#%%
# Function to predict quantity for a given item on a future date
def predict_quantity(models, item, size, date, time_group):
    # Create a sample input
    sample = pd.DataFrame(index=[0])
    
    # Update the sample with the given information
    sample['Standardized_Item'] = le_dict['Standardized_Item'].transform([item])[0]
    sample['Size'] = le_dict['Size'].transform([size])[0]
    sample['Category'] = le_dict['Category'].transform([scaled_df[scaled_df['Standardized_Item'] == item]['Category'].iloc[0]])[0]
    sample['Year'] = date.year
    sample['Month'] = date.month
    sample['Day'] = date.day
    sample['DayOfWeek'] = date.weekday()
    sample['WeekOfYear'] = date.isocalendar()[1]
    sample['Quarter'] = (date.month - 1) // 3 + 1
    sample['IsWeekend'] = 1 if date.weekday() >= 5 else 0
    
    # Time group encoding
    for tg in ['11:00-15:00', '15:00-19:00', '19:00-23:00', '23:00-03:00']:
        sample[f'TimeGroup_{tg}'] = 1 if tg == time_group else 0
    
    # Fill in other features with the last known values for this item and size
    last_known = scaled_df[(scaled_df['Standardized_Item'] == item) & (scaled_df['Size'] == size)].iloc[-1]
    for col in X.columns:
        if col not in sample.columns:
            sample[col] = last_known[col]
    
    # Make predictions with all models
    predictions = [model.predict(sample)[0] for model in models]
    
    # Return the average prediction
    return np.mean(predictions)

# Example usage for Caramel Custard
print("\nMaking example prediction...")
example_date = pd.to_datetime('2024-07-20')
example_item = "Caramel Custard"
example_size = "Regular"
example_time_group = '15:00-19:00'

predicted_quantity = predict_quantity(models, example_item, example_size, example_date, example_time_group)
print(f"\nPredicted quantity for {example_item} ({example_size}) on {example_date.date()} during {example_time_group}: {predicted_quantity:.2f}")

#%%
# Batch predictions with progress bar
def batch_predict(models, num_predictions=1000):
    print(f"\nMaking {num_predictions} predictions...")
    predictions = []
    for _ in tqdm(range(num_predictions), desc="Batch predictions"):
        date = pd.to_datetime(np.random.choice(scaled_df['Date']))
        item = np.random.choice(scaled_df['Standardized_Item'])
        size = np.random.choice(scaled_df['Size'])
        time_group = np.random.choice(['11:00-15:00', '15:00-19:00', '19:00-23:00', '23:00-03:00'])
        pred = predict_quantity(models, item, size, date, time_group)
        predictions.append(pred)
    return predictions

# Run batch predictions
batch_predictions = batch_predict(models)
print(f"Average prediction: {np.mean(batch_predictions):.2f}")