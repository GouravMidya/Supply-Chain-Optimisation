import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tqdm import tqdm
import time

# Assume scaled_df is your DataFrame

# Prepare the data
print("Preparing data...")
X = scaled_df.drop(['Date', 'Quantity'], axis=1)
y = scaled_df['Quantity']

# Encode categorical variables
le_dict = {}
for col in tqdm(X.select_dtypes(include=['object']).columns, desc="Encoding categorical variables"):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

# Function to train model with progress bar
def train_model_with_progress(model, X_train, y_train, model_name):
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"{model_name} training completed in {end_time - start_time:.2f} seconds")
    return model

# Linear Regression
lr_model = train_model_with_progress(LinearRegression(), X_train, y_train, "Linear Regression")
lr_pred = lr_model.predict(X_test)
evaluate_model(y_test, lr_pred, "Linear Regression")

# Random Forest
rf_model = train_model_with_progress(RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train, "Random Forest")
rf_pred = rf_model.predict(X_test)
evaluate_model(y_test, rf_pred, "Random Forest")

# XGBoost
xgb_model = train_model_with_progress(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), X_train, y_train, "XGBoost")
xgb_pred = xgb_model.predict(X_test)
evaluate_model(y_test, xgb_pred, "XGBoost")

# Feature importance for Random Forest
print("\nCalculating feature importance...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(feature_importance.head(10))

# Function to predict quantity for a given item on a future date
def predict_quantity(model, item, size, date, time_group):
    # Create a sample input
    sample = X.iloc[0].copy()  # Use the first row as a template
    
    # Update the sample with the given information
    sample['Standardized_Item'] = le_dict['Standardized_Item'].transform([item])[0]
    sample['Size'] = le_dict['Size'].transform([size])[0]
    sample['Category'] = le_dict['Category'].transform([sample['Category']])[0]  # Assuming category doesn't change
    sample['Year'] = date.year
    sample['Month'] = date.month
    sample['Day'] = date.day
    sample['DayOfWeek'] = date.weekday()
    sample['WeekOfYear'] = date.isocalendar()[1]
    sample['Quarter'] = (date.month - 1) // 3 + 1
    sample['IsWeekend'] = 1 if date.weekday() >= 5 else 0
    sample['ItemMonth'] = le_dict['ItemMonth'].transform([f"{item}_{date.month}"])[0]
    sample['ItemDayOfWeek'] = le_dict['ItemDayOfWeek'].transform([f"{item}_{date.weekday()}"])[0]
    sample['SizeSeason'] = le_dict['SizeSeason'].transform([f"{size}_{'Summer' if date.month in [3,4,5] else 'Monsoon' if date.month in [6,7,8,9] else 'Winter' if date.month in [11,12,1,2] else 'Post-Monsoon'}"])[0]
    sample['ItemTimeGroup'] = le_dict['ItemTimeGroup'].transform([f"{item}_{time_group}"])[0]
    
    # Set the TimeGroup
    for tg in ['11:00-15:00', '15:00-19:00', '19:00-23:00', '23:00-03:00']:
        sample[f'TimeGroup_{tg}'] = 1 if tg == time_group else 0
    
    # Make prediction
    prediction = model.predict(sample.values.reshape(1, -1))[0]
    return prediction

# Example usage
print("\nMaking example prediction...")
example_date = pd.to_datetime('2024-07-20')
example_item = scaled_df['Standardized_Item'].iloc[0]  # Use an item that exists in the data
example_size = scaled_df['Size'].iloc[0]  # Use a size that exists in the data
example_time_group = '15:00-19:00'

predicted_quantity = predict_quantity(rf_model, example_item, example_size, example_date, example_time_group)
print(f"\nPredicted quantity for {example_item} ({example_size}) on {example_date.date()} during {example_time_group}: {predicted_quantity:.2f}")

# Batch predictions with progress bar
def batch_predict(model, num_predictions=1000):
    print(f"\nMaking {num_predictions} predictions...")
    predictions = []
    for _ in tqdm(range(num_predictions), desc="Batch predictions"):
        date = pd.to_datetime(np.random.choice(scaled_df['Date']))
        item = np.random.choice(scaled_df['Standardized_Item'])
        size = np.random.choice(scaled_df['Size'])
        time_group = np.random.choice(['11:00-15:00', '15:00-19:00', '19:00-23:00', '23:00-03:00'])
        pred = predict_quantity(model, item, size, date, time_group)
        predictions.append(pred)
    return predictions

# Run batch predictions
batch_predictions = batch_predict(rf_model)
print(f"Average prediction: {np.mean(batch_predictions):.2f}")