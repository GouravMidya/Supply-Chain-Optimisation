import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
#%%
# Assuming your data is in a DataFrame called 'df'
# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'])
#%%
# Encode categorical variables
le_season = LabelEncoder()
le_size = LabelEncoder()
le_timegroup = LabelEncoder()

df['Season_encoded'] = le_season.fit_transform(df['Season'])
df['Size_encoded'] = le_size.fit_transform(df['Size'])
df['TimeGroup_encoded'] = le_timegroup.fit_transform(df['TimeGroup'])
#%%
# Select features for the model
features = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'IsWeekend',
            'DaysSinceLastSale', 'RollingCount7Day', 'RollingCount30Day',
            'RollingCount90Day', 'PrevDayQuantity', 'PrevWeekQuantity', 'PrevMonthQuantity',
            'ItemPopularity', 'ExpMovingAverage', 'Season_encoded', 'Size_encoded', 'TimeGroup_encoded']
#%%
X = df[features]
y = df['Quantity']

# Handle missing values
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#%%
# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#%%
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
#%%
# Function to predict quantity for a given date, time group, and size
def predict_quantity(date, time_group, size):
    # Convert input to datetime
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
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return prediction
#%%
# Example usage
predicted_quantity = predict_quantity('2023-09-06', '19:00-23:00', 'Regular')
print(f"Predicted quantity: {predicted_quantity}")
#%%
# Print unique values of Size and TimeGroup for reference
print("\nUnique Size values:")
print(df['Size'].unique())
print("\nUnique TimeGroup values:")
print(df['TimeGroup'].unique())