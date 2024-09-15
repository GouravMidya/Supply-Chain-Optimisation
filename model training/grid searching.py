# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:51:19 2024

@author: goura
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
#%%
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Create a base model
rf = RandomForestRegressor(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Get the best model
best_rf = grid_search.best_estimator_

# Make predictions on the test set
y_pred_rf = best_rf.predict(X_test_scaled)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"Optimized Random Forest - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
#%%
# Save the best model (optional)
import joblib
joblib.dump(best_rf, 'best_random_forest_model.joblib')