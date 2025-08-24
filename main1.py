import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("wheat_monthly_prices_production_2021_2025.csv")
data['Year'] = pd.to_datetime(data['Year-Month']).dt.year
data['Month'] = pd.to_datetime(data['Year-Month']).dt.month

# ----------- Step 1: Predict Production ----------- #
X_prod = data[['Year', 'Month']]
y_prod = data['Production_Tonnes']

X_prod_train, X_prod_test, y_prod_train, y_prod_test = train_test_split(X_prod, y_prod, test_size=0.2, random_state=42)

# Hyperparameter grid for XGB
param_grid_prod = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

print("🔍 Tuning Production Model...")
xgb_prod = XGBRegressor(random_state=42)
grid_search_prod = GridSearchCV(xgb_prod, param_grid_prod, cv=3, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
grid_search_prod.fit(X_prod_train, y_prod_train)

best_prod_model = grid_search_prod.best_estimator_
joblib.dump(best_prod_model, "production_model_xgb.pkl")

# Evaluate
y_prod_pred = best_prod_model.predict(X_prod_test)
rmse_prod = np.sqrt(mean_squared_error(y_prod_test, y_prod_pred))
print(f"✅ Best Production Model RMSE: {rmse_prod:.2f}")

# ----------- Step 2: Predict Price ----------- #
data['Predicted_Production'] = best_prod_model.predict(X_prod)

X_price = data[['Year', 'Month', 'Predicted_Production']]
y_price = data['AvgPrice_INR/ton']

X_train, X_test, y_train, y_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

param_grid_price = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

print("\n🔍 Tuning Price Model...")
xgb_price = XGBRegressor(random_state=42)
grid_search_price = GridSearchCV(xgb_price, param_grid_price, cv=3, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
grid_search_price.fit(X_train, y_train)

best_price_model = grid_search_price.best_estimator_
joblib.dump(best_price_model, "price_model_xgb.pkl")

# Evaluate
y_price_pred = best_price_model.predict(X_test)
rmse_price = np.sqrt(mean_squared_error(y_test, y_price_pred))
print(f"✅ Best Price Model RMSE: {rmse_price:.2f}")

# ----------- Step 3: Predict Future Price ----------- #
def predict_future_price(year, month):
    # Load models
    prod_model = joblib.load("production_model_xgb.pkl")
    price_model = joblib.load("price_model_xgb.pkl")

    # Predict production
    est_prod = prod_model.predict([[year, month]])[0]

    # Predict price
    est_price = price_model.predict([[year, month, est_prod]])[0]

    print(f"\n📅 Predicted wheat production for {year}/{month}: {est_prod:.2f} Tonnes")
    print(f"💰 Predicted wheat price for {year}/{month}: ₹{est_price:.2f}/ton")

# Example
predict_future_price(2025, 9)
