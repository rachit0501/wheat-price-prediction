import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("wheat_monthly_prices_production_2021_2025.csv")
data['Year'] = pd.to_datetime(data['Year-Month']).dt.year
data['Month'] = pd.to_datetime(data['Year-Month']).dt.month

# Step 1: Predict Production from Year + Month
X_prod = data[['Year', 'Month']]
y_prod = data['Production_Tonnes']

X_prod_train, X_prod_test, y_prod_train, y_prod_test = train_test_split(X_prod, y_prod, test_size=0.2, random_state=42)

# Try different hyperparameters for production model
best_prod_rmse = float('inf')
best_prod_model = None

print("🔍 Tuning Production Model...")
for depth in [5, 10, 15]:
    for trees in [100, 200, 300]:
        model = RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=42)
        model.fit(X_prod_train, y_prod_train)
        y_pred = model.predict(X_prod_test)
        rmse = np.sqrt(mean_squared_error(y_prod_test, y_pred))

        print(f"Prod -> Trees: {trees}, Depth: {depth}, RMSE: {rmse:.2f}")
        if rmse < best_prod_rmse:
            best_prod_rmse = rmse
            best_prod_model = model

# Save the best production model
joblib.dump(best_prod_model, "production_model.pkl")

# Step 2: Predict Price from Year + Month + Estimated Production
data['Predicted_Production'] = best_prod_model.predict(X_prod)
X_price = data[['Year', 'Month', 'Predicted_Production']]
y_price = data['AvgPrice_INR/ton']

X_train, X_test, y_train, y_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

# Try different hyperparameters for price model
best_price_rmse = float('inf')
best_price_model = None

print("\n🔍 Tuning Price Model...")
for depth in [5, 10, 15]:
    for trees in [100, 200, 300]:
        model = RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Price -> Trees: {trees}, Depth: {depth}, RMSE: {rmse:.2f}")
        if rmse < best_price_rmse:
            best_price_rmse = rmse
            best_price_model = model

# Save the best price model
joblib.dump(best_price_model, "price_model.pkl")

# Final evaluation
print(f"\n✅ Best Price Model RMSE: {best_price_rmse:.2f}")

# Step 3: Predict Future Price
def predict_future_price(year, month):
    # Load models
    prod_model = joblib.load("production_model.pkl")
    price_model = joblib.load("price_model.pkl")

    # Predict production
    estimated_prod = prod_model.predict([[year, month]])[0]

    # Predict price
    predicted_price = price_model.predict([[year, month, estimated_prod]])[0]

    print(f"\n📅 Predicted wheat production for {year}/{month}: {estimated_prod:.2f} Tonnes")
    print(f"💰 Predicted wheat price for {year}/{month}: ₹{predicted_price:.2f}/ton")

# Example usage
predict_future_price(2025, 9)
