import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
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

prod_model = LinearRegression()
prod_model.fit(X_prod_train, y_prod_train)

# Save the production model
joblib.dump(prod_model, "production_model.pkl")

# Step 2: Predict Price from Year + Month + Estimated Production
data['Predicted_Production'] = prod_model.predict(X_prod)
X_price = data[['Year', 'Month', 'Predicted_Production']]
y_price = data['AvgPrice_INR/ton']

X_train, X_test, y_train, y_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

price_model = LinearRegression()
price_model.fit(X_train, y_train)

# Save the price model
joblib.dump(price_model, "price_model.pkl")

# Evaluation
y_pred = price_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model Evaluation:\nRMSE: {rmse:.2f}")

# Step 3: Predict Future Price (Just Year and Month)
def predict_future_price(year, month):
    # Load models
    prod_model = joblib.load("production_model.pkl")
    price_model = joblib.load("price_model.pkl")

    # Predict production
    future_input_prod = pd.DataFrame([[year, month]], columns=['Year', 'Month'])
    estimated_prod = prod_model.predict(future_input_prod)[0]

    # Predict price
    future_input_price = pd.DataFrame([[year, month, estimated_prod]],
                                      columns=['Year', 'Month', 'Predicted_Production'])
    predicted_price = price_model.predict(future_input_price)[0]

    print(f"Predicted wheat production for {year}/{month}: {estimated_prod:.2f} (Tonnes)")
    print(f"Predicted wheat price for {year}/{month}: ₹{predicted_price:.2f}/ton")

# Example usage
predict_future_price(2026, 9)
