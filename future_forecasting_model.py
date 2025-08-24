# I'm Currently working on this one, so it won't work
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# ========================
# CONFIG
# ========================
DATA_FILE = "new_wheat_monthly_prices_production_2021_2025.csv"
FUTURE_MONTHS = [(2025, 9), (2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2)]

# ========================
# LOAD DATA
# ========================
df = pd.read_csv(DATA_FILE)

# Ensure columns exist
required_cols = [
    'Year', 'Month', 'Production_Tonnes', 'AvgPrice_INR_per_ton',
    'Rainfall_mm', 'Avg_Temperature_C', 'Fertilizer_kg_per_ha', 'CPI_Index'
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# ========================
# FUNCTION TO TRAIN & PREDICT
# ========================
def train_and_predict_feature(df, target_col, future_dates):
    """Train XGBRegressor for a target and predict future values."""
    X = df[['Year', 'Month']]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    preds_test = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds_test))
    print(f"✅ {target_col} Model RMSE: {rmse:.2f}")
    
    # Predict for future
    future_df = pd.DataFrame(future_dates, columns=['Year', 'Month'])
    future_preds = model.predict(future_df)
    
    return future_preds

# ========================
# STEP 1: PREDICT FUTURE CONDITIONS
# ========================
pred_rainfall = train_and_predict_feature(df, 'Rainfall_mm', FUTURE_MONTHS)
pred_temp = train_and_predict_feature(df, 'Avg_Temperature_C', FUTURE_MONTHS)
pred_fert = train_and_predict_feature(df, 'Fertilizer_kg_per_ha', FUTURE_MONTHS)
pred_cpi = train_and_predict_feature(df, 'CPI_Index', FUTURE_MONTHS)

future_conditions = pd.DataFrame(FUTURE_MONTHS, columns=['Year', 'Month'])
future_conditions['Rainfall_mm'] = pred_rainfall
future_conditions['Avg_Temperature_C'] = pred_temp
future_conditions['Fertilizer_kg_per_ha'] = pred_fert
future_conditions['CPI_Index'] = pred_cpi

print("\n📊 Future Conditions Predictions:")
print(future_conditions)

# ========================
# STEP 2: PRODUCTION MODEL
# ========================
X_prod = df[['Year', 'Month', 'Rainfall_mm', 'Avg_Temperature_C', 'Fertilizer_kg_per_ha', 'CPI_Index']]
y_prod = df['Production_Tonnes']

X_train, X_test, y_train, y_test = train_test_split(X_prod, y_prod, test_size=0.2, shuffle=False)

prod_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
prod_model.fit(X_train, y_train)

prod_preds_test = prod_model.predict(X_test)
rmse_prod = math.sqrt(mean_squared_error(y_test, prod_preds_test))
print(f"\n✅ Production Model RMSE: {rmse_prod:.2f}")

# Predict future production
future_conditions['Predicted_Production'] = prod_model.predict(future_conditions)

# ========================
# STEP 3: PRICE MODEL
# ========================
X_price = df[['Year', 'Month', 'Rainfall_mm', 'Avg_Temperature_C', 'Fertilizer_kg_per_ha', 'CPI_Index', 'Production_Tonnes']]
y_price = df['AvgPrice_INR_per_ton']

X_train, X_test, y_train, y_test = train_test_split(X_price, y_price, test_size=0.2, shuffle=False)

price_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
price_model.fit(X_train, y_train)

price_preds_test = price_model.predict(X_test)
rmse_price = math.sqrt(mean_squared_error(y_test, price_preds_test))
print(f"✅ Price Model RMSE: {rmse_price:.2f}")

# Predict future prices
future_price_input = future_conditions.rename(columns={'Predicted_Production': 'Production_Tonnes'})
future_conditions['Predicted_Price'] = price_model.predict(future_price_input)

# ========================
# STEP 4: OUTPUT RESULTS
# ========================
print("\n📅 Final Predictions:")
for i, row in future_conditions.iterrows():
    print(f"{int(row['Year'])}/{int(row['Month'])} -> "
          f"Rainfall: {row['Rainfall_mm']:.1f} mm, "
          f"Temp: {row['Avg_Temperature_C']:.1f} °C, "
          f"Fertilizer: {row['Fertilizer_kg_per_ha']:.1f} kg/ha, "
          f"CPI: {row['CPI_Index']:.2f}, "
          f"Production: {row['Predicted_Production']:.2f} Tonnes, "
          f"Price: ₹{row['Predicted_Price']:.2f}/ton")

# Save to CSV
future_conditions.to_csv("future_predictions.csv", index=False)
print("\n💾 Saved future predictions to future_predictions.csv")

