# main_pipeline.py
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# ---------------- Utility functions ----------------
def parse_year_month(df, col='Year-Month'):
    # Try various formats and normalize to YYYY-MM
    try:
        df[col] = pd.to_datetime(df[col], format='%y-%b')  # 21-Jan
    except Exception:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['Year'] = df[col].dt.year
    df['Month'] = df[col].dt.month
    return df

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

# ---------------- Load & prep dataset ----------------
CSV_PATH = "wheat_monthly_prices_production_2021_2025.csv"  # <--- change path if needed
df = pd.read_csv(CSV_PATH)

# Normalize date and create Year/Month
if 'Year-Month' not in df.columns:
    raise ValueError("CSV must contain 'Year-Month' column")
df = parse_year_month(df, 'Year-Month')

# Ensure price column exists and create price per tonne
if 'AvgPrice_INR/ton' not in df.columns:
    if 'Modal_Price_RsPerQtl' in df.columns:
        df['AvgPrice_INR/ton'] = df['Modal_Price_RsPerQtl'] * 10.0
    else:
        raise ValueError("CSV must contain 'AvgPrice_INR/ton' or 'Modal_Price_RsPerQtl'")

# Ensure production column exists
if 'Production_Tonnes' not in df.columns:
    raise ValueError("CSV must contain 'Production_Tonnes'")

# Target features to predict (weather / macro)
FEATURES_TO_FORECAST = ['Rainfall_mm', 'Avg_Temperature_C', 'Fertilizer_kg_per_ha', 'CPI_Index']

# If any of these are missing, create them as monthly averages (fallback)
for f in FEATURES_TO_FORECAST:
    if f not in df.columns:
        print(f"[INFO] Column '{f}' missing — filling with monthly average placeholder.")
        # placeholder: use month-wise mean of available months (or global mean)
        df[f] = np.nan

# Create lag features for price and production (helps models)
df = df.sort_values('Year-Month').reset_index(drop=True)
df['Prod_lag1'] = df['Production_Tonnes'].shift(1)
df['Price_lag1'] = df['AvgPrice_INR/ton'].shift(1)
# fill lag NaNs with median
df['Prod_lag1'].fillna(df['Production_Tonnes'].median(), inplace=True)
df['Price_lag1'].fillna(df['AvgPrice_INR/ton'].median(), inplace=True)

# Impute missing feature values by month-wise mean, then overall mean
for f in FEATURES_TO_FORECAST:
    if df[f].isna().any():
        df[f] = df.groupby(df['Month'])[f].transform(lambda x: x.fillna(x.mean()))
        df[f].fillna(df[f].mean(), inplace=True)

# Drop any remaining NaNs
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# ---------------- Train separate feature models ----------------
feature_models = {}
feature_rmse = {}

for feature in FEATURES_TO_FORECAST:
    X = df[['Year', 'Month', 'Prod_lag1', 'Price_lag1']].copy()
    y = df[feature].values
    # Simple split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    f_rmse = rmse(y_te, y_pred)
    feature_models[feature] = model
    feature_rmse[feature] = f_rmse
    joblib.dump(model, f"{feature}_model_xgb.pkl")
    print(f"[FEATURE] Trained {feature} model — RMSE: {f_rmse:.2f}")

# ---------------- Train production model ----------------
# Use year, month, predicted (or actual) features + lags
X_prod = df[['Year', 'Month', 'Prod_lag1', 'Price_lag1'] + FEATURES_TO_FORECAST].copy()
y_prod = df['Production_Tonnes'].values
X_tr, X_te, y_tr, y_te = train_test_split(X_prod, y_prod, test_size=0.2, random_state=42)

prod_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
prod_model.fit(X_tr, y_tr)
y_pred = prod_model.predict(X_te)
prod_rmse = rmse(y_te, y_pred)
joblib.dump(prod_model, "production_model_xgb_full.pkl")
print(f"[PRODUCTION] Trained production model — RMSE: {prod_rmse:.2f}")

# ---------------- Train price model ----------------
# Price model uses year,month, prod (actual/predicted), weather, macro, lags
X_price = df[['Year', 'Month', 'Prod_lag1', 'Price_lag1', 'Production_Tonnes'] + FEATURES_TO_FORECAST].copy()
y_price = df['AvgPrice_INR/ton'].values
X_tr, X_te, y_tr, y_te = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

price_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
price_model.fit(X_tr, y_tr)
y_price_pred = price_model.predict(X_te)
price_rmse = rmse(y_te, y_price_pred)
joblib.dump(price_model, "price_model_xgb_full.pkl")
print(f"[PRICE] Trained price model — RMSE: {price_rmse:.2f}")

# ---------------- Prediction pipeline (future) ----------------
def predict_future_all(year, month):
    """
    Returns dict with predicted features, production and price for given year and month.
    """
    # Build base input row (use last known lags from dataset tail)
    last_row = df.iloc[-1]
    base_prod_lag = last_row['Production_Tonnes']
    base_price_lag = last_row['AvgPrice_INR/ton']

    future_X_base = pd.DataFrame([[year, month, base_prod_lag, base_price_lag]],
                                 columns=['Year', 'Month', 'Prod_lag1', 'Price_lag1'])

    # Predict each auxiliary feature
    predicted_features = {}
    for f in FEATURES_TO_FORECAST:
        model = joblib.load(f"{f}_model_xgb.pkl")
        pf = model.predict(future_X_base)[0]
        predicted_features[f] = float(pf)

    # Build production input using predicted features
    prod_input = future_X_base.copy()
    for f in FEATURES_TO_FORECAST:
        prod_input[f] = predicted_features[f]

    prod_model_local = joblib.load("production_model_xgb_full.pkl")
    predicted_prod = float(prod_model_local.predict(prod_input)[0])

    # Build price input using predicted_prod and predicted features
    price_input = future_X_base.copy()
    price_input['Production_Tonnes'] = predicted_prod
    for f in FEATURES_TO_FORECAST:
        price_input[f] = predicted_features[f]

    price_model_local = joblib.load("price_model_xgb_full.pkl")
    predicted_price = float(price_model_local.predict(price_input)[0])

    return {
        'Year': int(year),
        'Month': int(month),
        'Predicted_Features': predicted_features,
        'Predicted_Production_Tonnes': predicted_prod,
        'Predicted_Price_RsPerTonne': predicted_price
    }

# Example usage:
if __name__ == "__main__":
    print("\n--- Model RMSE summary ---")
    for k, v in feature_rmse.items():
        print(f"{k}: {v:.2f}")
    print(f"Production RMSE: {prod_rmse:.2f}")
    print(f"Price RMSE: {price_rmse:.2f}")

    out = predict_future_all(2025, 9)
    print("\nPrediction for 2025-09:")
    print(out)
