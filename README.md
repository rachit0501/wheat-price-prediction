# 🌾 Wheat Price Prediction (2021–2025)

## 📌 Overview
This project predicts **wheat prices in India** for the period 2021–2025 using **Machine Learning models**.  
The model leverages factors such as **rainfall, temperature, fertilizer usage, and CPI index** to forecast wheat price trends.  
Built with **Python, Pandas, Scikit-learn, Matplotlib, and XGBoost**.

---

## 📂 Dataset
The dataset includes:
- **Year-Month**: Time index (2021–2025)
- **Wheat Price (₹/quintal)**
- **Rainfall (mm)**
- **Average Temperature (°C)**
- **Fertilizer Usage (kg/ha)**
- **CPI Index (base 2012=100)**

---

## ⚙️ Methodology
1. **Data Preprocessing**  
   - Filled missing values  
   - Converted dates into `datetime` format  
   - Normalized features where required  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized seasonal trends  
   - Checked correlations between features  

3. **Model Training**  
   - **Linear Regression** (baseline)  
   - **Random Forest Regressor** (nonlinear patterns)  
   - **XGBoost Regressor** (boosted performance)  

4. **Evaluation**  
   - RMSE (Root Mean Squared Error)  
   - R² Score  

---

## 📊 Results
- ✅ Rainfall model RMSE: ~45.2  
- ✅ XGBoost provided the most accurate predictions  
- ✅ Seasonal trends in wheat prices were captured successfully  

---

## 🛠️ Installation & Usage
Clone this repository:
```bash
git clone https://github.com/rachit0501/wheat-price-prediction.git
cd wheat-price-prediction
