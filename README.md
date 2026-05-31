# Bitcoin_Visualizarion_Using_PowerBi_-with-Machine-learning-


# 📈 Bitcoin Price Forecasting — Complete Project Documentation

> **ARIMA + Linear Regression se Bitcoin ka 90-din ka future forecast**
> Power BI Dashboard + Python ML Pipeline + CSV Data Export

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset Description](#3-dataset-description)
4. [Requirements & Installation](#4-requirements--installation)
5. [Step-by-Step Code Walkthrough](#5-step-by-step-code-walkthrough)
   - [Step 1: Libraries Import](#step-1-libraries-import)
   - [Step 2: Data Loading](#step-2-data-loading)
   - [Step 3: Data Preprocessing](#step-3-data-preprocessing)
   - [Step 4: Feature Engineering](#step-4-feature-engineering)
   - [Step 5: Train/Test Split](#step-5-traintest-split)
   - [Step 6: Linear Regression Model](#step-6-linear-regression-model)
   - [Step 7: ARIMA Model](#step-7-arima-model)
   - [Step 8: 90-Day Future Forecast](#step-8-90-day-future-forecast)
   - [Step 9: Visualization](#step-9-visualization)
   - [Step 10: Export Outputs](#step-10-export-outputs)
6. [Model Evaluation Metrics](#6-model-evaluation-metrics)
7. [Output Files](#7-output-files)
8. [Power BI Dashboard](#8-power-bi-dashboard)
9. [Key Insights & Findings](#9-key-insights--findings)
10. [Limitations & Future Improvements](#10-limitations--future-improvements)

---

## 1. Project Overview

Is project mein Bitcoin (BTC/USD) ki historical price data use kar ke **machine learning** aur **time series forecasting** se future price predict ki gayi hai.

### 🎯 Goals
- Bitcoin ki actual price ko historical data se samajhna
- **Linear Regression** aur **ARIMA** models se price predict karna
- Aglay **90 din** ka forecast generate karna
- Results ko **Power BI** dashboard mein visualize karna

### 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data manipulation & cleaning |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization / plots |
| **Scikit-learn** | Linear Regression model |
| **Statsmodels** | ARIMA time series model |
| **Power BI** | Interactive dashboard |
| **Jupyter Notebook** | Code execution environment |

---

## 2. Project Structure

```
Bitcoin-Forecasting-Project/
│
├── 📓 bitcoin.ipynb              # Main Jupyter Notebook (all code)
│
├── 📊 Bitcoin-data.csv           # Raw input dataset (366 rows)
│
├── 📈 bitcoin_forecast.csv       # Output: Actual + Predicted + Forecast data
│
├── 🖼️  bitcoin_forecast.png      # Output: Final visualization chart
│
└── 📋 README.md                  # This documentation file
```

---

## 3. Dataset Description

### File: `Bitcoin-data.csv`

| Column | Description | Example Value |
|---|---|---|
| `Date` | Date of the record (datetime) | `2025-06-06 00:00:00` |
| `Price_USD` | Bitcoin closing price in US Dollars | `101,650.74` |
| `Volume_USD` | Total trading volume in USD | `142,403,234,254` |
| `MarketCap_USD` | Total market capitalization | `1,262,033,077,144` |
| `Growth_Rate_%` | Percentage growth from a base point | `-40.56` |

### 📊 Dataset Statistics (from Power BI Dashboard)

| Metric | Value |
|---|---|
| **Latest Price** | $74,160 |
| **Average Price** | $94,450 |
| **Maximum Price** | $124,770 (Aug–Oct 2025 peak) |
| **Minimum Price** | $62,850 (Feb 2026 dip) |
| **Total Volume** | $17.07 Trillion |
| **Total Records** | ~366 rows (approx. 1 year of data) |
| **Date Range** | Jun 2025 → May 2026 |

---

## 4. Requirements & Installation

### Prerequisites
- Python 3.8 ya usse upar
- Jupyter Notebook / JupyterLab

### Install Libraries

Notebook ka pehla cell ye command run karta hai:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels jupyter
```

### Library Versions (Recommended)

```
pandas >= 1.5.0
numpy >= 1.23.0
matplotlib >= 3.5.0
scikit-learn >= 1.1.0
statsmodels >= 0.13.0
```

---

## 5. Step-by-Step Code Walkthrough

### Step 1: Libraries Import

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
```

**Kya ho raha hai yahan?**
- `pandas` → data tables (DataFrames) ke liye
- `numpy` → mathematical operations ke liye
- `matplotlib.pyplot` → graphs aur charts banane ke liye
- `sklearn` → Linear Regression model aur accuracy metrics ke liye
- `statsmodels.ARIMA` → time series forecasting ke liye
- `warnings.filterwarnings('ignore')` → unnecessary warnings ko suppress karta hai

---

### Step 2: Data Loading

```python
df = pd.read_csv(r"C:\Users\abc\Desktop\ADV quiz 3 and 4\Bitcoin-data.csv")

print(df.columns)   # Column names dekhna
print(df.head())    # Pehli 5 rows dekhna
print(df.shape)     # Rows aur Columns count
```

**Output:**
```
Index(['Date', 'Price_USD', 'Volume_USD', 'MarketCap_USD', 'Growth_Rate_%'], dtype='object')
(366, 5)
```

> ⚠️ **Note:** File path apne system ke according change karein.

---

### Step 3: Data Preprocessing

```python
df['Date'] = pd.to_datetime(df['Date'])    # Date ko datetime format mein convert karo
df = df.sort_values('Date').reset_index(drop=True)   # Date ke order mein sort karo
df = df[['Date', 'Price_USD']].dropna()   # Sirf zaruri columns rakho, NaN rows hataao
```

**Kya ho raha hai?**
- Date column string se datetime object ban jaata hai
- Data chronological (time) order mein sort hota hai
- Sirf `Date` aur `Price_USD` rakha jaata hai — baki columns is analysis mein use nahi hain
- `dropna()` → missing values wali rows hata deta hai

---

### Step 4: Feature Engineering

```python
# Time-based features
df['Day_Index']   = range(len(df))           # 0, 1, 2, 3... sequential day number
df['Year']        = df['Date'].dt.year
df['Month']       = df['Date'].dt.month
df['Quarter']     = df['Date'].dt.quarter
df['Day_of_Week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday

# Lag features (previous day prices as input)
df['Lag_1']         = df['Price_USD'].shift(1)   # Kal ka price
df['Lag_7']         = df['Price_USD'].shift(7)   # 7 din pehle ka price

# Rolling statistics
df['Rolling_Mean_7'] = df['Price_USD'].rolling(window=7).mean()  # 7-din ka average
df['Rolling_Std_7']  = df['Price_USD'].rolling(window=7).std()   # 7-din ka standard deviation

df = df.dropna().reset_index(drop=True)   # Lag se bane NaN rows hataao
```

**Feature Engineering kyun zaroori hai?**

| Feature | Reason |
|---|---|
| `Day_Index` | Model ko sequential time ka andaza deta hai |
| `Month`, `Quarter` | Seasonal patterns pakadne ke liye |
| `Lag_1`, `Lag_7` | Purane prices naye prices ko influence karte hain |
| `Rolling_Mean_7` | Recent trend smooth karta hai (noise kam karta hai) |

---

### Step 5: Train/Test Split

```python
split = int(len(df) * 0.8)   # 80% training, 20% testing
train = df[:split]
test  = df[split:]

print(f"Training rows: {len(train)}")
print(f"Testing rows:  {len(test)}")
```

**Data Division:**
```
Total rows after feature engineering: ~359
Training set: ~287 rows  (80%) → approx Jun 2025 to Mar 2026
Testing set:  ~72 rows   (20%) → approx Mar 2026 to May 2026
```

> **Kyun 80/20?** Yeh machine learning mein standard split hai. Model training data se seekhta hai aur testing data pe evaluate hota hai.

---

### Step 6: Linear Regression Model

```python
features = ['Day_Index', 'Month', 'Quarter', 'Lag_1', 'Lag_7', 'Rolling_Mean_7']

X_train = train[features]
y_train = train['Price_USD']
X_test  = test[features]
y_test  = test['Price_USD']

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)          # Model train karo
lr_pred  = lr_model.predict(X_test)    # Prediction karo

# Accuracy metrics
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)
```

**Linear Regression kya hai?**
- Ek mathematical formula: `Price = a×Feature1 + b×Feature2 + ... + c`
- Model best-fit line/surface dhundta hai jo features se price predict kare
- Simple but effective jab data mein clear patterns ho

---

### Step 7: ARIMA Model

```python
train_series = train['Price_USD'].values

arima_model = ARIMA(train_series, order=(5, 1, 0))
arima_fit   = arima_model.fit()

arima_pred  = arima_fit.forecast(steps=len(test))  # Test period ke liye predict karo
```

**ARIMA(5, 1, 0) ka matlab:**

| Parameter | Value | Meaning |
|---|---|---|
| **p** (AR) | 5 | Pichle 5 time steps ka effect use karo |
| **d** (I) | 1 | Data ko ek baar difference karo (stationary banane ke liye) |
| **q** (MA) | 0 | Moving Average terms use nahi kar rahe |

**ARIMA kyun Linear Regression se better hai time series mein?**
- ARIMA specifically time series data ke liye bana hai
- Yeh data ke sequential nature ko samajhta hai
- Seasonality aur trends ko better capture karta hai

---

### Step 8: 90-Day Future Forecast

```python
future_steps = 90

# ARIMA se test + future forecast ek saath generate karo, phir future slice karo
full_forecast   = arima_fit.forecast(steps=len(test) + future_steps)
future_forecast = full_forecast[len(test):]  # Sirf future 90 days

# Future dates generate karo
last_date    = df['Date'].max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=future_steps
)

future_df = pd.DataFrame({
    'Date':  future_dates,
    'Price': future_forecast,
    'Type':  'Forecast'
})
```

**Forecast Logic:**
- ARIMA model training data ke patterns se agla trend estimate karta hai
- `forecast(steps=N)` → N time steps aage ki prediction
- Chart mein dekha gaya: forecast ~$71,000 ke aas paas stable rehta hai (Aug–Sep 2026 tak)

---

### Step 9: Visualization

```python
plt.figure(figsize=(16, 7))

plt.plot(train['Date'], train['Price_USD'],
         label='Training Data', color='blue', linewidth=1.5)

plt.plot(test['Date'], test['Price_USD'],
         label='Actual Price', color='green', linewidth=2)

plt.plot(test['Date'], arima_pred,
         label='ARIMA Predicted', color='red',
         linestyle='--', linewidth=2)

plt.plot(future_dates, future_forecast,
         label='Future Forecast (90 Days)', color='orange',
         linestyle='--', linewidth=2)

plt.title('Bitcoin Price — Actual vs Predicted vs 90-Day Forecast',
          fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(r"...\bitcoin_forecast.png", dpi=150)
plt.show()
```

**Chart mein 4 lines:**

| Color | Line | Description |
|---|---|---|
| 🔵 Blue (solid) | Training Data | Jun 2025 – Feb 2026 historical prices |
| 🟢 Green (solid) | Actual Price | Mar–May 2026 real test data |
| 🔴 Red (dashed) | ARIMA Predicted | Model ki test period prediction |
| 🟠 Orange (dashed) | Future Forecast | 90-day ahead forecast (Jun–Sep 2026) |

**Chart kya dikhata hai:**
- Bitcoin ne Aug–Oct 2025 mein ~$124,000 peak touch kiya
- Feb 2026 mein ~$63,000 tak gira (bear market)
- May 2026 tak ~$74,000–82,000 recover hua
- ARIMA forecast ~$71,000 ke aas paas stable predict karta hai

---

### Step 10: Export Outputs

```python
# Actual + Predicted + Forecast ek CSV mein combine karo
actual_df = df[['Date', 'Price_USD']].copy()
actual_df.rename(columns={'Price_USD': 'Price'}, inplace=True)
actual_df['Type'] = 'Actual'

pred_df = test[['Date']].copy()
pred_df['Price'] = arima_pred
pred_df['Type']  = 'Predicted'

final_df = pd.concat([actual_df, pred_df, future_df], ignore_index=True)
final_df.to_csv(r"...\bitcoin_forecast.csv", index=False)

# Metrics summary CSV
metrics_df = pd.DataFrame({
    'Model':    ['Linear Regression', 'ARIMA'],
    'MAE':      [round(lr_mae, 2),    round(arima_mae, 2)],
    'RMSE':     [round(lr_rmse, 2),   round(arima_rmse, 2)],
    'R2_Score': [round(lr_r2, 4),     round(arima_r2, 4)]
})
metrics_df.to_csv(r"...\model_metrics.csv", index=False)
```

---

## 6. Model Evaluation Metrics

### Kya hote hain ye metrics?

| Metric | Full Name | Matlab | Ideal Value |
|---|---|---|---|
| **MAE** | Mean Absolute Error | Average prediction error (USD mein) | Jitna kam utna acha |
| **RMSE** | Root Mean Squared Error | Bade errors ko zyada penalize karta hai | Jitna kam utna acha |
| **R²** | R-Squared (Coefficient of Determination) | Model kitni accuracy se data explain karta hai | 1.0 = perfect |

### ARIMA vs Linear Regression Comparison

| Model | Strength | Weakness |
|---|---|---|
| **Linear Regression** | Fast, simple, interpretable | Assume karta hai linear relationship — Bitcoin mein clearly nahi hai |
| **ARIMA** | Time series ke liye designed, sequential patterns pakadta hai | Flat forecast deta hai (mean-reverting tendency) |

> **Chart observation:** ARIMA ki predicted line (red dashed) actual green line se neeche hai, jo dikhata hai ki ARIMA ne May 2026 recovery ko fully capture nahi kiya — lekin overall direction sahi thi.

---

## 7. Output Files

### `bitcoin_forecast.csv`
Complete combined data file:

| Column | Values |
|---|---|
| `Date` | Jun 2025 → Sep 2026 |
| `Price` | Actual / Predicted / Forecasted price in USD |
| `Type` | `Actual` / `Predicted` / `Forecast` |

### `bitcoin_forecast.png`
High-resolution (150 DPI) line chart showing all 4 data series on one plot.

### `model_metrics.csv`
Accuracy comparison table for both models (MAE, RMSE, R²).

---

## 8. Power BI Dashboard

Dashboard (`WhatsApp_Image_2026-05-31...jpeg`) mein dikhaya gaya:

### KPI Cards (Top Row)
| Card | Value |
|---|---|
| Latest Price | **$74.16K** |
| Avg Price | **$94.45K** |
| Max Price | **$124.77K** |
| Min Price | **$62.85K** |
| Total Volume | **$17.07 Trillion** |

### Visuals
1. **Line Chart** — `Sum of Price_USD by Date` → full price trend (Jun 2025–May 2026)
2. **Bar Chart** — `Avg Price by Month` → August & July highest, April lowest
3. **Data Table** — Bottom 10 lowest price dates (Feb–Mar 2026 bear market dates)
4. **Pie Chart** — `Volume by Quarter` → Q4 2025 mein 33.99% volume (most active)

### Filters (Slicers)
- **Year:** 2025 / 2026
- **Month Name:** April se September tak
- **Quarter:** Q1, Q2, Q3, Q4

---

## 9. Key Insights & Findings

### 📊 Price Trends
- Bitcoin ne **August–October 2025** mein **$124,770** ka all-time high touch kiya
- **November 2025 – February 2026** mein sharp decline aaya (~50% drop)
- **February 2026 mein lowest point: $62,854** (near-term bottom)
- **March–May 2026** mein gradual recovery ($62K → $82K)

### 📅 Monthly Patterns (Power BI)
- **August & July** → Highest average prices (summer bull run 2025)
- **April** → Lowest monthly average (worst performing month)
- **Q4 2025** → Highest trading volume (33.99% of total) — peak market activity

### 🔮 90-Day Forecast (ARIMA)
- Model predicts: **~$71,000 stable price** through August–September 2026
- Flat forecast nature ARIMA ki limitation dikhati hai (no major uptrend predicted)
- Real market movement zyada volatile hogi

---

## 10. Limitations & Future Improvements

### ⚠️ Current Limitations

| Limitation | Description |
|---|---|
| ARIMA flat forecast | ARIMA conservative rahta hai, extreme movements predict nahi karta |
| No external factors | News, regulations, macroeconomics data use nahi kiya |
| Limited data | ~366 rows — zyada historical data better results deta |
| Single asset | Sirf BTC — other crypto correlations ignore |

### 🚀 Future Improvements

1. **LSTM / Deep Learning** → Neural networks volatile assets ke liye better hote hain
2. **Prophet (Facebook)** → Seasonality automatically detect karta hai
3. **Sentiment Analysis** → Twitter/Reddit sentiment as a feature
4. **Multi-variate ARIMAX** → Volume, market cap, Gold price as exogenous variables
5. **Ensemble Model** → ARIMA + Linear Regression + ML combine karein
6. **Real-time API** → CoinGecko/Binance API se live data feed

---

## 📁 How to Run This Project

```bash
# 1. Clone/download the project
# 2. Install dependencies
pip install pandas numpy matplotlib scikit-learn statsmodels jupyter

# 3. Open Jupyter Notebook
jupyter notebook bitcoin.ipynb

# 4. Update the file path in Cell 1
df = pd.read_csv(r"YOUR_PATH\Bitcoin-data.csv")

# 5. Run all cells (Kernel → Restart & Run All)
```

---

## 👤 Author

**Author Name:** Hassan Mehmood
**Project:** Bitcoin Price Forecasting with ARIMA & Linear Regression  
**Tools:** Python, Jupyter Notebook, Power BI  
**Date:** 29th May 2026

---

*Is README mein poora project A to Z cover kiya gaya hai — data se lekar models, outputs, dashboard, insights, aur future improvements tak.*
