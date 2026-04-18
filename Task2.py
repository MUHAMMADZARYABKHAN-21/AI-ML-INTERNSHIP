# ============================================================
#   TASK 2: Predict Future Stock Prices (Short-Term)
#   Stock: Apple (AAPL) | Model: Linear Regression + Random Forest
#   Tools: yfinance, pandas, scikit-learn, matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# STEP 1: LOAD STOCK DATA
# ─────────────────────────────────────────

TICKER  = 'AAPL'       # Change to any ticker: 'TSLA', 'GOOGL', etc.
PERIOD  = '2y'         # 2 years of historical data
TEST_RATIO = 0.2       # 20% data used for testing

print(f"📈 Downloading {TICKER} stock data...")
df = yf.download(TICKER, period=PERIOD, progress=False)

# Flatten MultiIndex columns if present (yfinance sometimes returns them)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"✅ Downloaded {len(df)} trading days of data\n")

# ─────────────────────────────────────────
# STEP 2: INSPECT THE DATA
# ─────────────────────────────────────────

print("📐 Shape:", df.shape)
print("\n📋 Columns:", df.columns.tolist())
print("\n🔍 First 5 Rows:")
print(df.head())
print("\n📊 Dataset Info:")
df.info()
print("\n📈 Descriptive Statistics:")
print(df.describe().round(2))

# ─────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────

# Target: predict tomorrow's Close price
# Shift Close by -1 so each row's target = next day's close
df['Target'] = df['Close'].shift(-1)

# Additional technical features
df['Price_Change']  = df['Close'] - df['Open']           # Daily movement
df['High_Low_Range']= df['High'] - df['Low']             # Volatility indicator
df['MA_5']          = df['Close'].rolling(window=5).mean()   # 5-day moving average
df['MA_10']         = df['Close'].rolling(window=10).mean()  # 10-day moving average
df['Daily_Return']  = df['Close'].pct_change()           # % change day-over-day

# Drop rows with NaN values (caused by shift and rolling)
df.dropna(inplace=True)

print(f"\n✅ Features engineered. Working with {len(df)} rows after cleaning.")

# ─────────────────────────────────────────
# STEP 4: DEFINE FEATURES & TARGET
# ─────────────────────────────────────────

FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Price_Change', 'High_Low_Range', 'MA_5', 'MA_10', 'Daily_Return']

X = df[FEATURES].values
y = df['Target'].values.ravel()   # Next day's Close price

# ─────────────────────────────────────────
# STEP 5: TRAIN / TEST SPLIT (time-ordered)
# ─────────────────────────────────────────
# IMPORTANT: For time series, we do NOT shuffle — keep chronological order

split_idx = int(len(X) * (1 - TEST_RATIO))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

dates_test = df.index[split_idx:]   # Keep dates for x-axis in plot

print(f"\n📦 Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────
# STEP 6: SCALE FEATURES
# ─────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled  = scaler.transform(X_test)       # apply same scale to test

# ─────────────────────────────────────────
# STEP 7: TRAIN MODELS
# ─────────────────────────────────────────

# ── Model 1: Linear Regression ───────────
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

# ── Model 2: Random Forest ────────────────
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)    # RF doesn't need scaling
rf_preds = rf_model.predict(X_test)

# ─────────────────────────────────────────
# STEP 8: EVALUATE MODELS
# ─────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    print(f"\n  {name}")
    print(f"    MAE  : ${mae:.2f}  ← avg dollar error")
    print(f"    RMSE : ${rmse:.2f}  ← penalizes large errors")
    print(f"    R²   : {r2:.4f}  ← 1.0 = perfect fit")
    return mae, rmse, r2

print("\n" + "=" * 50)
print("  MODEL EVALUATION ON TEST SET")
print("=" * 50)
lr_scores = evaluate("Linear Regression", y_test, lr_preds)
rf_scores = evaluate("Random Forest",     y_test, rf_preds)

# ─────────────────────────────────────────
# STEP 9: FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────

importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=False)

print("\n🌲 Random Forest — Feature Importance:")
for feat, score in importances_sorted.items():
    bar = '█' * int(score * 40)
    print(f"  {feat:<18} {bar} {score:.4f}")

# ─────────────────────────────────────────
# STEP 10: PREDICT NEXT DAY'S PRICE
# ─────────────────────────────────────────

last_row = df[FEATURES].iloc[-1].values.reshape(1, -1)

lr_tomorrow = lr_model.predict(scaler.transform(last_row))[0]
rf_tomorrow = rf_model.predict(last_row)[0]
actual_last = df['Close'].iloc[-1]

print("\n" + "=" * 50)
print("  NEXT DAY PRICE PREDICTION")
print("=" * 50)
print(f"  Last Known Close  : ${float(actual_last):.2f}")
print(f"  Linear Regression : ${lr_tomorrow:.2f}")
print(f"  Random Forest     : ${rf_tomorrow:.2f}")

# ─────────────────────────────────────────
# STEP 11: PLOTTING
# ─────────────────────────────────────────

plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444466',
    'axes.labelcolor':  '#e0e0ff',
    'xtick.color':      '#aaaacc',
    'ytick.color':      '#aaaacc',
    'text.color':       '#e0e0ff',
    'grid.color':       '#2a2a4a',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})

fig, axes = plt.subplots(3, 1, figsize=(14, 16))
fig.suptitle(f'{TICKER} — Stock Price Prediction', fontsize=18,
             fontweight='bold', color='#e0e0ff', y=1.01)

# ── Plot A: Full Close Price History ─────
ax1 = axes[0]
full_dates = df.index
ax1.plot(full_dates, df['Close'], color='#00f5d4', linewidth=1.2, label='Close Price')
ax1.plot(full_dates, df['MA_5'],  color='#f72585', linewidth=0.8,
         linestyle='--', alpha=0.8, label='MA 5-day')
ax1.plot(full_dates, df['MA_10'], color='#7209b7', linewidth=0.8,
         linestyle='--', alpha=0.8, label='MA 10-day')
ax1.axvline(x=dates_test[0], color='yellow', linestyle=':', linewidth=1.5,
            label='Train/Test Split')
ax1.set_title('Full Price History + Moving Averages', color='#c0c0ff')
ax1.set_ylabel('Price (USD)')
ax1.legend(fontsize=9, framealpha=0.2)
ax1.grid(True)

# ── Plot B: Actual vs Predicted (LR) ─────
ax2 = axes[1]
ax2.plot(dates_test, y_test,    color='#00f5d4', linewidth=1.5,
         label='Actual Close')
ax2.plot(dates_test, lr_preds,  color='#f72585', linewidth=1.2,
         linestyle='--', label=f'Linear Regression  (R²={lr_scores[2]:.4f})')
ax2.set_title('Actual vs Predicted — Linear Regression', color='#c0c0ff')
ax2.set_ylabel('Price (USD)')
ax2.legend(fontsize=9, framealpha=0.2)
ax2.grid(True)

# ── Plot C: Actual vs Predicted (RF) ─────
ax3 = axes[2]
ax3.plot(dates_test, y_test,    color='#00f5d4', linewidth=1.5,
         label='Actual Close')
ax3.plot(dates_test, rf_preds,  color='#ffd60a', linewidth=1.2,
         linestyle='--', label=f'Random Forest  (R²={rf_scores[2]:.4f})')
ax3.set_title('Actual vs Predicted — Random Forest', color='#c0c0ff')
ax3.set_ylabel('Price (USD)')
ax3.set_xlabel('Date')
ax3.legend(fontsize=9, framealpha=0.2)
ax3.grid(True)

plt.tight_layout()
plt.savefig('stock_prediction.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.show()

# ── Plot D: Feature Importance Bar Chart ─
fig2, ax = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('#0f0f1a')
colors = ['#00f5d4' if i == 0 else '#7209b7' for i in range(len(importances_sorted))]
importances_sorted.plot(kind='bar', ax=ax, color=colors, edgecolor='white',
                        linewidth=0.5)
ax.set_title('Random Forest — Feature Importance', fontsize=13,
             fontweight='bold', color='#e0e0ff')
ax.set_ylabel('Importance Score')
ax.set_xlabel('Feature')
ax.tick_params(axis='x', rotation=30)
ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.show()

print("\n✅ Plots saved: stock_prediction.png, feature_importance.png")