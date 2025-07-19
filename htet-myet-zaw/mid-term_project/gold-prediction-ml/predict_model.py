import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# Load and clean the data
df = pd.read_csv('data/weekly_merged_data.csv')
df = df.dropna()

print("Dataset shape:", df.shape)
print(df.columns.tolist())

# Convert index to datetime if it's not already
if 'index' in df.columns:
    df['Date'] = pd.to_datetime(df['index'])
    df = df.sort_values('Date')
else:
    df['Date'] = pd.date_range(start='2010-01-01', periods=len(df), freq='W')

# Define features and target
features = ['CPIAUCSL', 'Brent_Oil', 'USD_Index', 'SP500']
target = 'Gold_Price'

print("\nFEATURE ENGINEERING")

# Add lag features (previous week's values)
for feature in features:
    df[f'{feature}_lag1'] = df[feature].shift(1)
    df[f'{feature}_lag2'] = df[feature].shift(2)

# Add rolling averages
for feature in features:
    df[f'{feature}_ma4'] = df[feature].rolling(window=4).mean()
    df[f'{feature}_ma12'] = df[feature].rolling(window=12).mean()

# Add price momentum features
df['Gold_momentum'] = df[target].pct_change()
df['Gold_momentum_ma4'] = df['Gold_momentum'].rolling(window=4).mean()

# Add volatility features
df['Gold_volatility'] = df[target].rolling(window=12).std()

# Add interaction features
df['Oil_USD_interaction'] = df['Brent_Oil'] * df['USD_Index']
df['CPI_SP500_interaction'] = df['CPIAUCSL'] * df['SP500']

# Drop NaN values after feature engineering
df = df.dropna()
print(df.head(10))

# Updated features list
enhanced_features = [
    'CPIAUCSL', 'Brent_Oil', 'USD_Index', 'SP500',
    'CPIAUCSL_lag1', 'Brent_Oil_lag1', 'USD_Index_lag1', 'SP500_lag1',
    'CPIAUCSL_lag2', 'Brent_Oil_lag2', 'USD_Index_lag2', 'SP500_lag2',
    'CPIAUCSL_ma4', 'Brent_Oil_ma4', 'USD_Index_ma4', 'SP500_ma4',
    'CPIAUCSL_ma12', 'Brent_Oil_ma12', 'USD_Index_ma12', 'SP500_ma12',
    'Gold_momentum_ma4', 'Gold_volatility',
    'Oil_USD_interaction', 'CPI_SP500_interaction'
]

print(f"Original features: {len(features)}")
print(f"Enhanced features: {len(enhanced_features)}")

# Extract features and target
X = df[enhanced_features]
y = df[target]

print(f"\nEnhanced dataset shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Time-series aware train-test split (use last 20% for testing)
split_idx = int(len(df) * 0.7)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\nTime-series split:")
print(f"Train set: {X_train.shape[0]} samples (first 70%)")
print(f"Test set: {X_test.shape[0]} samples (last 30%)")

# Using RobustScaler for better handling of outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTarget variable statistics:")
print(f"Train set - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"Test set - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

# Initialize and train Ridge Regression model
print("RIDGE REGRESSION MODEL")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = ridge_model.predict(X_train_scaled)
y_test_pred = ridge_model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Calculate percentage errors
train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print(f"\nModel Performance Metrics:")
print(f"Training Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  MAE: {train_mae:.2f}")
print(f"  MAPE: {train_mape:.2f}%")

print(f"\nTest Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  MAE: {test_mae:.2f}")
print(f"  MAPE: {test_mape:.2f}%")

# Feature importance (coefficients)
print(f"\nFeature Coefficients (Top 10 by absolute value):")
coefs = ridge_model.coef_
coef_df = pd.DataFrame({
    'feature': enhanced_features,
    'coefficient': coefs
})
coef_df['abs_coef'] = abs(coef_df['coefficient'])
coef_df = coef_df.sort_values('abs_coef', ascending=False)

for i, row in coef_df.head(10).iterrows():
    print(f"  {row['feature']}: {row['coefficient']:.4f}")

print(f"\nIntercept: {ridge_model.intercept_:.4f}")

# Sample predictions
print(f"\nSample Predictions (First 10 test samples):")
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100
    print(f"  Actual: {actual:.2f}, Predicted: {predicted:.2f}, Error: {error:.2f} ({error_pct:.1f}%)")

# Time series cross-validation for more robust evaluation
print("TIME SERIES CROSS-VALIDATION")

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for train_idx, val_idx in tscv.split(X_train_scaled):
    X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train Ridge model on CV fold
    cv_model = Ridge(alpha=1.0)
    cv_model.fit(X_cv_train, y_cv_train)
    
    # Predict on validation fold
    y_cv_pred = cv_model.predict(X_cv_val)
    cv_r2 = r2_score(y_cv_val, y_cv_pred)
    cv_scores.append(cv_r2)

print(f"Cross-validation R² scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV R²: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Save the model for Streamlit app
print("SAVING MODEL FOR STREAMLIT")

model_data = {
    'model': ridge_model,
    'scaler': scaler,
    'feature_names': enhanced_features,
    'performance': {
        'test_r2': test_r2,
        'test_mape': test_mape,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }
}

with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ Model saved as 'ridge_model.pkl'")

test_dates = df.loc[y_test.index, 'Date']

plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test, label='Actual Gold Price', marker='o')
plt.plot(test_dates, y_test_pred, label='Predicted Gold Price', marker='x')
plt.title('Actual vs. Predicted Gold Price (Test Set)')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.legend()
plt.tight_layout()
plt.show()

residuals = y_test - y_test_pred
plt.figure(figsize=(10, 5))
plt.scatter(df['Date'].iloc[split_idx:], residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals (Actual - Predicted) on Test Set')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
coef_df.head(10).plot(kind='bar', x='feature', y='abs_coef', legend=False)
plt.title('Top 10 Feature Importances (Absolute Coefficient Values)')
plt.ylabel('Absolute Coefficient')
plt.tight_layout()
plt.show()

if len(cv_scores) > 0:
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cv_scores)+1), cv_scores, marker='o')
    plt.title('Time Series Cross-Validation R² Scores')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.ylim(min(cv_scores) - 1, 1)
    plt.tight_layout()
    plt.show()
else:
    print("No cross-validation scores to plot. Check your data size or cross-validation loop.")


