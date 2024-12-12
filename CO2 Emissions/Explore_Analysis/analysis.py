import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data('../../data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv')

# 1. Show initial dataframe
print("🔍 Initial Data")
print(df)

# 2. Data preprocessing: Convert year to datetime, sort, and set index
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df = df.sort_values(by='Year')
df = df.set_index('Year')

# Show updated dataframe after preprocessing
print("🚠 Data after DateTime Conversion, Sorting, and Setting Index")
print(df)

# 3. Select country and check available emission categories
country = "UNITED KINGDOM"  # Example of a selected country

# Filter the dataframe based on selected country
filtered_df = df[df['Country'] == country]

# 4. Select emission category ('Total' in this case)
category = 'Total'

# 5. Filter data based on user selection and show head of filtered dataframe
filtered_df = filtered_df[['Total']]
print(f"📂 Filtered Data for {country} - {category} (After Selecting Columns)")
print(filtered_df.head())

# Analysis
# 1. Apply log transformation
filtered_df_log = np.log(filtered_df)
print(f"⟳ Log-transformed Data for {country} - {category}")
print(filtered_df_log)

# 2. Calculate SMA and EMA with window size 10
window_size = 10
filtered_df_log['SMA10'] = filtered_df_log['Total'].rolling(window=window_size).mean()
filtered_df_log['EMA10'] = filtered_df_log['Total'].ewm(span=window_size, adjust=False).mean()

# 3. Plot Original Log Data, SMA10, and EMA10
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(filtered_df_log.index, filtered_df_log['Total'], label='Original Log Data', color='blue')
ax.plot(filtered_df_log.index, filtered_df_log['SMA10'], label='SMA10', color='orange')
ax.plot(filtered_df_log.index, filtered_df_log['EMA10'], label='EMA10', color='green')
ax.set_title(f"SMA10 and EMA10 for {country} - {category}")
ax.set_xlabel("Year")
ax.set_ylabel("Log Emission Value")
ax.legend()
plt.show()

# 4. ETS Decomposition
print("🔍 ETS Decomposition")
try:
    result = seasonal_decompose(filtered_df_log['Total'], model='additive', period=1)
    result.plot()
    plt.show()
except Exception as e:
    print(f"Error in Seasonal Decomposition: {e}")

# 5. Split data into train and test sets
print("✂️ Splitting Data into Train and Test Sets")
train_size = int(len(filtered_df_log) * 0.8)
train = filtered_df_log.iloc[:train_size]
test = filtered_df_log.iloc[train_size:]
print(f"Training data points: {len(train)}")
print(f"Testing data points: {len(test)}")

# 6. Perform ADF test on training data
print("🧪 ADF Test on Training Data")
result = adfuller(train['Total'])
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"{key}: {value:.4f}")
if result[1] <= 0.05:
    print("Result: Reject the null hypothesis. The time series is stationary.")
else:
    print("Result: Fail to reject the null hypothesis. The time series is non-stationary.")

# 7. Differencing to make the series stationary if necessary
print("⟳ Differencing to Achieve Stationarity")
train_diff = train['Total'].diff().dropna()
plt.plot(train_diff)
plt.title("Differenced Data")
plt.show()

adf_test = adfuller(train_diff)
print(f"**ADF Statistic (Differenced):** {adf_test[0]:.4f}")
print(f"**p-value (Differenced):** {adf_test[1]:.4f}")

# 8. ACF and PACF Plots for Differenced Data
print("📈 ACF and PACF Plots for Differenced Data")
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(train_diff, ax=ax[0])
plot_pacf(train_diff, ax=ax[1])
plt.tight_layout()
plt.show()

# 9. Fit ARIMA model manually
print("🔧 ARIMA Model (Manual) Summary")
try:
    model = ARIMA(train['Total'], order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
except Exception as e:
    print(f"Error fitting ARIMA model: {e}")

# 10. Forecasting using ARIMA model
print("🔮 Forecasting with Manual ARIMA Model")
try:
    forecast_manual = model_fit.forecast(steps=len(test))
    forecast_manual_exp = np.exp(forecast_manual)
    test_exp = np.exp(test['Total'])

    plt.plot(test_exp, label='Actual', color='blue')
    plt.plot(forecast_manual_exp, label='Forecast Manual ARIMA', color='orange')
    plt.legend()
    plt.title("Forecast vs Actual - Manual ARIMA")
    plt.show()
except Exception as e:
    print(f"Error during forecasting with manual ARIMA: {e}")

# Further steps and analyses can be added following the same pattern

# 11. Residual Analysis for ARIMA
print("📉 Residual Analysis for ARIMA")
try:
    residuals = model_fit.resid
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1], color='red')
    plt.tight_layout()
    plt.show()

    # ACF and PACF for residuals
    print("📈 ACF and PACF for Residuals (ARIMA)")
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(residuals, ax=ax[0])
    plot_pacf(residuals, ax=ax[1])
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error during residual analysis for ARIMA: {e}")

# 12. Auto ARIMA model
print("🔧 Auto ARIMA Model Summary")
try:
    auto_arima_model = auto_arima(train['Total'], seasonal=False, stepwise=True, suppress_warnings=True)
    print(auto_arima_model.summary())
except Exception as e:
    print(f"Error fitting Auto ARIMA model: {e}")

# 13. Forecast using Auto ARIMA
print("🔮 Forecasting with Auto ARIMA Model")
try:
    forecast_auto = auto_arima_model.predict(n_periods=len(test))
    forecast_auto_exp = np.exp(forecast_auto)

    plt.plot(test_exp, label='Actual', color='blue')
    plt.plot(forecast_manual_exp, label='Forecast Manual ARIMA', color='orange')
    plt.plot(forecast_auto_exp, label='Forecast Auto ARIMA', color='green')
    plt.legend()
    plt.title("Forecast vs Actual - Manual and Auto ARIMA")
    plt.show()
except Exception as e:
    print(f"Error during forecasting with Auto ARIMA: {e}")

# 14. Fit Exponential Smoothing (ES) Model
print("🔧 Exponential Smoothing (ES) Model Summary")
try:
    es_model = ExponentialSmoothing(train['Total'], trend='add', seasonal=None)
    es_fit = es_model.fit()
    print(es_fit.summary())
except Exception as e:
    print(f"Error fitting Exponential Smoothing model: {e}")

# 15. Forecasting using ES model
print("🔮 Forecasting with Exponential Smoothing (ES) Model")
try:
    forecast_es = es_fit.forecast(steps=len(test))
    forecast_es_exp = np.exp(forecast_es)

    plt.plot(test_exp, label='Actual', color='blue')
    plt.plot(forecast_manual_exp, label='Forecast Manual ARIMA', color='orange')
    plt.plot(forecast_auto_exp, label='Forecast Auto ARIMA', color='green')
    plt.plot(forecast_es_exp, label='Forecast ES', color='purple')
    plt.legend()
    plt.title("Forecast vs Actual - Manual ARIMA, Auto ARIMA, and ES")
    plt.show()
except Exception as e:
    print(f"Error during forecasting with Exponential Smoothing: {e}")

# 16. Performance Metrics - Manual ARIMA
print("📊 Performance Metrics - Manual ARIMA")
try:
    mse_manual = mean_squared_error(test_exp, forecast_manual_exp)
    mae_manual = mean_absolute_error(test_exp, forecast_manual_exp)
    mape_manual = mean_absolute_percentage_error(test_exp, forecast_manual_exp)
    rmse_manual = np.sqrt(mse_manual)

    print(f"Manual ARIMA Performance Metrics:")
    print(f"  - MSE: {mse_manual:.2f}")
    print(f"  - MAE: {mae_manual:.2f}")
    print(f"  - MAPE: {mape_manual:.4f}")
    print(f"  - RMSE: {rmse_manual:.2f}")
except Exception as e:
    print(f"Error calculating performance metrics for Manual ARIMA: {e}")

# 17. Performance Metrics - Auto ARIMA
print("📊 Performance Metrics - Auto ARIMA")
try:
    mse_auto = mean_squared_error(test_exp, forecast_auto_exp)
    mae_auto = mean_absolute_error(test_exp, forecast_auto_exp)
    mape_auto = mean_absolute_percentage_error(test_exp, forecast_auto_exp)
    rmse_auto = np.sqrt(mse_auto)

    print(f"Auto ARIMA Performance Metrics:")
    print(f"  - MSE: {mse_auto:.2f}")
    print(f"  - MAE: {mae_auto:.2f}")
    print(f"  - MAPE: {mape_auto:.4f}")
    print(f"  - RMSE: {rmse_auto:.2f}")
except Exception as e:
    print(f"Error calculating performance metrics for Auto ARIMA: {e}")

# 18. Performance Metrics - Exponential Smoothing (ES)
print("📊 Performance Metrics - Exponential Smoothing (ES)")
try:
    mse_es = mean_squared_error(test_exp, forecast_es_exp)
    mae_es = mean_absolute_error(test_exp, forecast_es_exp)
    mape_es = mean_absolute_percentage_error(test_exp, forecast_es_exp)
    rmse_es = np.sqrt(mse_es)

    print(f"Exponential Smoothing (ES) Performance Metrics:")
    print(f"  - MSE: {mse_es:.2f}")
    print(f"  - MAE: {mae_es:.2f}")
    print(f"  - MAPE: {mape_es:.4f}")
    print(f"  - RMSE: {rmse_es:.2f}")
except Exception as e:
    print(f"Error calculating performance metrics for Exponential Smoothing: {e}")

# 19. Complete Data with Forecasts
print("📊 Complete Data with Forecasts")
try:
    complete_df = pd.DataFrame({
        'Actual': test_exp,
        'Forecast Manual ARIMA': forecast_manual_exp,
        'Forecast Auto ARIMA': forecast_auto_exp,
        'Forecast ES': forecast_es_exp
    })
    print(complete_df)
except Exception as e:
    print(f"Error displaying complete data with forecasts: {e}")
