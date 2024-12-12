import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Streamlit UI setup
st.title("📈 Time Series Analysis and Forecasting with Machine Learning")

# Load your data with caching to improve performance
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data('data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv')

# 1. Show initial dataframe
st.subheader("🔍 Initial Data")
st.write(df)

# 2. Data preprocessing: Convert year to datetime, sort, and set index
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df = df.sort_values(by='Year')
df = df.set_index('Year')

# Show updated dataframe after preprocessing
st.subheader("🚠 Data after DateTime Conversion, Sorting, and Setting Index")
st.write(df)

# 3. Select country and check available emission categories
st.sidebar.header("🔧 Select Parameters")
country = st.sidebar.selectbox('Select Country:', df['Country'].unique())

# Filter the dataframe based on selected country
filtered_df = df[df['Country'] == country]

# 4. Select emission category ('Total' in this case)
category = 'Total'

# 5. Filter data based on user selection and show head of filtered dataframe
filtered_df = filtered_df[['Total']]
st.subheader(f"📂 Filtered Data for **{country}** - **{category}** (After Selecting Columns)")
st.write(filtered_df.head())

# Button to start analysis
if st.button("🚀 Start Analysis"):
    st.success("Starting the analysis...")

    # 1. Apply log transformation
    filtered_df_log = np.log(filtered_df)
    st.subheader(f"⟳ Log-transformed Data for **{country}** - **{category}**")
    st.line_chart(filtered_df_log)

    # 2. Calculate SMA and EMA with window size 10
    window_size = 10
    filtered_df_log['SMA10'] = filtered_df_log['Total'].rolling(window=window_size).mean()
    filtered_df_log['EMA10'] = filtered_df_log['Total'].ewm(span=window_size, adjust=False).mean()

    # 3. Plot Original Log Data, SMA10, and EMA10
    st.subheader("📊 Simple Moving Average (SMA) and Exponential Moving Average (EMA)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_df_log.index, filtered_df_log['Total'], label='Original Log Data', color='blue')
    ax.plot(filtered_df_log.index, filtered_df_log['SMA10'], label='SMA10', color='orange')
    ax.plot(filtered_df_log.index, filtered_df_log['EMA10'], label='EMA10', color='green')
    ax.set_title(f"SMA10 and EMA10 for {country} - {category}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Log Emission Value")
    ax.legend()
    st.pyplot(fig)

    # 4. ETS Decomposition
    st.subheader("🔍 ETS Decomposition")
    try:
        result = seasonal_decompose(filtered_df_log['Total'], model='additive', period=1)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
        result.observed.plot(ax=ax1, title='Observed')
        result.trend.plot(ax=ax2, title='Trend')
        result.seasonal.plot(ax=ax3, title='Seasonal')
        result.resid.plot(ax=ax4, title='Residual')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in Seasonal Decomposition: {e}")

    # 5. Split data into train and test sets
    st.subheader("✂️ Splitting Data into Train and Test Sets")
    train_size = int(len(filtered_df_log) * 0.8)
    train = filtered_df_log.iloc[:train_size]
    test = filtered_df_log.iloc[train_size:]
    st.write(f"Training data points: {len(train)}")
    st.write(f"Testing data points: {len(test)}")

    # 6. Perform ADF test on training data
    st.subheader("🧪 ADF Test on Training Data")
    result = adfuller(train['Total'])
    st.write(f"**ADF Statistic:** {result[0]:.4f}")
    st.write(f"**p-value:** {result[1]:.4f}")
    st.write("**Critical Values:**")
    for key, value in result[4].items():
        st.write(f"\t{key}: {value:.4f}")
    if result[1] <= 0.05:
        st.write("**Result:** Reject the null hypothesis. The time series is stationary.")
    else:
        st.write("**Result:** Fail to reject the null hypothesis. The time series is non-stationary.")

    # 7. Differencing to make the series stationary if necessary
    st.subheader("⟳ Differencing to Achieve Stationarity")
    train_diff = train['Total'].diff().dropna()
    st.line_chart(train_diff)

    # 8. ACF and PACF Plots for Differenced Data
    st.subheader("📈 ACF and PACF Plots for Differenced Data")
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(train_diff, ax=ax[0])
    plot_pacf(train_diff, ax=ax[1])
    plt.tight_layout()
    st.pyplot(fig)

    # Perform ADF test on differenced data
    adf_test = adfuller(train_diff)
    st.write(f"**ADF Statistic (Differenced):** {adf_test[0]:.4f}")
    st.write(f"**p-value (Differenced):** {adf_test[1]:.4f}")
    if adf_test[1] <= 0.05:
        st.write("**Result:** Reject the null hypothesis. The differenced time series is stationary.")
    else:
        st.write("**Result:** Fail to reject the null hypothesis. The differenced time series is non-stationary.")

    # 9. Fit ARIMA model manually
    st.subheader("🔧 ARIMA Model (Manual) Summary")
    try:
        model = ARIMA(train['Total'], order=(1, 1, 1))
        model_fit = model.fit()
        st.text(model_fit.summary())
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {e}")

    # 10. Forecasting using ARIMA model
    st.subheader("🔮 Forecasting with Manual ARIMA Model")
    try:
        forecast_manual = model_fit.forecast(steps=len(test))
        # Exponentiate to revert log-transformation
        forecast_manual_exp = np.exp(forecast_manual)
        test_exp = np.exp(test['Total'])

        # Prepare dataframe for plotting
        forecast_manual_plot = pd.DataFrame({
            'Actual': test_exp,
            'Forecast Manual ARIMA': forecast_manual_exp
        }, index=test.index)

        st.line_chart(forecast_manual_plot)

    except Exception as e:
        st.error(f"Error during forecasting with manual ARIMA: {e}")

    # 11. Residual Analysis for ARIMA
    st.subheader("📉 Residual Analysis for ARIMA")
    try:
        residuals = model_fit.resid
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        residuals.plot(title="Residuals", ax=ax[0])
        residuals.plot(kind='kde', title='Density', ax=ax[1], color='red')
        plt.tight_layout()
        st.pyplot(fig)

        # ACF and PACF for residuals
        st.subheader("📈 ACF and PACF for Residuals (ARIMA)")
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(residuals, ax=ax[0])
        plot_pacf(residuals, ax=ax[1])
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during residual analysis for ARIMA: {e}")

    # 12. Auto ARIMA model
    st.subheader("🔧 Auto ARIMA Model Summary")
    try:
        auto_arima_model = auto_arima(train['Total'], seasonal=False, stepwise=True, suppress_warnings=True)
        st.text(auto_arima_model.summary())
    except Exception as e:
        st.error(f"Error fitting Auto ARIMA model: {e}")

    # 13. Forecast using Auto ARIMA
    st.subheader("🔮 Forecasting with Auto ARIMA Model")
    try:
        forecast_auto = auto_arima_model.predict(n_periods=len(test))
        forecast_auto_exp = np.exp(forecast_auto)

        # Prepare dataframe for plotting
        forecast_auto_plot = pd.DataFrame({
            'Actual': test_exp,
            'Forecast Manual ARIMA': forecast_manual_exp,
            'Forecast Auto ARIMA': forecast_auto_exp
        }, index=test.index)

        st.line_chart(forecast_auto_plot)

    except Exception as e:
        st.error(f"Error during forecasting with Auto ARIMA: {e}")

    # 14. Fit Exponential Smoothing (ES) Model
    st.subheader("🔧 Exponential Smoothing (ES) Model Summary")
    try:
        es_model = ExponentialSmoothing(train['Total'], trend='add', seasonal=None)
        es_fit = es_model.fit()
        st.text(es_fit.summary())
    except Exception as e:
        st.error(f"Error fitting Exponential Smoothing model: {e}")

    # 15. Forecasting using ES model
    st.subheader("🔮 Forecasting with Exponential Smoothing (ES) Model")
    try:
        forecast_es = es_fit.forecast(steps=len(test))
        forecast_es_exp = np.exp(forecast_es)

        # Prepare dataframe for plotting
        forecast_es_plot = pd.DataFrame({
            'Actual': test_exp,
            'Forecast Manual ARIMA': forecast_manual_exp,
            'Forecast Auto ARIMA': forecast_auto_exp,
            'Forecast ES': forecast_es_exp
        }, index=test.index)

        st.line_chart(forecast_es_plot)

    except Exception as e:
        st.error(f"Error during forecasting with Exponential Smoothing: {e}")

    # 16. Performance Metrics - Manual ARIMA
    st.subheader("📊 Performance Metrics - Manual ARIMA")
    try:
        mse_manual = mean_squared_error(test_exp, forecast_manual_exp)
        mae_manual = mean_absolute_error(test_exp, forecast_manual_exp)
        mape_manual = mean_absolute_percentage_error(test_exp, forecast_manual_exp)
        rmse_manual = np.sqrt(mse_manual)

        st.markdown(
            f"""
            <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border: 1px solid #e6e6e6;">
                <h4 style="color:#333;">**Manual ARIMA Performance Metrics:**</h4>
                <ul style="list-style-type:none;">
                    <li><strong>MSE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mse_manual:.2f}</span></li>
                    <li><strong>MAE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mae_manual:.2f}</span></li>
                    <li><strong>MAPE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mape_manual:.4f}</span></li>
                    <li><strong>RMSE:</strong> <span style="color:#FF6347; font-size:1.2em;">{rmse_manual:.2f}</span></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error calculating performance metrics for Manual ARIMA: {e}")

    # 17. Performance Metrics - Auto ARIMA
    st.subheader("📊 Performance Metrics - Auto ARIMA")
    try:
        mse_auto = mean_squared_error(test_exp, forecast_auto_exp)
        mae_auto = mean_absolute_error(test_exp, forecast_auto_exp)
        mape_auto = mean_absolute_percentage_error(test_exp, forecast_auto_exp)
        rmse_auto = np.sqrt(mse_auto)

        st.markdown(
            f"""
            <div style="background-color:#f0f0f7; padding:10px; border-radius:5px; border: 1px solid #e6e6e6;">
                <h4 style="color:#333;">**Auto ARIMA Performance Metrics:**</h4>
                <ul style="list-style-type:none;">
                    <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse_auto:.2f}</span></li>
                    <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae_auto:.2f}</span></li>
                    <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape_auto:.4f}</span></li>
                    <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse_auto:.2f}</span></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error calculating performance metrics for Auto ARIMA: {e}")

    # 18. Performance Metrics - Exponential Smoothing (ES)
    st.subheader("📊 Performance Metrics - Exponential Smoothing (ES)")
    try:
        mse_es = mean_squared_error(test_exp, forecast_es_exp)
        mae_es = mean_absolute_error(test_exp, forecast_es_exp)
        mape_es = mean_absolute_percentage_error(test_exp, forecast_es_exp)
        rmse_es = np.sqrt(mse_es)

        st.markdown(
            f"""
            <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; border: 1px solid #b3e0ff;">
                <h4 style="color:#333;">**Exponential Smoothing (ES) Performance Metrics:**</h4>
                <ul style="list-style-type:none;">
                    <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse_es:.2f}</span></li>
                    <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae_es:.2f}</span></li>
                    <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape_es:.4f}</span></li>
                    <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse_es:.2f}</span></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error calculating performance metrics for Exponential Smoothing: {e}")

    # 19. Complete Data with Forecasts
    st.subheader("📊 Complete Data with Forecasts")
    try:
        complete_df = pd.DataFrame({
            'Actual': test_exp,
            'Forecast Manual ARIMA': forecast_manual_exp,
            'Forecast Auto ARIMA': forecast_auto_exp,
            'Forecast ES': forecast_es_exp
        })
        st.write(complete_df)
    except Exception as e:
        st.error(f"Error displaying complete data with forecasts: {e}")

    st.markdown("""
    ---
    **Data Source:** `fossil_fuel_co2_emissions-by-nation_with_continent.csv`
    """)
