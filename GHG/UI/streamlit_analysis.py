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

st.set_page_config(page_title="Greenhouse Gas Emissions Analysis", layout="wide")


# Streamlit UI setup
st.title("📈 Time Series Analysis and Forecasting with Machine Learning")

# Load your data with caching to improve performance
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data('data/raw/greenhouse_gas_inventory_data_completed.csv')

# 1. Show initial dataframe
st.subheader("🔍 Initial Data")
st.write(df)

# 2. Data preprocessing: Convert year to datetime, sort, and set index
df['year'] = pd.to_datetime(df['year'], format='%Y')
df = df.sort_values(by='year')
df = df.set_index('year')

# Show updated dataframe after preprocessing
st.subheader("🛠️ Data after DateTime Conversion, Sorting, and Setting Index")
st.write(df)

# 3. **Đặt giá trị mặc định cho Country và Category**
default_country = 'Germany'
default_category = 'CO2 Emissions'

# Kiểm tra xem giá trị mặc định có tồn tại trong dữ liệu không
if default_country not in df['country_or_area'].unique():
    st.error(f"Country '{default_country}' không tồn tại trong dữ liệu.")
else:
    # 4. **Lọc dữ liệu dựa trên Country và Category mặc định**
    available_categories = df[df['country_or_area'] == default_country]['category'].unique()
    
    if default_category not in available_categories:
        st.error(f"Category '{default_category}' không tồn tại cho country '{default_country}'.")
    else:
        # 5. Lọc dữ liệu dựa trên chọn lọc mặc định và hiển thị head của dataframe đã lọc
        filtered_df = df[(df['country_or_area'] == default_country) & (df['category'] == default_category)]
        st.subheader(f"📂 Filtered Data for **{default_country}** - **{default_category}** (Before Dropping Columns)")
        st.write(filtered_df.head())
        
        # 6. Drop unnecessary columns and show updated head of filtered dataframe
        filtered_df = filtered_df.drop(['country_or_area', 'category', 'continent'], axis=1)
        st.subheader(f"📂 Filtered Data for **{default_country}** - **{default_category}** (After Dropping Columns)")
        st.write(filtered_df.head())
        
        # **Tự động bắt đầu phân tích mà không cần nút "Start Analysis"**
        st.success("Start analyze auto")
        
        # 1. Apply log transformation
        filtered_df_log = np.log(filtered_df)
        st.subheader(f"🔄 Log-transformed Data for **{default_country}** - **{default_category}**")
        st.line_chart(filtered_df_log)
        
        # 2. Calculate SMA and EMA with window size 5
        window_size = 5
        filtered_df_log['SMA5'] = filtered_df_log['value'].rolling(window=window_size).mean()
        filtered_df_log['EMA5'] = filtered_df_log['value'].ewm(span=window_size, adjust=False).mean()
        
        # 3. Plot Original Log Data, SMA5, and EMA5
        st.subheader("📊 Simple Moving Average (SMA) và Exponential Moving Average (EMA)")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(filtered_df_log.index, filtered_df_log['value'], label='Original Log Data', color='blue')
        ax.plot(filtered_df_log.index, filtered_df_log['SMA5'], label='SMA5', color='orange')
        ax.plot(filtered_df_log.index, filtered_df_log['EMA5'], label='EMA5', color='green')
        ax.set_title(f"SMA5 và EMA5 cho {default_country} - {default_category}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Log Emission Value")
        ax.legend()
        st.pyplot(fig)
        
        # 4. ETS Decomposition
        st.subheader("🔍 ETS Decomposition")
        try:
            # Xác định period dựa trên tần suất dữ liệu của bạn
            result = seasonal_decompose(filtered_df_log['value'], model='additive', period=1)
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
        result = adfuller(train['value'])
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
        st.subheader("🔄 Differencing to Achieve Stationarity")
        train_diff = train['value'].diff().dropna()
        st.line_chart(train_diff)
        
        # 8. ACF and PACF Plots for Differenced Data
        st.subheader("📈 ACF và PACF Plots cho Differenced Data")
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
        #st.subheader("🔧 ARIMA Model (Manual) Summary")
        try:
            model = ARIMA(train['value'], order=(1, 1, 1))
            model_fit = model.fit()
            #st.text(model_fit.summary())
        except Exception as e:
            st.error(f"Error fitting ARIMA model: {e}")
        
        # 10. Residual Analysis for ARIMA
        # st.subheader("📉 Residual Analysis for ARIMA")
        # try:
        #     residuals = model_fit.resid
        #     fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        #     residuals.plot(title="Residuals", ax=ax[0])
        #     residuals.plot(kind='kde', title='Density', ax=ax[1], color='red')
        #     plt.tight_layout()
        #     st.pyplot(fig)
        
        #     # ACF và PACF cho residuals
        #     st.subheader("📈 ACF và PACF cho Residuals (ARIMA)")
        #     fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        #     plot_acf(residuals, ax=ax[0])
        #     plot_pacf(residuals, ax=ax[1])
        #     plt.tight_layout()
        #     st.pyplot(fig)
        # except Exception as e:
        #     st.error(f"Error during residual analysis for ARIMA: {e}")
        
        # 11. Forecasting using ARIMA model
        # st.subheader("🔮 Forecasting with Manual ARIMA Model")
        try:
            forecast_manual = model_fit.forecast(steps=len(test))
            # Exponentiate to revert log-transformation
            forecast_manual_exp = np.exp(forecast_manual)
            test_exp = np.exp(test['value'])
        
            # Prepare dataframe for plotting
            # forecast_manual_plot = pd.DataFrame({
            #     'Actual': test_exp,
            #     'Forecast Manual ARIMA': forecast_manual_exp
            # }, index=test.index)
        
            # st.line_chart(forecast_manual_plot)
        
            # Add to original dataframe
            filtered_df['Forecast Manual ARIMA'] = [None]*train_size + list(forecast_manual_exp)
        except Exception as e:
            st.error(f"Error during forecasting with manual ARIMA: {e}")
        
        # 12. Auto ARIMA model
        # st.subheader("🔧 Auto ARIMA Model Summary")
        try:
            auto_arima_model = auto_arima(train['value'], seasonal=False, stepwise=True, suppress_warnings=True)
            #st.text(auto_arima_model.summary())
        except Exception as e:
            st.error(f"Error fitting Auto ARIMA model: {e}")
        
        # 13. Forecast using Auto ARIMA
        #st.subheader("🔮 Forecasting with Auto ARIMA Model")
        try:
            forecast_auto = auto_arima_model.predict(n_periods=len(test))
            forecast_auto_exp = np.exp(forecast_auto)
        
            # # Prepare dataframe for plotting
            # forecast_auto_plot = pd.DataFrame({
            #     'Actual': test_exp,
            #     'Forecast Manual ARIMA': forecast_manual_exp,
            #     'Forecast Auto ARIMA': forecast_auto_exp
            # }, index=test.index)
        
            # st.line_chart(forecast_auto_plot)
        
            # Add to original dataframe
            filtered_df['Forecast Auto ARIMA'] = [None]*train_size + list(forecast_auto_exp)
        except Exception as e:
            st.error(f"Error during forecasting with Auto ARIMA: {e}")
        
        # 14. Fit Exponential Smoothing (ES) Model
        #st.subheader("🔧 Exponential Smoothing (ES) Model Summary")
        try:
            # Initialize and fit the ES model on the log-transformed training data
            # Adjust the trend và seasonal components dựa trên đặc điểm dữ liệu
            es_model = ExponentialSmoothing(train['value'], trend='add', seasonal=None)
            es_fit = es_model.fit()
            #st.text(es_fit.summary())
        except Exception as e:
            st.error(f"Error fitting Exponential Smoothing model: {e}")
        
        # 15. Forecasting using ES model
        st.subheader("🔮 Forecasting with Time Series Model")
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
        

            # # Tạo biểu đồ trực quan với matplotlib
            # plt.figure(figsize=(12, 6))

            # # Plot dữ liệu thực tế
            # plt.plot(test.index, test_exp, label='Actual', color='blue', linewidth=2, linestyle='-')

            # # Plot các dự báo
            # plt.plot(test.index, forecast_manual_exp, label='Forecast Manual ARIMA', color='red', linewidth=2, linestyle='--')
            # plt.plot(test.index, forecast_auto_exp, label='Forecast Auto ARIMA', color='green', linewidth=2, linestyle='-.')
            # plt.plot(test.index, forecast_es_exp, label='Forecast ES', color='purple', linewidth=2, linestyle=':')

            # # Thêm tiêu đề, trục, và chú thích
            # plt.title('Forecasting with Time Series Models', fontsize=16)
            # plt.xlabel('Year', fontsize=12)
            # plt.ylabel('Emission Values', fontsize=12)
            # plt.legend(loc='lower left', fontsize=10)
            # plt.grid(True, linestyle='--', alpha=0.6)

            # # Hiển thị biểu đồ trong Streamlit
            # st.pyplot(plt)
            
        
            # Add to original dataframe
            filtered_df['Forecast ES'] = [None]*train_size + list(forecast_es_exp)
        except Exception as e:
            st.error(f"Error during forecasting with Exponential Smoothing: {e}")
        
        # 16. Metrics calculation - Manual ARIMA
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
        
        # 17. Metrics calculation - Auto ARIMA
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
        
        # 18. Metrics calculation - Exponential Smoothing (ES)
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
        
        # 19. Display the complete dataframe with forecasts
        st.subheader("📊 Complete Data with Forecasts")
        st.write(filtered_df)
        
        st.markdown("""
        ---
        **Data Source:** `greenhouse_gas_inventory_data_completed.csv`
        """)

