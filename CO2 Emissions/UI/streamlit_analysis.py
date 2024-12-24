# # import numpy as np
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import streamlit as st
# # import os
# # from statsmodels.tsa.stattools import adfuller
# # from statsmodels.tsa.seasonal import seasonal_decompose
# # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# # from pmdarima.arima import auto_arima
# # from statsmodels.tsa.arima.model import ARIMA
# # from statsmodels.tsa.holtwinters import ExponentialSmoothing

# # from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# # output_dir_result = './CO2 Emissions/Result/'
# # if not os.path.exists(output_dir_result):
# #     os.makedirs(output_dir_result)

# # output_dir_model = './CO2 Emissions/Models/'
# # if not os.path.exists(output_dir_model):
# #     os.makedirs(output_dir_model)

# # # Streamlit UI setup
# # st.title("📈 Time Series Analysis and Forecasting with Machine Learning")

# # # Load your data with caching to improve performance
# # @st.cache_data
# # def load_data(file_path):
# #     return pd.read_csv(file_path)

# # df = load_data('data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv')

# # # 1. Show initial dataframe
# # st.subheader("🔍 Initial Data")
# # st.write(df)

# # # 2. Data preprocessing: Convert year to datetime, sort, and set index
# # df['Year'] = pd.to_datetime(df['Year'], format='%Y')
# # df = df.sort_values(by='Year')
# # df = df.set_index('Year')

# # # Show updated dataframe after preprocessing
# # st.subheader("🚠 Data after DateTime Conversion, Sorting, and Setting Index")
# # st.write(df)

# # # 3. Select country and check available emission categories
# # st.sidebar.header("🔧 Select Parameters")
# # country = st.sidebar.selectbox('Select Country:', df['Country'].unique())

# # # Filter the dataframe based on selected country
# # filtered_df = df[df['Country'] == country]

# # # 4. Select emission category ('Total' in this case)
# # category = 'Total'

# # # 5. Filter data based on user selection and show head of filtered dataframe
# # filtered_df = filtered_df[['Total']]
# # st.subheader(f"📂 Filtered Data for **{country}** - **{category}** (After Selecting Columns)")
# # st.write(filtered_df.head())

# # # Button to start analysis
# # if st.button("🚀 Start Analysis"):
# #     st.success("Starting the analysis...")

# #     # 1. Apply log transformation
# #     filtered_df_log = np.log(filtered_df)
# #     st.subheader(f"⟳ Log-transformed Data for **{country}** - **{category}**")
# #     st.line_chart(filtered_df_log)

# #     # 2. Calculate SMA and EMA with window size 10
# #     window_size = 10
# #     filtered_df_log['SMA10'] = filtered_df_log['Total'].rolling(window=window_size).mean()
# #     filtered_df_log['EMA10'] = filtered_df_log['Total'].ewm(span=window_size, adjust=False).mean()

# #     # 3. Plot Original Log Data, SMA10, and EMA10
# #     st.subheader("📊 Simple Moving Average (SMA) and Exponential Moving Average (EMA)")
# #     fig, ax = plt.subplots(figsize=(12, 6))
# #     ax.plot(filtered_df_log.index, filtered_df_log['Total'], label='Original Log Data', color='blue')
# #     ax.plot(filtered_df_log.index, filtered_df_log['SMA10'], label='SMA10', color='orange')
# #     ax.plot(filtered_df_log.index, filtered_df_log['EMA10'], label='EMA10', color='green')
# #     ax.set_title(f"SMA10 and EMA10 for {country} - {category}")
# #     ax.set_xlabel("Year")
# #     ax.set_ylabel("Log Emission Value")
# #     ax.legend()
# #     st.pyplot(fig)

# #     # 4. ETS Decomposition
# #     st.subheader("🔍 ETS Decomposition")
# #     try:
# #         result = seasonal_decompose(filtered_df_log['Total'], model='additive', period=1)
# #         fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
# #         result.observed.plot(ax=ax1, title='Observed')
# #         result.trend.plot(ax=ax2, title='Trend')
# #         result.seasonal.plot(ax=ax3, title='Seasonal')
# #         result.resid.plot(ax=ax4, title='Residual')
# #         plt.tight_layout()
# #         st.pyplot(fig)
# #     except Exception as e:
# #         st.error(f"Error in Seasonal Decomposition: {e}")

# #     # 5. Split data into train and test sets
# #     st.subheader("✂️ Splitting Data into Train and Test Sets")
# #     train_size = int(len(filtered_df_log) * 0.8)
# #     train = filtered_df_log.iloc[:train_size]
# #     test = filtered_df_log.iloc[train_size:]
# #     st.write(f"Training data points: {len(train)}")
# #     st.write(f"Testing data points: {len(test)}")

# #     # 6. Perform ADF test on training data
# #     st.subheader("🧪 ADF Test on Training Data")
# #     result = adfuller(train['Total'])
# #     st.write(f"**ADF Statistic:** {result[0]:.4f}")
# #     st.write(f"**p-value:** {result[1]:.4f}")
# #     st.write("**Critical Values:**")
# #     for key, value in result[4].items():
# #         st.write(f"\t{key}: {value:.4f}")
# #     if result[1] <= 0.05:
# #         st.write("**Result:** Reject the null hypothesis. The time series is stationary.")
# #     else:
# #         st.write("**Result:** Fail to reject the null hypothesis. The time series is non-stationary.")

# #     # 7. Differencing to make the series stationary if necessary
# #     st.subheader("⟳ Differencing to Achieve Stationarity")
# #     train_diff = train['Total'].diff().dropna()
# #     st.line_chart(train_diff)

# #     # 8. ACF and PACF Plots for Differenced Data
# #     st.subheader("📈 ACF and PACF Plots for Differenced Data")
# #     fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# #     plot_acf(train_diff, ax=ax[0])
# #     plot_pacf(train_diff, ax=ax[1])
# #     plt.tight_layout()
# #     st.pyplot(fig)

# #     # Perform ADF test on differenced data
# #     adf_test = adfuller(train_diff)
# #     st.write(f"**ADF Statistic (Differenced):** {adf_test[0]:.4f}")
# #     st.write(f"**p-value (Differenced):** {adf_test[1]:.4f}")
# #     if adf_test[1] <= 0.05:
# #         st.write("**Result:** Reject the null hypothesis. The differenced time series is stationary.")
# #     else:
# #         st.write("**Result:** Fail to reject the null hypothesis. The differenced time series is non-stationary.")

# #     # 9. Fit ARIMA model manually
# #     st.subheader("🔧 ARIMA Model (Manual) Summary")
# #     try:
# #         model = ARIMA(train['Total'], order=(1, 1, 1))
# #         model_fit = model.fit()
# #         st.text(model_fit.summary())
# #     except Exception as e:
# #         st.error(f"Error fitting ARIMA model: {e}")

# #     # 10. Forecasting using ARIMA model
# #     st.subheader("🔮 Forecasting with Manual ARIMA Model")
# #     try:
# #         forecast_manual = model_fit.forecast(steps=len(test))
# #         # Exponentiate to revert log-transformation
# #         forecast_manual_exp = np.exp(forecast_manual)
# #         test_exp = np.exp(test['Total'])

# #         # Prepare dataframe for plotting
# #         forecast_manual_plot = pd.DataFrame({
# #             'Actual': test_exp,
# #             'Forecast Manual ARIMA': forecast_manual_exp
# #         }, index=test.index)

# #         st.line_chart(forecast_manual_plot)

# #     except Exception as e:
# #         st.error(f"Error during forecasting with manual ARIMA: {e}")

# #     # 11. Residual Analysis for ARIMA
# #     st.subheader("📉 Residual Analysis for ARIMA")
# #     try:
# #         residuals = model_fit.resid
# #         fig, ax = plt.subplots(1, 2, figsize=(14, 5))
# #         residuals.plot(title="Residuals", ax=ax[0])
# #         residuals.plot(kind='kde', title='Density', ax=ax[1], color='red')
# #         plt.tight_layout()
# #         st.pyplot(fig)

# #         # ACF and PACF for residuals
# #         st.subheader("📈 ACF and PACF for Residuals (ARIMA)")
# #         fig, ax = plt.subplots(2, 1, figsize=(12, 8))
# #         plot_acf(residuals, ax=ax[0])
# #         plot_pacf(residuals, ax=ax[1])
# #         plt.tight_layout()
# #         st.pyplot(fig)
# #     except Exception as e:
# #         st.error(f"Error during residual analysis for ARIMA: {e}")

# #     # 12. Auto ARIMA model
# #     st.subheader("🔧 Auto ARIMA Model Summary")
# #     try:
# #         auto_arima_model = auto_arima(train['Total'], seasonal=False, stepwise=True, suppress_warnings=True)
# #         st.text(auto_arima_model.summary())
# #     except Exception as e:
# #         st.error(f"Error fitting Auto ARIMA model: {e}")

# #     # 13. Forecast using Auto ARIMA
# #     st.subheader("🔮 Forecasting with Auto ARIMA Model")
# #     try:
# #         forecast_auto = auto_arima_model.predict(n_periods=len(test))
# #         forecast_auto_exp = np.exp(forecast_auto)

# #         # Prepare dataframe for plotting
# #         forecast_auto_plot = pd.DataFrame({
# #             'Actual': test_exp,
# #             'Forecast Manual ARIMA': forecast_manual_exp,
# #             'Forecast Auto ARIMA': forecast_auto_exp
# #         }, index=test.index)

# #         st.line_chart(forecast_auto_plot)

# #     except Exception as e:
# #         st.error(f"Error during forecasting with Auto ARIMA: {e}")

# #     # 14. Fit Exponential Smoothing (ES) Model
# #     st.subheader("🔧 Exponential Smoothing (ES) Model Summary")
# #     try:
# #         es_model = ExponentialSmoothing(train['Total'], trend='add', seasonal=None)
# #         es_fit = es_model.fit()
# #         st.text(es_fit.summary())
# #     except Exception as e:
# #         st.error(f"Error fitting Exponential Smoothing model: {e}")

# #     # 15. Forecasting using ES model
# #     st.subheader("🔮 Forecasting with Exponential Smoothing (ES) Model")
# #     try:
# #         forecast_es = es_fit.forecast(steps=len(test))
# #         forecast_es_exp = np.exp(forecast_es)

# #         # Prepare dataframe for plotting
# #         forecast_es_plot = pd.DataFrame({
# #             'Actual': test_exp,
# #             'Forecast Manual ARIMA': forecast_manual_exp,
# #             'Forecast Auto ARIMA': forecast_auto_exp,
# #             'Forecast ES': forecast_es_exp
# #         }, index=test.index)

# #         st.line_chart(forecast_es_plot)

# #     except Exception as e:
# #         st.error(f"Error during forecasting with Exponential Smoothing: {e}")

# #     # 16. Performance Metrics - Manual ARIMA
# #     st.subheader("📊 Performance Metrics - Manual ARIMA")
# #     try:
# #         mse_manual = mean_squared_error(test_exp, forecast_manual_exp)
# #         mae_manual = mean_absolute_error(test_exp, forecast_manual_exp)
# #         mape_manual = mean_absolute_percentage_error(test_exp, forecast_manual_exp)
# #         rmse_manual = np.sqrt(mse_manual)

# #         st.markdown(
# #             f"""
# #             <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border: 1px solid #e6e6e6;">
# #                 <h4 style="color:#333;">**Manual ARIMA Performance Metrics:**</h4>
# #                 <ul style="list-style-type:none;">
# #                     <li><strong>MSE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mse_manual:.2f}</span></li>
# #                     <li><strong>MAE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mae_manual:.2f}</span></li>
# #                     <li><strong>MAPE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mape_manual:.4f}</span></li>
# #                     <li><strong>RMSE:</strong> <span style="color:#FF6347; font-size:1.2em;">{rmse_manual:.2f}</span></li>
# #                 </ul>
# #             </div>
# #             """,
# #             unsafe_allow_html=True
# #         )
# #     except Exception as e:
# #         st.error(f"Error calculating performance metrics for Manual ARIMA: {e}")

# #     # 17. Performance Metrics - Auto ARIMA
# #     st.subheader("📊 Performance Metrics - Auto ARIMA")
# #     try:
# #         mse_auto = mean_squared_error(test_exp, forecast_auto_exp)
# #         mae_auto = mean_absolute_error(test_exp, forecast_auto_exp)
# #         mape_auto = mean_absolute_percentage_error(test_exp, forecast_auto_exp)
# #         rmse_auto = np.sqrt(mse_auto)

# #         st.markdown(
# #             f"""
# #             <div style="background-color:#f0f0f7; padding:10px; border-radius:5px; border: 1px solid #e6e6e6;">
# #                 <h4 style="color:#333;">**Auto ARIMA Performance Metrics:**</h4>
# #                 <ul style="list-style-type:none;">
# #                     <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse_auto:.2f}</span></li>
# #                     <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae_auto:.2f}</span></li>
# #                     <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape_auto:.4f}</span></li>
# #                     <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse_auto:.2f}</span></li>
# #                 </ul>
# #             </div>
# #             """,
# #             unsafe_allow_html=True
# #         )
# #     except Exception as e:
# #         st.error(f"Error calculating performance metrics for Auto ARIMA: {e}")

# #     # 18. Performance Metrics - Exponential Smoothing (ES)
# #     st.subheader("📊 Performance Metrics - Exponential Smoothing (ES)")
# #     try:
# #         mse_es = mean_squared_error(test_exp, forecast_es_exp)
# #         mae_es = mean_absolute_error(test_exp, forecast_es_exp)
# #         mape_es = mean_absolute_percentage_error(test_exp, forecast_es_exp)
# #         rmse_es = np.sqrt(mse_es)

# #         st.markdown(
# #             f"""
# #             <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; border: 1px solid #b3e0ff;">
# #                 <h4 style="color:#333;">**Exponential Smoothing (ES) Performance Metrics:**</h4>
# #                 <ul style="list-style-type:none;">
# #                     <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse_es:.2f}</span></li>
# #                     <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae_es:.2f}</span></li>
# #                     <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape_es:.4f}</span></li>
# #                     <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse_es:.2f}</span></li>
# #                 </ul>
# #             </div>
# #             """,
# #             unsafe_allow_html=True
# #         )
# #     except Exception as e:
# #         st.error(f"Error calculating performance metrics for Exponential Smoothing: {e}")

# #     # 19. Complete Data with Forecasts
# #     st.subheader("📊 Complete Data with Forecasts")
# #     try:
# #         complete_df = pd.DataFrame({
# #             'Actual': test_exp,
# #             'Forecast Manual ARIMA': forecast_manual_exp,
# #             'Forecast Auto ARIMA': forecast_auto_exp,
# #             'Forecast ES': forecast_es_exp
# #         })
# #         st.write(complete_df)
# #     except Exception as e:
# #         st.error(f"Error displaying complete data with forecasts: {e}")

# #     st.markdown("""
# #     ---
# #     **Data Source:** `fossil_fuel_co2_emissions-by-nation_with_continent.csv`
# #     """)



# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import streamlit as st
# import joblib

# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# from pmdarima.arima import auto_arima
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing

# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# # Ensure output directories exist
# output_dir_result = './CO2 Emissions/Result/'
# if not os.path.exists(output_dir_result):
#     os.makedirs(output_dir_result)

# output_dir_model = './CO2 Emissions/Models/'
# if not os.path.exists(output_dir_model):
#     os.makedirs(output_dir_model)

# # Streamlit UI setup
# st.title("📈 Time Series Analysis and Forecasting with Machine Learning")

# # Load your data with caching to improve performance
# @st.cache_data
# def load_data(file_path):
#     return pd.read_csv(file_path)

# df = load_data('data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv')

# # 1. Show initial dataframe
# st.subheader("🔍 Initial Data")
# st.write(df)

# # 2. Data preprocessing: Convert year to datetime, sort, and set index
# df['Year'] = pd.to_datetime(df['Year'], format='%Y')
# df = df.sort_values(by='Year')
# df = df.set_index('Year')

# # Show updated dataframe after preprocessing
# st.subheader("🚠 Data after DateTime Conversion, Sorting, and Setting Index")
# st.write(df)

# # 3. Select country and check available emission categories
# st.sidebar.header("🔧 Select Parameters")
# country = st.sidebar.selectbox('Select Country:', df['Country'].unique())

# # Filter the dataframe based on selected country
# filtered_df = df[df['Country'] == country]

# # 4. Select emission category ('Total' in this case)
# category = 'Total'

# # 5. Filter data based on user selection and show head of filtered dataframe
# filtered_df = filtered_df[['Total']]
# st.subheader(f"📂 Filtered Data for **{country}** - **{category}** (After Selecting Columns)")
# st.write(filtered_df.head())

# # Button to start analysis
# if st.button("🚀 Start Analysis"):
#     st.success("Starting the analysis...")

#     # 1. Apply log transformation
#     filtered_df_log = np.log(filtered_df)
#     st.subheader(f"⟳ Log-transformed Data for **{country}** - **{category}**")
#     st.line_chart(filtered_df_log)

#     # 2. Calculate SMA and EMA with window size 10
#     window_size = 10
#     filtered_df_log['SMA10'] = filtered_df_log['Total'].rolling(window=window_size).mean()
#     filtered_df_log['EMA10'] = filtered_df_log['Total'].ewm(span=window_size, adjust=False).mean()

#     # 3. Plot Original Log Data, SMA10, and EMA10
#     st.subheader("📊 Simple Moving Average (SMA) and Exponential Moving Average (EMA)")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(filtered_df_log.index, filtered_df_log['Total'], label='Original Log Data', color='blue')
#     ax.plot(filtered_df_log.index, filtered_df_log['SMA10'], label='SMA10', color='orange')
#     ax.plot(filtered_df_log.index, filtered_df_log['EMA10'], label='EMA10', color='green')
#     ax.set_title(f"SMA10 and EMA10 for {country} - {category}")
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Log Emission Value")
#     ax.legend()
#     st.pyplot(fig)

#     # 4. ETS Decomposition
#     st.subheader("🔍 ETS Decomposition")
#     try:
#         result = seasonal_decompose(filtered_df_log['Total'], model='additive', period=1)
#         fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
#         result.observed.plot(ax=ax1, title='Observed')
#         result.trend.plot(ax=ax2, title='Trend')
#         result.seasonal.plot(ax=ax3, title='Seasonal')
#         result.resid.plot(ax=ax4, title='Residual')
#         plt.tight_layout()
#         st.pyplot(fig)
#     except Exception as e:
#         st.error(f"Error in Seasonal Decomposition: {e}")

#     # 5. Split data into train and test sets
#     st.subheader("✂️ Splitting Data into Train and Test Sets")
#     train_size = int(len(filtered_df_log) * 0.8)
#     train = filtered_df_log.iloc[:train_size]
#     test = filtered_df_log.iloc[train_size:]
#     st.write(f"Training data points: {len(train)}")
#     st.write(f"Testing data points: {len(test)}")

#     # 6. Perform ADF test on training data
#     st.subheader("🧪 ADF Test on Training Data")
#     result = adfuller(train['Total'])
#     st.write(f"**ADF Statistic:** {result[0]:.4f}")
#     st.write(f"**p-value:** {result[1]:.4f}")
#     st.write("**Critical Values:**")
#     for key, value in result[4].items():
#         st.write(f"\t{key}: {value:.4f}")
#     if result[1] <= 0.05:
#         st.write("**Result:** Reject the null hypothesis. The time series is stationary.")
#     else:
#         st.write("**Result:** Fail to reject the null hypothesis. The time series is non-stationary.")

#     # 7. Differencing to make the series stationary if necessary
#     st.subheader("⟳ Differencing to Achieve Stationarity")
#     train_diff = train['Total'].diff().dropna()
#     st.line_chart(train_diff)

#     # 8. ACF and PACF Plots for Differenced Data
#     st.subheader("📈 ACF and PACF Plots for Differenced Data")
#     fig, ax = plt.subplots(2, 1, figsize=(12, 8))
#     plot_acf(train_diff, ax=ax[0])
#     plot_pacf(train_diff, ax=ax[1])
#     plt.tight_layout()
#     st.pyplot(fig)

#     # Perform ADF test on differenced data
#     adf_test = adfuller(train_diff)
#     st.write(f"**ADF Statistic (Differenced):** {adf_test[0]:.4f}")
#     st.write(f"**p-value (Differenced):** {adf_test[1]:.4f}")
#     if adf_test[1] <= 0.05:
#         st.write("**Result:** Reject the null hypothesis. The differenced time series is stationary.")
#     else:
#         st.write("**Result:** Fail to reject the null hypothesis. The differenced time series is non-stationary.")

#     # 9. Fit ARIMA model manually
#     st.subheader("🔧 ARIMA Model (Manual) Summary")
#     try:
#         model = ARIMA(train['Total'], order=(1, 1, 1))
#         model_fit = model.fit()
#         st.text(model_fit.summary())
#     except Exception as e:
#         st.error(f"Error fitting ARIMA model: {e}")

#     # 10. Forecasting using ARIMA model
#     st.subheader("🔮 Forecasting with Manual ARIMA Model")
#     try:
#         forecast_manual = model_fit.forecast(steps=len(test))
#         # Exponentiate to revert log-transformation
#         forecast_manual_exp = np.exp(forecast_manual)
#         test_exp = np.exp(test['Total'])

#         # Prepare dataframe for plotting
#         forecast_manual_plot = pd.DataFrame({
#             'Actual': test_exp,
#             'Forecast Manual ARIMA': forecast_manual_exp
#         }, index=test.index)

#         st.line_chart(forecast_manual_plot)

#     except Exception as e:
#         st.error(f"Error during forecasting with manual ARIMA: {e}")

#     # 11. Residual Analysis for ARIMA
#     st.subheader("📉 Residual Analysis for ARIMA")
#     try:
#         residuals = model_fit.resid
#         fig, ax = plt.subplots(1, 2, figsize=(14, 5))
#         residuals.plot(title="Residuals", ax=ax[0])
#         residuals.plot(kind='kde', title='Density', ax=ax[1], color='red')
#         plt.tight_layout()
#         st.pyplot(fig)

#         # ACF and PACF for residuals
#         st.subheader("📈 ACF and PACF for Residuals (ARIMA)")
#         fig, ax = plt.subplots(2, 1, figsize=(12, 8))
#         plot_acf(residuals, ax=ax[0])
#         plot_pacf(residuals, ax=ax[1])
#         plt.tight_layout()
#         st.pyplot(fig)
#     except Exception as e:
#         st.error(f"Error during residual analysis for ARIMA: {e}")

#     # 12. Auto ARIMA model
#     st.subheader("🔧 Auto ARIMA Model Summary")
#     try:
#         auto_arima_model = auto_arima(train['Total'], seasonal=False, stepwise=True, suppress_warnings=True)
#         st.text(auto_arima_model.summary())
#     except Exception as e:
#         st.error(f"Error fitting Auto ARIMA model: {e}")

#     # 13. Forecast using Auto ARIMA
#     st.subheader("🔮 Forecasting with Auto ARIMA Model")
#     try:
#         forecast_auto = auto_arima_model.predict(n_periods=len(test))
#         forecast_auto_exp = np.exp(forecast_auto)

#         # Prepare dataframe for plotting
#         forecast_auto_plot = pd.DataFrame({
#             'Actual': test_exp,
#             'Forecast Manual ARIMA': forecast_manual_exp,
#             'Forecast Auto ARIMA': forecast_auto_exp
#         }, index=test.index)

#         st.line_chart(forecast_auto_plot)

#     except Exception as e:
#         st.error(f"Error during forecasting with Auto ARIMA: {e}")

#     # 14. Fit Exponential Smoothing (ES) Model
#     st.subheader("🔧 Exponential Smoothing (ES) Model Summary")
#     try:
#         es_model = ExponentialSmoothing(train['Total'], trend='add', seasonal=None)
#         es_fit = es_model.fit()
#         st.text(es_fit.summary())
#     except Exception as e:
#         st.error(f"Error fitting Exponential Smoothing model: {e}")

#     # 15. Forecasting using ES model
#     st.subheader("🔮 Forecasting with Exponential Smoothing (ES) Model")
#     try:
#         forecast_es = es_fit.forecast(steps=len(test))
#         forecast_es_exp = np.exp(forecast_es)

#         # Prepare dataframe for plotting
#         forecast_es_plot = pd.DataFrame({
#             'Actual': test_exp,
#             'Forecast Manual ARIMA': forecast_manual_exp,
#             'Forecast Auto ARIMA': forecast_auto_exp,
#             'Forecast ES': forecast_es_exp
#         }, index=test.index)

#         st.line_chart(forecast_es_plot)

#     except Exception as e:
#         st.error(f"Error during forecasting with Exponential Smoothing: {e}")

#     # 16. Performance Metrics - Manual ARIMA
#     st.subheader("📊 Performance Metrics - Manual ARIMA")
#     try:
#         mse_manual = mean_squared_error(test_exp, forecast_manual_exp)
#         mae_manual = mean_absolute_error(test_exp, forecast_manual_exp)
#         mape_manual = mean_absolute_percentage_error(test_exp, forecast_manual_exp)
#         rmse_manual = np.sqrt(mse_manual)

#         st.markdown(
#             f"""
#             <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border: 1px solid #e6e6e6;">
#                 <h4 style="color:#333;">**Manual ARIMA Performance Metrics:**</h4>
#                 <ul style="list-style-type:none;">
#                     <li><strong>MSE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mse_manual:.2f}</span></li>
#                     <li><strong>MAE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mae_manual:.2f}</span></li>
#                     <li><strong>MAPE:</strong> <span style="color:#FF6347; font-size:1.2em;">{mape_manual:.4f}</span></li>
#                     <li><strong>RMSE:</strong> <span style="color:#FF6347; font-size:1.2em;">{rmse_manual:.2f}</span></li>
#                 </ul>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     except Exception as e:
#         st.error(f"Error calculating performance metrics for Manual ARIMA: {e}")

#     # 17. Performance Metrics - Auto ARIMA
#     st.subheader("📊 Performance Metrics - Auto ARIMA")
#     try:
#         mse_auto = mean_squared_error(test_exp, forecast_auto_exp)
#         mae_auto = mean_absolute_error(test_exp, forecast_auto_exp)
#         mape_auto = mean_absolute_percentage_error(test_exp, forecast_auto_exp)
#         rmse_auto = np.sqrt(mse_auto)

#         st.markdown(
#             f"""
#             <div style="background-color:#f0f0f7; padding:10px; border-radius:5px; border: 1px solid #e6e6e6;">
#                 <h4 style="color:#333;">**Auto ARIMA Performance Metrics:**</h4>
#                 <ul style="list-style-type:none;">
#                     <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse_auto:.2f}</span></li>
#                     <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae_auto:.2f}</span></li>
#                     <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape_auto:.4f}</span></li>
#                     <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse_auto:.2f}</span></li>
#                 </ul>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     except Exception as e:
#         st.error(f"Error calculating performance metrics for Auto ARIMA: {e}")

#     # 18. Performance Metrics - Exponential Smoothing (ES)
#     st.subheader("📊 Performance Metrics - Exponential Smoothing (ES)")
#     try:
#         mse_es = mean_squared_error(test_exp, forecast_es_exp)
#         mae_es = mean_absolute_error(test_exp, forecast_es_exp)
#         mape_es = mean_absolute_percentage_error(test_exp, forecast_es_exp)
#         rmse_es = np.sqrt(mse_es)

#         st.markdown(
#             f"""
#             <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; border: 1px solid #b3e0ff;">
#                 <h4 style="color:#333;">**Exponential Smoothing (ES) Performance Metrics:**</h4>
#                 <ul style="list-style-type:none;">
#                     <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse_es:.2f}</span></li>
#                     <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae_es:.2f}</span></li>
#                     <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape_es:.4f}</span></li>
#                     <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse_es:.2f}</span></li>
#                 </ul>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     except Exception as e:
#         st.error(f"Error calculating performance metrics for Exponential Smoothing: {e}")

#     # 19. Complete Data with Forecasts
#     st.subheader("📊 Complete Data with Forecasts")
#     try:
#         complete_df = pd.DataFrame({
#             'Actual': test_exp,
#             'Forecast Manual ARIMA': forecast_manual_exp,
#             'Forecast Auto ARIMA': forecast_auto_exp,
#             'Forecast ES': forecast_es_exp
#         })
#         st.write(complete_df)
#     except Exception as e:
#         st.error(f"Error displaying complete data with forecasts: {e}")

#     # 20. Save Models
#     st.subheader("💾 Saving Models")
#     try:
#         # Save Manual ARIMA Model
#         manual_arima_path = os.path.join(output_dir_model, f'manual_arima_model_{country}_{category}.keras')
#         joblib.dump(model_fit, manual_arima_path)
#         st.success(f"Manual ARIMA model saved to {manual_arima_path}")

#         # Save Auto ARIMA Model
#         auto_arima_path = os.path.join(output_dir_model, f'auto_arima_model_{country}_{category}.keras')
#         joblib.dump(auto_arima_model, auto_arima_path)
#         st.success(f"Auto ARIMA model saved to {auto_arima_path}")

#         # Save Exponential Smoothing Model
#         es_path = os.path.join(output_dir_model, f'exponential_smoothing_model_{country}_{category}.keras')
#         joblib.dump(es_fit, es_path)
#         st.success(f"Exponential Smoothing model saved to {es_path}")

#     except Exception as e:
#         st.error(f"Error saving models: {e}")

#     # 21. Export Performance Metrics and Forecasts to CSV
#     st.subheader("📤 Exporting Results to CSV")
#     try:
#         # Create Performance Metrics DataFrame
#         metrics_data = {
#             'Model': ['Manual ARIMA', 'Auto ARIMA', 'Exponential Smoothing (ES)'],
#             'MSE': [mse_manual, mse_auto, mse_es],
#             'MAE': [mae_manual, mae_auto, mae_es],
#             'MAPE': [mape_manual, mape_auto, mape_es],
#             'RMSE': [rmse_manual, rmse_auto, rmse_es]
#         }
#         metrics_df = pd.DataFrame(metrics_data)

#         # Save Performance Metrics
#         metrics_file = os.path.join(output_dir_result, f'performance_metrics_{country}_{category}.csv')
#         metrics_df.to_csv(metrics_file, index=False)
#         st.success(f"Performance metrics saved to {metrics_file}")
#         st.write(metrics_df)

#         # Create Actual vs Forecast DataFrame
#         forecast_df = complete_df.reset_index()
#         forecast_df.rename(columns={'index': 'Year'}, inplace=True)

#         # Save Forecasts
#         forecasts_file = os.path.join(output_dir_result, f'actual_vs_forecast_{country}_{category}.csv')
#         forecast_df.to_csv(forecasts_file, index=False)
#         st.success(f"Actual vs Forecast data saved to {forecasts_file}")
#         st.write(forecast_df)

#     except Exception as e:
#         st.error(f"Error exporting results to CSV: {e}")

#     st.markdown("""
#     ---
#     **Data Source:** `fossil_fuel_co2_emissions-by-nation_with_continent.csv`
#     """)


#current ver
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import joblib

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Ensure output directories exist
output_dir_result = './CO2 Emissions/Result/'
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result)

output_dir_model = './CO2 Emissions/Models/'
if not os.path.exists(output_dir_model):
    os.makedirs(output_dir_model)

st.set_page_config(page_title="CO2 Emissions Analysis", layout="wide")

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

# 3. Set default country and category
country = 'UNITED KINGDOM'
category = 'Total'

# 4. Filter data based on default country and category, and show head of filtered dataframe
filtered_df = df[df['Country'] == country]
filtered_df = filtered_df[[category]]
st.subheader(f"📂 Filtered Data for **{country}** - **{category}** (After Selecting Columns)")
st.write(filtered_df.head())

# Automatically start analysis without button
st.success("Starting the analysis...")

# 5. Apply log transformation
filtered_df_log = np.log(filtered_df)
st.subheader(f"⟳ Log-transformed Data for **{country}** - **{category}**")
st.line_chart(filtered_df_log)

# 6. Calculate SMA and EMA with window size 10
window_size = 10
filtered_df_log['SMA10'] = filtered_df_log[category].rolling(window=window_size).mean()
filtered_df_log['EMA10'] = filtered_df_log[category].ewm(span=window_size, adjust=False).mean()

# 7. Plot Original Log Data, SMA10, and EMA10
st.subheader("📊 Simple Moving Average (SMA) and Exponential Moving Average (EMA)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(filtered_df_log.index, filtered_df_log[category], label='Original Log Data', color='blue')
ax.plot(filtered_df_log.index, filtered_df_log['SMA10'], label='SMA10', color='orange')
ax.plot(filtered_df_log.index, filtered_df_log['EMA10'], label='EMA10', color='green')
ax.set_title(f"SMA10 and EMA10 for {country} - {category}")
ax.set_xlabel("Year")
ax.set_ylabel("Log Emission Value")
ax.legend()
st.pyplot(fig)

# 8. ETS Decomposition
st.subheader("🔍 ETS Decomposition")
try:
    result = seasonal_decompose(filtered_df_log[category], model='additive', period=1)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
    result.observed.plot(ax=ax1, title='Observed')
    result.trend.plot(ax=ax2, title='Trend')
    result.seasonal.plot(ax=ax3, title='Seasonal')
    result.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error in Seasonal Decomposition: {e}")

# 9. Split data into train and test sets
st.subheader("✂️ Splitting Data into Train and Test Sets")
train_size = int(len(filtered_df_log) * 0.8)
train = filtered_df_log.iloc[:train_size]
test = filtered_df_log.iloc[train_size:]
st.write(f"Training data points: {len(train)}")
st.write(f"Testing data points: {len(test)}")

# 10. Perform ADF test on training data
st.subheader("🧪 ADF Test on Training Data")
result = adfuller(train[category])
st.write(f"**ADF Statistic:** {result[0]:.4f}")
st.write(f"**p-value:** {result[1]:.4f}")
st.write("**Critical Values:**")
for key, value in result[4].items():
    st.write(f"\t{key}: {value:.4f}")
if result[1] <= 0.05:
    st.write("**Result:** Reject the null hypothesis. The time series is stationary.")
else:
    st.write("**Result:** Fail to reject the null hypothesis. The time series is non-stationary.")

# 11. Differencing to make the series stationary if necessary
st.subheader("⟳ Differencing to Achieve Stationarity")
train_diff = train[category].diff().dropna()
st.line_chart(train_diff)

# 12. ACF and PACF Plots for Differenced Data
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

# 13. Load ARIMA model manually
#st.subheader("🔧 ARIMA Model (Manual) Summary")
try:
    manual_arima_path = os.path.join(output_dir_model, f'manual_arima_model_{country}_{category}.keras')
    model_fit = joblib.load(manual_arima_path)
    st.success(f"Manual ARIMA model loaded from {manual_arima_path}")
    #st.text(model_fit.summary())
except Exception as e:
    st.error(f"Error loading Manual ARIMA model: {e}")

# 14. Forecasting using loaded ARIMA model
# st.subheader("🔮 Forecasting with Manual ARIMA Model")
try:
    forecast_manual = model_fit.forecast(steps=len(test))
    # Exponentiate to revert log-transformation
    forecast_manual_exp = np.exp(forecast_manual)
    test_exp = np.exp(test[category])

    # # Prepare dataframe for plotting
    # forecast_manual_plot = pd.DataFrame({
    #     'Actual': test_exp,
    #     'Forecast Manual ARIMA': forecast_manual_exp
    # }, index=test.index)

    # st.line_chart(forecast_manual_plot)
except Exception as e:
    st.error(f"Error during forecasting with manual ARIMA: {e}")

# 15. Residual Analysis for ARIMA
# st.subheader("📉 Residual Analysis for ARIMA")
# try:
#     residuals = model_fit.resid
#     fig, ax = plt.subplots(1, 2, figsize=(14, 5))
#     residuals.plot(title="Residuals", ax=ax[0])
#     residuals.plot(kind='kde', title='Density', ax=ax[1], color='red')
#     plt.tight_layout()
#     st.pyplot(fig)

#     # ACF and PACF for residuals
#     st.subheader("📈 ACF and PACF for Residuals (ARIMA)")
#     fig, ax = plt.subplots(2, 1, figsize=(12, 8))
#     plot_acf(residuals, ax=ax[0])
#     plot_pacf(residuals, ax=ax[1])
#     plt.tight_layout()
#     st.pyplot(fig)
# except Exception as e:
#     st.error(f"Error during residual analysis for ARIMA: {e}")

# 16. Load Auto ARIMA model
#st.subheader("🔧 Auto ARIMA Model Summary")
try:
    auto_arima_path = os.path.join(output_dir_model, f'auto_arima_model_{country}_{category}.keras')
    auto_arima_model = joblib.load(auto_arima_path)
    st.success(f"Auto ARIMA model loaded from {auto_arima_path}")
    #st.text(auto_arima_model.summary())
except Exception as e:
    st.error(f"Error loading Auto ARIMA model: {e}")

# 17. Forecast using loaded Auto ARIMA
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
except Exception as e:
    st.error(f"Error during forecasting with Auto ARIMA: {e}")

# 18. Load Exponential Smoothing (ES) model
#st.subheader("🔧 Exponential Smoothing (ES) Model Summary")
try:
    es_path = os.path.join(output_dir_model, f'exponential_smoothing_model_{country}_{category}.keras')
    es_fit = joblib.load(es_path)
    st.success(f"Exponential Smoothing model loaded from {es_path}")
    #st.text(es_fit.summary())
except Exception as e:
    st.error(f"Error loading Exponential Smoothing model: {e}")

# 19. Forecasting using loaded ES model
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
except Exception as e:
    st.error(f"Error during forecasting with Exponential Smoothing: {e}")

# 20. Performance Metrics - Manual ARIMA
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

# 21. Performance Metrics - Auto ARIMA
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

# 22. Performance Metrics - Exponential Smoothing (ES)
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

# 23. Complete Data with Forecasts
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

# 24. Export Performance Metrics and Forecasts to CSV
#st.subheader("📤 Exporting Results to CSV")
try:
    # Create Performance Metrics DataFrame
    metrics_data = {
        'Model': ['Manual ARIMA', 'Auto ARIMA', 'Exponential Smoothing (ES)'],
        'MSE': [mse_manual, mse_auto, mse_es],
        'MAE': [mae_manual, mae_auto, mae_es],
        'MAPE': [mape_manual, mape_auto, mape_es],
        'RMSE': [rmse_manual, rmse_auto, rmse_es]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # Save Performance Metrics
    metrics_file = os.path.join(output_dir_result, f'performance_metrics_{country}_{category}.csv')
    metrics_df.to_csv(metrics_file, index=False)
    st.success(f"Performance metrics saved to {metrics_file}")
    # st.write(metrics_df)

    # Create Actual vs Forecast DataFrame
    forecast_df = complete_df.reset_index()
    forecast_df.rename(columns={'index': 'Year'}, inplace=True)

    # Save Forecasts
    forecasts_file = os.path.join(output_dir_result, f'actual_vs_forecast_{country}_{category}.csv')
    forecast_df.to_csv(forecasts_file, index=False)
    st.success(f"Actual vs Forecast data saved to {forecasts_file}")
    #st.write(forecast_df)

except Exception as e:
    st.error(f"Error exporting results to CSV: {e}")

st.markdown("""
---
**Data Source:** `fossil_fuel_co2_emissions-by-nation_with_continent.csv`
""")
