# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import os
# # from statsmodels.tsa.stattools import adfuller
# # from statsmodels.tsa.seasonal import seasonal_decompose
# # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# # from pmdarima.arima import auto_arima
# # from statsmodels.tsa.arima.model import ARIMA
# # from statsmodels.tsa.holtwinters import ExponentialSmoothing
# # from statsmodels.tsa.statespace.sarimax import SARIMAX 

# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# # import warnings
# # warnings.filterwarnings("ignore")

# # output_dir_result = './Global Temperature/Result/'
# # if not os.path.exists(output_dir_result):
# #     os.makedirs(output_dir_result, exist_ok= True)

# # output_dir_model = './Global Temperature/Models/'
# # if not os.path.exists(output_dir_model):
# #     os.makedirs(output_dir_model, exist_ok= True)

# # # Set Streamlit page configuration
# # st.set_page_config(
# #     page_title="Temperature Time Series Analysis",
# #     layout="wide",
# #     initial_sidebar_state="expanded",
# # )

# # # Title of the App
# # st.title("📈 Temperature Time Series Analysis")

# # @st.cache_data
# # def load_data(file_path):
# #     df = pd.read_csv(file_path)
# #     return df

# # # Load data
# # file_path = 'data/processed/updated_data_with_time.csv'
# # df = load_data(file_path)

# # # Sidebar for country selection
# # st.sidebar.header("Select Country")
# # countries = df['Country'].unique().tolist()
# # selected_country = st.sidebar.selectbox("Country", countries, index=countries.index("Germany") if "Germany" in countries else 0)

# # # Filter data based on selected country
# # country_data = df[df['Country'] == selected_country].copy()
# # country_data['Time'] = pd.to_datetime(country_data['Time'])
# # country_data = country_data.sort_values(by='Time')
# # country_data = country_data.set_index('Time', drop=True)

# # st.header(f"📊 Data for {selected_country}")
# # st.dataframe(country_data.head())

# # # Select the Temperature column
# # if 'TempC' not in country_data.columns:
# #     st.error("The selected dataset does not contain 'TempC' column.")
# #     st.stop()

# # temperature_df = country_data[['TempC']].copy()

# # # Scaling
# # st.subheader("🔄 Data Scaling")
# # scaler = MinMaxScaler(feature_range=(-1, 1))
# # temperature_df['TempC_Scaled'] = scaler.fit_transform(temperature_df[['TempC']])
# # st.write("Scaled Temperature Data (TempC_Scaled):")
# # st.dataframe(temperature_df.head())

# # # Moving Averages
# # st.subheader("📉 Moving Averages")
# # temperature_df['SMA_12'] = temperature_df['TempC_Scaled'].rolling(window=12).mean()
# # temperature_df['EMA_12'] = temperature_df['TempC_Scaled'].ewm(span=12, adjust=False).mean()

# # fig, ax = plt.subplots(figsize=(12, 6))
# # ax.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
# # ax.plot(temperature_df['SMA_12'], label='SMA 12', color='yellow', linestyle='-')
# # ax.plot(temperature_df['EMA_12'], label='EMA 12', color='red', linestyle=':')
# # ax.set_title(f'{selected_country} Temperature Change Trend Over Time')
# # ax.set_xlabel('Year')
# # ax.set_ylabel('Scaled Temperature (TempC_Scaled)')
# # ax.legend()
# # st.pyplot(fig)

# # # ETS Decomposition
# # st.subheader("🔍 ETS Decomposition")
# # decomposition = seasonal_decompose(temperature_df['TempC_Scaled'], model='additive', period=12)
# # fig_decompose = decomposition.plot()
# # fig_decompose.set_size_inches(14, 8)
# # st.pyplot(fig_decompose)

# # # Split data into train and test
# # st.subheader("✂️ Train-Test Split")
# # test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5)
# # split_index = int((100 - test_size) / 100 * len(temperature_df))
# # train = temperature_df.iloc[:split_index]
# # test = temperature_df.iloc[split_index:]

# # st.write(f"Training Data: {train.shape[0]} samples")
# # st.write(f"Testing Data: {test.shape[0]} samples")

# # # ADF Test
# # st.subheader("🧪 Augmented Dickey-Fuller Test")
# # adf_result = adfuller(train['TempC_Scaled'])
# # st.write(f"**ADF Statistic:** {adf_result[0]:.6f}")
# # st.write(f"**p-value:** {adf_result[1]:.6f}")
# # st.write("**Critical Values:**")
# # for key, value in adf_result[4].items():
# #     st.write(f"   {key}: {value:.3f}")

# # if adf_result[1] <= 0.05:
# #     st.success("Reject the null hypothesis. The time series is stationary.")
# # else:
# #     st.warning("Fail to reject the null hypothesis. The time series is non-stationary.")

# # # ACF and PACF Plots
# # st.subheader("📈 Autocorrelation and Partial Autocorrelation")
# # fig_acf, ax_acf = plt.subplots(1, 2, figsize=(16, 4))
# # plot_acf(train['TempC_Scaled'], ax=ax_acf[0], lags=40)
# # plot_pacf(train['TempC_Scaled'], ax=ax_acf[1], lags=40)
# # st.pyplot(fig_acf)

# # # ARIMA Model
# # st.subheader("📉 ARIMA Model")
# # arima_order = st.text_input("Enter ARIMA order (p,d,q)", "(2,0,2)")
# # try:
# #     order = tuple(map(int, arima_order.strip("()").split(',')))
# #     arima_model = ARIMA(train['TempC_Scaled'], order=order)
# #     arima_fit = arima_model.fit()
# #     st.write("**ARIMA Model Summary:**")
# #     st.text(arima_fit.summary())
# # except:
# #     st.error("Invalid ARIMA order. Please enter in the format (p,d,q) e.g., (2,0,2).")

# # # Residuals Analysis
# # st.subheader("🔍 Residuals Analysis")
# # if 'arima_fit' in locals():
# #     residuals = arima_fit.resid[1:]
# #     fig_res, ax_res = plt.subplots(1, 2, figsize=(14, 5))
# #     residuals.plot(title="Residuals", ax=ax_res[0])
# #     residuals.plot(kind='kde', title='Density', ax=ax_res[1])
# #     st.pyplot(fig_res)

# #     fig_acf_res, ax_acf_res = plt.subplots(1, 2, figsize=(16, 4))
# #     plot_acf(residuals, ax=ax_acf_res[0], lags=40)
# #     plot_pacf(residuals, ax=ax_acf_res[1], lags=40)
# #     st.pyplot(fig_acf_res)

# #     # Forecast
# #     forecast_steps = len(test)
# #     forecast_arima = arima_fit.forecast(steps=forecast_steps)
# #     temperature_df['forecast_manual'] = [np.nan]*len(train) + list(forecast_arima)

# #     fig_forecast, ax_forecast = plt.subplots(figsize=(12,6))
# #     ax_forecast.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
# #     ax_forecast.plot(temperature_df['forecast_manual'], label='ARIMA Forecast', color='green')
# #     ax_forecast.set_title("ARIMA Forecast vs Actual")
# #     ax_forecast.set_xlabel("Time")
# #     ax_forecast.set_ylabel("Scaled Temperature")
# #     ax_forecast.legend()
# #     st.pyplot(fig_forecast)

# # # Auto ARIMA
# # st.subheader("🤖 Auto ARIMA Model")
# # with st.spinner("Fitting Auto ARIMA... This may take a while!"):
# #     try:
# #         auto_arima_model = auto_arima(train['TempC_Scaled'], stepwise=False, seasonal=False, suppress_warnings=True)
# #         st.write("**Auto ARIMA Model Summary:**")
# #         st.text(auto_arima_model.summary())

# #         # Forecast
# #         forecast_auto = auto_arima_model.predict(n_periods=len(test))
# #         temperature_df['forecast_auto'] = [np.nan]*len(train) + list(forecast_auto)

# #         # Plot Auto ARIMA Forecast
# #         fig_forecast_auto, ax_forecast_auto = plt.subplots(figsize=(12,6))
# #         ax_forecast_auto.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
# #         ax_forecast_auto.plot(temperature_df['forecast_auto'], label='Auto ARIMA Forecast', color='orange')
# #         ax_forecast_auto.set_title("Auto ARIMA Forecast vs Actual")
# #         ax_forecast_auto.set_xlabel("Time")
# #         ax_forecast_auto.set_ylabel("Scaled Temperature")
# #         ax_forecast_auto.legend()
# #         st.pyplot(fig_forecast_auto)

# #     except Exception as e:
# #         st.error(f"Auto ARIMA failed: {e}")

# # # Inverse Scaling for Metrics
# # st.subheader("🔄 Inverse Scaling for Performance Metrics")
# # train_exp = scaler.inverse_transform(train[['TempC_Scaled']])
# # test_exp = scaler.inverse_transform(test[['TempC_Scaled']])

# # if 'forecast_arima' in locals():
# #     forecast_arima_exp = scaler.inverse_transform(np.array(forecast_arima).reshape(-1, 1))
# # if 'forecast_auto' in locals():
# #     forecast_auto_exp = scaler.inverse_transform(np.array(forecast_auto).reshape(-1, 1))

# # # Function to display metrics with enhanced styling
# # def display_metrics(title, mse, mae, mape, rmse, r2, color):
# #     st.markdown(
# #         f"""
# #         <div style="background-color:{color}; padding:15px; border-radius:10px; margin-bottom:20px;">
# #             <h3 style="color:#333;">{title}</h3>
# #             <ul style="list-style-type:none; padding:0;">
# #                 <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse:.4f}</span></li>
# #                 <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae:.4f}</span></li>
# #                 <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape:.4f}</span></li>
# #                 <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse:.4f}</span></li>
# #                 <li><strong>R²:</strong> <span style="color:#4682B4; font-size:1.2em;">{r2:.4f}</span></li>
# #             </ul>
# #         </div>
# #         """,
# #         unsafe_allow_html=True
# #     )

# # # Performance Metrics for Manual ARIMA
# # if 'forecast_arima_exp' in locals():
# #     st.subheader("📊 Performance Metrics for Manual ARIMA")
# #     mse_arima = mean_squared_error(test_exp, forecast_arima_exp)
# #     mae_arima = mean_absolute_error(test_exp, forecast_arima_exp)
# #     mape_arima = mean_absolute_percentage_error(test_exp, forecast_arima_exp)
# #     rmse_arima = np.sqrt(mse_arima)
# #     r2_arima = r2_score(test_exp, forecast_arima_exp)

# #     display_metrics(
# #         title="Manual ARIMA Performance Metrics",
# #         mse=mse_arima,
# #         mae=mae_arima,
# #         mape=mape_arima,
# #         rmse=rmse_arima,
# #         r2=r2_arima,
# #         color="#d4edda"  # Light green background
# #     )

# # # Performance Metrics for Auto ARIMA
# # if 'forecast_auto_exp' in locals():
# #     st.subheader("📊 Performance Metrics for Auto ARIMA")
# #     mse_auto = mean_squared_error(test_exp, forecast_auto_exp)
# #     mae_auto = mean_absolute_error(test_exp, forecast_auto_exp)
# #     mape_auto = mean_absolute_percentage_error(test_exp, forecast_auto_exp)
# #     rmse_auto = np.sqrt(mse_auto)
# #     r2_auto = r2_score(test_exp, forecast_auto_exp)

# #     display_metrics(
# #         title="Auto ARIMA Performance Metrics",
# #         mse=mse_auto,
# #         mae=mae_auto,
# #         mape=mape_auto,
# #         rmse=rmse_auto,
# #         r2=r2_auto,
# #         color="#cce5ff"  # Light blue background
# #     )

# # # SARIMAX Model
# # st.subheader("🔧 SARIMAX Model")
# # with st.spinner("Fitting SARIMAX... This may take a while!"):
# #     try:
# #         # Refit Auto ARIMA with seasonal components if needed
# #         auto_arima_final = auto_arima(
# #             temperature_df['TempC_Scaled'],
# #             start_p=1,
# #             start_q=1,
# #             max_p=3,
# #             max_q=3,
# #             m=12,
# #             start_P=0,
# #             seasonal=True,
# #             d=None,
# #             D=1,
# #             trace=False,
# #             error_action='ignore',
# #             suppress_warnings=True,
# #             stepwise=True
# #         )

# #         st.write("**Final Auto ARIMA Model Summary with Seasonality:**")
# #         st.text(auto_arima_final.summary())

# #         # Extract orders
# #         best_order = auto_arima_final.order
# #         best_seasonal_order = auto_arima_final.seasonal_order
# #         st.write(f"**Best ARIMA order:** {best_order}")
# #         st.write(f"**Best Seasonal Order:** {best_seasonal_order}")

# #         # Fit SARIMAX
# #         sarimax_model = SARIMAX(train['TempC_Scaled'],  
# #                                 order=best_order,  
# #                                 seasonal_order=best_seasonal_order) 
# #         sarimax_fit = sarimax_model.fit()
# #         st.write("**SARIMAX Model Summary:**")
# #         st.text(sarimax_fit.summary())

# #         # Predictions
# #         predictions = sarimax_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
# #         temperature_df['forecast_sarimax'] = [np.nan]*len(train) + list(predictions)

# #         # Plot SARIMAX Forecast
# #         fig_sarimax, ax_sarimax = plt.subplots(figsize=(12,6))
# #         ax_sarimax.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
# #         ax_sarimax.plot(temperature_df['forecast_sarimax'], label='SARIMAX Forecast', color='purple')
# #         ax_sarimax.set_title("SARIMAX Forecast vs Actual")
# #         ax_sarimax.set_xlabel("Time")
# #         ax_sarimax.set_ylabel("Scaled Temperature")
# #         ax_sarimax.legend()
# #         st.pyplot(fig_sarimax)

# #         # Inverse Scaling for SARIMAX
# #         predictions_exp = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# #         # Performance Metrics for SARIMAX
# #         st.subheader("📊 Performance Metrics for SARIMAX")
# #         mse_sarimax = mean_squared_error(test_exp, predictions_exp)
# #         mae_sarimax = mean_absolute_error(test_exp, predictions_exp)
# #         mape_sarimax = mean_absolute_percentage_error(test_exp, predictions_exp)
# #         rmse_sarimax = np.sqrt(mse_sarimax)
# #         r2_sarimax = r2_score(test_exp, predictions_exp)

# #         display_metrics(
# #             title="SARIMAX Performance Metrics",
# #             mse=mse_sarimax,
# #             mae=mae_sarimax,
# #             mape=mape_sarimax,
# #             rmse=rmse_sarimax,
# #             r2=r2_sarimax,
# #             color="#f8d7da"  # Light red background
# #         )

# #     except Exception as e:
# #         st.error(f"SARIMAX failed: {e}")

# # # Exponential Smoothing
# # st.subheader("🌊 Exponential Smoothing")
# # with st.spinner("Fitting Exponential Smoothing..."):
# #     try:
# #         # Reconstruct the full scaled temperature for Exponential Smoothing
# #         temperature_df_full = temperature_df.copy()
# #         temperature_df_full['TempC_Scaled'] = list(train['TempC_Scaled']) + list(test['TempC_Scaled'])

# #         # Fit Exponential Smoothing
# #         es_model = ExponentialSmoothing(train['TempC_Scaled'], trend='add', seasonal='add', seasonal_periods=12)
# #         es_fit = es_model.fit()
# #         es_predictions = es_fit.forecast(len(test))
# #         temperature_df_full['forecast_es'] = [np.nan]*len(train) + list(es_predictions)

# #         # Plot Exponential Smoothing Forecast
# #         fig_es, ax_es = plt.subplots(figsize=(12,6))
# #         ax_es.plot(test.index, test['TempC_Scaled'], label='Test', color='orange')
# #         ax_es.plot(test.index, temperature_df_full['forecast_es'].dropna(), label='Exponential Smoothing Predictions', color='green')
# #         ax_es.set_title("Exponential Smoothing Forecast vs Actual")
# #         ax_es.set_xlabel("Time")
# #         ax_es.set_ylabel("Scaled Temperature")
# #         ax_es.legend()
# #         st.pyplot(fig_es)

# #         # Inverse Scaling for Exponential Smoothing
# #         es_predictions_exp = scaler.inverse_transform(np.array(es_predictions).reshape(-1, 1))

# #         # Performance Metrics for Exponential Smoothing
# #         st.subheader("📊 Performance Metrics for Exponential Smoothing")
# #         mse_es = mean_squared_error(test_exp, es_predictions_exp)
# #         mae_es = mean_absolute_error(test_exp, es_predictions_exp)
# #         mape_es = mean_absolute_percentage_error(test_exp, es_predictions_exp)
# #         rmse_es = np.sqrt(mse_es)
# #         r2_es = r2_score(test_exp, es_predictions_exp)

# #         display_metrics(
# #             title="Exponential Smoothing Performance Metrics",
# #             mse=mse_es,
# #             mae=mae_es,
# #             mape=mape_es,
# #             rmse=rmse_es,
# #             r2=r2_es,
# #             color="#d1ecf1"  # Light cyan background
# #         )

# #     except Exception as e:
# #         st.error(f"Exponential Smoothing failed: {e}")

# # # Final Data Display with Predictions
# # st.subheader("📈 Final Data with Forecasts")

# # # Prepare the final DataFrame with actual and predicted values
# # final_df = pd.DataFrame({
# #     "Actual": test_exp.flatten()
# # })

# # if 'forecast_arima_exp' in locals():
# #     final_df["ARIMA Forecast"] = forecast_arima_exp.flatten()
# # if 'forecast_auto_exp' in locals():
# #     final_df["Auto ARIMA Forecast"] = forecast_auto_exp.flatten()
# # if 'predictions_exp' in locals():
# #     final_df["SARIMAX Forecast"] = predictions_exp.flatten()
# # if 'es_predictions_exp' in locals():
# #     final_df["Exponential Smoothing Forecast"] = es_predictions_exp.flatten()

# # st.write("**Actual vs Predicted Values on Test Set:**")
# # st.dataframe(final_df.head(10))  # Display first 10 rows for brevity

# # # Save the final_df to the output_dir_result
# # csv_path = os.path.join(output_dir_result, 'predictions.csv')
# # final_df.to_csv(csv_path, index=False)
# # st.success(f"Predictions have been saved to `{csv_path}`.")

# # # Optionally, allow users to download the final_df as CSV
# # # csv = final_df.to_csv(index=False).encode('utf-8')
# # # st.download_button(
# # #     label="📥 Download Predictions as CSV",
# # #     data=csv,
# # #     file_name='predictions.csv',
# # #     mime='text/csv',
# # # )

# # # Footer
# # st.markdown("---")
# # st.markdown("© 2024 Temperature Time Series Analysis App")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# import pickle
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# from pmdarima.arima import auto_arima
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.statespace.sarimax import SARIMAX 

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# import warnings
# warnings.filterwarnings("ignore")

# # Directories for saving results and models
# output_dir_result = './Global Temperature/Result/'
# if not os.path.exists(output_dir_result):
#     os.makedirs(output_dir_result, exist_ok=True)

# output_dir_model = './Global Temperature/Models/'
# if not os.path.exists(output_dir_model):
#     os.makedirs(output_dir_model, exist_ok=True)

# # Set Streamlit page configuration
# st.set_page_config(
#     page_title="Temperature Time Series Analysis",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Title of the App
# st.title("📈 Temperature Time Series Analysis")

# @st.cache_data
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     return df

# # Load data
# file_path = 'data/processed/updated_data_with_time.csv'
# df = load_data(file_path)

# # Sidebar for country selection
# st.sidebar.header("Select Country")
# countries = df['Country'].unique().tolist()
# selected_country = st.sidebar.selectbox("Country", countries, index=countries.index("Germany") if "Germany" in countries else 0)

# # Filter data based on selected country
# country_data = df[df['Country'] == selected_country].copy()
# country_data['Time'] = pd.to_datetime(country_data['Time'])
# country_data = country_data.sort_values(by='Time')
# country_data = country_data.set_index('Time', drop=True)

# st.header(f"📊 Data for {selected_country}")
# st.dataframe(country_data.head())

# # Select the Temperature column
# if 'TempC' not in country_data.columns:
#     st.error("The selected dataset does not contain 'TempC' column.")
#     st.stop()

# temperature_df = country_data[['TempC']].copy()

# # Scaling
# st.subheader("🔄 Data Scaling")
# scaler = MinMaxScaler(feature_range=(-1, 1))
# temperature_df['TempC_Scaled'] = scaler.fit_transform(temperature_df[['TempC']])
# st.write("Scaled Temperature Data (TempC_Scaled):")
# st.dataframe(temperature_df.head())

# # Moving Averages
# st.subheader("📉 Moving Averages")
# temperature_df['SMA_12'] = temperature_df['TempC_Scaled'].rolling(window=12).mean()
# temperature_df['EMA_12'] = temperature_df['TempC_Scaled'].ewm(span=12, adjust=False).mean()

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
# ax.plot(temperature_df['SMA_12'], label='SMA 12', color='yellow', linestyle='-')
# ax.plot(temperature_df['EMA_12'], label='EMA 12', color='red', linestyle=':')
# ax.set_title(f'{selected_country} Temperature Change Trend Over Time')
# ax.set_xlabel('Year')
# ax.set_ylabel('Scaled Temperature (TempC_Scaled)')
# ax.legend()
# st.pyplot(fig)

# # ETS Decomposition
# st.subheader("🔍 ETS Decomposition")
# decomposition = seasonal_decompose(temperature_df['TempC_Scaled'], model='additive', period=12)
# fig_decompose = decomposition.plot()
# fig_decompose.set_size_inches(14, 8)
# st.pyplot(fig_decompose)

# # Split data into train and test
# st.subheader("✂️ Train-Test Split")
# test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5)
# split_index = int((100 - test_size) / 100 * len(temperature_df))
# train = temperature_df.iloc[:split_index]
# test = temperature_df.iloc[split_index:]

# st.write(f"Training Data: {train.shape[0]} samples")
# st.write(f"Testing Data: {test.shape[0]} samples")

# # ADF Test
# st.subheader("🧪 Augmented Dickey-Fuller Test")
# adf_result = adfuller(train['TempC_Scaled'])
# st.write(f"**ADF Statistic:** {adf_result[0]:.6f}")
# st.write(f"**p-value:** {adf_result[1]:.6f}")
# st.write("**Critical Values:**")
# for key, value in adf_result[4].items():
#     st.write(f"   {key}: {value:.3f}")

# if adf_result[1] <= 0.05:
#     st.success("Reject the null hypothesis. The time series is stationary.")
# else:
#     st.warning("Fail to reject the null hypothesis. The time series is non-stationary.")

# # ACF and PACF Plots
# st.subheader("📈 Autocorrelation and Partial Autocorrelation")
# fig_acf, ax_acf = plt.subplots(1, 2, figsize=(16, 4))
# plot_acf(train['TempC_Scaled'], ax=ax_acf[0], lags=40)
# plot_pacf(train['TempC_Scaled'], ax=ax_acf[1], lags=40)
# st.pyplot(fig_acf)

# # Initialize a dictionary to store performance metrics
# metrics_dict = {
#     "Model": [],
#     "MSE": [],
#     "MAE": [],
#     "MAPE": [],
#     "RMSE": [],
#     "R2": []
# }

# # ARIMA Model
# st.subheader("📉 Manual ARIMA Model")
# arima_order = st.text_input("Enter ARIMA order (p,d,q)", "(2,0,2)")
# try:
#     order = tuple(map(int, arima_order.strip("()").split(',')))
#     arima_model = ARIMA(train['TempC_Scaled'], order=order)
#     arima_fit = arima_model.fit()
#     st.write("**ARIMA Model Summary:**")
#     st.text(arima_fit.summary())
    
#     # Save ARIMA model
#     arima_model_path = os.path.join(output_dir_model, 'arima_model.keras')
#     with open(arima_model_path, 'wb') as f:
#         pickle.dump(arima_fit, f)
#     st.success(f"ARIMA model saved to `{arima_model_path}`.")
# except Exception as e:
#     st.error(f"ARIMA model fitting failed: {e}")

# # Residuals Analysis
# st.subheader("🔍 Residuals Analysis")
# if 'arima_fit' in locals():
#     residuals = arima_fit.resid[1:]
#     fig_res, ax_res = plt.subplots(1, 2, figsize=(14, 5))
#     residuals.plot(title="Residuals", ax=ax_res[0])
#     residuals.plot(kind='kde', title='Density', ax=ax_res[1])
#     st.pyplot(fig_res)

#     fig_acf_res, ax_acf_res = plt.subplots(1, 2, figsize=(16, 4))
#     plot_acf(residuals, ax=ax_acf_res[0], lags=40)
#     plot_pacf(residuals, ax=ax_acf_res[1], lags=40)
#     st.pyplot(fig_acf_res)

#     # Forecast
#     forecast_steps = len(test)
#     forecast_arima = arima_fit.forecast(steps=forecast_steps)
#     temperature_df['forecast_manual'] = [np.nan]*len(train) + list(forecast_arima)
    
#     # Save ARIMA forecast
#     arima_forecast_path = os.path.join(output_dir_result, 'arima_forecast.csv')
#     pd.DataFrame({
#         'Actual': scaler.inverse_transform(test[['TempC_Scaled']]).flatten(),
#         'ARIMA Forecast': scaler.inverse_transform(np.array(forecast_arima).reshape(-1, 1)).flatten()
#     }).to_csv(arima_forecast_path, index=False)
#     st.success(f"ARIMA forecast saved to `{arima_forecast_path}`.")

#     fig_forecast, ax_forecast = plt.subplots(figsize=(12,6))
#     ax_forecast.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
#     ax_forecast.plot(temperature_df['forecast_manual'], label='ARIMA Forecast', color='green')
#     ax_forecast.set_title("ARIMA Forecast vs Actual")
#     ax_forecast.set_xlabel("Time")
#     ax_forecast.set_ylabel("Scaled Temperature")
#     ax_forecast.legend()
#     st.pyplot(fig_forecast)

# # Auto ARIMA
# st.subheader("🤖 Auto ARIMA Model")
# with st.spinner("Fitting Auto ARIMA... This may take a while!"):
#     try:
#         auto_arima_model = auto_arima(train['TempC_Scaled'], stepwise=False, seasonal=False, suppress_warnings=True)
#         st.write("**Auto ARIMA Model Summary:**")
#         st.text(auto_arima_model.summary())
        
#         # Save Auto ARIMA model
#         auto_arima_model_path = os.path.join(output_dir_model, 'auto_arima_model.keras')
#         with open(auto_arima_model_path, 'wb') as f:
#             pickle.dump(auto_arima_model, f)
#         st.success(f"Auto ARIMA model saved to `{auto_arima_model_path}`.")
        
#         # Forecast
#         forecast_auto = auto_arima_model.predict(n_periods=len(test))
#         temperature_df['forecast_auto'] = [np.nan]*len(train) + list(forecast_auto)
        
#         # Save Auto ARIMA forecast
#         auto_arima_forecast_path = os.path.join(output_dir_result, 'auto_arima_forecast.csv')
#         pd.DataFrame({
#             'Actual': scaler.inverse_transform(test[['TempC_Scaled']]).flatten(),
#             'Auto ARIMA Forecast': scaler.inverse_transform(np.array(forecast_auto).reshape(-1, 1)).flatten()
#         }).to_csv(auto_arima_forecast_path, index=False)
#         st.success(f"Auto ARIMA forecast saved to `{auto_arima_forecast_path}`.")
        
#         # Plot Auto ARIMA Forecast
#         fig_forecast_auto, ax_forecast_auto = plt.subplots(figsize=(12,6))
#         ax_forecast_auto.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
#         ax_forecast_auto.plot(temperature_df['forecast_auto'], label='Auto ARIMA Forecast', color='orange')
#         ax_forecast_auto.set_title("Auto ARIMA Forecast vs Actual")
#         ax_forecast_auto.set_xlabel("Time")
#         ax_forecast_auto.set_ylabel("Scaled Temperature")
#         ax_forecast_auto.legend()
#         st.pyplot(fig_forecast_auto)
    
#     except Exception as e:
#         st.error(f"Auto ARIMA failed: {e}")

# # Inverse Scaling for Metrics
# st.subheader("🔄 Inverse Scaling for Performance Metrics")
# train_exp = scaler.inverse_transform(train[['TempC_Scaled']])
# test_exp = scaler.inverse_transform(test[['TempC_Scaled']])

# if 'forecast_arima' in locals():
#     forecast_arima_exp = scaler.inverse_transform(np.array(forecast_arima).reshape(-1, 1))
# if 'forecast_auto' in locals():
#     forecast_auto_exp = scaler.inverse_transform(np.array(forecast_auto).reshape(-1, 1))

# # Function to display metrics with enhanced styling
# def display_metrics(title, mse, mae, mape, rmse, r2, color):
#     st.markdown(
#         f"""
#         <div style="background-color:{color}; padding:15px; border-radius:10px; margin-bottom:20px;">
#             <h3 style="color:#333;">{title}</h3>
#             <ul style="list-style-type:none; padding:0;">
#                 <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse:.4f}</span></li>
#                 <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae:.4f}</span></li>
#                 <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape:.4f}</span></li>
#                 <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse:.4f}</span></li>
#                 <li><strong>R²:</strong> <span style="color:#4682B4; font-size:1.2em;">{r2:.4f}</span></li>
#             </ul>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# # Performance Metrics for Manual ARIMA
# if 'forecast_arima_exp' in locals():
#     st.subheader("📊 Performance Metrics for Manual ARIMA")
#     mse_arima = mean_squared_error(test_exp, forecast_arima_exp)
#     mae_arima = mean_absolute_error(test_exp, forecast_arima_exp)
#     mape_arima = mean_absolute_percentage_error(test_exp, forecast_arima_exp)
#     rmse_arima = np.sqrt(mse_arima)
#     r2_arima = r2_score(test_exp, forecast_arima_exp)
    
#     # Display metrics
#     display_metrics(
#         title="Manual ARIMA Performance Metrics",
#         mse=mse_arima,
#         mae=mae_arima,
#         mape=mape_arima,
#         rmse=rmse_arima,
#         r2=r2_arima,
#         color="#d4edda"  # Light green background
#     )
    
#     # Save metrics
#     metrics_dict["Model"].append("Manual ARIMA")
#     metrics_dict["MSE"].append(mse_arima)
#     metrics_dict["MAE"].append(mae_arima)
#     metrics_dict["MAPE"].append(mape_arima)
#     metrics_dict["RMSE"].append(rmse_arima)
#     metrics_dict["R2"].append(r2_arima)

# # Performance Metrics for Auto ARIMA
# if 'forecast_auto_exp' in locals():
#     st.subheader("📊 Performance Metrics for Auto ARIMA")
#     mse_auto = mean_squared_error(test_exp, forecast_auto_exp)
#     mae_auto = mean_absolute_error(test_exp, forecast_auto_exp)
#     mape_auto = mean_absolute_percentage_error(test_exp, forecast_auto_exp)
#     rmse_auto = np.sqrt(mse_auto)
#     r2_auto = r2_score(test_exp, forecast_auto_exp)
    
#     # Display metrics
#     display_metrics(
#         title="Auto ARIMA Performance Metrics",
#         mse=mse_auto,
#         mae=mae_auto,
#         mape=mape_auto,
#         rmse=rmse_auto,
#         r2=r2_auto,
#         color="#cce5ff"  # Light blue background
#     )
    
#     # Save metrics
#     metrics_dict["Model"].append("Auto ARIMA")
#     metrics_dict["MSE"].append(mse_auto)
#     metrics_dict["MAE"].append(mae_auto)
#     metrics_dict["MAPE"].append(mape_auto)
#     metrics_dict["RMSE"].append(rmse_auto)
#     metrics_dict["R2"].append(r2_auto)

# # Auto ARIMA has been fitted; proceed to SARIMAX
# # SARIMAX Model
# st.subheader("🔧 SARIMAX Model")
# with st.spinner("Fitting SARIMAX... This may take a while!"):
#     try:
#         # Refit Auto ARIMA with seasonal components if needed
#         auto_arima_final = auto_arima(
#             temperature_df['TempC_Scaled'],
#             start_p=1,
#             start_q=1,
#             max_p=3,
#             max_q=3,
#             m=12,
#             start_P=0,
#             seasonal=True,
#             d=None,
#             D=1,
#             trace=False,
#             error_action='ignore',
#             suppress_warnings=True,
#             stepwise=True
#         )

#         st.write("**Final Auto ARIMA Model Summary with Seasonality:**")
#         st.text(auto_arima_final.summary())

#         # Extract orders
#         best_order = auto_arima_final.order
#         best_seasonal_order = auto_arima_final.seasonal_order
#         st.write(f"**Best ARIMA order:** {best_order}")
#         st.write(f"**Best Seasonal Order:** {best_seasonal_order}")

#         # Fit SARIMAX
#         sarimax_model = SARIMAX(train['TempC_Scaled'],  
#                                 order=best_order,  
#                                 seasonal_order=best_seasonal_order) 
#         sarimax_fit = sarimax_model.fit()
#         st.write("**SARIMAX Model Summary:**")
#         st.text(sarimax_fit.summary())

#         # Save SARIMAX model
#         sarimax_model_path = os.path.join(output_dir_model, 'sarimax_model.keras')
#         with open(sarimax_model_path, 'wb') as f:
#             pickle.dump(sarimax_fit, f)
#         st.success(f"SARIMAX model saved to `{sarimax_model_path}`.")

#         # Predictions
#         predictions = sarimax_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
#         temperature_df['forecast_sarimax'] = [np.nan]*len(train) + list(predictions)
        
#         # Save SARIMAX forecast
#         sarimax_forecast_path = os.path.join(output_dir_result, 'sarimax_forecast.csv')
#         pd.DataFrame({
#             'Actual': scaler.inverse_transform(test[['TempC_Scaled']]).flatten(),
#             'SARIMAX Forecast': scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
#         }).to_csv(sarimax_forecast_path, index=False)
#         st.success(f"SARIMAX forecast saved to `{sarimax_forecast_path}`.")

#         # Plot SARIMAX Forecast
#         fig_sarimax, ax_sarimax = plt.subplots(figsize=(12,6))
#         ax_sarimax.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
#         ax_sarimax.plot(temperature_df['forecast_sarimax'], label='SARIMAX Forecast', color='purple')
#         ax_sarimax.set_title("SARIMAX Forecast vs Actual")
#         ax_sarimax.set_xlabel("Time")
#         ax_sarimax.set_ylabel("Scaled Temperature")
#         ax_sarimax.legend()
#         st.pyplot(fig_sarimax)

#         # Inverse Scaling for SARIMAX
#         predictions_exp = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
#         # Performance Metrics for SARIMAX
#         st.subheader("📊 Performance Metrics for SARIMAX")
#         mse_sarimax = mean_squared_error(test_exp, predictions_exp)
#         mae_sarimax = mean_absolute_error(test_exp, predictions_exp)
#         mape_sarimax = mean_absolute_percentage_error(test_exp, predictions_exp)
#         rmse_sarimax = np.sqrt(mse_sarimax)
#         r2_sarimax = r2_score(test_exp, predictions_exp)

#         # Display metrics
#         display_metrics(
#             title="SARIMAX Performance Metrics",
#             mse=mse_sarimax,
#             mae=mae_sarimax,
#             mape=mape_sarimax,
#             rmse=rmse_sarimax,
#             r2=r2_sarimax,
#             color="#f8d7da"  # Light red background
#         )
        
#         # Save metrics
#         metrics_dict["Model"].append("SARIMAX")
#         metrics_dict["MSE"].append(mse_sarimax)
#         metrics_dict["MAE"].append(mae_sarimax)
#         metrics_dict["MAPE"].append(mape_sarimax)
#         metrics_dict["RMSE"].append(rmse_sarimax)
#         metrics_dict["R2"].append(r2_sarimax)

#     except Exception as e:
#         st.error(f"SARIMAX failed: {e}")

# # Exponential Smoothing
# st.subheader("🌊 Exponential Smoothing")
# with st.spinner("Fitting Exponential Smoothing..."):
#     try:
#         # Reconstruct the full scaled temperature for Exponential Smoothing
#         temperature_df_full = temperature_df.copy()
#         temperature_df_full['TempC_Scaled'] = list(train['TempC_Scaled']) + list(test['TempC_Scaled'])

#         # Fit Exponential Smoothing
#         es_model = ExponentialSmoothing(train['TempC_Scaled'], trend='add', seasonal='add', seasonal_periods=12)
#         es_fit = es_model.fit()
#         es_predictions = es_fit.forecast(len(test))
#         temperature_df_full['forecast_es'] = [np.nan]*len(train) + list(es_predictions)
        
#         # Save Exponential Smoothing model
#         es_model_path = os.path.join(output_dir_model, 'exponential_smoothing_model.keras')
#         with open(es_model_path, 'wb') as f:
#             pickle.dump(es_fit, f)
#         st.success(f"Exponential Smoothing model saved to `{es_model_path}`.")

#         # Save Exponential Smoothing forecast
#         es_forecast_path = os.path.join(output_dir_result, 'exponential_smoothing_forecast.csv')
#         pd.DataFrame({
#             'Actual': scaler.inverse_transform(test[['TempC_Scaled']]).flatten(),
#             'Exponential Smoothing Forecast': scaler.inverse_transform(np.array(es_predictions).reshape(-1, 1)).flatten()
#         }).to_csv(es_forecast_path, index=False)
#         st.success(f"Exponential Smoothing forecast saved to `{es_forecast_path}`.")

#         # Plot Exponential Smoothing Forecast
#         fig_es, ax_es = plt.subplots(figsize=(12,6))
#         ax_es.plot(test.index, test['TempC_Scaled'], label='Test', color='orange')
#         ax_es.plot(test.index, temperature_df_full['forecast_es'].dropna(), label='Exponential Smoothing Predictions', color='green')
#         ax_es.set_title("Exponential Smoothing Forecast vs Actual")
#         ax_es.set_xlabel("Time")
#         ax_es.set_ylabel("Scaled Temperature")
#         ax_es.legend()
#         st.pyplot(fig_es)

#         # Inverse Scaling for Exponential Smoothing
#         es_predictions_exp = scaler.inverse_transform(np.array(es_predictions).reshape(-1, 1))

#         # Performance Metrics for Exponential Smoothing
#         st.subheader("📊 Performance Metrics for Exponential Smoothing")
#         mse_es = mean_squared_error(test_exp, es_predictions_exp)
#         mae_es = mean_absolute_error(test_exp, es_predictions_exp)
#         mape_es = mean_absolute_percentage_error(test_exp, es_predictions_exp)
#         rmse_es = np.sqrt(mse_es)
#         r2_es = r2_score(test_exp, es_predictions_exp)

#         # Display metrics
#         display_metrics(
#             title="Exponential Smoothing Performance Metrics",
#             mse=mse_es,
#             mae=mae_es,
#             mape=mape_es,
#             rmse=rmse_es,
#             r2=r2_es,
#             color="#d1ecf1"  # Light cyan background
#         )
        
#         # Save metrics
#         metrics_dict["Model"].append("Exponential Smoothing")
#         metrics_dict["MSE"].append(mse_es)
#         metrics_dict["MAE"].append(mae_es)
#         metrics_dict["MAPE"].append(mape_es)
#         metrics_dict["RMSE"].append(rmse_es)
#         metrics_dict["R2"].append(r2_es)

#     except Exception as e:
#         st.error(f"Exponential Smoothing failed: {e}")


#     combined_df = pd.DataFrame({
#         'Actual': test_exp,
#         'Forecast Manual ARIMA': ,
#         'Forecast Auto ARIMA': ,
#         'Forecast SARIMA': ,
#         'Forecast Exponential Smôthing': ,
#     })

#     st.line_chart(combined_df)

# # Final Data Display with Predictions
# st.subheader("📈 Final Data with Forecasts")

# # Prepare the final DataFrame with actual and predicted values
# final_df = pd.DataFrame({
#     "Actual": test_exp.flatten()
# })

# if 'forecast_arima_exp' in locals():
#     final_df["ARIMA Forecast"] = forecast_arima_exp.flatten()
# if 'forecast_auto_exp' in locals():
#     final_df["Auto ARIMA Forecast"] = forecast_auto_exp.flatten()
# if 'predictions_exp' in locals():
#     final_df["SARIMAX Forecast"] = predictions_exp.flatten()
# if 'es_predictions_exp' in locals():
#     final_df["Exponential Smoothing Forecast"] = es_predictions_exp.flatten()

# st.write("**Actual vs Predicted Values on Test Set:**")
# st.dataframe(final_df.head(10))  # Display first 10 rows for brevity

# # Save the final_df to the output_dir_result
# csv_path = os.path.join(output_dir_result, 'predictions.csv')
# final_df.to_csv(csv_path, index=False)
# st.success(f"Predictions have been saved to `{csv_path}`.")

# # Save performance metrics to CSV
# if metrics_dict["Model"]:
#     metrics_df = pd.DataFrame(metrics_dict)
#     metrics_csv_path = os.path.join(output_dir_result, 'metrics.csv')
#     metrics_df.to_csv(metrics_csv_path, index=False)
#     st.success(f"Performance metrics have been saved to `{metrics_csv_path}`.")
# else:
#     st.warning("No performance metrics to save.")

# # Optionally, allow users to download the final_df as CSV
# # csv = final_df.to_csv(index=False).encode('utf-8')
# # st.download_button(
# #     label="📥 Download Predictions as CSV",
# #     data=csv,
# #     file_name='predictions.csv',
# #     mime='text/csv',
# # )

# # Footer
# st.markdown("---")
# st.markdown("© 2024 Temperature Time Series Analysis App")



import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# Directories for saving results and models
output_dir_result = './Global Temperature/Result/'
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok=True)

output_dir_model = './Global Temperature/Models/'
if not os.path.exists(output_dir_model):
    os.makedirs(output_dir_model, exist_ok=True)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Temperature Time Series Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the App
st.title("📈 Temperature Time Series Analysis")

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Load data
file_path = 'data/processed/updated_data_with_time.csv'
df = load_data(file_path)

# Set default country to Germany
selected_country = "Germany"

# Filter data based on selected country
country_data = df[df['Country'] == selected_country].copy()
country_data['Time'] = pd.to_datetime(country_data['Time'])
country_data = country_data.sort_values(by='Time')
country_data = country_data.set_index('Time', drop=True)

st.header(f"📊 Data for {selected_country}")
st.dataframe(country_data.head())

# Select the Temperature column
if 'TempC' not in country_data.columns:
    st.error("The selected dataset does not contain 'TempC' column.")
    st.stop()

temperature_df = country_data[['TempC']].copy()

# Scaling
st.subheader("🔄 Data Scaling")
scaler = MinMaxScaler(feature_range=(-1, 1))
temperature_df['TempC_Scaled'] = scaler.fit_transform(temperature_df[['TempC']])
st.write("Scaled Temperature Data (TempC_Scaled):")
st.dataframe(temperature_df.head())

# Moving Averages
st.subheader("📉 Moving Averages")
temperature_df['SMA_12'] = temperature_df['TempC_Scaled'].rolling(window=12).mean()
temperature_df['EMA_12'] = temperature_df['TempC_Scaled'].ewm(span=12, adjust=False).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
ax.plot(temperature_df['SMA_12'], label='SMA 12', color='yellow', linestyle='-')
ax.plot(temperature_df['EMA_12'], label='EMA 12', color='red', linestyle=':')
ax.set_title(f'{selected_country} Temperature Change Trend Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Scaled Temperature (TempC_Scaled)')
ax.legend()
st.pyplot(fig)

# ETS Decomposition
st.subheader("🔍 ETS Decomposition")
decomposition = seasonal_decompose(temperature_df['TempC_Scaled'], model='additive', period=12)
fig_decompose = decomposition.plot()
fig_decompose.set_size_inches(14, 8)
st.pyplot(fig_decompose)

# Split data into train and test
st.subheader("✂️ Train-Test Split")
test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5)
split_index = int((100 - test_size) / 100 * len(temperature_df))
train = temperature_df.iloc[:split_index]
test = temperature_df.iloc[split_index:]

st.write(f"Training Data: {train.shape[0]} samples")
st.write(f"Testing Data: {test.shape[0]} samples")

# ADF Test
st.subheader("🧪 Augmented Dickey-Fuller Test")
adf_result = adfuller(train['TempC_Scaled'])
st.write(f"**ADF Statistic:** {adf_result[0]:.6f}")
st.write(f"**p-value:** {adf_result[1]:.6f}")
st.write("**Critical Values:**")
for key, value in adf_result[4].items():
    st.write(f"   {key}: {value:.3f}")

if adf_result[1] <= 0.05:
    st.success("Reject the null hypothesis. The time series is stationary.")
else:
    st.warning("Fail to reject the null hypothesis. The time series is non-stationary.")

# ACF and PACF Plots
st.subheader("📈 Autocorrelation and Partial Autocorrelation")
fig_acf, ax_acf = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(train['TempC_Scaled'], ax=ax_acf[0], lags=40)
plot_pacf(train['TempC_Scaled'], ax=ax_acf[1], lags=40)
st.pyplot(fig_acf)

# Initialize a dictionary to store performance metrics
metrics_dict = {
    "Model": [],
    "MSE": [],
    "MAE": [],
    "MAPE": [],
    "RMSE": [],
    "R2": []
}

# Function to display metrics with enhanced styling
def display_metrics(title, mse, mae, mape, rmse, r2, color):
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h3 style="color:#333;">{title}</h3>
            <ul style="list-style-type:none; padding:0;">
                <li><strong>MSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mse:.4f}</span></li>
                <li><strong>MAE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mae:.4f}</span></li>
                <li><strong>MAPE:</strong> <span style="color:#4682B4; font-size:1.2em;">{mape:.4f}</span></li>
                <li><strong>RMSE:</strong> <span style="color:#4682B4; font-size:1.2em;">{rmse:.4f}</span></li>
                <li><strong>R²:</strong> <span style="color:#4682B4; font-size:1.2em;">{r2:.4f}</span></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------ Load and Predict with Existing Models ------------------

# Manual ARIMA Model
#st.subheader("📉 Manual ARIMA Model")
try:
    arima_model_path = os.path.join(output_dir_model, 'arima_model.keras')
    with open(arima_model_path, 'rb') as f:
        arima_fit = pickle.load(f)
    #st.write("**ARIMA Model Summary:**")
    #st.text(arima_fit.summary())
    
    # Forecast
    forecast_steps = len(test)
    forecast_arima = arima_fit.forecast(steps=forecast_steps)
    temperature_df['forecast_manual'] = [np.nan]*len(train) + list(forecast_arima)
    
    # Save ARIMA forecast
    arima_forecast_path = os.path.join(output_dir_result, 'arima_forecast.csv')
    pd.DataFrame({
        'Actual': scaler.inverse_transform(test[['TempC_Scaled']]).flatten(),
        'ARIMA Forecast': scaler.inverse_transform(np.array(forecast_arima).reshape(-1, 1)).flatten()
    }).to_csv(arima_forecast_path, index=False)
    st.success(f"ARIMA forecast saved to `{arima_forecast_path}`.")
    
    # fig_forecast, ax_forecast = plt.subplots(figsize=(12,6))
    # ax_forecast.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
    # ax_forecast.plot(temperature_df['forecast_manual'], label='ARIMA Forecast', color='green')
    # ax_forecast.set_title("ARIMA Forecast vs Actual")
    # ax_forecast.set_xlabel("Time")
    # ax_forecast.set_ylabel("Scaled Temperature")
    # ax_forecast.legend()
    # st.pyplot(fig_forecast)
    
except Exception as e:
    st.error(f"ARIMA model loading failed: {e}")

# Auto ARIMA Model
#st.subheader("🤖 Auto ARIMA Model")
try:
    auto_arima_model_path = os.path.join(output_dir_model, 'auto_arima_model.keras')
    with open(auto_arima_model_path, 'rb') as f:
        auto_arima_fit = pickle.load(f)
    #st.write("**Auto ARIMA Model Summary:**")
    #st.text(auto_arima_fit.summary())
    
    # Forecast
    forecast_auto = auto_arima_fit.predict(n_periods=len(test))
    temperature_df['forecast_auto'] = [np.nan]*len(train) + list(forecast_auto)
    
    # Save Auto ARIMA forecast
    auto_arima_forecast_path = os.path.join(output_dir_result, 'auto_arima_forecast.csv')
    pd.DataFrame({
        'Actual': scaler.inverse_transform(test[['TempC_Scaled']]).flatten(),
        'Auto ARIMA Forecast': scaler.inverse_transform(np.array(forecast_auto).reshape(-1, 1)).flatten()
    }).to_csv(auto_arima_forecast_path, index=False)
    st.success(f"Auto ARIMA forecast saved to `{auto_arima_forecast_path}`.")
    
    # Plot Auto ARIMA Forecast
    # fig_forecast_auto, ax_forecast_auto = plt.subplots(figsize=(12,6))
    # ax_forecast_auto.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
    # ax_forecast_auto.plot(temperature_df['forecast_auto'], label='Auto ARIMA Forecast', color='orange')
    # ax_forecast_auto.set_title("Auto ARIMA Forecast vs Actual")
    # ax_forecast_auto.set_xlabel("Time")
    # ax_forecast_auto.set_ylabel("Scaled Temperature")
    # ax_forecast_auto.legend()
    # st.pyplot(fig_forecast_auto)

except Exception as e:
    st.error(f"Auto ARIMA model loading failed: {e}")

# SARIMAX Model
#st.subheader("🔧 SARIMAX Model")
try:
    sarimax_model_path = os.path.join(output_dir_model, 'sarimax_model.keras')
    with open(sarimax_model_path, 'rb') as f:
        sarimax_fit = pickle.load(f)
    #st.write("**SARIMAX Model Summary:**")
    #st.text(sarimax_fit.summary())
    
    # Predictions
    predictions = sarimax_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    temperature_df['forecast_sarimax'] = [np.nan]*len(train) + list(predictions)
    
    # Save SARIMAX forecast
    sarimax_forecast_path = os.path.join(output_dir_result, 'sarimax_forecast.csv')
    pd.DataFrame({
        'Actual': scaler.inverse_transform(test[['TempC_Scaled']]).flatten(),
        'SARIMAX Forecast': scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    }).to_csv(sarimax_forecast_path, index=False)
    st.success(f"SARIMAX forecast saved to `{sarimax_forecast_path}`.")
    
    # Plot SARIMAX Forecast
    # fig_sarimax, ax_sarimax = plt.subplots(figsize=(12,6))
    # ax_sarimax.plot(temperature_df['TempC_Scaled'], label='Actual', color='blue')
    # ax_sarimax.plot(temperature_df['forecast_sarimax'], label='SARIMAX Forecast', color='purple')
    # ax_sarimax.set_title("SARIMAX Forecast vs Actual")
    # ax_sarimax.set_xlabel("Time")
    # ax_sarimax.set_ylabel("Scaled Temperature")
    # ax_sarimax.legend()
    # st.pyplot(fig_sarimax)

except Exception as e:
    st.error(f"SARIMAX model loading failed: {e}")

# Exponential Smoothing
#st.subheader("🌊 Exponential Smoothing")
try:
    es_model_path = os.path.join(output_dir_model, 'exponential_smoothing_model.keras')
    with open(es_model_path, 'rb') as f:
        es_fit = pickle.load(f)
    #st.write("**Exponential Smoothing Model Loaded Successfully.**")
    
    # Forecast
    es_predictions = es_fit.forecast(len(test))
    temperature_df['forecast_es'] = [np.nan]*len(train) + list(es_predictions)
    
    # Save Exponential Smoothing forecast
    es_forecast_path = os.path.join(output_dir_result, 'exponential_smoothing_forecast.csv')
    pd.DataFrame({
        'Actual': scaler.inverse_transform(test[['TempC_Scaled']]).flatten(),
        'Exponential Smoothing Forecast': scaler.inverse_transform(np.array(es_predictions).reshape(-1, 1)).flatten()
    }).to_csv(es_forecast_path, index=False)
    st.success(f"Exponential Smoothing forecast saved to `{es_forecast_path}`.")
    
    # Plot Exponential Smoothing Forecast
    # fig_es, ax_es = plt.subplots(figsize=(12,6))
    # ax_es.plot(test.index, test['TempC_Scaled'], label='Test', color='orange')
    # ax_es.plot(test.index, temperature_df['forecast_es'].dropna(), label='Exponential Smoothing Predictions', color='green')
    # ax_es.set_title("Exponential Smoothing Forecast vs Actual")
    # ax_es.set_xlabel("Time")
    # ax_es.set_ylabel("Scaled Temperature")
    # ax_es.legend()
    # st.pyplot(fig_es)

except Exception as e:
    st.error(f"Exponential Smoothing model loading failed: {e}")

# Inverse Scaling for Metrics
#st.subheader("🔄 Inverse Scaling for Performance Metrics")
train_exp = scaler.inverse_transform(train[['TempC_Scaled']])
test_exp = scaler.inverse_transform(test[['TempC_Scaled']])

if 'forecast_arima' in locals():
    forecast_arima_exp = scaler.inverse_transform(np.array(forecast_arima).reshape(-1, 1))
if 'forecast_auto' in locals():
    forecast_auto_exp = scaler.inverse_transform(np.array(forecast_auto).reshape(-1, 1))
if 'predictions' in locals():
    predictions_exp = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
if 'es_predictions' in locals():
    es_predictions_exp = scaler.inverse_transform(np.array(es_predictions).reshape(-1, 1))

# Performance Metrics for Manual ARIMA
if 'forecast_arima_exp' in locals():
    st.subheader("📊 Performance Metrics for Manual ARIMA")
    mse_arima = mean_squared_error(test_exp, forecast_arima_exp)
    mae_arima = mean_absolute_error(test_exp, forecast_arima_exp)
    mape_arima = mean_absolute_percentage_error(test_exp, forecast_arima_exp)
    rmse_arima = np.sqrt(mse_arima)
    r2_arima = r2_score(test_exp, forecast_arima_exp)
    
    # Display metrics
    display_metrics(
        title="Manual ARIMA Performance Metrics",
        mse=mse_arima,
        mae=mae_arima,
        mape=mape_arima,
        rmse=rmse_arima,
        r2=r2_arima,
        color="#d4edda"  # Light green background
    )
    
    # Save metrics
    metrics_dict["Model"].append("Manual ARIMA")
    metrics_dict["MSE"].append(mse_arima)
    metrics_dict["MAE"].append(mae_arima)
    metrics_dict["MAPE"].append(mape_arima)
    metrics_dict["RMSE"].append(rmse_arima)
    metrics_dict["R2"].append(r2_arima)

# Performance Metrics for Auto ARIMA
if 'forecast_auto_exp' in locals():
    st.subheader("📊 Performance Metrics for Auto ARIMA")
    mse_auto = mean_squared_error(test_exp, forecast_auto_exp)
    mae_auto = mean_absolute_error(test_exp, forecast_auto_exp)
    mape_auto = mean_absolute_percentage_error(test_exp, forecast_auto_exp)
    rmse_auto = np.sqrt(mse_auto)
    r2_auto = r2_score(test_exp, forecast_auto_exp)
    
    # Display metrics
    display_metrics(
        title="Auto ARIMA Performance Metrics",
        mse=mse_auto,
        mae=mae_auto,
        mape=mape_auto,
        rmse=rmse_auto,
        r2=r2_auto,
        color="#cce5ff"  # Light blue background
    )
    
    # Save metrics
    metrics_dict["Model"].append("Auto ARIMA")
    metrics_dict["MSE"].append(mse_auto)
    metrics_dict["MAE"].append(mae_auto)
    metrics_dict["MAPE"].append(mape_auto)
    metrics_dict["RMSE"].append(rmse_auto)
    metrics_dict["R2"].append(r2_auto)

# Performance Metrics for SARIMAX
if 'predictions_exp' in locals():
    st.subheader("📊 Performance Metrics for SARIMAX")
    mse_sarimax = mean_squared_error(test_exp, predictions_exp)
    mae_sarimax = mean_absolute_error(test_exp, predictions_exp)
    mape_sarimax = mean_absolute_percentage_error(test_exp, predictions_exp)
    rmse_sarimax = np.sqrt(mse_sarimax)
    r2_sarimax = r2_score(test_exp, predictions_exp)

    # Display metrics
    display_metrics(
        title="SARIMAX Performance Metrics",
        mse=mse_sarimax,
        mae=mae_sarimax,
        mape=mape_sarimax,
        rmse=rmse_sarimax,
        r2=r2_sarimax,
        color="#f8d7da"  # Light red background
    )
    
    # Save metrics
    metrics_dict["Model"].append("SARIMAX")
    metrics_dict["MSE"].append(mse_sarimax)
    metrics_dict["MAE"].append(mae_sarimax)
    metrics_dict["MAPE"].append(mape_sarimax)
    metrics_dict["RMSE"].append(rmse_sarimax)
    metrics_dict["R2"].append(r2_sarimax)

# Performance Metrics for Exponential Smoothing
if 'es_predictions_exp' in locals():
    st.subheader("📊 Performance Metrics for Exponential Smoothing")
    mse_es = mean_squared_error(test_exp, es_predictions_exp)
    mae_es = mean_absolute_error(test_exp, es_predictions_exp)
    mape_es = mean_absolute_percentage_error(test_exp, es_predictions_exp)
    rmse_es = np.sqrt(mse_es)
    r2_es = r2_score(test_exp, es_predictions_exp)

    # Display metrics
    display_metrics(
        title="Exponential Smoothing Performance Metrics",
        mse=mse_es,
        mae=mae_es,
        mape=mape_es,
        rmse=rmse_es,
        r2=r2_es,
        color="#d1ecf1"  # Light cyan background
    )
    
    # Save metrics
    metrics_dict["Model"].append("Exponential Smoothing")
    metrics_dict["MSE"].append(mse_es)
    metrics_dict["MAE"].append(mae_es)
    metrics_dict["MAPE"].append(mape_es)
    metrics_dict["RMSE"].append(rmse_es)
    metrics_dict["R2"].append(r2_es)

# Combined Forecasts DataFrame
st.subheader("📊 Combined Forecasts")
combined_df = pd.DataFrame({
    'Actual': test_exp.flatten()
})

if 'forecast_arima_exp' in locals():
    combined_df["Forecast Manual ARIMA"] = forecast_arima_exp.flatten()
if 'forecast_auto_exp' in locals():
    combined_df["Forecast Auto ARIMA"] = forecast_auto_exp.flatten()
if 'predictions_exp' in locals():
    combined_df["Forecast SARIMAX"] = predictions_exp.flatten()
if 'es_predictions_exp' in locals():
    combined_df["Forecast Exponential Smoothing"] = es_predictions_exp.flatten()

st.line_chart(combined_df)

# Final Data Display with Predictions
st.subheader("📈 Final Data with Forecasts")

# Prepare the final DataFrame with actual and predicted values
final_df = pd.DataFrame({
    "Actual": test_exp.flatten()
})

if 'forecast_arima_exp' in locals():
    final_df["ARIMA Forecast"] = forecast_arima_exp.flatten()
if 'forecast_auto_exp' in locals():
    final_df["Auto ARIMA Forecast"] = forecast_auto_exp.flatten()
if 'predictions_exp' in locals():
    final_df["SARIMAX Forecast"] = predictions_exp.flatten()
if 'es_predictions_exp' in locals():
    final_df["Exponential Smoothing Forecast"] = es_predictions_exp.flatten()

st.write("**Actual vs Predicted Values on Test Set:**")
st.dataframe(final_df)  # Display first 10 rows for brevity

# # Vẽ biểu đồ bằng Matplotlib
# fig_final, ax_final = plt.subplots(figsize=(14, 7))

# # Vẽ dữ liệu thực tế
# ax_final.plot(test.index, final_df["Actual"], label='Actual', color='blue')

# # Vẽ các dự báo nếu tồn tại
# if 'ARIMA Forecast' in final_df.columns:
#     ax_final.plot(test.index, final_df["ARIMA Forecast"], label='ARIMA Forecast', color='green')
# if 'Auto ARIMA Forecast' in final_df.columns:
#     ax_final.plot(test.index, final_df["Auto ARIMA Forecast"], label='Auto ARIMA Forecast', color='orange')
# if 'SARIMAX Forecast' in final_df.columns:
#     ax_final.plot(test.index, final_df["SARIMAX Forecast"], label='SARIMAX Forecast', color='purple')
# if 'Exponential Smoothing Forecast' in final_df.columns:
#     ax_final.plot(test.index, final_df["Exponential Smoothing Forecast"], label='Exponential Smoothing Forecast', color='red')

# # Thiết lập tiêu đề và nhãn trục
# ax_final.set_title("Actual vs Predicted Values on Test Set", fontsize=16)
# ax_final.set_xlabel("Time", fontsize=14)
# ax_final.set_ylabel("Temperature", fontsize=14)

# # Thêm legend
# ax_final.legend()

# # Hiển thị biểu đồ trong Streamlit
# st.pyplot(fig_final)


# Save the final_df to the output_dir_result
csv_path = os.path.join(output_dir_result, 'predictions.csv')
final_df.to_csv(csv_path, index=False)
st.success(f"Predictions have been saved to `{csv_path}`.")

# Save performance metrics to CSV
if metrics_dict["Model"]:
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_path = os.path.join(output_dir_result, 'metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    st.success(f"Performance metrics have been saved to `{metrics_csv_path}`.")
else:
    st.warning("No performance metrics to save.")

# Optionally, allow users to download the final_df as CSV
# csv = final_df.to_csv(index=False).encode('utf-8')
# st.download_button(
#     label="📥 Download Predictions as CSV",
#     data=csv,
#     file_name='predictions.csv',
#     mime='text/csv',
# )

# Footer
st.markdown("---")
st.markdown("© 2024 Temperature Time Series Analysis App")
