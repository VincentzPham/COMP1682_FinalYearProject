import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load your data
df = pd.read_csv('../../data/raw/greenhouse_gas_inventory_data_completed.csv')
df.head()

df['year'] = pd.to_datetime(df['year'], format='%Y')
df = df.sort_values(by='year')
df = df.set_index('year')
df.head()

germany = df[(df['country_or_area'] == 'Germany') & (df['category'] == 'CO2 Emissions')]
germany.head()

germany = germany.drop(['country_or_area', 'category','continent'], axis=1)
germany.head()

germany = np.log(germany)
germany.plot()

# ETS Decomposition
result = seasonal_decompose(germany['value'],
                            model ='additive', period = 1)

# ETS plot
result.plot();

train = germany.iloc[:22]
test = germany.iloc[22:]

train, test

# Perform the ADF test on the training data
result = adfuller(train['value'])

# Print the test results
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Interpret the results
if result[1] <= 0.05:
    print("Reject the null hypothesis. The time series is stationary.")
else:
    print("Fail to reject the null hypothesis. The time series is non-stationary.")

acf_original = plot_acf(train['value'])

pacf_original = plot_pacf(train['value'])

train_diff = train.diff().dropna()
train_diff.plot()

acf_diff = plot_acf(train_diff['value'])

pacf_diff = plot_pacf(train_diff['value'])

adf_test = adfuller(train_diff['value'])
print(f'p-value: {adf_test[1]}')

model = ARIMA(train, order = (1,1,1))
model_fit = model.fit()
print(model_fit.summary())

residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

acf_res = plot_acf(residuals)

pacf_res = plot_pacf(residuals)

forecast_test = model_fit.forecast(steps=len(test))

germany['forecast_manual'] = [None]*len(train) + list(forecast_test)

germany.plot()

auto_arima = auto_arima(train, seasonal=False)
auto_arima

auto_arima.summary()

forecast_test_auto = auto_arima.predict(n_periods=len(test))
germany['forecast_auto'] = [None]*len(train) + list(forecast_test_auto)

germany.plot()

"""Metrics"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

mse = mean_squared_error(test, forecast_test)
mae = mean_absolute_error(test, forecast_test)
mape = mean_absolute_percentage_error(test, forecast_test)
rmse = np.sqrt(mean_squared_error(test, forecast_test))

print(f'mse - manual: {mse}')
print(f'mae - manual: {mae}')
print(f'mape - manual: {mape}')
print(f'rmse - manual: {rmse}')

mse = mean_squared_error(test, forecast_test_auto)
mae = mean_absolute_error(test, forecast_test_auto)
mape = mean_absolute_percentage_error(test, forecast_test_auto)
rmse = np.sqrt(mean_squared_error(test, forecast_test_auto))

print(f'mse - auto: {mse}')
print(f'mae - auto: {mae}')
print(f'mape - auto: {mape}')
print(f'rmse - auto: {rmse}')

# Chuyển đổi ngược về giá trị ban đầu từ log
train_exp = np.exp(train)
test_exp = np.exp(test)
forecast_test_exp = np.exp(forecast_test)
forecast_test_auto_exp = np.exp(forecast_test_auto)

# Cập nhật lại cột forecast_manual và forecast_auto để lưu kết quả ngược từ log về giá trị ban đầu
germany['value'] = list(train_exp['value']) + list(test_exp['value'])
germany['forecast_manual'] = [None]*len(train) + list(forecast_test_exp)
germany['forecast_auto'] = [None]*len(train) + list(forecast_test_auto_exp)

# Vẽ biểu đồ để thấy giá trị thực và giá trị dự báo sau khi đã chuyển đổi ngược về
germany.plot(title="Dự báo so với giá trị thực (đã chuyển về giá trị ban đầu)")
plt.show()

# Tính toán lại các metric sau khi chuyển đổi ngược
mse_manual = mean_squared_error(test_exp, forecast_test_exp)
mae_manual = mean_absolute_error(test_exp, forecast_test_exp)
mape_manual = mean_absolute_percentage_error(test_exp, forecast_test_exp)
rmse_manual = np.sqrt(mse_manual)

print(f'MSE - manual (back to original scale): {mse_manual}')
print(f'MAE - manual (back to original scale): {mae_manual}')
print(f'MAPE - manual (back to original scale): {mape_manual}')
print(f'RMSE - manual (back to original scale): {rmse_manual}')

mse_auto = mean_squared_error(test_exp, forecast_test_auto_exp)
mae_auto = mean_absolute_error(test_exp, forecast_test_auto_exp)
mape_auto = mean_absolute_percentage_error(test_exp, forecast_test_auto_exp)
rmse_auto = np.sqrt(mse_auto)

print(f'MSE - auto (back to original scale): {mse_auto}')
print(f'MAE - auto (back to original scale): {mae_auto}')
print(f'MAPE - auto (back to original scale): {mape_auto}')
print(f'RMSE - auto (back to original scale): {rmse_auto}')
