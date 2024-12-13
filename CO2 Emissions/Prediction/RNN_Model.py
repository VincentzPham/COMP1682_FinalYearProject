import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# file_path = "Dataset/data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv"
file_path = "../../data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv"

df = pd.read_csv(file_path)
df

uk_data = df[df['Country'] == 'UNITED KINGDOM']
uk_data['Year'] = pd.to_datetime(uk_data['Year'], format='%Y')
uk_data = uk_data.sort_values(by='Year')
uk_data = uk_data.reset_index(drop=True)
uk_data = uk_data.set_index('Year')
uk_data

# Drop 'Country' and 'Continent' columns to focus on numerical features
numerical_features = uk_data.drop(['Country', 'Continent'], axis=1)

# Calculate the correlation matrix
correlation_matrix = numerical_features.corr()

# Plot the heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(uk_data.index, uk_data['Total'], label='Total Fuel Usage')
plt.title('Total Fossil Fuel Usage in the United Kingdom Over Time')
plt.xlabel('Year')
plt.ylabel('Total Fuel Usage (in thousands of metric tons)')
plt.grid()
plt.legend()
plt.show()

data = uk_data[['Total']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences for LSTM model
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data)-sequence_length-1):
        x = data[i:(i+sequence_length)]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Prepare the data
sequence_length = 30
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(SimpleRNN(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# Predicting and inverse transforming the predictions
predicted_temperature = model.predict(X_test)
predicted_temperature = scaler.inverse_transform(predicted_temperature)

# Inverse transform the actual temperature for comparison
actual_temperature = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualization
plt.figure(figsize=(10,6))
plt.plot(actual_temperature, color='blue', label='Actual Temperature')
plt.plot(predicted_temperature, color='red', linestyle='--', label='Predicted Temperature')
plt.title('Temperature Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# prompt: show me MAPE, MAE, RMSE, MSE

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred)**2)

mape = mean_absolute_percentage_error(actual_temperature, predicted_temperature)
mae = mean_absolute_error(actual_temperature, predicted_temperature)
rmse = root_mean_squared_error(actual_temperature, predicted_temperature)
mse = mean_squared_error(actual_temperature, predicted_temperature)

print(f"MAPE: {mape}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MSE: {mse}")

