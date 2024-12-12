import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Kiểm tra thư mục hiện tại và các tệp
print("Current Working Directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

# Đường dẫn tới tệp CSV
file_path = '../../data/raw/greenhouse_gas_inventory_data_completed.csv'

# Kiểm tra sự tồn tại của tệp
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Đọc dữ liệu
print("Loading data...")
df = pd.read_csv(file_path)
print("DataFrame loaded successfully.")
print(df.head())

# Đảm bảo các cột cần thiết tồn tại
required_columns = ['country_or_area', 'category', 'year', 'value']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Lọc dữ liệu và sắp xếp theo năm
df_germany_co2 = df[
    (df['country_or_area'] == 'Germany') & 
    (df['category'] == 'CO2 Emissions')
].sort_values('year')

print("Filtered DataFrame:")
print(df_germany_co2.head())
print("Number of records after filtering:", len(df_germany_co2))

if df_germany_co2.empty:
    raise ValueError("Filtered DataFrame is empty. Check your filter conditions.")

# Reset chỉ mục
df_germany_co2 = df_germany_co2.reset_index(drop=True)

# Chuyển đổi 'year' thành datetime
try:
    df_germany_co2['year'] = pd.to_datetime(df_germany_co2['year'], format='%Y')
    print("Conversion to datetime successful.")
except Exception as e:
    print("Error converting 'year' to datetime:", e)
    raise

print(df_germany_co2.head())

# Đặt 'year' làm chỉ mục
df_germany_co2_sorted = df_germany_co2[['year','value']].set_index('year')
print("Sorted DataFrame:")
print(df_germany_co2_sorted.head())

# Lấy dữ liệu
data = df_germany_co2_sorted['value'].values

# Tạo DataFrame cho scaling
df_data = pd.DataFrame({'value': data})
print("Data before scaling:")
print(df_data.head())

# Kiểm tra giá trị thiếu
print("Missing values in data:", df_data.isnull().sum())

# Nếu có giá trị thiếu, loại bỏ hoặc điền giá trị
if df_data.isnull().sum().any():
    df_data = df_data.dropna()
    print("Dropped rows with missing values.")

# Scaling dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
df_data['scaled_value'] = scaler.fit_transform(df_data[['value']])
print("Data after scaling:")
print(df_data.head())

# Lấy dữ liệu đã scale
scaled_data = df_data['scaled_value'].values

# Tạo sequences
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length - 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 5
X, y = create_sequences(scaled_data, sequence_length)
print("Sequences created.")
print("X shape:", X.shape)
print("y shape:", y.shape)

if X.size == 0:
    raise ValueError("No sequences created. Adjust sequence_length or provide more data.")

# Split dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split into training and testing sets.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Reshape dữ liệu cho LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print("Data reshaped for LSTM.")

# Xây dựng mô hình
model = Sequential()
model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, activation='tanh'))
model.add(Dense(8, 'relu'))
model.add(Dense(1, activation='linear'))
print("Model built successfully.")

# Compile mô hình
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
print("Model compiled.")

# Đào tạo mô hình với EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("Starting model training...")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=4, 
    validation_split=0.1, 
    callbacks=[early_stop],
    verbose=1
)
print("Model trained.")

def plot_training_history(model_history):
    plt.figure(figsize=(10, 5))
    plt.plot(model_history.history['loss'], label='Training Loss')
    plt.plot(model_history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#Hàm plot model đã train
plot_training_history(history)

# Dự đoán và inverse transform
print("Predicting emissions...")
predicted_emissions = model.predict(X_test)
predicted_emissions = scaler.inverse_transform(predicted_emissions)
print("Predicted emissions:", predicted_emissions[:5])

# Inverse transform actual emissions
actual_emissions = scaler.inverse_transform(y_test.reshape(-1, 1))
print("Actual emissions:", actual_emissions[:5])

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(actual_emissions, color='blue', label='Actual Emissions')
plt.plot(predicted_emissions, color='red', linestyle='--', label='Predicted Emissions')
plt.title('Emissions Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Emissions')
plt.legend()
plt.show()

# prompt: use metrics to compare actual_emissions and predicted_emissions

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Calculate metrics
mse = mean_squared_error(actual_emissions, predicted_emissions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_emissions, predicted_emissions)
mape = mean_absolute_percentage_error(actual_emissions, predicted_emissions)
r2 = r2_score(actual_emissions, predicted_emissions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")