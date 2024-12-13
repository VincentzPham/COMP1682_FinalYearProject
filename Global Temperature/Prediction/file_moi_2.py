import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib  # Thư viện để lưu scaler_params.pkl
warnings.filterwarnings('ignore')

# Import các modules của sklearn và tensorflow cần thiết
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
import keras_tuner as kt

# Đọc dữ liệu từ file CSV
file_path = './updated_data_with_time.csv'
df = pd.read_csv(file_path, encoding='latin-1')

# Chỉ lấy dữ liệu từ Đức và sắp xếp theo thời gian
df = df[df['Country'] == 'Germany']
df = df.sort_values(by='Time')
df['Time'] = pd.to_datetime(df['Time'])

# Chọn cột 'Time' và 'TempC' và đặt 'Time' làm chỉ số
df = df[['Time', 'TempC']]
df = df.set_index('Time')

# Lấy dữ liệu nhiệt độ
data = df['TempC'].values

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# Lưu các tham số của MinMaxScaler vào scaler_params.pkl
scaler_params = {
    'min_': scaler.min_,
    'scale_': scaler.scale_
}
joblib.dump(scaler_params, 'scaler_params.pkl')
print("Đã lưu scaler_params.pkl với các tham số sau:")
print(scaler_params)

# Hàm tạo các chuỗi dữ liệu cho mô hình LSTM
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Chuẩn bị dữ liệu
sequence_length = 24
X, y = create_sequences(scaled_data, sequence_length)

# Chia dữ liệu thành bộ huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm xây dựng mô hình LSTM với siêu tham số và tối ưu hóa bias
def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', min_value=1, max_value=3, step=1)):
        model.add(LSTM(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                       activation=hp.Choice('activation', values=['relu', 'tanh']),
                       return_sequences=True if i < hp.get('num_layers') - 1 else False,
                       bias_initializer=Constant(value=hp.Choice('bias_initializer', values=[0.1, 0.2, 0.3, 0.5, 0.7]))))
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1,
                    bias_initializer=Constant(value=hp.Choice('bias_initializer', values=[0.1, 0.2, 0.3, 0.5, 0.7])),
                    kernel_regularizer=l2(0.01)))
    model.compile(
        optimizer=Adam(
            hp.Float('learning_rate', min_value=0.001, max_value=0.01, sampling='log')
        ),
        loss='mse'
    )
    return model

# Định nghĩa tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=100, 
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='tempC_prediction'
)

# Tìm kiếm siêu tham số tốt nhất
tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test),
             callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# Lấy siêu tham số tốt nhất
best_hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
print("Best hyperparameters found: ", best_hps)

# Hàm xây dựng mô hình với siêu tham số tốt nhất
def build_model_from_best_hyperparameters(best_hps):
    model = Sequential()
    num_layers = best_hps['num_layers']
    for i in range(num_layers):
        model.add(LSTM(units=best_hps[f'units_{i}'],
                       activation=best_hps['activation'],
                       return_sequences=True if i < num_layers - 1 else False,
                       bias_initializer=Constant(value=0.1)))
    model.add(Dropout(best_hps['dropout']))
    model.add(Dense(1, bias_initializer=Constant(value=0.1), kernel_regularizer=l2(0.01)))
    model.compile(
        optimizer=Adam(learning_rate=best_hps['learning_rate']),
        loss='mse'
    )
    return model

# Xây dựng và lưu mô hình tốt nhất (chưa huấn luyện)
best_model = build_model_from_best_hyperparameters(best_hps)
best_model.save('pretrained_best_model.keras', save_format='keras')

# Huấn luyện mô hình tốt nhất
history = best_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# Lưu mô hình đã huấn luyện
best_model.save('best_model_trained.keras', save_format='keras')

# Đánh giá mô hình
test_loss = best_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

# Dự đoán và chuyển đổi giá trị
y_pred_scaled = best_model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Đánh giá hiệu suất
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")

# Lưu kết quả và chỉ số đánh giá
results_df = pd.DataFrame({'True Values': y_test.flatten(), 'Predictions': y_pred.flatten()})
results_df.to_csv('temperature_predictions_vs_real.csv', index=False)

evaluation_metrics = pd.DataFrame({'RMSE': [rmse], 'MAE': [mae], 'MSE': [mse], 'MAPE': [mape]})
evaluation_metrics.to_csv('evaluation_metrics.csv', index=False)

# Biểu đồ so sánh
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values')
plt.plot(y_pred, label='Predictions')
plt.title('LSTM Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
from matplotlib.ticker import MaxNLocator
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

