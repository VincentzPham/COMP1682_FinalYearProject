# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# import streamlit as st

# # -------------- Setup Output Directory --------------
# output_dir_result = './GHG/Result/'
# if not os.path.exists(output_dir_result):
#     os.makedirs(output_dir_result)

# # -------------- Streamlit Application --------------
# # Set the title and description of the application
# st.title("Greenhouse Gas Emissions Forecasting Using LSTM and SimpleRNN Models")
# st.markdown("""
# This application uses LSTM and SimpleRNN models to forecast CO₂ emissions for a specific country based on historical data. 
# You can select a country and emission category to train the models and view the forecast results.
# """)

# # 1. Load Data
# st.header("1. Load Data")

# # Path to the CSV file
# file_path = 'data/raw/greenhouse_gas_inventory_data_completed.csv'

# # Check if the file exists
# if not os.path.isfile(file_path):
#     st.error(f"The file `{file_path}` does not exist in the current directory. Please check the file path.")
#     st.stop()

# # Read data with caching for faster load times
# @st.cache_data
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     return df

# df = load_data(file_path)

# st.success("Data loaded successfully!")
# st.write("### Original Data", df.head())

# # 2. Select Country and Emission Category
# st.header("2. Select Country and Emission Category")

# # Get the list of countries and emission categories from the dataset
# countries = df['country_or_area'].unique()
# categories = df['category'].unique()

# # Create selection widgets
# selected_country = st.selectbox("Select Country", options=sorted(countries))
# selected_category = st.selectbox("Select Emission Category", options=sorted(categories))

# st.write(f"**Selected Country:** {selected_country}")
# st.write(f"**Selected Emission Category:** {selected_category}")

# # 3. Data Processing
# st.header("3. Data Processing")

# # Filter data based on user selections
# df_filtered = df[
#     (df['country_or_area'] == selected_country) & 
#     (df['category'] == selected_category)
# ].sort_values('year')

# st.write("### Filtered Data", df_filtered.head())
# st.write(f"Number of records after filtering: {len(df_filtered)}")

# if df_filtered.empty:
#     st.error("Filtered data is empty. Please select another country or emission category.")
#     st.stop()

# # Reset index
# df_filtered = df_filtered.reset_index(drop=True)

# # Convert 'year' to datetime
# try:
#     df_filtered['year'] = pd.to_datetime(df_filtered['year'], format='%Y')
#     st.success("Converted 'year' column to datetime successfully.")
# except Exception as e:
#     st.error(f"Error converting 'year' to datetime: {e}")
#     st.stop()

# st.write("### Data after converting 'year'", df_filtered.head())

# # Set 'year' as the index
# df_sorted = df_filtered[['year', 'value']].set_index('year')
# st.write("### Data sorted by year", df_sorted.head())

# # Get data values
# data = df_sorted['value'].values

# # Create DataFrame for scaling
# df_data = pd.DataFrame({'value': data})
# st.write("### Data before scaling", df_data.head())

# # Check for missing values
# missing_values = df_data.isnull().sum()
# st.write("### Check Missing Values", missing_values)

# # If there are missing values, drop or fill them
# if df_data.isnull().sum().any():
#     df_data = df_data.dropna()
#     st.warning("Dropped rows with missing values.")

# # Scale the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# df_data['scaled_value'] = scaler.fit_transform(df_data[['value']])
# st.write("### Data after scaling", df_data.head())

# # Get scaled data
# scaled_data = df_data['scaled_value'].values

# # Create sequences
# def create_sequences(data, sequence_length):
#     xs = []
#     ys = []
#     for i in range(len(data) - sequence_length - 1):
#         x = data[i:(i + sequence_length)]
#         y = data[i + sequence_length]
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

# sequence_length = 5
# X, y = create_sequences(scaled_data, sequence_length)
# st.write("### Sequences Created Successfully")
# st.write(f"Shape of X: {X.shape}")
# st.write(f"Shape of y: {y.shape}")

# if X.size == 0:
#     st.error("Could not create sequences. Please adjust `sequence_length` or provide more data.")
#     st.stop()

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# st.write("### Data Split Successfully")
# st.write(f"Shape of X_train: {X_train.shape}")
# st.write(f"Shape of X_test: {X_test.shape}")
# st.write(f"Shape of y_train: {y_train.shape}")
# st.write(f"Shape of y_test: {y_test.shape}")

# # Reshape data for RNN and LSTM
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# st.write("### Data Reshaped Successfully")

# # 4. Train Models
# st.header("4. Train Models")

# if st.button("Start Training"):
#     with st.spinner('Training the models...'):
#         # ---------------------
#         # Train SimpleRNN Model
#         # ---------------------
#         st.subheader("4.1. Training SimpleRNN Model")
#         try:
#             # Build the SimpleRNN model
#             rnn_model = Sequential()
#             rnn_model.add(SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
#             rnn_model.add(SimpleRNN(units=50, activation='relu'))
#             rnn_model.add(Dense(8, activation='relu'))
#             rnn_model.add(Dense(1, activation='linear'))
            
#             # Compile the model
#             rnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
            
#             # Train the model with EarlyStopping
#             early_stop_rnn = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#             history_rnn = rnn_model.fit(
#                 X_train, y_train, 
#                 epochs=100, 
#                 batch_size=4, 
#                 validation_split=0.1, 
#                 callbacks=[early_stop_rnn],
#                 verbose=0
#             )
#             st.success("SimpleRNN model training completed!")
#         except Exception as e:
#             st.error(f"Error training SimpleRNN model: {e}")
#             st.stop()
        
#         # ---------------------
#         # Train LSTM Model
#         # ---------------------
#         st.subheader("4.2. Training LSTM Model")
#         try:
#             # Build the LSTM model
#             lstm_model = Sequential()
#             lstm_model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
#             lstm_model.add(LSTM(units=50, activation='relu'))
#             lstm_model.add(Dense(8, activation='relu'))
#             lstm_model.add(Dense(1, activation='linear'))
            
#             # Compile the model
#             lstm_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
            
#             # Train the model with EarlyStopping
#             early_stop_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#             history_lstm = lstm_model.fit(
#                 X_train, y_train, 
#                 epochs=100, 
#                 batch_size=4, 
#                 validation_split=0.1, 
#                 callbacks=[early_stop_lstm],
#                 verbose=0
#             )
#             st.success("LSTM model training completed!")
#         except Exception as e:
#             st.error(f"Error training LSTM model: {e}")
#             st.stop()
    
#     # 5. Display Results and Visualizations
#     st.header("5. Results and Visualizations")
    
#     # ---------------------
#     # SimpleRNN Results
#     # ---------------------
#     st.subheader("5.1. SimpleRNN Training History")
#     def plot_training_history_rnn(history):
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(history.history['loss'], label='Training Loss')
#         ax.plot(history.history['val_loss'], label='Validation Loss')
#         ax.set_title('SimpleRNN Loss over Epochs')
#         ax.set_xlabel('Epochs')
#         ax.set_ylabel('Loss')
#         ax.legend()
#         st.pyplot(fig)
    
#     plot_training_history_rnn(history_rnn)
    
#     st.subheader("5.2. SimpleRNN Predictions vs Actual")
#     try:
#         # Predictions and inverse transform for SimpleRNN
#         predicted_rnn = rnn_model.predict(X_test)
#         predicted_rnn = scaler.inverse_transform(predicted_rnn)
#         actual_emissions = scaler.inverse_transform(y_test.reshape(-1, 1))
    
#         st.write("### SimpleRNN: Sample Predictions and Actual Values")
#         comparison_rnn_df = pd.DataFrame({
#             'Actual Emissions': actual_emissions.flatten(),
#             'Predicted Emissions': predicted_rnn.flatten()
#         })
#         st.write(comparison_rnn_df.head())
    
#         # Visualization for SimpleRNN
#         st.write("### SimpleRNN: Comparison Chart")
#         fig_rnn, ax_rnn = plt.subplots(figsize=(10, 6))
#         ax_rnn.plot(actual_emissions, color='blue', label='Actual Emissions')
#         ax_rnn.plot(predicted_rnn, color='red', linestyle='--', label='Predicted Emissions')
#         ax_rnn.set_title('SimpleRNN Emissions Prediction')
#         ax_rnn.set_xlabel('Time Steps')
#         ax_rnn.set_ylabel('Emissions')
#         ax_rnn.legend()
#         st.pyplot(fig_rnn)
        
#         # Save SimpleRNN comparison DataFrame
#         comparison_rnn_path = os.path.join(output_dir_result, 'SimpleRNN_Predictions_vs_Actual.csv')
#         comparison_rnn_df.to_csv(comparison_rnn_path, index=False)
#         st.success(f"SimpleRNN predictions saved to `{comparison_rnn_path}`")
        
#         # Download Button for SimpleRNN Predictions
#         # st.download_button(
#         #     label="Download SimpleRNN Predictions CSV",
#         #     data=comparison_rnn_df.to_csv(index=False).encode('utf-8'),
#         #     file_name='SimpleRNN_Predictions_vs_Actual.csv',
#         #     mime='text/csv',
#         # )
        
#         # Optional: Save SimpleRNN model
#         rnn_model_path = os.path.join(output_dir_result, 'SimpleRNN_Model.keras')
#         rnn_model.save(rnn_model_path, save_format='keras')
#         st.success(f"SimpleRNN model saved to `{rnn_model_path}`")
#     except Exception as e:
#         st.error(f"Error in SimpleRNN predictions: {e}")
    
#     # ---------------------
#     # LSTM Results
#     # ---------------------
#     st.subheader("5.3. LSTM Training History")
#     def plot_training_history_lstm(history):
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(history.history['loss'], label='Training Loss')
#         ax.plot(history.history['val_loss'], label='Validation Loss')
#         ax.set_title('LSTM Loss over Epochs')
#         ax.set_xlabel('Epochs')
#         ax.set_ylabel('Loss')
#         ax.legend()
#         st.pyplot(fig)
    
#     plot_training_history_lstm(history_lstm)
    
#     st.subheader("5.4. LSTM Predictions vs Actual")
#     try:
#         # Predictions and inverse transform for LSTM
#         predicted_lstm = lstm_model.predict(X_test)
#         predicted_lstm = scaler.inverse_transform(predicted_lstm)
#         actual_emissions = scaler.inverse_transform(y_test.reshape(-1, 1))
    
#         st.write("### LSTM: Sample Predictions and Actual Values")
#         comparison_lstm_df = pd.DataFrame({
#             'Actual Emissions': actual_emissions.flatten(),
#             'Predicted Emissions': predicted_lstm.flatten()
#         })
#         st.write(comparison_lstm_df.head())
    
#         # Visualization for LSTM
#         st.write("### LSTM: Comparison Chart")
#         fig_lstm, ax_lstm = plt.subplots(figsize=(10, 6))
#         ax_lstm.plot(actual_emissions, color='blue', label='Actual Emissions')
#         ax_lstm.plot(predicted_lstm, color='green', linestyle='--', label='Predicted Emissions')
#         ax_lstm.set_title('LSTM Emissions Prediction')
#         ax_lstm.set_xlabel('Time Steps')
#         ax_lstm.set_ylabel('Emissions')
#         ax_lstm.legend()
#         st.pyplot(fig_lstm)
        
#         # Save LSTM comparison DataFrame
#         comparison_lstm_path = os.path.join(output_dir_result, 'LSTM_Predictions_vs_Actual.csv')
#         comparison_lstm_df.to_csv(comparison_lstm_path, index=False)
#         st.success(f"LSTM predictions saved to `{comparison_lstm_path}`")
        
#         # Download Button for LSTM Predictions
#         # st.download_button(
#         #     label="Download LSTM Predictions CSV",
#         #     data=comparison_lstm_df.to_csv(index=False).encode('utf-8'),
#         #     file_name='LSTM_Predictions_vs_Actual.csv',
#         #     mime='text/csv',
#         # )
        
#         # Optional: Save LSTM model
#         lstm_model_path = os.path.join(output_dir_result, 'LSTM_Model.keras')
#         lstm_model.save(lstm_model_path, save_format='keras')
#         st.success(f"LSTM model saved to `{lstm_model_path}`")
#     except Exception as e:
#         st.error(f"Error in LSTM predictions: {e}")
    
#     # ---------------------
#     # Model Evaluation Metrics
#     # ---------------------
#     st.subheader("5.5. Model Evaluation Metrics")
#     try:
#         # Metrics for SimpleRNN
#         mse_rnn = mean_squared_error(actual_emissions, predicted_rnn)
#         rmse_rnn = np.sqrt(mse_rnn)
#         mae_rnn = mean_absolute_error(actual_emissions, predicted_rnn)
#         mape_rnn = mean_absolute_percentage_error(actual_emissions, predicted_rnn)
#         r2_rnn = r2_score(actual_emissions, predicted_rnn)
    
#         metrics_rnn_df = pd.DataFrame({
#             'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 
#                        'Mean Absolute Percentage Error (MAPE)', 'Mean Absolute Error (MAE)', 
#                        'R-squared (R2)'],
#             'SimpleRNN': [mse_rnn, rmse_rnn, mape_rnn, mae_rnn, r2_rnn]
#         })
    
#         # Metrics for LSTM
#         mse_lstm = mean_squared_error(actual_emissions, predicted_lstm)
#         rmse_lstm = np.sqrt(mse_lstm)
#         mae_lstm = mean_absolute_error(actual_emissions, predicted_lstm)
#         mape_lstm = mean_absolute_percentage_error(actual_emissions, predicted_lstm)
#         r2_lstm = r2_score(actual_emissions, predicted_lstm)
    
#         metrics_lstm_df = pd.DataFrame({
#             'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 
#                        'Mean Absolute Percentage Error (MAPE)', 'Mean Absolute Error (MAE)', 
#                        'R-squared (R2)'],
#             'LSTM': [mse_lstm, rmse_lstm, mape_lstm, mae_lstm, r2_lstm]
#         })
    
#         # Combine metrics
#         combined_metrics_df = metrics_rnn_df.merge(metrics_lstm_df, on='Metric')
    
#         st.write("### Model Evaluation Metrics")
#         st.table(combined_metrics_df)
    
#         # Save combined metrics DataFrame
#         metrics_path = os.path.join(output_dir_result, 'Model_Evaluation_Metrics.csv')
#         combined_metrics_df.to_csv(metrics_path, index=False)
#         st.success(f"Model evaluation metrics saved to `{metrics_path}`")
        
#         # Download Button for Metrics
#         # st.download_button(
#         #     label="Download Model Evaluation Metrics CSV",
#         #     data=combined_metrics_df.to_csv(index=False).encode('utf-8'),
#         #     file_name='Model_Evaluation_Metrics.csv',
#         #     mime='text/csv',
#         # )
#     except Exception as e:
#         st.error(f"Error calculating metrics: {e}")


#current code
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import streamlit as st

# -------------- Setup Output Directory --------------
output_dir_result = './GHG/Result/'
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result)

output_dir_model = './GHG/Models/'
if not os.path.exists(output_dir_model):
    os.makedirs(output_dir_model)

st.set_page_config(page_title="Greenhouse Gas Emissions Deep Learning", layout="wide")

# -------------- Streamlit Application --------------
# Set the title and description of the application
st.title("Greenhouse Gas Emissions Forecasting Using LSTM and SimpleRNN Models")
st.markdown("""
This application uses pre-trained LSTM and SimpleRNN models to forecast CO₂ emissions for a specific country based on historical data. 
The models are automatically loaded and predictions are generated without the need for user interaction.
""")

# 1. Load Data
st.header("1. Load Data")

# Path to the CSV file
file_path = 'data/raw/greenhouse_gas_inventory_data_completed.csv'

# Check if the file exists
if not os.path.isfile(file_path):
    st.error(f"The file `{file_path}` does not exist in the current directory. Please check the file path.")
    st.stop()

# Read data with caching for faster load times
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

df = load_data(file_path)

st.success("Data loaded successfully!")
st.write("### Original Data", df.head())

# 2. Set Default Country and Emission Category
st.header("2. Configuration")

# Set default country and emission category
default_country = 'Germany'  # Replace with your desired default country
default_category = 'CO2 Emissions'   # Replace with your desired default category

st.write(f"**Country:** {default_country}")
st.write(f"**Emission Category:** {default_category}")

# 3. Data Processing
st.header("3. Data Processing")

# Filter data based on default selections
df_filtered = df[
    (df['country_or_area'] == default_country) & 
    (df['category'] == default_category)
].sort_values('year')

st.write("### Filtered Data", df_filtered.head())
st.write(f"Number of records after filtering: {len(df_filtered)}")

if df_filtered.empty:
    st.error("Filtered data is empty. Please check the default country and emission category.")
    st.stop()

# Reset index
df_filtered = df_filtered.reset_index(drop=True)

# Convert 'year' to datetime
try:
    df_filtered['year'] = pd.to_datetime(df_filtered['year'], format='%Y')
    st.success("Converted 'year' column to datetime successfully.")
except Exception as e:
    st.error(f"Error converting 'year' to datetime: {e}")
    st.stop()

st.write("### Data after converting 'year'", df_filtered.head())

# Set 'year' as the index
df_sorted = df_filtered[['year', 'value']].set_index('year')
st.write("### Data sorted by year", df_sorted.head())

# Get data values
data = df_sorted['value'].values

# Create DataFrame for scaling
df_data = pd.DataFrame({'value': data})
st.write("### Data before scaling", df_data.head())

# Check for missing values
missing_values = df_data.isnull().sum()
st.write("### Check Missing Values", missing_values)

# If there are missing values, drop or fill them
if df_data.isnull().sum().any():
    df_data = df_data.dropna()
    st.warning("Dropped rows with missing values.")

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_data['scaled_value'] = scaler.fit_transform(df_data[['value']])
st.write("### Data after scaling", df_data.head())

# Get scaled data
scaled_data = df_data['scaled_value'].values

# Create sequences
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
st.write("### Sequences Created Successfully")
st.write(f"Shape of X: {X.shape}")
st.write(f"Shape of y: {y.shape}")

if X.size == 0:
    st.error("Could not create sequences. Please adjust `sequence_length` or provide more data.")
    st.stop()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
st.write("### Data Split Successfully")
st.write(f"Shape of X_train: {X_train.shape}")
st.write(f"Shape of X_test: {X_test.shape}")
st.write(f"Shape of y_train: {y_train.shape}")
st.write(f"Shape of y_test: {y_test.shape}")

# Reshape data for RNN and LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
st.write("### Data Reshaped Successfully")

# 4. Load Pre-trained Models and Make Predictions
st.header("4. Load Models and Make Predictions")

# Paths to the pre-trained models
rnn_model_path = os.path.join(output_dir_model, 'SimpleRNN_Model.keras')
lstm_model_path = os.path.join(output_dir_model, 'LSTM_Model.keras')

# Check if model files exist
if not os.path.isfile(rnn_model_path):
    st.error(f"The RNN model file `{rnn_model_path}` does not exist.")
    st.stop()

if not os.path.isfile(lstm_model_path):
    st.error(f"The LSTM model file `{lstm_model_path}` does not exist.")
    st.stop()

# Load the models
try:
    rnn_model = load_model(rnn_model_path)
    st.success("SimpleRNN model loaded successfully!")
except Exception as e:
    st.error(f"Error loading SimpleRNN model: {e}")
    st.stop()

try:
    lstm_model = load_model(lstm_model_path)
    st.success("LSTM model loaded successfully!")
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")
    st.stop()

# Make predictions
try:
    with st.spinner('Making predictions with SimpleRNN model...'):
        predicted_rnn = rnn_model.predict(X_test)
        predicted_rnn = scaler.inverse_transform(predicted_rnn)
    st.success("SimpleRNN predictions completed!")

    with st.spinner('Making predictions with LSTM model...'):
        predicted_lstm = lstm_model.predict(X_test)
        predicted_lstm = scaler.inverse_transform(predicted_lstm)
    st.success("LSTM predictions completed!")

    # Inverse transform actual values
    actual_emissions = scaler.inverse_transform(y_test.reshape(-1, 1))
except Exception as e:
    st.error(f"Error during predictions: {e}")
    st.stop()

# 5. Display Results and Visualizations
st.header("5. Results and Visualizations")

# ---------------------
# SimpleRNN Results
# ---------------------
st.subheader("5.1. SimpleRNN Predictions vs Actual")
try:
    st.write("### SimpleRNN: Sample Predictions and Actual Values")
    comparison_rnn_df = pd.DataFrame({
        'Actual Emissions': actual_emissions.flatten(),
        'Predicted Emissions': predicted_rnn.flatten()
    })
    st.write(comparison_rnn_df.head())

    # Visualization for SimpleRNN
    st.write("### SimpleRNN: Comparison Chart")
    fig_rnn, ax_rnn = plt.subplots(figsize=(10, 6))
    ax_rnn.plot(actual_emissions, color='blue', label='Actual Emissions')
    ax_rnn.plot(predicted_rnn, color='red', linestyle='--', label='Predicted Emissions')
    ax_rnn.set_title('SimpleRNN Emissions Prediction')
    ax_rnn.set_xlabel('Time Steps')
    ax_rnn.set_ylabel('Emissions')
    ax_rnn.legend()
    
    ax_rnn.set_ylim(bottom = 0)
    
    st.pyplot(fig_rnn)
    
    # fig_rnn_path = os.path.join(output_dir_result, 'SimpleRNN_Predictions.png')
    # fig_rnn.savefig(fig_rnn_path)  # Lưu biểu đồ vào tệp

    
    # Save SimpleRNN comparison DataFrame
    comparison_rnn_path = os.path.join(output_dir_result, 'SimpleRNN_Predictions_vs_Actual.csv')
    comparison_rnn_df.to_csv(comparison_rnn_path, index=False)
    st.success(f"SimpleRNN predictions saved to `{comparison_rnn_path}`")
except Exception as e:
    st.error(f"Error in SimpleRNN predictions: {e}")

# ---------------------
# LSTM Results
# ---------------------
st.subheader("5.2. LSTM Predictions vs Actual")
try:
    st.write("### LSTM: Sample Predictions and Actual Values")
    comparison_lstm_df = pd.DataFrame({
        'Actual Emissions': actual_emissions.flatten(),
        'Predicted Emissions': predicted_lstm.flatten()
    })
    st.write(comparison_lstm_df.head())

    # Visualization for LSTM
    st.write("### LSTM: Comparison Chart")
    fig_lstm, ax_lstm = plt.subplots(figsize=(10, 6))
    ax_lstm.plot(actual_emissions, color='blue', label='Actual Emissions')
    ax_lstm.plot(predicted_lstm, color='green', linestyle='--', label='Predicted Emissions')
    ax_lstm.set_title('LSTM Emissions Prediction')
    ax_lstm.set_xlabel('Time Steps')
    ax_lstm.set_ylabel('Emissions')
    ax_lstm.legend()
    
    ax_lstm.set_ylim(bottom=0)
    
    st.pyplot(fig_lstm)
    
    # Save LSTM comparison DataFrame
    comparison_lstm_path = os.path.join(output_dir_result, 'LSTM_Predictions_vs_Actual.csv')
    comparison_lstm_df.to_csv(comparison_lstm_path, index=False)
    st.success(f"LSTM predictions saved to `{comparison_lstm_path}`")
except Exception as e:
    st.error(f"Error in LSTM predictions: {e}")

# ---------------------
# Model Evaluation Metrics
# ---------------------
st.subheader("5.3. Model Evaluation Metrics")
try:
    # Metrics for SimpleRNN
    mse_rnn = mean_squared_error(actual_emissions, predicted_rnn)
    rmse_rnn = np.sqrt(mse_rnn)
    mae_rnn = mean_absolute_error(actual_emissions, predicted_rnn)
    mape_rnn = mean_absolute_percentage_error(actual_emissions, predicted_rnn)
    r2_rnn = r2_score(actual_emissions, predicted_rnn)

    metrics_rnn_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 
                   'Mean Absolute Percentage Error (MAPE)', 'Mean Absolute Error (MAE)', 
                   'R-squared (R2)'],
        'SimpleRNN': [mse_rnn, rmse_rnn, mape_rnn, mae_rnn, r2_rnn]
    })

    # Metrics for LSTM
    mse_lstm = mean_squared_error(actual_emissions, predicted_lstm)
    rmse_lstm = np.sqrt(mse_lstm)
    mae_lstm = mean_absolute_error(actual_emissions, predicted_lstm)
    mape_lstm = mean_absolute_percentage_error(actual_emissions, predicted_lstm)
    r2_lstm = r2_score(actual_emissions, predicted_lstm)

    metrics_lstm_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 
                   'Mean Absolute Percentage Error (MAPE)', 'Mean Absolute Error (MAE)', 
                   'R-squared (R2)'],
        'LSTM': [mse_lstm, rmse_lstm, mape_lstm, mae_lstm, r2_lstm]
    })

    # Combine metrics
    combined_metrics_df = metrics_rnn_df.merge(metrics_lstm_df, on='Metric')

    st.write("### Model Evaluation Metrics")
    st.table(combined_metrics_df)

    # Save combined metrics DataFrame
    metrics_path = os.path.join(output_dir_result, 'Model_Evaluation_Metrics.csv')
    combined_metrics_df.to_csv(metrics_path, index=False)
    st.success(f"Model evaluation metrics saved to `{metrics_path}`")
except Exception as e:
    st.error(f"Error calculating metrics: {e}")
