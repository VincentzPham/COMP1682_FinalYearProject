import streamlit as st
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib  # To save scaler parameters
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN  # Added SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import keras_tuner as kt  # For hyperparameter tuning

# Ensure output directory exists
output_dir = './CO2 Emissions/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir_result = './CO2 Emissions/Result/'
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result)

output_dir_model = './CO2 Emissions/Models/'
if not os.path.exists(output_dir_model):
    os.makedirs(output_dir_model)

# Set page configuration
st.set_page_config(
    page_title="Fossil Fuel CO2 Emissions Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("Fossil Fuel CO2 Emissions Analysis for Various Countries")

# Sidebar for country and column selection
st.sidebar.header("Select Options for Analysis")

# Load the dataset
@st.cache_data
def load_data():
    # file_path = "./fossil_fuel_co2_emissions-by-nation_with_continent.csv"
    file_path = "data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv"
    if not os.path.exists(file_path):
        st.error(f"Data file not found at `{file_path}`. Please ensure the file is in the correct directory.")
        st.stop()
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Extract unique countries from the dataset
countries = df['Country'].unique()

# Add a selectbox to the sidebar for country selection
selected_country = st.sidebar.selectbox("Choose a Country", options=sorted(countries))

# Filter data for the selected country
country_data = df[df['Country'] == selected_country]

if country_data.empty:
    st.warning(f"No data available for {selected_country}. Please select a different country.")
    st.stop()

# Convert 'Year' to datetime and sort
country_data['Year'] = pd.to_datetime(country_data['Year'], format='%Y')
country_data = country_data.sort_values(by='Year')
country_data = country_data.reset_index(drop=True)
country_data = country_data.set_index('Year')

st.subheader(f"Data for {selected_country}")
st.write(country_data)

# Identify numerical columns excluding 'Country' and 'Continent'
numerical_columns = country_data.select_dtypes(include=[np.number]).columns.tolist()

# Check if 'Total' is in numerical columns; if not, handle accordingly
if 'Total' not in numerical_columns:
    st.warning(f"'Total' column not found in the dataset for {selected_country}. Please select a different column.")

# Add a selectbox for column selection with 'Total' as default
default_column = 'Total' if 'Total' in numerical_columns else numerical_columns[0]
selected_column = st.sidebar.selectbox(
    "Select Column to Train the Model On",
    options=sorted(numerical_columns),
    index=sorted(numerical_columns).index(default_column) if default_column in sorted(numerical_columns) else 0
)

st.write(f"**Selected Column for Training:** `{selected_column}`")

# Check if the selected column exists
if selected_column not in country_data.columns:
    st.error(f"The selected column `{selected_column}` does not exist in the dataset.")
    st.stop()

# Check for missing values in the selected column
if country_data[selected_column].isnull().sum() > 0:
    st.warning(f"The selected column `{selected_column}` contains missing values. These will be filled using forward fill method.")
    country_data[selected_column].fillna(method='ffill', inplace=True)

# Plot selected column over time
st.subheader(f"Fossil Fuel Usage Over Time ({selected_column})")
plt.figure(figsize=(10, 6))
plt.plot(country_data.index, country_data[selected_column], label=selected_column)
plt.title(f'{selected_column} in {selected_country} Over Time')
plt.xlabel('Year')
plt.ylabel(f'{selected_column} (in thousands of metric tons)')
plt.grid(True)
plt.legend()
st.pyplot(plt)

# Prepare the data for modeling
data = country_data[[selected_column]].values

# Check if there is enough data for the sequence length
sequence_length = 30
if len(data) < sequence_length + 2:
    st.error(f"Not enough data to create sequences with a length of {sequence_length}. Please select a country with more data points.")
    st.stop()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save scaler parameters using joblib
scaler_params = {
    'min_': scaler.min_,
    'scale_': scaler.scale_
}
# joblib.dump(scaler_params, 'scaler_params.pkl')
joblib.dump(scaler_params, os.path.join(output_dir_model, 'scaler_params.pkl'))
st.success("Scaler parameters saved as `scaler_params.pkl`.")

# Function to create sequences for modeling
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
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

st.subheader("Model Training with Hyperparameter Tuning")

# Button to start training
if st.button("Train Model"):
    ### Begin Simple RNN Model Training ###
    st.write("**Simple RNN Model Training**")
    
    # Define the Simple RNN model
    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(
        units=50,  # You can adjust the number of units as needed
        activation='relu',
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    rnn_model.add(Dense(1))
    
    # Compile the RNN model
    rnn_model.compile(
        optimizer=Adam(),
        loss='mean_squared_error'
    )
    
    st.write("**Training Simple RNN Model...**")
    
    # Train the RNN model
    with st.spinner('Training Simple RNN model...'):
        rnn_history = rnn_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            ],
            verbose=0
        )
    st.success('Simple RNN model training completed!')
    
    # Plot RNN training and validation loss
    st.subheader("Simple RNN Training and Validation Loss")
    plt.figure(figsize=(10,6))
    plt.plot(rnn_history.history['loss'], label='Training Loss')
    plt.plot(rnn_history.history['val_loss'], label='Validation Loss')
    plt.title('Simple RNN Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Save the trained RNN model
    rnn_model_path = 'simple_rnn_trained.keras'
    rnn_model.save(os.path.join(output_dir_model, 'simple_rnn_trained.keras'), save_format='keras')
    # rnn_model.save(rnn_model_path, save_format='keras')
    st.write(f"**Trained Simple RNN model saved as `{rnn_model_path}`**")
    
    # Predicting and inverse transforming the RNN predictions
    rnn_predicted = rnn_model.predict(X_test)
    rnn_predicted = scaler.inverse_transform(rnn_predicted)
    
    # Inverse transform the actual values for comparison
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Visualization of RNN predictions vs actual
    st.subheader("Simple RNN Prediction vs Actual")
    plt.figure(figsize=(10,6))
    plt.plot(actual, color='blue', label='Actual')
    plt.plot(rnn_predicted, color='red', linestyle='--', label='RNN Predicted')
    plt.title(f'Simple RNN Prediction for {selected_country} - {selected_column}')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{selected_column} (in thousands of metric tons)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Define evaluation metrics for RNN
    def mean_absolute_percentage_error_custom(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero and handle very small y_true
        epsilon = np.finfo(np.float64).eps
        non_zero = y_true > epsilon
        if np.sum(non_zero) == 0:
            return np.nan
        return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

    def mean_absolute_error_metric(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    def root_mean_squared_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def mean_squared_error_metric(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean((y_true - y_pred)**2)

    # Calculate RNN metrics
    rnn_mape = mean_absolute_percentage_error_custom(actual, rnn_predicted)
    rnn_mae = mean_absolute_error_metric(actual, rnn_predicted)
    rnn_rmse = root_mean_squared_error(actual, rnn_predicted)
    rnn_mse = mean_squared_error_metric(actual, rnn_predicted)

    # Display RNN metrics
    st.subheader("Simple RNN Evaluation Metrics")
    rnn_col1, rnn_col2 = st.columns(2)
    with rnn_col1:
        if np.isnan(rnn_mape):
            st.metric("MAPE", "Undefined")
        else:
            st.metric("MAPE", f"{rnn_mape:.2f}%")
        st.metric("MAE", f"{rnn_mae:.2f}")
    with rnn_col2:
        st.metric("RMSE", f"{rnn_rmse:.2f}")
        st.metric("MSE", f"{rnn_mse:.2f}")

    # Save RNN evaluation metrics and predictions
    with st.spinner('Saving RNN evaluation metrics and predictions...'):
        # Round metrics to 2 decimal places before saving
        rnn_evaluation_metrics = pd.DataFrame({
            'RMSE': [round(rnn_rmse, 2)],
            'MAE': [round(rnn_mae, 2)],
            'MSE': [round(rnn_mse, 2)],
            'MAPE': [round(rnn_mape, 2) if not np.isnan(rnn_mape) else 'Undefined']
        })
        rnn_evaluation_metrics.to_csv(os.path.join(output_dir_result, 'rnn_evaluation_metrics.csv'), index=False)
        # rnn_evaluation_metrics.to_csv('rnn_evaluation_metrics.csv', index=False)
        st.write("**RNN Evaluation metrics saved as `rnn_evaluation_metrics.csv`**")

        # Save RNN prediction results
        rnn_results_df = pd.DataFrame({
            'True Values': actual.flatten(),
            'RNN Predictions': rnn_predicted.flatten()
        })
        rnn_results_df.to_csv(os.path.join(output_dir_result, 'rnn_predictions_vs_real.csv'), index=False)
        # rnn_results_df.to_csv('rnn_predictions_vs_real.csv', index=False)
        st.write("**RNN Prediction results saved as `rnn_predictions_vs_real.csv`**")
    
    ### End of Simple RNN Model Training ###
    
    # Existing LSTM Model Training with Hyperparameter Tuning continues here...
    
    # Define the model building function for Keras Tuner
    def build_model(hp):
        model = Sequential()
        # Number of LSTM layers
        num_layers = hp.Int('num_layers', min_value=1, max_value=2, step=1)
        for i in range(num_layers):
            # Number of units in LSTM layer
            units = hp.Int(f'units_{i}', min_value=30, max_value=150, step=20)
            # Activation function
            activation = hp.Choice('activation', values=['relu', 'tanh'])
            # Bias initializer
            bias_init = hp.Choice('bias_initializer', values=[0.1, 0.2, 0.3, 0.5, 0.7])
            # Add LSTM layer
            model.add(LSTM(
                units=units,
                activation=activation,
                return_sequences=True if i < num_layers - 1 else False,
                bias_initializer=Constant(value=bias_init)
            ))
        # Dropout layer
        dropout_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate))
        # Dense output layer
        model.add(Dense(
            1,
            bias_initializer=Constant(value=hp.Choice('bias_initializer', values=[0.1, 0.2, 0.3, 0.5, 0.7])),
            kernel_regularizer=l2(0.01)
        ))
        # Learning rate for optimizer
        learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='log')
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        return model

    # Define tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=100,  # Adjust as needed
        executions_per_trial=1,
        # directory='tuner_dir',
        directory=os.path.join(output_dir_model, 'tuner_dir'),  # Save in subdirectory
        project_name='fossil_fuel_emissions'
    )

    # Display tuner search progress
    st.write("**Hyperparameter Tuning:** Performing hyperparameter search. This may take several minutes...")

    # Perform hyperparameter search
    with st.spinner('Searching for the best hyperparameters...'):
        tuner.search(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        )
    st.success('Hyperparameter tuning completed!')

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    st.write("**Best Hyperparameters Found:**")
    best_hp_dict = {
        'num_layers': best_hps.get('num_layers'),
        'units': [best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))],
        'activation': best_hps.get('activation'),
        'bias_initializer': best_hps.get('bias_initializer'),
        'dropout': best_hps.get('dropout'),
        'learning_rate': best_hps.get('learning_rate')
    }
    st.write(best_hp_dict)

    # Function to build model with best hyperparameters
    def build_model_from_best_hyperparameters(best_hps):
        model = Sequential()
        num_layers = best_hps.get('num_layers')
        for i in range(num_layers):
            model.add(LSTM(
                units=best_hps.get(f'units_{i}'),
                activation=best_hps.get('activation'),
                return_sequences=True if i < num_layers - 1 else False,
                bias_initializer=Constant(value=best_hps.get('bias_initializer'))
            ))
        model.add(Dropout(best_hps.get('dropout')))
        model.add(Dense(
            1,
            bias_initializer=Constant(value=best_hps.get('bias_initializer')),
            kernel_regularizer=l2(0.01)
        ))
        model.compile(
            optimizer=Adam(learning_rate=best_hps.get('learning_rate')),
            loss='mean_squared_error'
        )
        return model

    # Build and save the best model (untrained)
    best_model = build_model_from_best_hyperparameters(best_hps)
    best_model_path = 'best_model_pretrained.keras'
    best_model.save(os.path.join(output_dir_model, 'best_model_pretrained.keras'), save_format='keras')
    # best_model.save(best_model_path, save_format='keras')
    st.write(f"**Best model architecture saved as `{best_model_path}`**")

    # Train the best model
    with st.spinner('Training the best model...'):
        history = best_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            ],
            verbose=0
        )
    st.success('Model training completed!')

    # Plot training and validation loss
    st.subheader("LSTM Training and Validation Loss")
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Save the trained model
    trained_model_path = 'best_model_trained.keras'
    best_model.save(os.path.join(output_dir_model, 'best_model_trained.keras'), save_format='keras')
    # best_model.save(trained_model_path, save_format='keras')
    st.write(f"**Trained LSTM model saved as `{trained_model_path}`**")

    # Predicting and inverse transforming the LSTM predictions
    predicted = best_model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)

    # Inverse transform the actual values for comparison
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualization of LSTM predictions vs actual
    st.subheader("LSTM Prediction vs Actual")
    plt.figure(figsize=(10,6))
    plt.plot(actual, color='blue', label='Actual')
    plt.plot(predicted, color='red', linestyle='--', label='LSTM Predicted')
    plt.title(f'LSTM Prediction for {selected_country} - {selected_column}')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{selected_column} (in thousands of metric tons)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Define evaluation metrics
    def mean_absolute_percentage_error_custom(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero and handle very small y_true
        epsilon = np.finfo(np.float64).eps
        non_zero = y_true > epsilon
        if np.sum(non_zero) == 0:
            return np.nan
        return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

    def mean_absolute_error_metric(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    def root_mean_squared_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def mean_squared_error_metric(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean((y_true - y_pred)**2)

    # Calculate LSTM metrics
    mape = mean_absolute_percentage_error_custom(actual, predicted)
    mae = mean_absolute_error_metric(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    mse = mean_squared_error_metric(actual, predicted)

    # Display LSTM metrics
    st.subheader("LSTM Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        if np.isnan(mape):
            st.metric("MAPE", "Undefined")
        else:
            st.metric("MAPE", f"{mape:.2f}%")
        st.metric("MAE", f"{mae:.2f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MSE", f"{mse:.2f}")

    # Save scaler parameters and evaluation metrics
    with st.spinner('Saving evaluation metrics and predictions...'):
        # Round metrics to 2 decimal places before saving
        evaluation_metrics = pd.DataFrame({
            'RMSE': [round(rmse, 2)],
            'MAE': [round(mae, 2)],
            'MSE': [round(mse, 2)],
            'MAPE': [round(mape, 2) if not np.isnan(mape) else 'Undefined']
        })
        evaluation_metrics.to_csv(os.path.join(output_dir_result, 'evaluation_metrics.csv'), index=False)
        # evaluation_metrics.to_csv('evaluation_metrics.csv', index=False)
        st.write("**Evaluation metrics saved as `evaluation_metrics.csv`**")

        # Save prediction results
        results_df = pd.DataFrame({
            'True Values': actual.flatten(),
            'LSTM Predictions': predicted.flatten()
        })
        results_df.to_csv(os.path.join(output_dir_result, 'lstm_predictions_vs_real.csv'), index=False)
        # results_df.to_csv('lstm_predictions_vs_real.csv', index=False)
        st.write("**LSTM Prediction results saved as `lstm_predictions_vs_real.csv`**")

    st.success("All processes completed successfully!")
