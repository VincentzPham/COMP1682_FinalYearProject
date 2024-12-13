import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
import keras_tuner as kt

# Suppress warnings
warnings.filterwarnings('ignore')

output_dir = './Global Temperature/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok= True)

output_dir_result = './Global Temperature/Result/'
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok= True)

output_dir_models = './Global Temperature/Models/'
if not os.path.exists(output_dir_models):
    os.makedirs(output_dir_models, exist_ok= True)


# Custom CSS for styling
def local_css(file_name):
    """Function to load custom CSS"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# You can create a separate CSS file and include it if needed
# For simplicity, I'll add CSS directly here
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-title {
        font-size: 16px;
        color: #555;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit App
st.title("📈 LSTM and SimpleRNN Temperature Forecasting for Countries")
st.write("""
This application trains both LSTM and SimpleRNN models to forecast temperature based on historical data from selected countries. 
It includes data preprocessing, hyperparameter tuning for LSTM, model training, evaluation, and visualization.
""")

@st.cache_data
def load_all_countries(file_path):
    """Load unique countries from the dataset."""
    df = pd.read_csv(file_path, encoding='latin-1', usecols=['Country'])
    countries = df['Country'].dropna().unique().tolist()
    return sorted(countries)

@st.cache_data
def load_data(file_path, country):
    """Load and filter data for the selected country."""
    df = pd.read_csv(file_path, encoding='latin-1')
    # Check if 'Country' column exists
    if 'Country' not in df.columns:
        st.error("The CSV file does not contain a 'Country' column.")
        return None
    # Filter for the selected country and sort by time
    df = df[df['Country'] == country]
    if df.empty:
        st.error(f"No data found for country: {country}")
        return None
    if 'Time' not in df.columns or 'TempC' not in df.columns:
        st.error("The CSV file must contain 'Time' and 'TempC' columns.")
        return None
    df = df.sort_values(by='Time')
    df['Time'] = pd.to_datetime(df['Time'])
    # Select relevant columns
    df = df[['Time', 'TempC']]
    df = df.set_index('Time')
    return df

@st.cache_data
def preprocess_data(df, output_dir_models = './Global Temperature/Models/', scaler_path='scaler_params.pkl'):
    """Scale the temperature data and save scaler parameters."""
    data = df['TempC'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Save scaler parameters
    scaler_params = {
        'min_': scaler.min_,
        'scale_': scaler.scale_
    }
    scaler_path_full = os.path.join(output_dir_models, scaler_path)
    joblib.dump(scaler_params, scaler_path_full)
    return scaled_data, scaler_params

def create_sequences(data, sequence_length):
    """Create input sequences and corresponding targets."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length - 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(hp):
    """Build LSTM model with hyperparameters from Keras Tuner."""
    model = Sequential()
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    for i in range(num_layers):
        model.add(LSTM(
            units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
            activation=hp.Choice('activation', values=['relu', 'tanh']),
            return_sequences=True if i < num_layers - 1 else False,
            bias_initializer=Constant(value=hp.Choice('bias_initializer', values=[0.1, 0.2, 0.3, 0.5, 0.7]))
        ))
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(
        1,
        bias_initializer=Constant(value=hp.Choice('bias_initializer', values=[0.1, 0.2, 0.3, 0.5, 0.7])),
        kernel_regularizer=l2(0.01)
    ))
    model.compile(
        optimizer=Adam(
            hp.Float('learning_rate', min_value=0.001, max_value=0.01, sampling='log')
        ),
        loss='mse'
    )
    return model

def build_lstm_model_from_best_hyperparameters(best_hps_dict):
    """Build the best LSTM model based on tuned hyperparameters."""
    model = Sequential()
    num_layers = best_hps_dict['num_layers']
    for i in range(num_layers):
        model.add(LSTM(
            units=best_hps_dict[f'units_{i}'],
            activation=best_hps_dict['activation'],
            return_sequences=True if i < num_layers - 1 else False,
            bias_initializer=Constant(value=best_hps_dict['bias_initializer'])
        ))
    model.add(Dropout(best_hps_dict['dropout']))
    model.add(Dense(
        1, 
        bias_initializer=Constant(value=best_hps_dict['bias_initializer']),
        kernel_regularizer=l2(0.01)
    ))
    model.compile(
        optimizer=Adam(learning_rate=best_hps_dict['learning_rate']),
        loss='mse'
    )
    return model

def build_rnn_model():
    """Build a SimpleRNN model with predefined parameters."""
    model = Sequential()
    # First SimpleRNN layer with return_sequences=True to stack another RNN layer
    model.add(SimpleRNN(
        units=50,
        activation='relu',
        return_sequences=True,
        input_shape=(None, 1)
    ))
    # Second SimpleRNN layer with return_sequences=False
    model.add(SimpleRNN(
        units=50,
        activation='relu',
        return_sequences=False
    ))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

def main():
    st.sidebar.title("🔧 Options")
    file_path = st.sidebar.text_input("📂 CSV File Path", "data/processed/updated_data_with_time.csv")
    
    # Initialize countries list
    countries = []
    if os.path.exists(file_path):
        try:
            countries = load_all_countries(file_path)
        except Exception as e:
            st.sidebar.error(f"❌ Error loading countries: {e}")
    else:
        st.sidebar.warning(f"⚠️ File not found: {file_path}")
    
    if countries:
        country = st.sidebar.selectbox("🌍 Select Country", countries)
    else:
        country = None
    
    sequence_length = st.sidebar.slider("🔢 Sequence Length", min_value=10, max_value=48, value=24, step=2)
    test_size = st.sidebar.slider("📊 Test Size (Fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("🎲 Random State", min_value=0, max_value=100, value=42, step=1)
    
    if st.sidebar.button("🚀 Run Models"):
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            return
        if not country:
            st.error("❌ Please select a country.")
            return
        
        with st.spinner('🔄 Loading and preprocessing data...'):
            df = load_data(file_path, country)
            if df is None:
                return
            st.success("✅ Data loaded successfully!")
            st.write(f"**Selected Country:** 🌍 {country}")
            st.write("**First few rows of the dataset:**")
            st.dataframe(df.head())
        
        with st.spinner('📈 Scaling data...'):
            scaled_data, scaler_params = preprocess_data(df)
            st.success("✅ Data scaled successfully!")
            st.write("**Scaler parameters:**")
            st.json(scaler_params)
        
        with st.spinner('🔗 Creating sequences...'):
            X, y = create_sequences(scaled_data, sequence_length)
            st.success(f"✅ Created sequences with length {sequence_length}.")
            st.write(f"**Total samples:** {X.shape[0]}")
        
        with st.spinner('✂️ Splitting data into train and test sets...'):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            st.success(f"✅ Data split into train and test sets with test size {test_size}.")
            st.write(f"**Training samples:** {X_train.shape[0]}, **Testing samples:** {X_test.shape[0]}")
        
        # Reshape for RNN and LSTM
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # =================== Train SimpleRNN Model ===================
        st.header("🤖 SimpleRNN Model Training")
        with st.spinner('🔧 Building and training SimpleRNN model...'):
            rnn_model = build_rnn_model()
            history_rnn = rnn_model.fit(
                X_train_reshaped, y_train, 
                epochs=50,  # Same as LSTM
                batch_size=32,  # Consistent batch size
                validation_data=(X_test_reshaped, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
                verbose=0
            )
            rnn_model.save(os.path.join(output_dir_models, 'simple_rnn_model_trained.keras'), save_format='keras')
            # rnn_model.save('simple_rnn_model_trained.keras', save_format='keras')
            st.success("✅ SimpleRNN model training completed and saved as 'simple_rnn_model_trained.keras'.")
        
        with st.spinner('📊 Evaluating SimpleRNN model...'):
            test_loss_rnn = rnn_model.evaluate(X_test_reshaped, y_test, verbose=0)
            st.write(f"**SimpleRNN Test Loss (MSE):** {test_loss_rnn:.4f}")
            
            y_pred_rnn_scaled = rnn_model.predict(X_test_reshaped)
            # Reconstruct the scaler using saved parameters
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.min_, scaler.scale_ = scaler_params['min_'], scaler_params['scale_']
            y_pred_rnn = scaler.inverse_transform(y_pred_rnn_scaled)
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            mae_rnn = mean_absolute_error(y_true, y_pred_rnn)
            mse_rnn = mean_squared_error(y_true, y_pred_rnn)
            rmse_rnn = np.sqrt(mse_rnn)
            mape_rnn = mean_absolute_percentage_error(y_true, y_pred_rnn)
        
        # =================== Display SimpleRNN Metrics ===================
        st.subheader("📈 SimpleRNN Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">MAE</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
            """.format(mae_rnn), unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">MSE</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
            """.format(mse_rnn), unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">RMSE</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
            """.format(rmse_rnn), unsafe_allow_html=True)
        with col4:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">MAPE</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
            """.format(mape_rnn), unsafe_allow_html=True)
        
        # =================== Plot SimpleRNN Loss ===================
        st.subheader("📉 SimpleRNN: Loss over Epochs")
        fig_rnn_loss, ax_rnn_loss = plt.subplots(figsize=(10, 6))
        ax_rnn_loss.plot(history_rnn.history['loss'], label='Training Loss', color='blue')
        ax_rnn_loss.plot(history_rnn.history['val_loss'], label='Validation Loss', color='orange')
        ax_rnn_loss.set_title('SimpleRNN Loss over Epochs')
        ax_rnn_loss.set_xlabel('Epoch')
        ax_rnn_loss.set_ylabel('Loss (MSE)')
        ax_rnn_loss.legend()
        ax_rnn_loss.grid(True)
        st.pyplot(fig_rnn_loss)
        plt.close(fig_rnn_loss)
        
        with st.spinner('💾 Saving SimpleRNN results...'):
            results_rnn_df = pd.DataFrame({
                'True Values': y_true.flatten(), 
                'Predictions': y_pred_rnn.flatten()
            })
            results_file_path = os.path.join(output_dir_result, 'rnn_temperature_predictions_vs_real.csv')
            results_rnn_df.to_csv(results_file_path, index=False)
            
            # results_rnn_df.to_csv(os.path.join(output_dir, 'rnn_temperature_predictions_vs_real.csv'), index=False)
            # # results_rnn_df.to_csv('rnn_temperature_predictions_vs_real.csv', index=False)
            
            evaluation_metrics_rnn = pd.DataFrame({
                'RMSE': [rmse_rnn], 
                'MAE': [mae_rnn], 
                'MSE': [mse_rnn], 
                'MAPE': [mape_rnn]
            })
            evaluation_metrics_file_path = os.path.join(output_dir_result, 'rnn_evaluation_metrics.csv')
            evaluation_metrics_rnn.to_csv(evaluation_metrics_file_path, index=False)
            
            # evaluation_metrics_rnn.to_csv(os.path.join(output_dir, 'rnn_evaluation_metrics.csv'), index=False)
            # # evaluation_metrics_rnn.to_csv('rnn_evaluation_metrics.csv', index=False)
            st.success("✅ SimpleRNN results saved as 'rnn_temperature_predictions_vs_real.csv' and 'rnn_evaluation_metrics.csv'.")
        
        st.subheader("📉 SimpleRNN: True Values vs Predictions")
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label='True Values', color='blue')
        plt.plot(y_pred_rnn, label='RNN Predictions', color='red')
        plt.title(f'SimpleRNN Temperature Prediction for {country}')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # =================== Train LSTM Model ===================
        st.header("🔮 LSTM Model Training with Hyperparameter Tuning")
        with st.spinner('⚙️ Starting hyperparameter tuning for LSTM model...'):
            tuner = kt.RandomSearch(
                build_lstm_model,
                objective='val_loss',
                max_trials=10,  # Adjust as needed
                executions_per_trial=1,
                # directory='tuner_dir',
                directory=os.path.join(output_dir_models, 'tuner_dir'),  # Save in subdirectory
                project_name='tempC_prediction'  # Keep the original project name
            )
            
            tuner.search(
                X_train_reshaped, y_train, 
                epochs=20,  # Adjust as needed
                batch_size=32,  # Consistent batch size
                validation_data=(X_test_reshaped, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                verbose=0
            )
            st.success("✅ Hyperparameter tuning for LSTM completed!")
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        st.write("**Best Hyperparameters Found for LSTM:**")
        st.json(best_hps.values)
        
        with st.spinner('🔧 Building the best LSTM model...'):
            best_lstm_model = build_lstm_model_from_best_hyperparameters(best_hps.values)
            # Optionally save the pretrained model
            best_lstm_model.save(os.path.join(output_dir_models, 'pretrained_best_model.keras'), save_format='keras')
            # best_lstm_model.save('pretrained_best_model.keras', save_format='keras')
            st.success("✅ Best LSTM model built and saved as 'pretrained_best_model.keras'.")
        
        with st.spinner('🏋️‍♂️ Training the best LSTM model...'):
            history = best_lstm_model.fit(
                X_train_reshaped, y_train, 
                epochs=50,  # Same as RNN
                batch_size=32,  # Consistent batch size
                validation_data=(X_test_reshaped, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=5)], 
                verbose=0
            )
            best_lstm_model.save(os.path.join(output_dir_models, 'best_model_trained.keras'), save_format='keras')
            # best_lstm_model.save('best_model_trained.keras', save_format='keras')
            st.success("✅ LSTM model training completed and saved as 'best_model_trained.keras'.")
        
        with st.spinner('📊 Evaluating the LSTM model...'):
            test_loss_lstm = best_lstm_model.evaluate(X_test_reshaped, y_test, verbose=0)
            st.write(f"**LSTM Test Loss (MSE):** {test_loss_lstm:.4f}")
            
            y_pred_lstm_scaled = best_lstm_model.predict(X_test_reshaped)
            y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)
            # y_true is already defined earlier
            
            mae_lstm = mean_absolute_error(y_true, y_pred_lstm)
            mse_lstm = mean_squared_error(y_true, y_pred_lstm)
            rmse_lstm = np.sqrt(mse_lstm)
            mape_lstm = mean_absolute_percentage_error(y_true, y_pred_lstm)
        
        # =================== Display LSTM Metrics ===================
        st.subheader("📈 LSTM Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">MAE</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
            """.format(mae_lstm), unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">MSE</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
            """.format(mse_lstm), unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">RMSE</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
            """.format(rmse_lstm), unsafe_allow_html=True)
        with col4:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">MAPE</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
            """.format(mape_lstm), unsafe_allow_html=True)
        
        # =================== Plot LSTM Loss ===================
        st.subheader("📉 LSTM: Loss over Epochs")
        fig_lstm_loss, ax_lstm_loss = plt.subplots(figsize=(10, 6))
        ax_lstm_loss.plot(history.history['loss'], label='Training Loss', color='blue')
        ax_lstm_loss.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        ax_lstm_loss.set_title('LSTM Loss over Epochs')
        ax_lstm_loss.set_xlabel('Epoch')
        ax_lstm_loss.set_ylabel('Loss (MSE)')
        ax_lstm_loss.legend()
        ax_lstm_loss.grid(True)
        st.pyplot(fig_lstm_loss)
        plt.close(fig_lstm_loss)
        
        with st.spinner('💾 Saving LSTM results...'):
            results_df = pd.DataFrame({
                'True Values': y_true.flatten(), 
                'Predictions': y_pred_lstm.flatten()
            })
            results_df.to_csv(os.path.join(output_dir_result, 'temperature_predictions_vs_real.csv'), index=False)
            # results_df.to_csv('temperature_predictions_vs_real.csv', index=False)
            
            evaluation_metrics = pd.DataFrame({
                'RMSE': [rmse_lstm], 
                'MAE': [mae_lstm], 
                'MSE': [mse_lstm], 
                'MAPE': [mape_lstm]
            })
            evaluation_metrics.to_csv(os.path.join(output_dir_result, 'evaluation_metrics.csv'), index=False)
            # evaluation_metrics.to_csv('evaluation_metrics.csv', index=False)
            st.success("✅ LSTM results saved as 'temperature_predictions_vs_real.csv' and 'evaluation_metrics.csv'.")
        
        st.subheader("📉 LSTM: True Values vs Predictions")
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label='True Values', color='blue')
        plt.plot(y_pred_lstm, label='LSTM Predictions', color='green')
        plt.title(f'LSTM Temperature Prediction for {country}')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # =================== Comparison ===================
        st.header("⚖️ Model Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE'],
            '🤖 SimpleRNN': [mae_rnn, mse_rnn, rmse_rnn, mape_rnn],
            '🔮 LSTM': [mae_lstm, mse_lstm, rmse_lstm, mape_lstm]
        })
        
        # Enhanced table with styling
        st.markdown("""
            <style>
            .comparison-table {
                border-collapse: collapse;
                width: 100%;
            }
            .comparison-table th, .comparison-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            .comparison-table th {
                padding-top: 12px;
                padding-bottom: 12px;
                background-color: #4CAF50;
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <table class="comparison-table">
                <tr>
                    <th>Metric</th>
                    <th>🤖 SimpleRNN</th>
                    <th>🔮 LSTM</th>
                </tr>
                <tr>
                    <td>MAE</td>
                    <td>{:.4f}</td>
                    <td>{:.4f}</td>
                </tr>
                <tr>
                    <td>MSE</td>
                    <td>{:.4f}</td>
                    <td>{:.4f}</td>
                </tr>
                <tr>
                    <td>RMSE</td>
                    <td>{:.4f}</td>
                    <td>{:.4f}</td>
                </tr>
                <tr>
                    <td>MAPE</td>
                    <td>{:.4f}</td>
                    <td>{:.4f}</td>
                </tr>
            </table>
        """.format(mae_rnn, mae_lstm, mse_rnn, mse_lstm, rmse_rnn, rmse_lstm, mape_rnn, mape_lstm), unsafe_allow_html=True)
        
        # =================== Download Buttons ===================
        st.header("📥 Download Results")

        # Provide download links for RNN
        st.subheader("🤖 SimpleRNN Results")
        with open(os.path.join(output_dir_result, 'rnn_temperature_predictions_vs_real.csv'), 'rb') as f:
            st.download_button(
                '📂 Download RNN Predictions vs True Values',
                f,
                'rnn_temperature_predictions_vs_real.csv',
                'text/csv'
            )

        with open(os.path.join(output_dir_result, 'rnn_evaluation_metrics.csv'), 'rb') as f:
            st.download_button(
                '📊 Download RNN Evaluation Metrics',
                f,
                'rnn_evaluation_metrics.csv',
                'text/csv'
            )

        # Provide download links for LSTM
        st.subheader("🔮 LSTM Results")
        with open(os.path.join(output_dir_result, 'temperature_predictions_vs_real.csv'), 'rb') as f:
            st.download_button(
                '📂 Download LSTM Predictions vs True Values',
                f,
                'temperature_predictions_vs_real.csv',
                'text/csv'
            )

        with open(os.path.join(output_dir_result, 'evaluation_metrics.csv'), 'rb') as f:
            st.download_button(
                '📊 Download LSTM Evaluation Metrics',
                f,
                'evaluation_metrics.csv',
                'text/csv'
            )

if __name__ == "__main__":
    main()
