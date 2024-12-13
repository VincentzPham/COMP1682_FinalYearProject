# Harnessing Machine Learning and Deep Learning for Environmental Solutions

**Author:** Pham Hoang Gia Khang

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [How to Run This Project](#how-to-run-this-project)
- [Time Series Analysis](#time-series-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Resources](#resources)

## Introduction
This is my final project focusing on environmental problems. We apply Machine Learning and Deep Learning algorithms to predict global temperature, greenhouse gas emissions, and CO₂ emissions from fossil fuels.

## Datasets
1. **Global Temperature:** Data sourced from [FAO](https://www.fao.org/faostat/en/#data/ET).
2. **Greenhouse Gas:** Data sourced from [Kaggle](https://www.kaggle.com/datasets/unitednations/international-greenhouse-gas-emissions).
3. **Fossil Fuels CO₂ Emissions:** Data sourced from [Datahub](https://datahub.io/core/co2-fossil-by-nation).

In this project, I work with numerical and time series data relevant to environmental metrics.

## How to Run This Project
1. **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
2. **Activate the Virtual Environment:**
    - **On Windows:**
      ```bash
      venv\Scripts\activate
      ```
    - **On macOS/Linux:**
      ```bash
      source venv/bin/activate
      ```
3. **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Application:**
    ```bash
    python MainAppRun.py
    ```

## Time Series Analysis
I implement the following Time Series Analysis approaches:
- **Simple Moving Average (SMA)**
- **Exponential Moving Average (EMA)**
- **Seasonal Decomposition**
- **Augmented Dickey-Fuller Test (ADF Test)**
- **ACF/PACF Plots**

## Machine Learning Models
- **ARIMA:** Autoregressive Integrated Moving Average.
- **SARIMA:** Seasonal ARIMA.
- **Exponential Smoothing:** A method for smoothing time series data.

## Deep Learning Models
- **Recurrent Neural Network (RNN):** A type of neural network suited for sequential data.
- **Long Short-Term Memory (LSTM):** An advanced RNN architecture capable of learning long-term dependencies.

**Resources**

| Path | Description
| :--- | :----------
| [COMP1682_FinalYearProject]() | Main folder.
| &boxvr;&nbsp; [data]() | data folder.
| &boxvr;&nbsp; [CO2 Emissions]() | data folder.
| &boxv;&nbsp; &boxvr;&nbsp; [Explore_Analysis]() | Exploratory Data Analysis - Time Series Analysis process.
| &boxv;&nbsp; &boxvr;&nbsp; [Models]() | The model file when running app, such as (.keras), (.pkl), and file directory running Hyperparameter Tuning.
| &boxv;&nbsp; &boxvr;&nbsp; [Prediction]() | The RNN and LSTM model prediction.
| &boxv;&nbsp; &boxvr;&nbsp; [Result]() | The result when running app, result is (.csv).
| &boxv;&nbsp; &boxvr;&nbsp; [UI]() | The GUI for Expore - Analysis - Train model.
| &boxvr;&nbsp; [GHG]() | data folder.
| &boxv;&nbsp; &boxvr;&nbsp; [Explore_Analysis]() | Exploratory Data Analysis - Time Series Analysis process.
| &boxv;&nbsp; &boxvr;&nbsp; [Models]() | The model file when running app, such as (.keras), (.pkl), and file directory running Hyperparameter Tuning.
| &boxv;&nbsp; &boxvr;&nbsp; [Prediction]() | The RNN and LSTM model prediction.
| &boxv;&nbsp; &boxvr;&nbsp; [Result]() | The result when running app, result is (.csv).
| &boxv;&nbsp; &boxvr;&nbsp; [UI]() | The GUI for Expore - Analysis - Train model.
| &boxvr;&nbsp; [Global Temperature]() | data folder.
| &boxv;&nbsp; &boxvr;&nbsp; [Explore_Analysis]() | Exploratory Data Analysis - Time Series Analysis process.
| &boxv;&nbsp; &boxvr;&nbsp; [Models]() | The model file when running app, such as (.keras), (.pkl), and file directory running Hyperparameter Tuning.
| &boxv;&nbsp; &boxvr;&nbsp; [Prediction]() | The RNN and LSTM model prediction.
| &boxv;&nbsp; &boxvr;&nbsp; [Result]() | The result when running app, result is (.csv).
| &boxv;&nbsp; &boxvr;&nbsp; [UI]() | The GUI for Expore - Analysis - Train model.
| &boxvr;&nbsp; [MainAppRun.py]() | run the test case.



