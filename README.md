# Harnessing Machine Learning and Deep Learning for Environmental Solutions

**Author:** Pham Hoang Gia Khang

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [How to Run This Project](#how-to-run-this-project)
- [Data Preprocessing](#data-preprocessing)
- [Time Series Analysis](#time-series-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Resources](#resources)

## Introduction
This is my final project focusing on environmental problems. We apply Machine Learning and Deep Learning algorithms to predict global temperature, greenhouse gas emissions, and CO₂ emissions from fossil fuels.

**Harnessing Machine Learning and Deep Learning for Environmental Solutions** is a comprehensive project aimed at leveraging advanced computational techniques to forecast key environmental indicators. By employing ML and DL algorithms, this project seeks to provide precise predictions of global temperature trends, greenhouse gas emissions, and CO₂ emissions from fossil fuels. These predictions can aid governments, organizations, and researchers in making informed decisions to combat climate change and promote sustainable practices.

### Objectives

1. **Accurate Prediction:** Utilize ML and DL models to achieve high-accuracy predictions of environmental metrics.
2. **Data-Driven Insights:** Analyze historical data to uncover patterns and trends that can inform future environmental policies.
3. **Scalability:** Develop models that can be scaled and adapted to incorporate additional environmental variables and datasets.
4. **User-Friendly Application:** Create a user interface that allows stakeholders to interact with the models and visualize predictions effortlessly.

### Significance

The integration of ML and DL in environmental science offers the potential to revolutionize how we understand and address climate-related issues. Predictive models can forecast future scenarios based on current and historical data, enabling proactive measures rather than reactive responses. This project not only contributes to the academic field but also provides practical tools for real-world applications in environmental management and policy formulation.

By advancing the accuracy and reliability of environmental predictions, this project supports the global effort to mitigate the adverse effects of climate change, ultimately contributing to a more sustainable and resilient future after.

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
## Data Preprocessing
Data preprocessing is a crucial step to ensure the quality and reliability of the data used for modeling. Below are the preprocessing steps tailored for each dataset.

### 1. Global Temperature
#### Handling Missing Values
In the Global Temperature dataset, some quarterly data points were missing. To address this, I computed the missing quarterly temperature (`x4`) by taking the average of the three corresponding monthly temperatures (`x1`, `x2`, `x3`). The formula used is:

$$
x4 = \frac{x1 + x2 + x3}{3}
$$

**Where:**
- **x1, x2, x3:** Temperatures for the individual months within the quarter.
- **x4:** Average temperature for the quarter.

**Steps Implemented:**
- **Correlation Analysis:** Analyzed the correlation between individual monthly temperatures and the quarterly average. The high correlation indicated that averaging the three months would be a reliable method for imputing missing quarterly values.
- **Imputation:** For each missing quarterly temperature, calculated the mean of the available monthly temperatures (`x1`, `x2`, `x3`) and assigned it to `x4`.

#### Data Normalization
To ensure that all features contribute equally to the model's performance, I normalized the numerical data using **Min-Max Scaling**. This scales the data to a fixed range, typically [-1, 1].

#### Feature Engineering
- **Time Features:** Extracted additional time-based features like month, quarter, and year to help the model capture seasonal patterns.
- **Lag Features:** Created lagged versions of the target variables to incorporate past information into the current prediction.

### 2. Greenhouse Gas
#### Data Normalization
Normalized the numerical data using **MinMaxScaler** to center the data around the mean with a unit standard deviation. This is particularly useful for algorithms that assume normally distributed data.

#### Feature Engineering
- **Time Features:** Partitioned the data by year to capture annual trends.
- **No Quarterly Data:** Since the dataset is partitioned by year, there is no need to create quarterly features.

### 3. Fossil Fuels CO₂ Emissions
#### Data Normalization
Applied **Min-Max Scaling** to normalize the emission values, ensuring all features are on the same scale for effective model training.

#### Feature Engineering
- **Time Features:** Partitioned the data by year to focus on annual emission trends.
- **No Quarterly Data:** Similar to the Greenhouse Gas dataset, there is no quarterly data to process.

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



