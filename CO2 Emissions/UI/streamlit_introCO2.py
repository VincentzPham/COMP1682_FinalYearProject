import streamlit as st
import pandas as pd

# Title and description
st.title("Fossil Fuels CO2 Emissions Explorer")
st.subheader("Introduction")
st.write("""
This app introduces the topic of fossil fuels and their impact on global CO2 emissions. 
The dataset used here contains CO2 emissions from various nations categorized by fuel types such as 
solid, liquid, gas, and cement production. The data spans multiple years and includes information about continents.
""")

# Load the dataset
file_path = "data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv"
try:
    df = pd.read_csv(file_path)
    
    # Dataset overview
    st.subheader("Dataset Overview")
    st.write("This dataset contains information on CO2 emissions from fossil fuel consumption. Below is a preview of the data:")

    # Display dataset shape
    st.write(f"**Number of rows:** {df.shape[0]}")
    st.write(f"**Number of columns:** {df.shape[1]}")
    
    # Display column names
    st.write("**Columns in the dataset:**")
    st.write(", ".join(df.columns))
    
    # Display first few rows of the dataset
    st.dataframe(df.head(10))

except FileNotFoundError:
    st.error("The dataset file was not found. Please make sure the file path is correct.")
