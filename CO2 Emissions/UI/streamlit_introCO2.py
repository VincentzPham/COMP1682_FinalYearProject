import streamlit as st
import pandas as pd

st.title("Fossil Fuels CO2 Emissions Introduction")

# Correctly load the CSV file
file_path = "data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv"
df = pd.read_csv(file_path)  # Use read_csv to read the file

st.write(df)
