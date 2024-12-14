import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="CO2 Emissions Explore", layout="wide")

# Function Definitions
def load_data(file_path):
    """Loads the dataset and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        st.success(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

def plot_mean_emissions_per_year(df):
    """Plots the mean CO2 emissions over the years."""
    mean_emissions_per_year = df.groupby('Year')['Total'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mean_emissions_per_year.index, mean_emissions_per_year.values, linewidth=2, marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean CO2 Emissions')
    ax.set_title('Trend of Mean CO2 Emissions Over the Years')
    ax.grid(True)
    st.pyplot(fig)

def plot_total_emissions_by_continent(df):
    """Plots total CO2 emissions by continent over time."""
    data_grouped_by_continent_year = df.groupby(['Year', 'Continent'])['Total'].sum().unstack()
    fig, ax = plt.subplots(figsize=(12, 8))
    data_grouped_by_continent_year.plot(ax=ax, linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total CO2 Emissions (in thousands of metric tons)')
    ax.set_title('Total CO2 Emissions by Continent Over Time')
    ax.legend(title='Continent', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

def plot_emissions_by_country_in_continent(df):
    """Plots total CO2 emissions by country in selected continents over time."""
    st.subheader("3. Total CO2 Emissions by Country in Selected Continent(s) Over Time")
    
    # Lấy danh sách các châu lục từ dữ liệu
    continents = sorted(df['Continent'].unique())
    continents.insert(0, "All")  # Thêm tùy chọn "All" vào đầu danh sách
    
    # Tạo widget lựa chọn châu lục
    selected_continents = st.multiselect(
        "Select Continent(s) to Display:",
        options=continents,
        default=["All"]
    )
    
    if "All" in selected_continents:
        selected_continents = continents[1:]  # Loại bỏ "All" nếu đã chọn

    if not selected_continents:
        st.warning("Please select at least one continent or choose 'All' to display data.")
        return
    
    # Lọc dữ liệu theo các châu lục đã chọn
    filtered_df = df[df['Continent'].isin(selected_continents)]
    
    if "All" in st.session_state.get('selected_continents', []):
        # Phân tách theo từng châu lục và vẽ biểu đồ cho từng châu lục
        unique_continents = filtered_df['Continent'].unique()
        for continent in unique_continents:
            st.markdown(f"**Emissions for {continent}**")
            continent_data = filtered_df[filtered_df['Continent'] == continent]
            emissions_by_country = continent_data.groupby(['Year', 'Country'])['Total'].sum().unstack()
            
            if emissions_by_country.empty:
                st.warning(f"No data available for continent: {continent}")
                continue
            
            fig, ax = plt.subplots(figsize=(14, 8))
            emissions_by_country.plot(ax=ax, linewidth=2)
            ax.set_xlabel('Year')
            ax.set_ylabel('Total CO2 Emissions (in thousands of metric tons)')
            ax.set_title(f'Total CO2 Emissions by Country in {continent} Over Time')
            ax.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)
    else:
        # Vẽ biểu đồ cho các châu lục đã chọn trong một biểu đồ duy nhất
        fig, ax = plt.subplots(figsize=(14, 8))
        for continent in selected_continents:
            continent_data = filtered_df[filtered_df['Continent'] == continent]
            emissions_by_country = continent_data.groupby(['Year', 'Country'])['Total'].sum().unstack()
            for country in emissions_by_country.columns:
                ax.plot(emissions_by_country.index, emissions_by_country[country], linewidth=1, label=f"{continent} - {country}")
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Total CO2 Emissions (in thousands of metric tons)')
        ax.set_title('Total CO2 Emissions by Country in Selected Continents Over Time')
        ax.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

def plot_emission_trends_by_source(df):
    """Plots trends of emissions by source over the years."""
    emissions_by_year = df.groupby('Year')[['Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring']].sum()
    fig, ax = plt.subplots(figsize=(12, 8))
    emissions_by_year.plot(ax=ax, linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Emissions (in thousands of metric tons)')
    ax.set_title('Trends of Emissions by Source Over the Years')
    ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

def plot_emissions_by_continent_and_source(df):
    """Plots total emissions by source for each continent."""
    emissions_by_continent = df.groupby('Continent')[['Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring', 'Bunker fuels (Not in Total)']].sum()
    fig, ax = plt.subplots(figsize=(14, 8))
    emissions_by_continent.plot(kind='bar', ax=ax)
    ax.set_xlabel('Continent')
    ax.set_ylabel('Total Emissions (in thousands of metric tons)')
    ax.set_title('Total Emissions by Source for Each Continent')
    ax.legend(title='Source')
    fig.tight_layout()
    st.pyplot(fig)

def plot_box_and_histograms(df, columns):
    """Plots boxplots and histograms for specified columns."""
    st.subheader("6. Boxplots and Histograms for Emission Variables")
    for column in columns:
        st.markdown(f"**{column} Distribution**")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot
        sns.boxplot(data=df, x=column, ax=axes[0])
        axes[0].set_title(f'Boxplot of {column}')
        axes[0].set_xlabel('')
        
        # Histogram
        axes[1].hist(df[column].dropna(), bins=30, alpha=0.7, color='steelblue')
        axes[1].set_title(f'Histogram of {column}')
        axes[1].set_xlabel(column)
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)

def descriptive_statistics(df):
    """Returns descriptive statistics for relevant columns."""
    st.subheader("7. Descriptive Statistics")
    stats = df[['Total', 'Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring', 'Bunker fuels (Not in Total)']].describe().transpose()
    st.dataframe(stats.style.format("{:.2f}"))

def plot_correlation_heatmap(df):
    """Plots a heatmap of correlations between emission variables."""
    correlation_matrix = df[['Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring', 'Bunker fuels (Not in Total)']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    ax.set_title('Correlation Heatmap of CO2 Emission Variables')
    st.pyplot(fig)

def analyze_calculated_vs_reported_totals(df):
    """Analyzes the relationship between calculated and reported totals."""
    df['Calculated Total'] = df['Solid Fuel'] + df['Liquid Fuel'] + df['Gas Fuel'] + df['Cement'] + df['Gas Flaring']
    df['Difference'] = df['Total'] - df['Calculated Total']
    
    correlation = df['Total'].corr(df['Calculated Total'])
    
    st.subheader("9. Analysis of Calculated vs Reported Totals")
    st.write(f"**Correlation between Reported Total and Calculated Total:** {correlation:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Calculated Total'], df['Total'], alpha=0.6, color='teal')
    ax.set_title('Calculated Total from 5 Fuels vs Reported Total')
    ax.set_xlabel('Calculated Total (Sum of components)')
    ax.set_ylabel('Reported Total')
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Difference Statistics")
    difference_stats = df['Difference'].describe()
    st.write(difference_stats)

# Main Application
def main():
    st.title("🌍 Fossil Fuels CO2 Emissions Explore Dashboard")
    st.markdown("""
    This dashboard provides a comprehensive analysis of CO2 emissions data across different continents, countries, and emission sources over the years.
    """)
    
    # Load Data
    DATA_FILE = "data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv"
    df = load_data(DATA_FILE)
    
    if df is not None:
        st.header("1. Mean CO2 Emissions Over the Years")
        plot_mean_emissions_per_year(df)
        
        st.header("2. Total CO2 Emissions by Continent Over Time")
        plot_total_emissions_by_continent(df)
        
        plot_emissions_by_country_in_continent(df)  # Updated function
        
        st.header("4. Trends of Emissions by Source Over the Years")
        plot_emission_trends_by_source(df)
        
        st.header("5. Total Emissions by Source for Each Continent")
        plot_emissions_by_continent_and_source(df)
        
        plot_box_and_histograms(df, ['Total', 'Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring', 'Bunker fuels (Not in Total)'])
        
        st.header("7. Descriptive Statistics")
        descriptive_statistics(df)
        
        st.header("8. Correlation Heatmap of Emission Variables")
        plot_correlation_heatmap(df)
        
        analyze_calculated_vs_reported_totals(df)
        
        st.markdown("""
        ---
        **Data Source:** *Ensure that the CSV file is up-to-date and accurately represents the CO2 emissions data.*
        """)
    else:
        st.stop()

if __name__ == "__main__":
    main()
