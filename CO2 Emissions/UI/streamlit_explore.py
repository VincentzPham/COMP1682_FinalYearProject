import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from scipy.stats import boxcox

# Set Streamlit page configuration
st.set_page_config(page_title="CO2 Emissions Explore", layout="wide")

# Function Definitions
def load_data(file_path):
    """Loads the dataset and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
        # df['Year'] = df['Year'].dt.year

        st.success(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

def summarize_data(df):
    st.write(df)
    """Summarize the dataset with rows, columns, and column information."""
    num_rows, num_columns = df.shape
    st.write(f"**Number of rows:** {num_rows}")
    st.write(f"**Number of columns:** {num_columns}")
    column_info = df.dtypes.to_frame(name='Data Type').join(df.nunique().to_frame(name='Unique Values'))
    column_info.reset_index(inplace=True)
    column_info.columns = ['Column Name', 'Data Type', 'Unique Values']
    column_info['Category'] = column_info.apply(
        lambda row: classify_column(row['Column Name'], row['Data Type'], row['Unique Values'], num_rows),
        axis=1
    )
    st.write("### Column Information")
    st.dataframe(column_info)
    return num_rows, num_columns, column_info

def classify_column(col_name, data_type, unique_values, num_rows):
    """Classify columns as Categorical or Numerical."""
    if data_type == 'object':
        return 'Categorical'
    elif data_type in ['int64', 'float64']:
        if unique_values < 0.05 * num_rows:
            return 'Categorical'
        else:
            return 'Numerical'
    else:
        return 'Categorical'

def describe_values(df):
    st.write("### Descriptive statitics")
    st.write(df.describe())

def check_missing_values(df):
    """Check and visualize missing values."""
    missing_values = df.isnull().sum()
    st.write("### Missing Values")
    st.write(missing_values)

def plot_mean_emissions_per_year(df):
    """Plots the mean Total CO2 emissions over the years."""
    mean_emissions_per_year = df.groupby('Year')['Total'].mean()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(mean_emissions_per_year.index, mean_emissions_per_year.values)
    ax.set_xlabel('Year', fontsize=6)
    ax.set_ylabel('Mean CO2 Emissions', fontsize=6)
    ax.set_title('Trend of Mean CO2 Emissions Over the Years', fontsize=6)
    ax.grid(True)
    st.pyplot(fig)
    
    st.markdown("""
    Conclusion: Average CO2 emissions have increased steadily over the years, especially since the mid-20th century. This shows that humans are using more and more fossil fuels, affecting the environment.
                """)
    
    
def plot_total_emissions_by_continent(df):
    """Plots total CO2 emissions by continent over time."""
    data_grouped_by_continent_year = df.groupby(['Year', 'Continent'])['Total'].sum().unstack()
    fig, ax = plt.subplots(figsize=(8, 3))
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
        default=["North America"]
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
            
            fig, ax = plt.subplots(figsize=(12, 8))
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
        fig, ax = plt.subplots(figsize=(12, 8))
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
    emissions_by_year = df.groupby('Year')[['Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring', 'Bunker fuels (Not in Total)']].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    emissions_by_year.plot(ax=ax, linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Emissions (in thousands of metric tons)')
    ax.set_title('Trends of Emissions by Source Over the Years')
    ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

def plot_continent_frequency(df):
    # Tạo figure
    fig = plt.figure(figsize=(8, 3)) 

    # Lấy giá trị của cột `Continent` từ DataFrame `df`
    continent_counts = df['Continent'].value_counts()

    # Tạo colormap
    cmap = cm.get_cmap('Blues')  # Chọn colormap 'Blues'
    colors = cmap(continent_counts / continent_counts.max()) # Chuẩn hóa dữ liệu để ánh xạ với colormap
    
    # Sử dụng `plot(kind='bar')` để tạo biểu đồ hình thanh hiển thị tần số của mỗi châu lục với màu sắc tương ứng
    continent_counts.plot(kind='bar', color=colors) 

    # Đặt tiêu đề cho biểu đồ là 'Frequency of Continent'
    plt.title('Frequency of Continent')

    # plt.gca().set_xlabel('Test', rotation='horizontal')
    
    # Hiển thị biểu đồ
    st.pyplot(fig)

def plot_emissions_by_continent_and_source(df):
    """Plots total emissions by source for each continent."""
    emissions_by_continent = df.groupby('Continent')[['Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring', 'Bunker fuels (Not in Total)']].sum()
    fig, ax = plt.subplots(figsize=(8, 6))
    emissions_by_continent.plot(kind='bar', ax=ax)
    ax.set_xlabel('Continent')
    ax.set_ylabel('Total Emissions (in thousands of metric tons)')
    ax.set_title('Total Emissions by Source for Each Continent')
    ax.legend(title='Source', fontsize='small', title_fontsize='medium')
    fig.tight_layout()
    st.pyplot(fig)

def plot_box_and_histograms(df, columns):
    """Plots boxplots and histograms for specified columns."""
    # st.subheader("6. Boxplots and Histograms for Emission Variables")
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
        
        mean_value = df[column].mean()
        axes[1].axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)


# def plot_box_and_histograms(df, columns): //box cox transform
#     """Plots boxplots and histograms for specified columns with Box-Cox Transformation."""
#     for column in columns:
#         st.markdown(f"**{column} Distribution Before and After Box-Cox Transformation**")
        
#         # Loại bỏ các giá trị NA và đảm bảo dữ liệu > 0
#         data = df[column].dropna()
#         data = data[data > 0]  # Box-Cox chỉ áp dụng cho dữ liệu dương
        
#         # Áp dụng Box-Cox Transformation
#         transformed_data, lambda_opt = boxcox(data)
        
#         # Tạo subplot cho Before và After
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
        
#         # Boxplot Before
#         sns.boxplot(data=data, ax=axes[0, 0], color='gray')
#         axes[0, 0].set_title(f'Boxplot of {column} (Original)')
        
#         # Histogram Before
#         axes[0, 1].hist(data, bins=30, alpha=0.7, color='steelblue')
#         axes[0, 1].set_title(f'Histogram of {column} (Original)')
#         mean_value = data.mean()
#         axes[0, 1].axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
#         axes[0, 1].legend()
        
#         # Boxplot After Box-Cox
#         sns.boxplot(data=transformed_data, ax=axes[1, 0], color='lightgreen')
#         axes[1, 0].set_title(f'Boxplot of {column} (Box-Cox Transformed)')
        
#         # Histogram After Box-Cox
#         axes[1, 1].hist(transformed_data, bins=30, alpha=0.7, color='orange')
#         axes[1, 1].set_title(f'Histogram of {column} (Box-Cox Transformed)')
#         mean_transformed = transformed_data.mean()
#         axes[1, 1].axvline(mean_transformed, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_transformed:.2f}')
#         axes[1, 1].legend()
        
#         # Lambda value
#         st.write(f"**Optimal Lambda for {column}: {lambda_opt:.2f}**")
        
#         plt.tight_layout()
#         st.pyplot(fig)


def plot_correlation_heatmap(df):
    """Plots a heatmap of correlations between emission variables."""
    correlation_matrix = df[['Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring', 'Bunker fuels (Not in Total)']].corr()
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    ax.set_title('Correlation Heatmap of CO2 Emission Variables')
    st.pyplot(fig)

def analyze_calculated_vs_reported_totals(df):
    """Analyzes the relationship between calculated and reported totals."""
    df['Calculated Total'] = df['Solid Fuel'] + df['Liquid Fuel'] + df['Gas Fuel'] + df['Cement'] + df['Gas Flaring']
    df['Difference'] = df['Total'] - df['Calculated Total']
    
    correlation = df['Total'].corr(df['Calculated Total'])
    
    # st.subheader("9. Analysis of Calculated vs Reported Totals")
    st.write(f"**Correlation between Reported Total and Calculated Total:** {correlation:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 4))
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
    # DATA_FILE = "data/raw/fossil_fuel_co2_emissions-by-nation_with_continent.csv"
    DATA_FILE = "data/raw/fossil_fuel_co2_emissions_by_nation_with_continent_cleaned.csv"
    df = load_data(DATA_FILE)
    
    if df is not None:
        summarize_data(df)
        
        describe_values(df)
        
        check_missing_values(df)
        
        st.header("1. Mean CO2 Emissions Over the Years")
        plot_mean_emissions_per_year(df)
        
        # st.header("2. Total CO2 Emissions by Continent Over Time")
        # plot_total_emissions_by_continent(df)
        
        # plot_emissions_by_country_in_continent(df)  # Updated function
        
        # st.header("4. Trends of Emissions by Source Over the Years")
        # plot_emission_trends_by_source(df)
        
        st.header("2. Frequency of Continent")
        plot_continent_frequency(df)
        
        st.header("3. Distribution of Fossil Fuels CO2 Emissions")
        # plot_emissions_by_continent_and_source(df)
        #plot_box_and_histograms(df, ['Total', 'Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring', 'Bunker fuels (Not in Total)'])
        plot_box_and_histograms(df, ['Total'])
        
        st.header("4. Correlation Heatmap of Emission Variables")
        plot_correlation_heatmap(df)
        
        st.header("5. Analyze the relationship between Total and 5 fossil fuels")
        analyze_calculated_vs_reported_totals(df)
        
        st.markdown("""
        ---
        **Data Source:** *Ensure that the CSV file is up-to-date and accurately represents the CO2 emissions data.*
        """)
    else:
        st.stop()

if __name__ == "__main__":
    main()
