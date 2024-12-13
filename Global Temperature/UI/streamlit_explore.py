import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

output_dir_result = './Global Temperature/Result/'
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok= True)


# Set Streamlit page configuration
st.set_page_config(
    page_title="Temperature Change Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Title of the Streamlit App
st.title("Temperature Change Explore Dashboard")

# ------------------------------
# Step 1: Load and Display Dataset
# ------------------------------

st.header("1. Load and Preview Dataset")

file_path = 'data/raw/Environment_Temperature_change_E_All_Data_NOFLAG.csv'

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    return df

df = load_data(file_path)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ------------------------------
# Step 2: Unique Months
# ------------------------------

st.header("2. Unique Months in Dataset")

unique_months = df['Months'].unique()
st.write("Unique Months in the Dataset:")
st.write(unique_months)

# ------------------------------
# Step 3: Analyze DataFrame
# ------------------------------

st.header("3. DataFrame Analysis")

def analyze_dataframe(df):
    # Bước 1: Lấy số lượng hàng và cột
    num_rows, num_columns = df.shape
    print(f"Number of rows: {num_rows}")  # In ra số lượng hàng
    print(f"Number of columns: {num_columns}")  # In ra số lượng cột

    # Bước 2: Tạo DataFrame với thông tin cột: tên cột, kiểu dữ liệu và số giá trị duy nhất
    column_info = df.dtypes.to_frame(name='Data Type').join(df.nunique().to_frame(name='Unique Values'))

    # Bước 3: Phân loại kiểu dữ liệu của các cột
    def classify_column(col_name, data_type, unique_values):
        # Nếu kiểu dữ liệu là object thì là Categorical
        if data_type == 'object':
            return 'Categorical'
        # Nếu kiểu dữ liệu là int64 hoặc float64
        elif data_type in ['int64', 'float64']:
            if unique_values < 0.05 * num_rows:  # Dựa trên số lượng giá trị duy nhất
                return 'Categorical'
            else:
                return 'Numerical'
        else:
            return 'Categorical'

    # Thêm cột phân loại kiểu dữ liệu
    column_info['Category'] = column_info.apply(
        lambda row: classify_column(row.name, row['Data Type'], row['Unique Values']), axis=1
    )

    # In thông tin chi tiết
    print(column_info)

    return num_rows, num_columns, column_info

num_rows, num_columns, column_info = analyze_dataframe(df)

st.write(f"**Number of rows:** {num_rows}")
st.write(f"**Number of columns:** {num_columns}")

st.subheader("Column Information")
st.dataframe(column_info)

# ------------------------------
# Step 4: Reshape Data
# ------------------------------

st.header("4. Data Reshaping")

# Reshaping the data: Converting year columns (Y1961 to Y2023) from wide to long format
reshaped_data = df.melt(
    id_vars=['Area Code', 'Area Code (M49)', 'Area', 'Continent', 'Months Code', 
             'Months', 'Element Code', 'Element', 'Unit'],
    var_name='Year',
    value_name='TempC'
)

st.subheader("Reshaped Data Preview")
st.dataframe(reshaped_data.head())

# Removing the 'Y' character from the 'Year' column and converting it to numeric
reshaped_data['Year'] = reshaped_data['Year'].str.lstrip('Y').astype(int)

st.subheader("Year Column After Cleaning")
st.write(reshaped_data['Year'].head())

# Dropping the specified columns
columns_to_drop = ['Area Code', 'Area Code (M49)', 'Months Code', 'Unit', 'Element Code']
reshaped_data = reshaped_data.drop(columns=columns_to_drop)

st.subheader("Reshaped Data After Dropping Columns")
st.dataframe(reshaped_data.head())

# Cleaning the 'Months' column
reshaped_data['Months'] = reshaped_data['Months'].str.replace('\x96', '-')

st.subheader("Unique Months After Cleaning")
st.write(reshaped_data['Months'].unique())

# ------------------------------
# Step 5: Define Plotting Functions
# ------------------------------

st.header("5. Data Visualization")

# Define all plotting functions without modification
def plot_temperature_trend(data, element='Temperature change', title='Average Temperature Change Trend Over Years'):
    # Filter data for the specified element and calculate the mean temperature change per year
    element_data = data[data['Element'] == element]
    mean_temp_by_year = element_data.groupby('Year')['TempC'].mean()

    # Plotting the trend
    plt.figure(figsize=(12, 6))
    plt.plot(mean_temp_by_year.index, mean_temp_by_year.values, label=f'Average {element}')
    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    return plt

def plot_continent_temperature_trend(data, element='Temperature change', title='Average Temperature Change Trend by Continent Over Years'):

    # Filter data for the specified element and calculate the mean temperature change per year by continent
    element_data = data[data['Element'] == element]
    continent_temp_trend = element_data.groupby(['Year', 'Continent'])['TempC'].mean().unstack()

    # Plotting the trends for each continent
    plt.figure(figsize=(14, 8))
    for continent in continent_temp_trend.columns:
        plt.plot(continent_temp_trend.index, continent_temp_trend[continent], label=continent)

    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.grid(True)
    plt.legend(title='Continent', fontsize=12)
    return plt

def plot_monthly_temperature_trend(data, element='Temperature change', title='Average Monthly Temperature Change Trend'):

    # Filter data for the specified element and calculate the mean temperature change per month
    element_data = data[data['Element'] == element]
    monthly_temp_trend = element_data.groupby('Months')['TempC'].mean()

    # Ensuring months are in proper order
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    monthly_temp_trend = monthly_temp_trend.reindex(month_order)

    # Plotting the trend
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_temp_trend.index, monthly_temp_trend.values, marker='o', linestyle='-', label='Average Monthly Temperature Change')
    plt.title(title, fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(fontsize=12)
    return plt

def plot_quarterly_trend_from_months(data, element='Temperature change', title='Average Quarterly Temperature Change Trend Over Years'):

    # Filter the data for rows corresponding to quarters and the specified element
    quarter_values = ['Dec-Jan-Feb', 'Mar-Apr-May', 'Jun-Jul-Aug', 'Sep-Oct-Nov']
    quarter_data = data[(data['Months'].isin(quarter_values)) & (data['Element'] == element)]

    # Calculate the mean temperature change for each quarter by year
    quarterly_temp_trend = quarter_data.groupby(['Year', 'Months'])['TempC'].mean().unstack()

    # Plot the trends for each quarter
    plt.figure(figsize=(14, 8))
    for quarter in quarterly_temp_trend.columns:
        plt.plot(quarterly_temp_trend.index, quarterly_temp_trend[quarter], label=quarter)

    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.grid(True)
    plt.legend(title='Quarter', fontsize=12)
    return plt

def plot_quarterly_temperature_trend(data, element='Temperature change', title='Average Quarterly Temperature Change Trend Over Years'):

    # Define mapping of months to quarters
    quarter_mapping = {
        'January': 'Q1', 'February': 'Q1', 'March': 'Q2',
        'April': 'Q2', 'May': 'Q2', 'June': 'Q3',
        'July': 'Q3', 'August': 'Q3', 'September': 'Q4',
        'October': 'Q4', 'November': 'Q4', 'December': 'Q1'
    }
    # Add a new column for quarters based on the month
    data['Quarter'] = data['Months'].map(quarter_mapping)

    # Filter data for the specified element and calculate the mean temperature change per year by quarter
    element_data = data[data['Element'] == element]
    quarterly_temp_trend = element_data.groupby(['Year', 'Quarter'])['TempC'].mean().unstack()

    # Plotting the trends for each quarter
    plt.figure(figsize=(14, 8))
    for quarter in quarterly_temp_trend.columns:
        plt.plot(quarterly_temp_trend.index, quarterly_temp_trend[quarter], label=quarter)

    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.grid(True)
    plt.legend(title='Quarter', fontsize=12)
    return plt

def plot_specific_quarter_trend(data, quarter, element='Temperature change', title_prefix='Average Temperature Change Trend for Quarter'):
    # Filter the data for the specified quarter and element
    quarter_data = data[(data['Months'] == quarter) & (data['Element'] == element)]

    # Calculate the mean temperature change for the specified quarter by year
    quarter_trend = quarter_data.groupby('Year')['TempC'].mean()

    # Plot the trend for the specified quarter
    plt.figure(figsize=(10, 6))
    plt.plot(quarter_trend.index, quarter_trend.values, marker='o', linestyle='-', label=quarter)
    plt.title(f"{title_prefix} ({quarter})", fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    return plt

def plot_continent_trend_for_quarter(data, quarter, element='Temperature change', title_prefix='Average Temperature Change by Continent for Quarter'):

    # Filter the data for the specified quarter and element
    quarter_data = data[(data['Months'] == quarter) & (data['Element'] == element)]

    # Calculate the mean temperature change for each continent by year
    continent_trend = quarter_data.groupby(['Year', 'Continent'])['TempC'].mean().unstack()

    # Plot the trends for each continent
    plt.figure(figsize=(14, 8))
    for continent in continent_trend.columns:
        plt.plot(continent_trend.index, continent_trend[continent], marker='o', linestyle='-', label=continent)

    plt.title(f"{title_prefix} ({quarter})", fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.grid(True)
    plt.legend(title='Continent', fontsize=12)
    return plt

def plot_all_countries_trend_for_continent(data, continent, element='Temperature change', title_prefix='Average Temperature Change by Country for Continent'):

    # Filter the data for the specified continent and element
    continent_data = data[(data['Continent'] == continent) & (data['Element'] == element)]

    # Calculate the mean temperature change for each country by year
    country_trend = continent_data.groupby(['Year', 'Area'])['TempC'].mean().unstack()

    # Plot the trends for each country
    plt.figure(figsize=(14, 8))
    for country in country_trend.columns:
        plt.plot(country_trend.index, country_trend[country], marker='', linestyle='-', linewidth=1, label=country)

    plt.title(f"{title_prefix} ({continent})", fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.grid(True)
    plt.legend(title='Country', fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    plt.tight_layout()
    return plt

def plot_top_countries_trend_for_continent(data, continent, top_n=10, element='Temperature change', title_prefix='Average Temperature Change for Top Countries in Continent'):

    # Filter the data for the specified continent and element
    continent_data = data[(data['Continent'] == continent) & (data['Element'] == element)]

    # Calculate the overall average temperature change for each country
    country_avg_temp_change = continent_data.groupby('Area')['TempC'].mean().sort_values(ascending=False).head(top_n)

    # Filter the data for only the top N countries
    top_countries_data = continent_data[continent_data['Area'].isin(country_avg_temp_change.index)]

    # Calculate the mean temperature change for each country by year
    country_trend = top_countries_data.groupby(['Year', 'Area'])['TempC'].mean().unstack()

    # Plot the trends for each country
    plt.figure(figsize=(14, 8))
    for country in country_trend.columns:
        plt.plot(country_trend.index, country_trend[country], marker='o', linestyle='-', label=country)

    plt.title(f"{title_prefix} ({continent}, Top {top_n})", fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature Change (°C)', fontsize=14)
    plt.grid(True)
    plt.legend(title='Country', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return plt

def plot_distribution(data, element='Temperature change', title='Distribution of Temperature Change'):
    """
    Creates a subplot with a histogram and a box plot to visualize the distribution of temperature change.

    Parameters:
    - data (DataFrame): The reshaped data containing 'Element' and 'TempC'.
    - element (str): The type of temperature element to filter (default: 'Temperature change').
    - title (str): Title of the plot.
    """
    # Filter data for the specified element
    element_data = data[data['Element'] == element]['TempC'].dropna()

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Histogram
    axes[0].hist(element_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'{title} - Histogram', fontsize=14)
    axes[0].set_xlabel('Temperature Change (°C)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    
    # Box plot
    axes[1].boxplot(element_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen', color='black'))
    axes[1].set_title(f'{title} - Box Plot', fontsize=14)
    axes[1].set_xlabel('Temperature Change (°C)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_refined_continent_distribution(data, element='Temperature change', title_prefix='Temperature Change by Continent'):
    """
    Creates a refined plot with a box plot and a histogram to visualize the temperature change distribution by continent.

    Parameters:
    - data (DataFrame): The reshaped data containing 'Continent', 'Element', and 'TempC'.
    - element (str): The type of temperature element to filter (default: 'Temperature change').
    - title_prefix (str): Prefix for the title of the plots.
    """
    # Filter data for the specified element
    element_data = data[data['Element'] == element]

    # Create the figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 2]})
    
    # Box plot
    sns.boxplot(
        ax=axes[0], 
        data=element_data, 
        x='TempC', 
        y='Continent', 
        orient='h', 
        palette='coolwarm', 
        showfliers=False
    )
    axes[0].set_title(f'Box Plot of {title_prefix}', fontsize=14)
    axes[0].set_xlabel('Temperature Change (°C)', fontsize=12)
    axes[0].set_ylabel('Continent', fontsize=12)
    
    # Refined Histogram
    sns.histplot(
        ax=axes[1],
        data=element_data,
        x='TempC',
        hue='Continent',
        element='bars',
        stat='density',
        common_norm=False,
        palette='tab10',
        bins=30,
        alpha=0.5
    )
    axes[1].set_title(f'Histogram of {title_prefix}', fontsize=14)
    axes[1].set_xlabel('Temperature Change (°C)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_refined_each_continent_distribution(data, continent, element='Temperature change', title_prefix='Temperature Change for Continent'):

    # Filter data for the specified element and continent
    element_data = data[(data['Element'] == element) & (data['Continent'] == continent)]

    # Create the figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    # Box plot
    sns.boxplot(
        ax=axes[0], 
        data=element_data, 
        x='TempC', 
        orient='h', 
        palette='coolwarm', 
        showfliers=False
    )
    axes[0].set_title(f'Box Plot of {title_prefix} - {continent}', fontsize=14)
    axes[0].set_xlabel('Temperature Change (°C)', fontsize=12)
    
    # Histogram
    sns.histplot(
        ax=axes[1],
        data=element_data,
        x='TempC',
        stat='density',
        common_norm=False,
        palette='tab10',
        bins=30,
        alpha=0.7
    )
    axes[1].set_title(f'Histogram of {title_prefix} - {continent}', fontsize=14)
    axes[1].set_xlabel('Temperature Change (°C)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(data, title='Correlation Heatmap', figsize=(10, 8)):
    """
    Plots a correlation heatmap for the given dataset.

    Parameters:
    - data (DataFrame): The reshaped data containing variables to analyze correlations.
    - title (str): Title of the heatmap.
    - figsize (tuple): Size of the heatmap figure.
    """
    # Calculate the correlation matrix for numerical variables
    correlation_matrix = data.corr(numeric_only=True)

    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(title, fontsize=16)
    return plt

# ------------------------------
# Step 6: Display Plots Sequentially
# ------------------------------

# 6.1 Temperature Trend Over Years
st.subheader("6.1 Average Temperature Change Trend Over Years")
fig1 = plot_temperature_trend(reshaped_data)
st.pyplot(fig1)

# 6.2 Temperature Trend by Continent Over Years
st.subheader("6.2 Average Temperature Change Trend by Continent Over Years")
fig2 = plot_continent_temperature_trend(reshaped_data)
st.pyplot(fig2)

# 6.3 Average Monthly Temperature Change Trend
st.subheader("6.3 Average Monthly Temperature Change Trend")
fig3 = plot_monthly_temperature_trend(reshaped_data)
st.pyplot(fig3)

# 6.4 Average Quarterly Temperature Change Trend From Months
st.subheader("6.4 Average Quarterly Temperature Change Trend Over Years")
fig4 = plot_quarterly_trend_from_months(reshaped_data)
st.pyplot(fig4)

# 6.5 Average Quarterly Temperature Change Trend
st.subheader("6.5 Average Quarterly Temperature Change Trend Over Years")
fig5 = plot_quarterly_temperature_trend(reshaped_data)
st.pyplot(fig5)

# 6.6 Average Temperature Change Trend for Specific Quarter: Dec-Jan-Feb
st.subheader("6.6 Average Temperature Change Trend for Quarter: Dec-Jan-Feb")
fig6 = plot_specific_quarter_trend(reshaped_data, 'Dec-Jan-Feb')
st.pyplot(fig6)

# 6.7 Average Temperature Change by Continent for Quarter: Dec-Jan-Feb
st.subheader("6.7 Average Temperature Change by Continent for Quarter: Dec-Jan-Feb")
fig7 = plot_continent_trend_for_quarter(reshaped_data, 'Dec-Jan-Feb')
st.pyplot(fig7)

# 6.8 Average Temperature Change by Country for Continent: Europe
st.subheader("6.8 Average Temperature Change by Country for Continent: Europe")
fig8 = plot_all_countries_trend_for_continent(reshaped_data, 'Europe')
st.pyplot(fig8)

# 6.9 Average Temperature Change for Top 10 Countries in Asia
st.subheader("6.9 Average Temperature Change for Top 10 Countries in Asia")
fig9 = plot_top_countries_trend_for_continent(reshaped_data, 'Asia', top_n=10)
st.pyplot(fig9)

# 6.10 Distribution of Temperature Change
st.subheader("6.10 Distribution of Temperature Change")
fig10 = plot_distribution(reshaped_data)
st.pyplot(fig10)

# 6.11 Temperature Change by Continent
st.subheader("6.11 Temperature Change by Continent")
fig11 = plot_refined_continent_distribution(reshaped_data)
st.pyplot(fig11)

# 6.12 Temperature Change for Continent: Asia
st.subheader("6.12 Temperature Change for Continent: Asia")
fig12 = plot_refined_each_continent_distribution(reshaped_data, 'Asia')
st.pyplot(fig12)

# 6.13 Correlation Heatmap
st.subheader("6.13 Correlation Heatmap for Reshaped Data")
fig13 = plot_correlation_heatmap(reshaped_data, title="Correlation Heatmap for Reshaped Data")
st.pyplot(fig13)

# ------------------------------
# Step 7: Correlation Analyses
# ------------------------------

st.header("7. Correlation Analyses")

def analyze_and_visualize_quarterly_correlation(df):

    # Step 1: Clean and filter the data
    data_copy = df.copy()
    data_copy['Months'] = data_copy['Months'].str.replace('\x96', '–')

    # Filter data for temperature change and valid months
    valid_months = [
        'January', 'February', 'December', 'Dec–Jan–Feb'
    ]
    data_filtered = data_copy[
        (data_copy['Element'] == 'Temperature change') &
        (data_copy['Months'].isin(valid_months))
    ]

    # Step 2: Identify missing data for years 2022 and 2023
    data_filtered = data_filtered[['Area', 'Months', 'Y2022', 'Y2023']]
    data_missing = data_filtered[
        (data_filtered['Y2022'].isnull()) | (data_filtered['Y2023'].isnull())
    ]
    missing_countries = data_missing['Area'].unique()

    # Step 3: Filter out rows for countries with missing data
    filtered_data = data_filtered[
        ~data_filtered['Area'].isin(missing_countries)
    ]

    # Step 4: Pivot the data to separate months and Q1 values
    grouped_data = filtered_data.groupby(['Area', 'Months']).mean().reset_index()
    pivoted_data = grouped_data.pivot(index='Area', columns='Months', values=['Y2022', 'Y2023']).reset_index()

    # Calculate the average of December 2022, January 2023, February 2023
    pivoted_data['Avg_3_Months'] = (
        pivoted_data[('Y2022', 'December')] +
        pivoted_data[('Y2023', 'January')] +
        pivoted_data[('Y2023', 'February')]
    ) / 3

    # Extract the Q1 value (Dec–Jan–Feb 2023)
    if ('Y2023', 'Dec–Jan–Feb') in pivoted_data.columns:
        pivoted_data['Q1_Value'] = pivoted_data[('Y2023', 'Dec–Jan–Feb')]
    else:
        pivoted_data['Q1_Value'] = np.nan  # Handle missing column

    # Step 5: Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pivoted_data['Avg_3_Months'], pivoted_data['Q1_Value'], alpha=0.7)
    
    # Add a diagonal reference line (y = x)
    min_val = min(pivoted_data['Avg_3_Months'].min(), pivoted_data['Q1_Value'].min())
    max_val = max(pivoted_data['Avg_3_Months'].max(), pivoted_data['Q1_Value'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

    # Set plot labels and title
    plt.xlabel('Average of (December + January + February) / 3')
    plt.ylabel('Q1 Value (Dec–Jan–Feb)')
    plt.title('Scatter Plot of Monthly Average vs Q1 Value for Countries')
    plt.legend()
    plt.grid(True)
    return plt, pivoted_data[['Area', 'Avg_3_Months', 'Q1_Value']]

st.subheader("7.1 Scatter Plot: Average of (December + January + February) vs Q1 Value")
fig14, result = analyze_and_visualize_quarterly_correlation(df)
st.pyplot(fig14)
st.write("**Correlation Results (First 5 Rows):**")
st.dataframe(result.head())

def analyze_and_visualize_quarterly2_correlation(df):
    """
    Analyzes the relationship between the average of three months and the Q2 value (Mar–Apr–May) for a given dataset.
    Produces a scatter plot to visualize the correlation.

    Parameters:
    - df (DataFrame): The raw dataset containing 'Area', 'Months', and relevant columns.

    Returns:
    - A scatter plot visualizing the relationship.
    """
    # Step 1: Clean and filter the data
    data_copy = df.copy()
    data_copy['Months'] = data_copy['Months'].str.replace('\x96', '–')

    # Filter data for temperature change and valid months
    valid_months = [
        'March', 'April', 'May', 'Mar–Apr–May'
    ]
    data_filtered = data_copy[
        (data_copy['Element'] == 'Temperature change') &
        (data_copy['Months'].isin(valid_months))
    ]

    # Step 2: Identify missing data for year 2010
    data_filtered = data_filtered[['Area', 'Months', 'Y2010']]
    data_missing = data_filtered[data_filtered['Y2010'].isnull()]
    missing_countries = data_missing['Area'].unique()

    # Step 3: Filter out rows for countries with missing data
    filtered_data = data_filtered[~data_filtered['Area'].isin(missing_countries)]

    # Step 4: Pivot the data to separate months and Q2 values
    grouped_data = filtered_data.groupby(['Area', 'Months']).mean().reset_index()
    pivoted_data = grouped_data.pivot(index='Area', columns='Months', values='Y2010').reset_index()

    # Check if required columns exist
    if {'March', 'April', 'May', 'Mar–Apr–May'}.issubset(pivoted_data.columns):
        # Calculate the average of March, April, May
        pivoted_data['Avg_3_Months'] = (
            pivoted_data['March'] +
            pivoted_data['April'] +
            pivoted_data['May']
        ) / 3

        # Extract the Q2 value (Mar–Apr–May)
        pivoted_data['Q2_Value'] = pivoted_data['Mar–Apr–May']

        # Step 5: Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(pivoted_data['Avg_3_Months'], pivoted_data['Q2_Value'], alpha=0.7)
        
        # Add a diagonal reference line (y = x)
        min_val = min(pivoted_data['Avg_3_Months'].min(), pivoted_data['Q2_Value'].min())
        max_val = max(pivoted_data['Avg_3_Months'].max(), pivoted_data['Q2_Value'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

        # Set plot labels and title
        plt.xlabel('Average of (March + April + May) / 3')
        plt.ylabel('Q2 Value (Mar–Apr–May)')
        plt.title('Scatter Plot of Monthly Average vs Q2 Value for Countries')
        plt.legend()
        plt.grid(True)
        return plt, pivoted_data[['Area', 'Avg_3_Months', 'Q2_Value']]
    else:
        st.warning("Required columns ('March', 'April', 'May', 'Mar–Apr–May') are missing.")
        return None, None

st.subheader("7.2 Scatter Plot: Average of (March + April + May) vs Q2 Value")
fig15, result2 = analyze_and_visualize_quarterly2_correlation(df)
if fig15:
    st.pyplot(fig15)
    st.write("**Correlation Results (First 5 Rows):**")
    st.dataframe(result2.head())
else:
    st.write("Correlation plot for Q2 could not be generated due to missing data.")

def analyze_and_visualize_quarterly3_correlation(df):
    """
    Analyzes the relationship between the average of three months and the Q3 value (Jun–Jul–Aug) for a given dataset.
    Produces a scatter plot to visualize the correlation.

    Parameters:
    - df (DataFrame): The raw dataset containing 'Area', 'Months', and relevant columns.

    Returns:
    - A scatter plot visualizing the relationship.
    """
    # Step 1: Clean and filter the data
    data_copy = df.copy()
    data_copy['Months'] = data_copy['Months'].str.replace('\x96', '–')

    # Filter data for temperature change and valid months
    valid_months = [
        'June', 'July', 'August', 'Jun–Jul–Aug'
    ]
    data_filtered = data_copy[
        (data_copy['Element'] == 'Temperature change') &
        (data_copy['Months'].isin(valid_months))
    ]

    # Step 2: Identify missing data for year 2010
    data_filtered = data_filtered[['Area', 'Months', 'Y2010']]
    data_missing = data_filtered[data_filtered['Y2010'].isnull()]
    missing_countries = data_missing['Area'].unique()

    # Step 3: Filter out rows for countries with missing data
    filtered_data = data_filtered[~data_filtered['Area'].isin(missing_countries)]

    # Step 4: Pivot the data to separate months and Q3 values
    grouped_data = filtered_data.groupby(['Area', 'Months']).mean().reset_index()
    pivoted_data = grouped_data.pivot(index='Area', columns='Months', values='Y2010').reset_index()

    # Check if required columns exist
    if {'June', 'July', 'August', 'Jun–Jul–Aug'}.issubset(pivoted_data.columns):
        # Calculate the average of June, July, August
        pivoted_data['Avg_3_Months'] = (
            pivoted_data['June'] +
            pivoted_data['July'] +
            pivoted_data['August']
        ) / 3

        # Extract the Q3 value (Jun–Jul–Aug)
        pivoted_data['Q3_Value'] = pivoted_data['Jun–Jul–Aug']

        # Step 5: Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(pivoted_data['Avg_3_Months'], pivoted_data['Q3_Value'], alpha=0.7)
        
        # Add a diagonal reference line (y = x)
        min_val = min(pivoted_data['Avg_3_Months'].min(), pivoted_data['Q3_Value'].min())
        max_val = max(pivoted_data['Avg_3_Months'].max(), pivoted_data['Q3_Value'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

        # Set plot labels and title
        plt.xlabel('Average of (June + July + August) / 3')
        plt.ylabel('Q3 Value (Jun–Jul–Aug)')
        plt.title('Scatter Plot of Monthly Average vs Q3 Value for Countries')
        plt.legend()
        plt.grid(True)
        return plt, pivoted_data[['Area', 'Avg_3_Months', 'Q3_Value']]
    else:
        st.warning("Required columns ('June', 'July', 'August', 'Jun–Jul–Aug') are missing.")
        return None, None

st.subheader("7.3 Scatter Plot: Average of (June + July + August) vs Q3 Value")
fig16, result3 = analyze_and_visualize_quarterly3_correlation(df)
if fig16:
    st.pyplot(fig16)
    st.write("**Correlation Results (First 5 Rows):**")
    st.dataframe(result3.head())
else:
    st.write("Correlation plot for Q3 could not be generated due to missing data.")

def analyze_and_visualize_quarterly4_correlation(df):
    """
    Analyzes the relationship between the average of three months and the Q4 value (Sep–Oct–Nov) for a given dataset.
    Produces a scatter plot to visualize the correlation.

    Parameters:
    - df (DataFrame): The raw dataset containing 'Area', 'Months', and relevant columns.

    Returns:
    - A scatter plot visualizing the relationship.
    """
    # Step 1: Clean and filter the data
    data_copy = df.copy()
    data_copy['Months'] = data_copy['Months'].str.replace('\x96', '–')

    # Filter data for temperature change and valid months
    valid_months = [
        'September', 'October', 'November', 'Sep–Oct–Nov'
    ]
    data_filtered = data_copy[
        (data_copy['Element'] == 'Temperature change') &
        (data_copy['Months'].isin(valid_months))
    ]

    # Step 2: Identify missing data for year 2010
    data_filtered = data_filtered[['Area', 'Months', 'Y2010']]
    data_missing = data_filtered[data_filtered['Y2010'].isnull()]
    missing_countries = data_missing['Area'].unique()

    # Step 3: Filter out rows for countries with missing data
    filtered_data = data_filtered[~data_filtered['Area'].isin(missing_countries)]

    # Step 4: Pivot the data to separate months and Q4 values
    grouped_data = filtered_data.groupby(['Area', 'Months']).mean().reset_index()
    pivoted_data = grouped_data.pivot(index='Area', columns='Months', values='Y2010').reset_index()

    # Check if required columns exist
    if {'September', 'October', 'November', 'Sep–Oct–Nov'}.issubset(pivoted_data.columns):
        # Calculate the average of September, October, November
        pivoted_data['Avg_3_Months'] = (
            pivoted_data['September'] +
            pivoted_data['October'] +
            pivoted_data['November']
        ) / 3

        # Extract the Q4 value (Sep–Oct–Nov)
        pivoted_data['Q4_Value'] = pivoted_data['Sep–Oct–Nov']

        # Step 5: Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(pivoted_data['Avg_3_Months'], pivoted_data['Q4_Value'], alpha=0.7)
        
        # Add a diagonal reference line (y = x)
        min_val = min(pivoted_data['Avg_3_Months'].min(), pivoted_data['Q4_Value'].min())
        max_val = max(pivoted_data['Avg_3_Months'].max(), pivoted_data['Q4_Value'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

        # Set plot labels and title
        plt.xlabel('Average of (September + October + November) / 3')
        plt.ylabel('Q4 Value (Sep–Oct–Nov)')
        plt.title('Scatter Plot of Monthly Average vs Q4 Value for Countries')
        plt.legend()
        plt.grid(True)
        return plt, pivoted_data[['Area', 'Avg_3_Months', 'Q4_Value']]
    else:
        st.warning("Required columns ('September', 'October', 'November', 'Sep–Oct–Nov') are missing.")
        return None, None

st.subheader("7.4 Scatter Plot: Average of (September + October + November) vs Q4 Value")
fig17, result4 = analyze_and_visualize_quarterly4_correlation(df)
if fig17:
    st.pyplot(fig17)
    st.write("**Correlation Results (First 5 Rows):**")
    st.dataframe(result4.head())
else:
    st.write("Correlation plot for Q4 could not be generated due to missing data.")


output_dir = './Global Temperature/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok= True)

st.subheader("8. Handle missing values")

# Display the original data
st.write("Step 1: Display the original dataset for the user to preview.")
data_handle = df.copy()
st.write(data_handle)

# Replace invalid characters in the 'Months' column and display unique values
st.write("Step 2: Replace invalid characters in the 'Months' column and display unique values.")
data_handle['Months'] = data_handle['Months'].str.replace('\x96', '-')
st.write(data_handle['Months'].unique())

# Rename 'Area' column to 'Country' and filter data to keep only 'Temperature change'
st.write("Step 3: Rename 'Area' column to 'Country' and filter data to keep only 'Temperature change'.")
data_handle = data_handle.rename(columns={'Area': 'Country'})
data_handle = data_handle[data_handle['Element'] == 'Temperature change']
data_handle = data_handle.drop(columns=['Area Code', 'Area Code (M49)', 'Months Code', 'Element Code', 'Unit'])

# Filter data by month and display
st.write("Step 4: Filter data to keep only valid months and display the result.")
TempC = data_handle.loc[data_handle.Months.isin([
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
    'Dec-Jan-Feb', 'Mar-Apr-May', 'Jun-Jul-Aug', 'Sep-Oct-Nov'
])]
st.write(TempC)

# Convert data to long format and display
st.write("Step 5: Convert data from wide format to long format and clean the 'Year' column.")
TempC = TempC.melt(id_vars=['Country', 'Continent', 'Months', 'Element'], var_name='Year', value_name='TempC')
TempC['Year'] = TempC['Year'].str[1:].astype('str')
TempC['Year'] = pd.to_numeric(TempC['Year'], errors='coerce')
st.write(TempC)

# Check for missing values in the 'TempC' column
st.write("Step 6: Check for missing values in the 'TempC' column.")
st.write(TempC.isna().sum())

# Step 1: Filter and split the data into two parts: Complete data and Incomplete data
st.write("Step 7: Filter and split the data into two groups: Complete data (no missing) and Incomplete data.")
countries_with_no_null = TempC.groupby('Country')['TempC'].apply(lambda x: x.notnull().all())
countries_no_null = countries_with_no_null[countries_with_no_null].index.tolist()
countries_with_null = countries_with_no_null[~countries_with_no_null].index.tolist()

data_complete = TempC[TempC['Country'].isin(countries_no_null)]
data_incomplete = TempC[TempC['Country'].isin(countries_with_null)]

# Define the mapping of months to quarters
st.write("Step 8: Define the quarters of the year and map months to quarters.")
quarter_mapping = {
    'Dec-Jan-Feb': ['December', 'January', 'February'],
    'Mar-Apr-May': ['March', 'April', 'May'],
    'Jun-Jul-Aug': ['June', 'July', 'August'],
    'Sep-Oct-Nov': ['September', 'October', 'November']
}

# Step 2: Function to handle missing values
st.write("Step 9: Apply the function to handle missing values for each group.")
def handle_missing(group):
    missing = group[group['TempC'].isnull()]['Months']
    if len(missing) > 1:
        return None
    elif len(missing) == 1:
        missing_month = missing.iloc[0]
        if missing_month in quarter_mapping:
            related = quarter_mapping[missing_month]
        else:
            related = [
                months for quarter, months in quarter_mapping.items() if missing_month in months
            ][0]
        
        mean_value = group.loc[group['Months'].isin(related), 'TempC'].mean()
        group.loc[group['Months'] == missing_month, 'TempC'] = mean_value
    return group

# Apply the function to handle missing values
data_incomplete = data_incomplete.groupby(['Country', 'Year']).apply(handle_missing)
data_incomplete = data_incomplete.dropna(subset=['TempC'])

# Step 3: Combine the complete and incomplete data into a final dataset
st.write("Step 10: Combine the processed complete and incomplete datasets into a final dataset.")
final_dataset = pd.concat([data_complete, data_incomplete])
st.write(final_dataset.isna().sum())

# Reset index after combining the data
final_dataset = final_dataset.reset_index(drop=True)
st.write(final_dataset.head(16))  # Display the first 16 rows

# Step 1: Filter out quarter values from 'Months' and keep only valid months
st.write("Step 11: Filter data to keep only valid months and remove quarter values.")
valid_months = [
    'January', 'February', 'March', 'April', 'May', 'June', 
    'July', 'August', 'September', 'October', 'November', 'December'
]
final_dataset = final_dataset[final_dataset['Months'].isin(valid_months)]
st.write(final_dataset)

# Step 2: Convert month names to numbers
st.write("Step 12: Convert month names to numbers for easier processing.")
month_to_number = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 
    'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
    'November': 11, 'December': 12
}
final_dataset['Months'] = final_dataset['Months'].map(month_to_number)
st.write(final_dataset.head(12))

# Create 'Time' column from 'Year' and 'Months'
st.write("Step 13: Create a 'Time' column from 'Year' and 'Months' for date processing.")
final_dataset['Time'] = pd.to_datetime(
    final_dataset[['Year', 'Months']].assign(day=1)  # Add a default day (1) for each month
)
st.write(final_dataset.head(12))

# Display dataset information
st.write(final_dataset.info())

# Sort the data by time ('Time' column)
final_dataset = final_dataset.sort_values(by='Time')
st.write(final_dataset.head())  # Display sorted data

# Save the CSV files and provide download buttons
st.write("Step 14: Save the processed data to CSV files and provide download buttons.")
# data_complete.to_csv('./complete_data.csv', index=False)
# data_incomplete.to_csv('./incomplete_data_handled.csv', index=False)  
# final_dataset.to_csv('./updated_data_with_time.csv', index=False)

data_complete.to_csv(os.path.join(output_dir_result, 'complete_data.csv'), index=False)
data_incomplete.to_csv(os.path.join(output_dir_result, 'incomplete_data_handled.csv'), index=False)
final_dataset.to_csv(os.path.join(output_dir_result, 'updated_data_with_time.csv'), index=False)

# 8.1 Download button for complete data
csv_complete = data_complete.to_csv(index=False)
st.download_button(
    label="Download Complete Data",  # Button for downloading complete data
    data=csv_complete,
    file_name='complete_data.csv',
    mime='text/csv',
)

# 8.2 Download button for handled incomplete data
csv_incomplete = data_incomplete.to_csv(index=False)
st.download_button(
    label="Download Handled Incomplete Data",  # Button for downloading handled incomplete data
    data=csv_incomplete,
    file_name='incomplete_data_handled.csv',
    mime='text/csv',
)

# 8.3 Download button for final dataset (with time)
csv_final = final_dataset.to_csv(index=False)
st.download_button(
    label="Download Final Dataset (with Time)",  # Button for downloading final dataset with time column
    data=csv_final,
    file_name='updated_data_with_time.csv',
    mime='text/csv',
)
