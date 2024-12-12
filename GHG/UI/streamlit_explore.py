import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

# Set the page configuration
st.set_page_config(
    page_title="Greenhouse Gas Emissions Explore",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function Definitions

def load_data(file_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'country_or_area': 'Country',
        'continent': 'Continent',
        'year': 'Year',
        'value': 'Value',
        'category': 'Category'
    })
    return df

def summarize_data(df):
    df
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

def check_missing_values(df):
    """Check and visualize missing values."""
    missing_values = df.isnull().sum()
    st.write("### Missing Values")
    st.write(missing_values[missing_values > 0])
    if missing_values.sum() > 0:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=missing_values.index[missing_values > 0], y=missing_values[missing_values > 0], palette='Reds')
        plt.title('Missing Values per Column')
        plt.ylabel('Count')
        plt.xlabel('Columns')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)
    else:
        st.write("No missing values detected.")

def visualize_category_counts(df):
    """Visualize the number of records per category."""
    st.write("### Number of Records per Emission Category")
    category_counts = df['Category'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='Blues', ax=ax)
    ax.set_title('Number of Records per Emission Category')
    ax.set_xlabel('Emission Category')
    ax.set_ylabel('Number of Records')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

def visualize_continent_counts(df):
    """Visualize the number of records per continent."""
    st.write("### Number of Records per Continent")
    continent_counts = df['Continent'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=continent_counts.index, y=continent_counts.values, palette='Greens', ax=ax)
    ax.set_title('Number of Records per Continent')
    ax.set_xlabel('Continent')
    ax.set_ylabel('Number of Records')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

def plot_emissions_distribution(df):
    """Visualize box plot and histogram of emissions by category."""
    st.write("### Emissions Distribution")
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

    ax0 = plt.subplot(gs[0])
    sns.boxplot(x='Value', y='Category', data=df, whis=[0, 100], width=0.6, orient='h', ax=ax0, linewidth=1.5, showmeans=True, palette='Set2')
    ax0.set_title('Box Plot of Emissions by Category')
    ax0.xaxis.grid(True)
    ax0.set_xlim(0, df['Value'].max() * 1.1)

    ax1 = plt.subplot(gs[1])
    sns.histplot(data=df, x='Value', hue='Category', element='step', stat='density', common_norm=False, bins=10, ax=ax1, alpha=0.7)
    ax1.set_title('Histogram of Emissions by Category')
    ax1.set_xlabel('Emission Value')
    ax1.set_ylabel('Density')
    ax1.xaxis.grid(True)
    ax1.set_xlim(0, df['Value'].max() * 1.1)

    plt.tight_layout()
    st.pyplot(fig)

def analyze_specific_emission(df, category):
    """Analyze specific emissions by category."""
    df_filtered = df[df['Category'] == category]
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 5])

    ax0 = plt.subplot(gs[0])
    sns.boxplot(x='Value', y='Category', data=df_filtered, whis=[0, 100], width=0.6, orient='h', ax=ax0, linewidth=1.5, palette='Set3')
    ax0.set_title(f'Box Plot of {category} Emissions')
    ax0.set_xlabel('Emission Value')
    ax0.set_ylabel('')

    ax1 = plt.subplot(gs[1])
    sns.histplot(data=df_filtered, x='Value', hue='Category', element='step', stat='density', common_norm=False, bins=20, ax=ax1, palette='Set3')
    ax1.set_title(f'Histogram of {category} Emissions')
    ax1.set_xlabel('Emission Value')
    ax1.set_ylabel('Density')

    plt.tight_layout()
    st.pyplot(fig)

def analyze_category_distribution(df):
    """Analyze distribution of all emission categories."""
    st.write("### Distribution of Each Emission Category")
    categories = df['Category'].unique()
    for category in categories:
        with st.expander(f"Analyze Category: {category}"):
            analyze_specific_emission(df, category)

def calculate_aggregations(df):
    """Aggregate data by category."""
    st.write("### Aggregations by Category")
    st.write("**Mean Emission Value by Category:**")
    mean_values = df.groupby(['Category'])['Value'].mean().reset_index()
    st.dataframe(mean_values)

    st.write("**Mean, Min, and Max Emission Values by Category:**")
    agg_values = df.groupby(['Category']).agg({'Value': ['mean', 'min', 'max']}).reset_index()
    agg_values.columns = ['Category', 'Mean', 'Min', 'Max']
    st.dataframe(agg_values)

def plot_histogram_log(df):
    """Plot histogram with log scale for wide range values."""
    st.write("### Histogram of Emission Values (Log Scale)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df['Value'], bins=50, color='lightblue', edgecolor='black', ax=ax)
    ax.set_title('Distribution of Emission Values')
    ax.set_xlabel('Emission Value')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    st.pyplot(fig)

def plot_mean_emissions_by_year(df):
    """Plot mean emissions over years."""
    st.write("### Mean Greenhouse Gas Emissions Over Years")
    mean_per_year = df.groupby('Year')['Value'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Year', y='Value', data=mean_per_year, marker='o', ax=ax)
    ax.set_title('Mean Greenhouse Gas Emissions Over Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Emission Value')
    ax.grid(True)
    st.pyplot(fig)

def plot_emissions_by_continent_year(df):
    """Visualize mean emissions by continent over years."""
    st.write("### Mean Greenhouse Gas Emissions by Continents Over Years")
    mean_emissions_per_year_continent = df.groupby(['Continent', 'Year'])['Value'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Year', y='Value', hue='Continent', data=mean_emissions_per_year_continent, marker='o', ax=ax)
    ax.set_title('Mean Greenhouse Gas Emissions by Continents Over Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Emission Value')
    ax.legend(title='Continent', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig)

def analyze_country_data(df, continent):
    """Analyze data for specific continent."""
    st.write(f"### Mean Greenhouse Gas Emissions by Countries in {continent} Over Years")
    continent_data = df[df['Continent'] == continent]
    mean_emissions_per_year_country = continent_data.groupby(['Country', 'Year'])['Value'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Year', y='Value', hue='Country', data=mean_emissions_per_year_country, marker='o', ax=ax)
    ax.set_title(f'Mean Greenhouse Gas Emissions by Countries in {continent} Over Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Emission Value')
    ax.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

def facet_grid_distribution(df):
    """Visualize distributions across continents."""
    st.write("### Emission Value Distributions by Continent")
    g = sns.FacetGrid(df, col="Continent", col_wrap=3, height=4, sharex=False, sharey=False)
    g.map(sns.histplot, "Value", bins=20, kde=True, color="skyblue")
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Emission Value Distributions by Continent")
    st.pyplot(g.fig)

def correlation_heatmap(df):
    """Calculate and plot correlation heatmap."""
    st.write("### Correlation Heatmap of Emission Categories")
    pivot_df = df.pivot_table(index='Year', columns='Category', values='Value', aggfunc='sum')
    correlation_matrix = pivot_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap of Greenhouse Gas Emissions Categories')
    st.pyplot(fig)

def scatter_plot_analysis(df):
    """Plot scatter comparisons between emissions."""
    st.write("### Scatter Plot Analysis Between Emission Categories")
    pivot_df = df.pivot_table(index='Year', columns='Category', values='Value', aggfunc='sum')
    pairs = [
        ('GHG Emissions (Excl. LULUCF)', 'GHG Emissions (Incl. Indirect CO2)'),
        ('PFC Emissions', 'N2O Emissions'),
        ('PFC Emissions', 'SF6 Emissions'),
        ('SF6 Emissions', 'N2O Emissions')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for i, (x, y) in enumerate(pairs):
        sns.scatterplot(x=pivot_df[x], y=pivot_df[y], ax=axes[i], palette='viridis')
        axes[i].set_title(f'{x} vs {y}')
        axes[i].set_xlabel(x)
        axes[i].set_ylabel(y)
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("🌍 Greenhouse Gas Emissions Explore Dashboard")
    st.markdown("""
    This dashboard provides an in-depth analysis of greenhouse gas emissions data.
    It includes data summaries, visualizations, and various analytical insights to understand emission trends across different categories and continents.
    """)

    # Load Data
    file_path = 'data/raw/greenhouse_gas_inventory_data_completed.csv'
    @st.cache_data
    def load_data_cached(file_path):
        return load_data(file_path)
    
    try:
        df = load_data_cached(file_path)
        st.success("Data loaded successfully!")
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return

    st.header("📊 Data Summary")
    summarize_data(df)

    st.header("🔍 Missing Values Analysis")
    check_missing_values(df)

    st.header("📈 Category and Continent Counts")
    visualize_category_counts(df)
    visualize_continent_counts(df)

    st.header("📉 Emissions Distribution")
    plot_emissions_distribution(df)

    st.header("🔬 Category-wise Distribution Analysis")
    analyze_category_distribution(df)

    st.header("📑 Data Aggregations")
    calculate_aggregations(df)

    st.header("📊 Histogram with Log Scale")
    plot_histogram_log(df)

    st.header("📆 Mean Emissions Over Years")
    plot_mean_emissions_by_year(df)

    st.header("🌐 Emissions by Continent Over Years")
    plot_emissions_by_continent_year(df)

    st.header("🏙️ Country-wise Emissions in North America")
    analyze_country_data(df, 'North America')

    st.header("🔲 Facet Grid: Emission Distributions by Continent")
    facet_grid_distribution(df)

    st.header("🗺️ Correlation Heatmap")
    correlation_heatmap(df)

    st.header("🔗 Scatter Plot Analysis")
    scatter_plot_analysis(df)

    st.markdown("""
    ---
    **Data Source:** Your dataset `greenhouse_gas_inventory_data_completed.csv`
    """)

if __name__ == "__main__":
    main()
