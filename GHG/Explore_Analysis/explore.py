import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

def load_data(file_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    df
    df = df.rename(columns={'country_or_area': 'Country',
                            'continent': 'Continent',
                            'year': 'Year',
                            'value': 'Value',
                            'category': 'Category'})
    return df

def summarize_data(df):
    """Summarize the dataset with rows, columns, and column information."""
    num_rows, num_columns = df.shape
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_columns}")
    column_info = df.dtypes.to_frame(name='Data Type').join(df.nunique().to_frame(name='Unique Values'))
    column_info.reset_index(inplace=True)
    column_info.columns = ['Column Name', 'Data Type', 'Unique Values']
    column_info['Category'] = column_info.apply(
        lambda row: classify_column(row['Column Name'], row['Data Type'], row['Unique Values'], num_rows),
        axis=1
    )
    print(column_info)
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
    print("Missing Values:")
    print(missing_values)
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        missing_values.plot(kind='bar', figsize=(8, 5), color='red')
        plt.title('Missing Values per Column')
        plt.ylabel('Count')
        plt.xlabel('Columns')
        plt.show()

def visualize_category_counts(df):
    """Visualize the number of records per category."""
    category_counts = df['Category'].value_counts()
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Records per Emission Category')
    plt.xlabel('Emission Category')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def visualize_continent_counts(df):
    """Visualize the number of records per continent."""
    continent_counts = df['Continent'].value_counts()
    plt.figure(figsize=(8, 5))
    continent_counts.plot(kind='bar', color='lightgreen')
    plt.title('Number of Records per Continent')
    plt.xlabel('Continent')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def plot_emissions_distribution(df):
    """Visualize box plot and histogram of emissions by category."""
    plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

    ax0 = plt.subplot(gs[0])
    sns.boxplot(x='Value', y='Category', data=df, whis=[0, 100], width=0.6, orient='h', ax=ax0, linewidth=1.5, showmeans=True)
    ax0.set_title('Box Plot of Emissions by Continent')
    ax0.xaxis.grid(True)
    ax0.set_xlim(0, df['Value'].max() * 1.1)

    ax1 = plt.subplot(gs[1])
    sns.histplot(data=df, x='Value', hue='Category', element='step', stat='density', common_norm=False, bins=10, ax=ax1, alpha=0.7)
    ax1.set_title('Histogram of Emissions by Continent')
    ax1.set_xlabel('Emission Value')
    ax1.set_ylabel('Density')
    ax1.xaxis.grid(True)
    ax1.set_xlim(0, df['Value'].max() * 1.1)

    plt.tight_layout()
    plt.show()


def analyze_specific_emission(df, category):
    """Analyze specific emissions by category."""
    df_filtered = df[df['Category'] == category]
    plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 5])

    ax0 = plt.subplot(gs[0])
    sns.boxplot(x='Value', y='Category', data=df_filtered, whis=[0, 100], width=0.6, orient='h', ax=ax0, linewidth=1.5)
    ax0.set_title(f'Box Plot of {category} Emissions')
    ax0.set_xlabel('Emission Value')
    ax0.set_ylabel('')

    ax1 = plt.subplot(gs[1])
    sns.histplot(data=df_filtered, x='Value', hue='Category', element='step', stat='density', common_norm=False, bins=20, ax=ax1)
    ax1.set_title(f'Histogram of {category} Emissions')
    ax1.set_xlabel('Emission Value')
    ax1.set_ylabel('Density')

    plt.tight_layout()
    plt.show()

def analyze_category_distribution(df):
    """Analyze distribution of all emission categories."""
    categories = df['Category'].unique()
    for category in categories:
        print(f"Analyzing Category: {category}")
        analyze_specific_emission(df, category)

def calculate_aggregations(df):
    """Aggregate data by category."""
    print(df.groupby(['Category'])['Value'].mean())
    print(df.groupby(['Category']).agg({'Value': ['mean', 'min', 'max']}))

def plot_histogram_log(df):
    """Plot histogram with log scale for wide range values."""
    plt.figure(figsize=(12, 6))
    plt.hist(df['Value'], bins=50, color='lightblue', edgecolor='black')
    plt.title('Distribution of Emission Values')
    plt.xlabel('Emission Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.show()

def plot_mean_emissions_by_year(df):
    """Plot mean emissions over years."""
    mean_per_year = df.groupby('Year')['Value'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(mean_per_year['Year'], mean_per_year['Value'], marker='o')
    plt.title('Mean Greenhouse Gas Emissions Over Years')
    plt.xlabel('Year')
    plt.ylabel('Mean Emission Value')
    plt.grid(True)
    plt.show()

def plot_emissions_by_continent_year(df):
    """Visualize mean emissions by continent over years."""
    mean_emissions_per_year_continent = df.groupby(['Continent', 'Year'])['Value'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='Value', hue='Continent', data=mean_emissions_per_year_continent, marker='o')
    plt.title('Mean Greenhouse Gas Emissions by Continents Over Years')
    plt.xlabel('Year')
    plt.ylabel('Mean Emission Value')
    plt.legend(title='Continent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

def analyze_country_data(df, continent):
    """Analyze data for specific continent."""
    continent_data = df[df['Continent'] == continent]
    mean_emissions_per_year_country = continent_data.groupby(['Country', 'Year'])['Value'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='Value', hue='Country', data=mean_emissions_per_year_country, marker='o')
    plt.title(f'Mean Greenhouse Gas Emissions by Countries in {continent} Over Years')
    plt.xlabel('Year')
    plt.ylabel('Mean Emission Value')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def facet_grid_distribution(df):
    """Visualize distributions across continents."""
    g = sns.FacetGrid(df, col="Continent", col_wrap=3, height=4, sharex=False, sharey=False)
    g.map(sns.histplot, "Value", bins=20, kde=True, color="skyblue")
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Emission Value Distributions by Continent")
    plt.show()

def correlation_heatmap(df):
    """Calculate and plot correlation heatmap."""
    pivot_df = df.pivot_table(index='Year', columns='Category', values='Value', aggfunc='sum')
    correlation_matrix = pivot_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Greenhouse Gas Emissions Categories')
    plt.show()

def scatter_plot_analysis(df):
    """Plot scatter comparisons between emissions."""
    pivot_df = df.pivot_table(index='Year', columns='Category', values='Value', aggfunc='sum')
    plt.figure(figsize=(15, 8))
    pairs = [
        ('GHG Emissions (Excl. LULUCF)', 'GHG Emissions (Incl. Indirect CO2)'),
        ('PFC Emissions', 'N2O Emissions'),
        ('PFC Emissions', 'SF6 Emissions'),
        ('SF6 Emissions', 'N2O Emissions')
    ]
    for i, (x, y) in enumerate(pairs):
        plt.subplot(2, 2, i + 1)
        plt.scatter(pivot_df[x], pivot_df[y])
        plt.title(f'{x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
    plt.tight_layout()
    plt.show()

def main():
    file_path = '../../data/raw/greenhouse_gas_inventory_data_completed.csv'
    df = load_data(file_path)
    summarize_data(df)
    check_missing_values(df)  # Analyze missing values
    visualize_category_counts(df)
    visualize_continent_counts(df)
    plot_emissions_distribution(df)
    analyze_category_distribution(df)  # Analyze each category
    calculate_aggregations(df)
    plot_histogram_log(df)
    plot_mean_emissions_by_year(df)
    plot_emissions_by_continent_year(df)
    analyze_country_data(df, 'North America')  # Specific continent analysis
    facet_grid_distribution(df)  # Distribution by continent
    correlation_heatmap(df)  # Heatmap of correlations
    scatter_plot_analysis(df)  # Scatter plot analysis

# Execute the updated main function
if __name__ == "__main__":
    main()
