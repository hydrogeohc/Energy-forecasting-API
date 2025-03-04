import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_display_data(file_path):
    """
    Load the dataset and display the first few rows, info and summary statistics
    """

    df = pd.read_csv(file_path)
    print("Dataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """

      # Convert the 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = df['timestamp'].fillna(method='ffill')  # Forward fill for timestamps
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    return df

def engineer_features(df):

    """
    Engineer new features in the dataset
    """
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create interaction features
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df['temperature_bins'] = pd.cut(df['temperature'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    print("New features added:")
    print(df[['hour', 'day_of_week', 'month', 'is_weekend', 'temp_humidity_interaction', 'temperature_bins']].head())
    
    return df

def visualize_data(df):
    """
    Visualize the dataset
    """

   # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Correlation heatmap

    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Energy Consumption Features')
    plt.show()

    # Pairplot of key features
    sns.pairplot(df[['temperature', 'humidity', 'energy_consumption', 'temp_humidity_interaction']])
    plt.suptitle('Pairplot of Key Features vs Energy Consumption', y=1.02)
    plt.show()

    # Distribution of energy consumption
    plt.figure(figsize=(10, 6))
    sns.histplot(df['energy_consumption'], kde=True)
    plt.title('Distribution of Energy Consumption')
    plt.xlabel('Energy Consumption')
    plt.show()

    # Distribution of temperature bins
    plt.figure(figsize=(10, 6))
    sns.countplot(x='temperature_bins', data=df, order=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    plt.title('Distribution of Temperature Bins')
    plt.xlabel('Temperature Bins')
    plt.ylabel('Count')
    plt.show()

    # Energy consumption by day of the week
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='day_of_week', y='energy_consumption', data=df)
    plt.title('Energy Consumption by Day of the Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Energy Consumption')
    plt.show()

    # Energy consumption: Weekday vs Weekend
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_weekend', y='energy_consumption', data=df)
    plt.title('Energy Consumption: Weekday vs Weekend')
    plt.xlabel('Is Weekend (0=Weekday, 1=Weekend)')
    plt.ylabel('Energy Consumption')
    plt.show()


def preprocess_and_explore_data(file_path):
    """
    Preprocess and explore the dataset
    """
    df = load_and_display_data(file_path)
    df = handle_missing_values(df)
    df = engineer_features(df)
    visualize_data(df)
    
    return df

if __name__ == '__main__':
    processed_df = preprocess_and_explore_data('./energy_consumption_data.csv')
    print("\nProcessed DataFrame:")
    print(processed_df.head())

