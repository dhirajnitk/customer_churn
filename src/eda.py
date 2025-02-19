import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df: pd.DataFrame):
    """
    Basic EDA steps: info, describe, missing values, distribution plots, correlation heatmap.
    
    Parameters:
    df (DataFrame): The input data.
    """
    # Info and summary statistics
    print("Data Info:")
    df.info()
    print("\nSummary Statistics:")
    print(df.describe())

    # Missing values
    print("\nMissing Values:")
    print(df.isna().sum())

    # Distribution of numeric features
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    # Correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='Blues')
    plt.title("Correlation Heatmap")
    plt.show()