import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import os

# Load params
params = yaml.safe_load(open("params.yaml"))

def perform_eda():
    df = pd.read_csv(params['data']['raw_path'])
    
    # Create directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Exited', data=df)
    plt.title('Class Distribution')
    plt.savefig('plots/class_distribution.png')
    
    # 2. Correlation Matrix
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('plots/correlation.png')
    
    print("EDA completed. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    perform_eda()