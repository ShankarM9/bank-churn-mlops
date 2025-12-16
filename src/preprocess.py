import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import os

params = yaml.safe_load(open("params.yaml"))

def preprocess_data():
    df = pd.read_csv(params['data']['raw_path'])
    
    # Drop irrelevant columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Encode Categorical Variables
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Geography'] = le.fit_transform(df['Geography'])
    
    # Save processed data
    os.makedirs(os.path.dirname(params['data']['processed_path']), exist_ok=True)
    df.to_csv(params['data']['processed_path'], index=False)
    print(f"Processed data saved to {params['data']['processed_path']}")

if __name__ == "__main__":
    preprocess_data()