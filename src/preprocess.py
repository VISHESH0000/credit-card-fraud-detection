import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Load the credit card dataset safely."""
    return pd.read_csv(path, on_bad_lines='skip', low_memory=False)  # Skip problematic lines

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    df = df.dropna()
    X = df.drop("Class", axis=1)
    y = df["Class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def split_data(X, y):
    """Split the data into train and test sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)