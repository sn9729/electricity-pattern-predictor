import pandas as pd
import numpy as np
import os
import requests
import zipfile
import io

def download_data(data_dir="data"):
    """Downloads the dataset from UCI repository if not present."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    target_path = os.path.join(data_dir, "household_power_consumption.txt")
    
    if os.path.exists(target_path):
        print(f"Data already exists at {target_path}")
        return target_path

    print("Downloading dataset...")
    os.makedirs(data_dir, exist_ok=True)
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(data_dir)
    print("Download complete.")
    return target_path

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the power consumption dataset."""
    print("Loading data...")
    use_cols = [
        "Date", "Time", "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity", "Sub_metering_1",
        "Sub_metering_2", "Sub_metering_3"
    ]
    
    # Read with low_memory=False to handle mixed types initially, or specify dtypes
    # The original script used a specific dtype map, let's replicate that logic but safer
    df = pd.read_csv(filepath, sep=";", usecols=use_cols, na_values="?", low_memory=False)
    df.dropna(inplace=True)

    # Convert columns to float
    float_cols = [c for c in use_cols if c not in ["Date", "Time"]]
    for col in float_cols:
        df[col] = df[col].astype("float32")

    # Datetime index
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df.set_index("DateTime", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)
    df.sort_index(inplace=True)

    return df

def create_features(df):
    """Creates time-based and lag features."""
    print("Feature engineering...")
    df = df.copy()
    df["hour"] = df.index.hour.astype("int8")
    df["day_of_week"] = df.index.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    
    # Lag features
    df["lag1"] = df["Global_active_power"].shift(1).astype("float32")
    df["lag2"] = df["Global_active_power"].shift(2).astype("float32")
    
    df.dropna(inplace=True)
    return df

def get_train_test_split(df, target_col="Global_active_power", split_ratio=0.8):
    """Splits data into train and test sets based on time."""
    # Exclude target and Global_intensity (to prevent leakage)
    drop_cols = [target_col, "Global_intensity"]
    features = [c for c in df.columns if c not in drop_cols]
    
    print(f"Training with features: {features}")
    
    X = df[features]
    y = df[target_col]
    
    cutoff = int(len(X) * split_ratio)
    X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
    X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]
    
    return X_train, y_train, X_test, y_test
