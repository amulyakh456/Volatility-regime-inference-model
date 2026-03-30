import yfinance as yf
import pandas as pd

def load_data():
    print("Downloading data...")

    data = yf.download("^GSPC", start="2015-01-01", end="2024-01-01")

    if data.empty:
        print("❌ Data download failed!")
    else:
        print("✅ Data downloaded successfully!")

    print(data.head())

    # Keep only closing price
    data = data[['Close']]

    return data


def add_features(data):
    data['returns'] = data['Close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=10).std()
    
    data = data.dropna()
    return data


if __name__ == "__main__":
    print("Loading data...")
    
    df = load_data()
    print("Raw data:")
    print(df.head())

    df = add_features(df)
    print("Processed data:")
    print(df.head())