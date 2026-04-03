import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from data_loader import load_data, add_features
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 🔥 Advanced feature engineering
def add_advanced_features(data):
    data['momentum'] = data['Close'].pct_change(5)

    data['ma_short'] = data['Close'].rolling(10).mean()
    data['ma_long'] = data['Close'].rolling(30).mean()
    data['ma_signal'] = data['ma_short'] - data['ma_long']

    data['vol_change'] = data['volatility'].pct_change()

    data = data.dropna()
    return data


# 🚀 Train improved HMM
def train_hmm():
    data = load_data()
    data = add_features(data)

    # Clean multi-index columns
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # Add advanced features
    data = add_advanced_features(data)

    features = ['returns', 'volatility', 'momentum', 'ma_signal', 'vol_change']
    X = data[features].values

    # Scale features (important)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=500,
        random_state=42
    )

    model.fit(X_scaled)

    hidden_states = model.predict(X_scaled)
    data['regime'] = hidden_states

    probs = model.predict_proba(X_scaled)
    data['regime_0_prob'] = probs[:, 0]
    data['regime_1_prob'] = probs[:, 1]
    data['regime_2_prob'] = probs[:, 2]

    return data


# 📊 Analyze regimes
def analyze_regimes(data):
    print("\nRegime Analysis:\n")

    for i in range(3):
        subset = data[data['regime'] == i]

        print(f"Regime {i}:")
        print(f"  Avg Return: {subset['returns'].mean():.5f}")
        print(f"  Avg Volatility: {subset['volatility'].mean():.5f}")
        print()


# 🧠 Label regimes automatically
def label_regimes(data):
    stats = {}

    for i in range(3):
        subset = data[data['regime'] == i]
        stats[i] = (
            subset['returns'].mean(),
            subset['volatility'].mean()
        )

    labels = {}

    for k, (ret, vol) in stats.items():
        if ret > 0 and vol < 0.01:
            labels[k] = "Bull 🟢"
        elif ret < 0 and vol > 0.015:
            labels[k] = "Crash 🔴"
        else:
            labels[k] = "Sideways 🟡"

    return labels


# 📈 Plot regimes
def plot_regimes(data, labels):
    plt.figure(figsize=(12,6))

    for regime in range(3):
        subset = data[data['regime'] == regime]
        plt.scatter(subset.index, subset['Close'], label=labels[regime])

    plt.legend()
    plt.title("Market Regimes (Upgraded Model)")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.show(block=True)


# ▶️ MAIN EXECUTION
if __name__ == "__main__":
    df = train_hmm()

    analyze_regimes(df)

    labels = label_regimes(df)

    latest = df.iloc[-1]

    print("\nCurrent Market State:\n")

    probs = [
        float(latest['regime_0_prob']),
        float(latest['regime_1_prob']),
        float(latest['regime_2_prob'])
    ]

    for i in range(3):
        print(f"{labels[i]}: {probs[i]:.2f}")

    plot_regimes(df, labels)