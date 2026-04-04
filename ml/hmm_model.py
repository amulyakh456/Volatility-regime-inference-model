import pandas as pd
import os
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

    # Fix multi-index columns
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data = add_advanced_features(data)

    features = ['returns', 'volatility', 'momentum', 'ma_signal', 'vol_change']
    X = data[features].values

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


# 🧠 FIXED label logic (always unique)
def label_regimes(data):
    stats = {}

    for i in range(3):
        subset = data[data['regime'] == i]
        stats[i] = (
            subset['returns'].mean(),
            subset['volatility'].mean()
        )

    # sort by return (low → high)
    sorted_regimes = sorted(stats.items(), key=lambda x: x[1][0])

    labels = {}
    labels[sorted_regimes[0][0]] = "Crash 🔴"
    labels[sorted_regimes[1][0]] = "Sideways 🟡"
    labels[sorted_regimes[2][0]] = "Bull 🟢"

    return labels


# 📈 Plot regimes
def plot_regimes(data, labels):
    plt.figure(figsize=(12,6))

    for regime in range(3):
        subset = data[data['regime'] == regime]
        plt.scatter(subset.index, subset['Close'], label=labels[regime])

    plt.legend()
    plt.title("Market Regimes")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.show(block=True)


# 🧠 Strategy
def apply_strategy(data, labels):
    positions = []

    for i in range(len(data)):
        probs = [
            data['regime_0_prob'].iloc[i],
            data['regime_1_prob'].iloc[i],
            data['regime_2_prob'].iloc[i]
        ]

        regime = np.argmax(probs)
        confidence = max(probs)

        label = labels[regime]

        if "Bull" in label:
            base = 1
        elif "Crash" in label:
            base = 0
        else:
            base = 0.5

        position = base * confidence

        vol = data['volatility'].iloc[i]
        position = position / (1 + vol * 50)

        positions.append(position)

    data['position'] = positions
    data['strategy_returns'] = data['position'] * data['returns']

    return data


# 📊 Performance
def evaluate_strategy(data):
    data['cum_market'] = (1 + data['returns']).cumprod()
    data['cum_strategy'] = (1 + data['strategy_returns']).cumprod()

    rolling_max = data['cum_strategy'].cummax()
    drawdown = (data['cum_strategy'] - rolling_max) / rolling_max

    max_drawdown = drawdown.min()
    sharpe = data['strategy_returns'].mean() / data['strategy_returns'].std()

    print("\nStrategy Performance:\n")
    print(f"Market Return: {data['cum_market'].iloc[-1]:.2f}x")
    print(f"Strategy Return: {data['cum_strategy'].iloc[-1]:.2f}x")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")


# 📈 Plot performance
def plot_strategy(data):
    plt.figure(figsize=(12,6))

    plt.plot(data.index, data['cum_market'], label="Market")
    plt.plot(data.index, data['cum_strategy'], label="Strategy")

    plt.legend()
    plt.title("Strategy vs Market Performance")
    plt.xlabel("Date")
    plt.ylabel("Growth")

    plt.show(block=True)

import os

def export_regimes(data):

    # Get project root (ml_proj)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Path to data folder
    data_dir = os.path.join(BASE_DIR, "data")

    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "regime_output.csv")

    df_out = pd.DataFrame({
        "date": data.index,
        "bull": data["regime_2_prob"],
        "crash": data["regime_0_prob"],
        "sideways": data["regime_1_prob"]
    })

    df_out = df_out.reset_index(drop=True)

    df_out.to_csv(output_path, index=False)

    print("✅ Regime probabilities exported to:", output_path)
# 🚀 MAIN
if __name__ == "__main__":
    df = train_hmm()

    analyze_regimes(df)

    labels = label_regimes(df)

    latest = df.iloc[-1]

    probs = [
        float(latest['regime_0_prob']),
        float(latest['regime_1_prob']),
        float(latest['regime_2_prob'])
    ]

    print("\nCurrent Market State:\n")

    for i in range(3):
        print(f"{labels[i]}: {probs[i]:.2f}")

    current_regime = np.argmax(probs)

    print(f"\n👉 Current Regime: {labels[current_regime]}")

    print("\n📌 INTERPRETATION:\n")

    if "Bull" in labels[current_regime]:
        print("Market is stable with low volatility → Favor growth / take more exposure.")
    elif "Crash" in labels[current_regime]:
        print("High risk environment → Reduce exposure, protect capital.")
    else:
        print("Uncertain / sideways market → Stay cautious and reduce risk.")

    df = apply_strategy(df, labels)

    evaluate_strategy(df)

    export_regimes(df)
    plot_regimes(df, labels)
    plot_strategy(df)
    