import pandas as pd
from hmmlearn.hmm import GaussianHMM
from data_loader import load_data, add_features
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def train_hmm():
    data = load_data()
    data = add_features(data)

    X = data[['returns', 'volatility']].values

    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=300)

    # ✅ TRAIN MODEL FIRST
    model.fit(X)

    # THEN predict
    hidden_states = model.predict(X)
    data['regime'] = hidden_states

    # Probabilities
    probs = model.predict_proba(X)
    data['regime_0_prob'] = probs[:, 0]
    data['regime_1_prob'] = probs[:, 1]
    data['regime_2_prob'] = probs[:, 2]

    return data, model, X
def plot_regimes(data):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))

    for regime in range(3):
        subset = data[data['regime'] == regime]
        plt.scatter(subset.index, subset['Close'], label=f'Regime {regime}')

    plt.legend()
    plt.title("Market Regimes")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.show(block=True)   # 🔥 important
    
def analyze_regimes(data):
    print("\nRegime Analysis:\n")

    for i in range(3):
        subset = data[data['regime'] == i]

        avg_return = subset['returns'].mean()
        avg_vol = subset['volatility'].mean()

        print(f"Regime {i}:")
        print(f"  Avg Return: {avg_return:.5f}")
        print(f"  Avg Volatility: {avg_vol:.5f}")
        print()
def get_regime_probabilities(model, X):
    probs = model.predict_proba(X)
    return probs

if __name__ == "__main__":
    df, model, X = train_hmm()

    analyze_regimes(df)   # optional but good

    latest = df.iloc[-1]

    print("\nCurrent Market State:\n")

    probs = [
        float(latest['regime_0_prob'].values[0]),
        float(latest['regime_1_prob'].values[0]),
        float(latest['regime_2_prob'].values[0])
    ]

    regime_names = ["Crash", "Bull", "Sideways"]

    for i in range(3):
        print(f"{regime_names[i]}: {probs[i]:.2f}")

    # 🚨 THIS LINE MUST EXIST
    plot_regimes(df)