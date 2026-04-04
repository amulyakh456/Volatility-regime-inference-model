from flask import Flask, jsonify
import pandas as pd
import os
from config import DATA_PATH

app = Flask(__name__)


@app.route("/")
def home():
    return "VRIM Market Regime API Running"


@app.route("/current-regime")
def current_regime():

    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "regime_output.csv not found"}), 404

    try:
        df = pd.read_csv(DATA_PATH)

        if df.empty:
            return jsonify({"error": "regime data file is empty"}), 400

        latest = df.iloc[-1]

        data = {
            "date": latest["date"],
            "bull": float(latest["bull"]),
            "crash": float(latest["crash"]),
            "sideways": float(latest["sideways"]),
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/regime-history")
def regime_history():

    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "regime_output.csv not found"}), 404

    try:
        df = pd.read_csv(DATA_PATH)

        if df.empty:
            return jsonify({"error": "regime data file is empty"}), 400

        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)