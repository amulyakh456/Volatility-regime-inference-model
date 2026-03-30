from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "VRIM Market Regime API Running"

@app.route("/current-regime")
def current_regime():
    data = {
        "bull": 0.62,
        "crash": 0.12,
        "sideways": 0.26
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)