import requests
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

API_URL = "http://127.0.0.1:5000/regime-history"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "regime_output.csv"
REGIME_COLORS = {
    "bull": "#00b050",
    "crash": "#ff4c4c",
    "sideways": "#ffab00",
}


def fetch_data():
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            raise ValueError(data["error"])

        df = pd.DataFrame(data)
    except Exception:
        df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def plot_dashboard(df):
    fig = px.area(
        df,
        x="date",
        y=["bull", "crash", "sideways"],
        title="Market Regime Probabilities",
        labels={
            "value": "Probability",
            "variable": "Regime",
            "date": "Date",
        },
        color_discrete_map=REGIME_COLORS,
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"count": 3, "label": "3y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ]
            ),
        ),
        yaxis=dict(title="Probability", tickformat=".0%"),
        legend=dict(title="Regime", orientation="h", y=1.08, x=0.01),
        margin=dict(l=50, r=30, t=80, b=40),
    )

    fig.show()


if __name__ == "__main__":
    df = fetch_data()
    plot_dashboard(df)
