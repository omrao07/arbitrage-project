import pandas as pd
import numpy as np
from datetime import datetime
from backend.utils.logger import log

class OilInventorySurpriseAlpha:
    """
    Strategy:
    - Trades oil futures based on EIA inventory surprises.
    - If reported inventory is significantly below estimates (surprise draw), go LONG oil.
    - If above estimates (surprise build), go SHORT oil.
    """

    def __init__(self, inventory_csv_path="backend/data/macro_data/eia_oil_inventory.csv", threshold=1.5):
        self.inventory_csv_path = inventory_csv_path
        self.threshold = threshold
        self.signals = {}

    def load_data(self):
        try:
            df = pd.read_csv(self.inventory_csv_path, parse_dates=["Date"])
            df.dropna(subset=["Actual", "Forecast"], inplace=True)
            df["Surprise"] = df["Actual"] - df["Forecast"]
            self.data = df
        except Exception as e:
            log(f"[OilInventorySurpriseAlpha] Failed to load inventory data: {e}")
            self.data = pd.DataFrame()

    def compute_signals(self):
        self.signals = {}
        for _, row in self.data.iterrows():
            date = row["Date"]
            surprise = row["Surprise"]
            if surprise < -self.threshold:
                signal = "LONG"
            elif surprise > self.threshold:
                signal = "SHORT"
            else:
                signal = "NEUTRAL"
            self.signals[date.strftime("%Y-%m-%d")] = {
                "Surprise": round(surprise, 2),
                "Signal": signal
            }
        return self.signals

    def latest_signal(self):
        if not self.signals:
            return None
        latest_date = max(self.signals.keys())
        return {latest_date: self.signals[latest_date]}

# Example usage
if __name__ == "__main__":
    strat = OilInventorySurpriseAlpha()
    strat.load_data()
    signals = strat.compute_signals()
    print("Latest signal:", strat.latest_signal())