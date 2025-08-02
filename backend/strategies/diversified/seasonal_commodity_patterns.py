import pandas as pd
from backend.utils.logger import log
from datetime import datetime

class SeasonalCommodityPatterns:
    """
    Strategy:
    - Exploits known seasonal trends in commodities (e.g., natural gas in winter, corn in summer).
    - Compares current month vs historical average performance to issue signals.
    """

    def __init__(self, seasonality_data_path="backend/data/seasonal_patterns.csv", threshold=0.03):
        self.seasonality_data_path = seasonality_data_path
        self.threshold = threshold  # minimum historical gain % to trigger signal
        self.signals = {}

    def load_seasonality_data(self):
        try:
            self.df = pd.read_csv(self.seasonality_data_path)
            if "month" not in self.df.columns:
                raise ValueError("Missing 'month' column in seasonal_patterns.csv")
        except Exception as e:
            log(f"[SeasonalCommodityPatterns] Error loading data: {e}")
            self.df = pd.DataFrame()

    def compute_signals(self):
        self.signals = {}
        if self.df.empty:
            return self.signals

        current_month = datetime.now().month

        for _, row in self.df.iterrows():
            commodity = row["commodity"]
            avg_return = row.get(f"month_{current_month}", 0)

            if avg_return >= self.threshold:
                signal = "LONG"
            elif avg_return <= -self.threshold:
                signal = "SHORT"
            else:
                signal = "NEUTRAL"

            self.signals[commodity] = {
                "AvgSeasonalReturn": round(avg_return, 4),
                "Signal": signal
            }

        return self.signals

    def latest_signal(self):
        return self.signals

# Example usage
if __name__ == "__main__":
    strat = SeasonalCommodityPatterns()
    strat.load_seasonality_data()
    strat.compute_signals()
    print(strat.latest_signal())