import pandas as pd
import yfinance as yf
from backend.utils.logger import log

class ShippingCostArbitrage:
    """
    Strategy:
    - Uses changes in the Baltic Dry Index (BDI) to identify arbitrage opportunities in global shipping and related stocks.
    - If BDI surges, go long shipping companies; if it collapses, short them.
    """

    def __init__(self, bdi_symbol="^BDI", related_assets=None, bdi_threshold=0.05):
        self.bdi_symbol = bdi_symbol
        self.related_assets = related_assets or ["ZIM", "SBLK", "DAC"]  # Common shipping stocks
        self.bdi_threshold = bdi_threshold  # % change to trigger action
        self.signals = {}

    def get_bdi_change(self, period="7d"):
        try:
            bdi_data = yf.download(self.bdi_symbol, period=period, interval="1d")["Close"]
            if len(bdi_data) < 2:
                raise ValueError("Insufficient BDI data")
            pct_change = (bdi_data[-1] - bdi_data[0]) / bdi_data[0]
            return pct_change
        except Exception as e:
            log(f"[ShippingCostArbitrage] Error fetching BDI: {e}")
            return None

    def generate_signals(self):
        bdi_change = self.get_bdi_change()

        if bdi_change is None:
            return {}

        direction = "NEUTRAL"
        if bdi_change > self.bdi_threshold:
            direction = "LONG"
        elif bdi_change < -self.bdi_threshold:
            direction = "SHORT"

        for stock in self.related_assets:
            self.signals[stock] = {
                "BDI_Change": round(bdi_change, 4),
                "Signal": direction
            }

        return self.signals

    def latest_signal(self):
        return self.signals

# Example
if __name__ == "__main__":
    strat = ShippingCostArbitrage()
    strat.generate_signals()
    print(strat.latest_signal())