import pandas as pd
import yfinance as yf
from backend.utils.logger import log
from datetime import datetime, timedelta

class MetalsDemandCycle:
    """
    Metals Demand Cycle Strategy:
    - Long metals (copper, aluminum, zinc) during expected industrial growth
    - Avoid or short during global slowdown periods
    """

    def __init__(self, tickers=["CPER", "JJU", "JJS"], lookback_days=90):
        """
        tickers: ETFs for copper, aluminum, zinc
        """
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.data = {}
        self.signals = {}

    def fetch_data(self):
        try:
            end = datetime.today()
            start = end - timedelta(days=self.lookback_days)
            for ticker in self.tickers:
                df = yf.download(ticker, start=start, end=end)
                if not df.empty:
                    self.data[ticker] = df["Adj Close"]
        except Exception as e:
            log(f"[MetalsDemandCycle] Data fetch failed: {e}")

    def compute_signals(self):
        """Signal: Long if 3-month momentum is positive"""
        for ticker, prices in self.data.items():
            try:
                momentum = (prices[-1] - prices[0]) / prices[0]
                self.signals[ticker] = {
                    "3M Momentum (%)": round(momentum * 100, 2),
                    "Signal": "LONG" if momentum > 0.03 else "NEUTRAL"
                }
            except Exception as e:
                log(f"[MetalsDemandCycle] Signal calc failed for {ticker}: {e}")

        return self.signals

    def get_trade_weights(self):
        """Equal weight to all active LONG metals"""
        active = [t for t, s in self.signals.items() if s["Signal"] == "LONG"]
        if not active:
            return {}
        weight = round(1 / len(active), 2)
        return {t: weight for t in active}

# Example usage
if __name__ == "__main__":
    strategy = MetalsDemandCycle()
    strategy.fetch_data()
    signals = strategy.compute_signals()
    print(pd.DataFrame(signals).T)
    print("Trade Weights:", strategy.get_trade_weights())