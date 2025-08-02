import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from backend.utils.logger import log

class MultiAssetTrendFollowing:
    """
    Multi-Asset Trend Following Strategy:
    - Long assets exhibiting strong upward trends (positive momentum)
    - Avoid or short assets showing persistent weakness
    - Universe includes equities, bonds, commodities, FX
    """

    def __init__(self, tickers=None, lookback_days=90):
        self.tickers = tickers or {
            "SPY": "US Equities",
            "TLT": "Long-term Bonds",
            "GLD": "Gold",
            "DBC": "Commodities",
            "UUP": "USD Index"
        }
        self.lookback_days = lookback_days
        self.data = {}
        self.signals = {}

    def fetch_data(self):
        end = datetime.today()
        start = end - timedelta(days=self.lookback_days)
        try:
            for ticker in self.tickers:
                df = yf.download(ticker, start=start, end=end)
                if not df.empty:
                    self.data[ticker] = df["Adj Close"]
        except Exception as e:
            log(f"[MultiAssetTrendFollowing] Data fetch failed: {e}")

    def compute_signals(self):
        """
        If asset has > 5% 3-month momentum → LONG
        Else → NEUTRAL
        """
        for ticker, prices in self.data.items():
            try:
                if len(prices) < 2:
                    continue
                momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                self.signals[ticker] = {
                    "3M Momentum (%)": round(momentum * 100, 2),
                    "Signal": "LONG" if momentum > 0.05 else "NEUTRAL"
                }
            except Exception as e:
                log(f"[MultiAssetTrendFollowing] Signal calc failed for {ticker}: {e}")
        return self.signals

    def get_trade_weights(self):
        """Assign equal weight to each 'LONG' asset"""
        active = [t for t, s in self.signals.items() if s["Signal"] == "LONG"]
        if not active:
            return {}
        weight = round(1 / len(active), 2)
        return {t: weight for t in active}

# Example Usage
if __name__ == "__main__":
    strategy = MultiAssetTrendFollowing()
    strategy.fetch_data()
    signals = strategy.compute_signals()
    print(pd.DataFrame(signals).T)
    print("Trade Weights:", strategy.get_trade_weights())