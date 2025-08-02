import yfinance as yf
import pandas as pd
from backend.utils.logger import log

class MeanReversion:
    """
    Long stocks that are extremely oversold and expected to revert to the mean.
    Uses z-score of price relative to rolling mean as entry signal.
    """

    def __init__(self, ticker="AAPL", lookback=20, z_threshold=-2):
        self.ticker = ticker
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.data = None
        self.signal = None

    def fetch_data(self):
        """Download historical data from Yahoo Finance"""
        try:
            self.data = yf.download(self.ticker, period="6mo", interval="1d")
            self.data = self.data[["Close"]].dropna()
        except Exception as e:
            log(f"[MeanReversion] Error fetching data: {e}")
            self.data = pd.DataFrame()

    def calculate_zscore(self):
        """Calculate z-score of the price vs rolling mean"""
        if self.data is None or self.data.empty:
            self.fetch_data()

        df = self.data.copy()
        df["Rolling_Mean"] = df["Close"].rolling(window=self.lookback).mean()
        df["Rolling_Std"] = df["Close"].rolling(window=self.lookback).std()
        df["Z_Score"] = (df["Close"] - df["Rolling_Mean"]) / df["Rolling_Std"]
        self.data = df

    def generate_signal(self):
        """Generate a buy signal if Z-score is below threshold"""
        self.calculate_zscore()
        latest = self.data.dropna().iloc[-1]
        z = latest["Z_Score"]

        self.signal = {
            "Ticker": self.ticker,
            "Latest Z-Score": round(z, 2),
            "Signal": "LONG" if z <= self.z_threshold else "NO_ACTION"
        }
        return self.signal

    def get_trade_recommendation(self):
        if self.signal is None:
            self.generate_signal()

        if self.signal["Signal"] == "LONG":
            return {self.ticker: 0.05}  # Allocate 5% of capital
        else:
            return {}

# Example usage
if __name__ == "__main__":
    strategy = MeanReversion(ticker="MSFT")
    print(strategy.generate_signal())
    print(strategy.get_trade_recommendation())