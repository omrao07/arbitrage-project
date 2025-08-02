import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

class CalendarSpreadStrategy:
    """
    Calendar Spread Strategy:
    Long near-term futures, short longer-dated ones (or vice versa) depending on curve shape.
    Here, we use ETF proxies for front- and long-dated oil futures.
    """

    def __init__(self, start_date="2022-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.symbols = {
            "front_month": "USO",  # U.S. Oil Fund (front-month proxy)
            "long_month": "DBO"    # Invesco DB Oil Fund (longer-dated proxy)
        }
        self.data = self.fetch_data()

    def fetch_data(self):
        prices = {}
        for label, ticker in self.symbols.items():
            prices[label] = yf.download(ticker, start=self.start_date, end=self.end_date)["Adj Close"]
        df = pd.DataFrame(prices).dropna()
        return df

    def generate_spread(self):
        df = self.data.copy()
        df["spread"] = df["front_month"] - df["long_month"]
        df["spread_z"] = (df["spread"] - df["spread"].rolling(30).mean()) / df["spread"].rolling(30).std()
        return df.dropna()

    def generate_signals(self, z_threshold=0.75):
        df = self.generate_spread()
        df["signal"] = 0
        df.loc[df["spread_z"] > z_threshold, "signal"] = -1  # Spread too wide → expect convergence (short spread)
        df.loc[df["spread_z"] < -z_threshold, "signal"] = 1  # Spread too narrow → expect divergence (long spread)
        return df

    def backtest(self):
        df = self.generate_signals()
        df["returns"] = (df["front_month"].pct_change() - df["long_month"].pct_change()).shift(-1)
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
        df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
        return df

    def latest_signal(self):
        df = self.generate_signals()
        latest = df.iloc[-1]
        signal = latest["signal"]
        if signal == 1:
            return "LONG Calendar Spread (expect widening)"
        elif signal == -1:
            return "SHORT Calendar Spread (expect narrowing)"
        else:
            return "No signal"

# EXAMPLE
if __name__ == "__main__":
    strat = CalendarSpreadStrategy()
    print(strat.latest_signal())