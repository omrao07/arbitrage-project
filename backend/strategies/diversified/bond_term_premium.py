import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class BondTermPremiumCompression:
    """
    Strategy:
    Long long-duration bonds when the yield curve is flattening (term premium compressing).
    Short when steepening.
    """

    def __init__(self, start_date="2022-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.symbols = {
            "2y": "^IRX",     # 13-week Treasury (proxy for short-term)
            "10y": "^TNX",    # 10-year Treasury yield
            "bond_etf": "TLT" # Long-duration bond ETF
        }
        self.data = self.download_data()

    def download_data(self):
        df = {}
        for label, ticker in self.symbols.items():
            df[label] = yf.download(ticker, start=self.start_date, end=self.end_date)["Close"]
        return pd.DataFrame(df).dropna()

    def calculate_term_spread(self):
        df = self.data.copy()
        df["term_spread"] = df["10y"] - df["2y"]
        df["spread_change"] = df["term_spread"].diff()
        df["spread_zscore"] = (df["spread_change"] - df["spread_change"].rolling(30).mean()) / df["spread_change"].rolling(30).std()
        return df.dropna()

    def generate_signals(self, z_threshold=0.5):
        df = self.calculate_term_spread()
        df["signal"] = 0
        df.loc[df["spread_zscore"] < -z_threshold, "signal"] = 1   # Flattening → long TLT
        df.loc[df["spread_zscore"] > z_threshold, "signal"] = -1  # Steepening → short TLT
        return df

    def backtest(self):
        df = self.generate_signals()
        df["returns"] = df["bond_etf"].pct_change().shift(-1)
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
        df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
        return df

    def latest_signal(self):
        df = self.generate_signals()
        latest = df.iloc[-1]
        signal = latest["signal"]
        if signal == 1:
            return "LONG TLT: Yield curve flattening expected"
        elif signal == -1:
            return "SHORT TLT: Yield curve steepening expected"
        else:
            return "No actionable signal"

# EXAMPLE
if __name__ == "__main__":
    strat = BondTermPremiumCompression()
    print(strat.latest_signal())