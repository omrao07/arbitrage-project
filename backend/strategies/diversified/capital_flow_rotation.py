import yfinance as yf
import pandas as pd
from datetime import datetime

class CapitalFlowRotation:
    """
    Capital Flow Rotation Strategy:
    Detects rotations between Emerging Markets (EM) and Developed Markets (DM)
    using ETF proxies to capture capital inflows/outflows.
    """

    def __init__(self, start_date="2022-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.symbols = {
            "EM": "EEM",   # iShares MSCI Emerging Markets ETF
            "DM": "VEA"    # Vanguard FTSE Developed Markets ETF
        }
        self.data = self._fetch_data()

    def _fetch_data(self):
        prices = {}
        for label, ticker in self.symbols.items():
            prices[label] = yf.download(ticker, start=self.start_date, end=self.end_date)["Adj Close"]
        df = pd.DataFrame(prices).dropna()
        return df

    def generate_relative_strength(self):
        df = self.data.copy()
        df["EM_returns"] = df["EM"].pct_change()
        df["DM_returns"] = df["DM"].pct_change()
        df["rolling_EM"] = df["EM_returns"].rolling(20).mean()
        df["rolling_DM"] = df["DM_returns"].rolling(20).mean()
        df["rotation_signal"] = df["rolling_EM"] - df["rolling_DM"]
        return df.dropna()

    def generate_signals(self, threshold=0.001):
        df = self.generate_relative_strength()
        df["signal"] = 0
        df.loc[df["rotation_signal"] > threshold, "signal"] = 1   # Favor EM
        df.loc[df["rotation_signal"] < -threshold, "signal"] = -1 # Favor DM
        return df

    def backtest(self):
        df = self.generate_signals()
        df["returns"] = df["EM_returns"] * (df["signal"].shift(1) == 1) + \
                        df["DM_returns"] * (df["signal"].shift(1) == -1)
        df["strategy_returns"] = df["returns"]
        df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
        return df

    def latest_signal(self):
        df = self.generate_signals()
        latest = df.iloc[-1]
        signal = latest["signal"]
        if signal == 1:
            return "LONG Emerging Markets (EM)"
        elif signal == -1:
            return "LONG Developed Markets (DM)"
        else:
            return "No signal"

# EXAMPLE
if __name__ == "__main__":
    strategy = CapitalFlowRotation()
    print("Latest Signal:", strategy.latest_signal())