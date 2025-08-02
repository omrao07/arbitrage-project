import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CrackSpreadTrading:
    """
    Crack Spread Arbitrage:
    Long gasoline and heating oil, short crude oil — when refined product margins widen abnormally.
    """

    def __init__(self, start_date="2022-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.symbols = {
            "crude_oil": "CL=F",       # WTI Crude Oil Futures
            "gasoline": "RB=F",        # RBOB Gasoline Futures
            "heating_oil": "HO=F"      # Heating Oil Futures
        }
        self.data = self.download_data()

    def download_data(self):
        df = {}
        for name, symbol in self.symbols.items():
            df[name] = yf.download(symbol, start=self.start_date, end=self.end_date)["Close"]
        return pd.DataFrame(df).dropna()

    def calculate_crack_spread(self):
        df = self.data.copy()
        # Crack Spread = (Gasoline + Heating Oil) - 2 * Crude
        df["crack_spread"] = (df["gasoline"] + df["heating_oil"]) - 2 * df["crude_oil"]
        df["spread_zscore"] = (df["crack_spread"] - df["crack_spread"].rolling(30).mean()) / df["crack_spread"].rolling(30).std()
        return df.dropna()

    def generate_signals(self, z_threshold=1.5):
        df = self.calculate_crack_spread()
        df["signal"] = 0
        df.loc[df["spread_zscore"] > z_threshold, "signal"] = -1  # Overbought crack spread → short
        df.loc[df["spread_zscore"] < -z_threshold, "signal"] = 1  # Undervalued → long
        return df

    def backtest(self):
        df = self.generate_signals()
        df["returns"] = (
            0.5 * df["gasoline"].pct_change().shift(-1) +
            0.5 * df["heating_oil"].pct_change().shift(-1) -
            df["crude_oil"].pct_change().shift(-1)
        )
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
        df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
        return df

    def latest_signal(self):
        df = self.generate_signals()
        latest = df.iloc[-1]
        signal = latest["signal"]
        if signal == 1:
            return "LONG crack spread: Buy Gasoline & Heating Oil, Sell Crude"
        elif signal == -1:
            return "SHORT crack spread: Sell Gasoline & Heating Oil, Buy Crude"
        else:
            return "No actionable signal"

# EXAMPLE USAGE
if __name__ == "__main__":
    strat = CrackSpreadTrading()
    print(strat.latest_signal())