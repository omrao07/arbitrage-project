import pandas as pd
import yfinance as yf
import numpy as np

class InterestRateDifferentialArbitrage:
    """
    Go long on the currency of the country with higher interest rate
    and short the one with a lower rate. Classic carry trade model.
    Example: Long USD/JPY when US rates > Japan rates.
    """
    def __init__(self, currency_pair="USDJPY=X", high_rate=5.25, low_rate=0.1):
        self.currency_pair = currency_pair
        self.high_rate = high_rate  # e.g., Fed Funds Rate
        self.low_rate = low_rate    # e.g., BoJ Rate
        self.data = None
        self.signals = None

    def load_data(self):
        """Download spot FX price"""
        fx = yf.download(self.currency_pair, period="1y", interval="1d")['Adj Close']
        fx = fx.rename("FX_Price")
        fx = fx.pct_change().dropna()
        self.data = fx.to_frame()

    def generate_signals(self):
        """Signal logic based on static rate differential"""
        if self.data is None:
            self.load_data()

        df = self.data.copy()
        diff = self.high_rate - self.low_rate

        # Signal: long pair if positive rate spread
        df['Signal'] = 1 if diff > 0 else -1
        df['Strategy_Return'] = df['Signal'] * df['FX_Price']

        self.signals = df
        return df

    def backtest(self):
        if self.signals is None:
            self.generate_signals()
        return self.signals

    def latest_signal(self):
        """Return most recent signal info"""
        if self.signals is None:
            self.generate_signals()
        latest = self.signals.iloc[-1]
        return {
            "Currency Pair": self.currency_pair,
            "Signal": latest["Signal"],
            "Return": latest["Strategy_Return"]
        }

# Example usage
if __name__ == "__main__":
    strat = InterestRateDifferentialArbitrage("USDJPY=X", high_rate=5.25, low_rate=0.1)
    strat.load_data()
    strat.generate_signals()
    print(strat.latest_signal())
    print(strat.backtest().tail())