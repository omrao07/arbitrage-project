import yfinance as yf
import pandas as pd
import numpy as np

class LivestockSpreadTrade:
    """
    Livestock-Hog-Corn Spread Arbitrage:
    Long lean hogs and short corn if hog-to-corn ratio falls below historical mean,
    implying favorable spread for hog producers.
    """

    def __init__(self, hog_ticker="HE=F", corn_ticker="ZC=F", window=60):
        self.hog_ticker = hog_ticker
        self.corn_ticker = corn_ticker
        self.window = window
        self.data = None
        self.signals = None

    def load_data(self):
        """Download lean hog and corn futures price data"""
        hog_data = yf.download(self.hog_ticker, period="6mo", interval="1d")['Adj Close']
        corn_data = yf.download(self.corn_ticker, period="6mo", interval="1d")['Adj Close']
        df = pd.DataFrame({
            'Lean_Hogs': hog_data,
            'Corn': corn_data
        }).dropna()
        df['Hog_Corn_Ratio'] = df['Lean_Hogs'] / df['Corn']
        self.data = df

    def generate_signals(self):
        """Create long/short signals based on hog/corn ratio mean reversion"""
        if self.data is None:
            self.load_data()
        df = self.data.copy()
        df['Ratio_Mean'] = df['Hog_Corn_Ratio'].rolling(self.window).mean()
        df['Ratio_STD'] = df['Hog_Corn_Ratio'].rolling(self.window).std()

        df['Z_Score'] = (df['Hog_Corn_Ratio'] - df['Ratio_Mean']) / df['Ratio_STD']
        df['Signal'] = 0

        # Long spread (Long hogs / Short corn) if spread is cheap
        df.loc[df['Z_Score'] < -1, 'Signal'] = 1
        # Short spread if it's expensive
        df.loc[df['Z_Score'] > 1, 'Signal'] = -1

        df['Return_Hog'] = df['Lean_Hogs'].pct_change()
        df['Return_Corn'] = df['Corn'].pct_change()
        df['Strategy_Return'] = df['Signal'].shift(1) * (df['Return_Hog'] - df['Return_Corn'])

        self.signals = df
        return df

    def latest_signal(self):
        """Return the latest signal output"""
        if self.signals is None:
            self.generate_signals()
        latest = self.signals.iloc[-1]
        return {
            "Signal": latest["Signal"],
            "Z_Score": round(latest["Z_Score"], 2),
            "Hog/Corn Ratio": round(latest["Hog_Corn_Ratio"], 2),
            "Strategy_Return": round(latest["Strategy_Return"], 4)
        }

# Example usage
if __name__ == "__main__":
    strat = LivestockSpreadTrade()
    strat.load_data()
    strat.generate_signals()
    print(strat.latest_signal())