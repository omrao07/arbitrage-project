import numpy as np
import pandas as pd
import yfinance as yf

class ImpliedVolatilityPremium:
    """
    Strategy that captures the implied volatility premium by comparing
    implied volatility (IV) with realized volatility (RV).
    """

    def __init__(self, symbol="SPY", iv_column="IV", lookback=20):
        """
        Parameters:
        - symbol: Underlying asset (default = SPY)
        - iv_column: The name of the column containing implied volatility
        - lookback: Number of days to compute realized volatility
        """
        self.symbol = symbol
        self.iv_column = iv_column
        self.lookback = lookback
        self.data = None
        self.signals = None

    def load_data(self):
        """Download historical data"""
        df = yf.download(self.symbol, period="6mo", interval="1d")
        df['Return'] = df['Adj Close'].pct_change()
        df['RV'] = df['Return'].rolling(self.lookback).std() * np.sqrt(252)

        # Simulate IV (in real case, fetch from options API)
        df[self.iv_column] = df['RV'] + np.random.normal(0.01, 0.005, size=len(df))
        self.data = df.dropna()

    def generate_signals(self):
        """Generate trade signals based on IV - RV spread"""
        if self.data is None:
            self.load_data()

        df = self.data.copy()
        df['IV_Premium'] = df[self.iv_column] - df['RV']

        # If IV significantly > RV → short volatility (mean reversion)
        df['Signal'] = np.where(df['IV_Premium'] > 0.02, -1,
                         np.where(df['IV_Premium'] < -0.01, 1, 0))
        
        self.signals = df[['Close', self.iv_column, 'RV', 'IV_Premium', 'Signal']]
        return self.signals

    def backtest(self):
        """Backtest performance of the IV premium strategy"""
        if self.signals is None:
            self.generate_signals()

        df = self.signals.copy()
        df['PnL'] = df['Signal'].shift(1) * df['Close'].pct_change()
        df.dropna(inplace=True)
        return df[['Close', 'IV_Premium', 'Signal', 'PnL']]

    def latest_signal(self):
        """Return latest trade decision"""
        if self.signals is None:
            self.generate_signals()
        latest = self.signals.iloc[-1]
        return {
            "IV": latest[self.iv_column],
            "RV": latest["RV"],
            "IV_Premium": latest["IV_Premium"],
            "Signal": latest["Signal"]
        }

# Example usage
if __name__ == "__main__":
    strat = ImpliedVolatilityPremium()
    strat.load_data()
    signals = strat.generate_signals()
    results = strat.backtest()
    print(strat.latest_signal())
    print(results.tail())