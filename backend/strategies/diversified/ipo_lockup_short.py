import pandas as pd
import yfinance as yf
import datetime

class IPOLockupShort:
    """
    Short stocks around the IPO lock-up expiry date, as insiders are allowed to sell.
    Empirical studies show increased supply leads to short-term negative price drift.
    """

    def __init__(self, ticker, ipo_date, lockup_period_days=180):
        self.ticker = ticker
        self.ipo_date = pd.to_datetime(ipo_date)
        self.lockup_expiry = self.ipo_date + pd.Timedelta(days=lockup_period_days)
        self.data = None
        self.signals = None

    def load_data(self):
        """Load stock price data around lock-up expiry"""
        start = self.lockup_expiry - pd.Timedelta(days=30)
        end = self.lockup_expiry + pd.Timedelta(days=30)
        df = yf.download(self.ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        df = df[['Adj Close']].rename(columns={'Adj Close': 'Price'})
        df['Return'] = df['Price'].pct_change()
        self.data = df

    def generate_signals(self):
        """Generate short signal at lockup expiry"""
        if self.data is None:
            self.load_data()

        df = self.data.copy()
        df['Signal'] = 0
        expiry_str = self.lockup_expiry.strftime('%Y-%m-%d')

        if expiry_str in df.index:
            df.at[expiry_str, 'Signal'] = -1
        else:
            # fallback to closest date
            nearest = df.index.get_indexer([self.lockup_expiry], method='nearest')[0]
            df.iloc[nearest, df.columns.get_loc('Signal')] = -1

        df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
        self.signals = df
        return df

    def latest_signal(self):
        if self.signals is None:
            self.generate_signals()
        latest = self.signals.iloc[-1]
        return {
            "Ticker": self.ticker,
            "Lock-Up Expiry": self.lockup_expiry.strftime('%Y-%m-%d'),
            "Signal": latest["Signal"],
            "Return": latest["Strategy_Return"]
        }

# Example usage
if __name__ == "__main__":
    strat = IPOLockupShort(ticker="ABNB", ipo_date="2020-12-10")  # Airbnb IPO
    strat.load_data()
    strat.generate_signals()
    print(strat.latest_signal())
    print(strat.signals.tail())