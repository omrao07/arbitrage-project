import pandas as pd
import numpy as np

class EarningsDriftPostReaction:
    """
    Strategy: Post-Earnings Announcement Drift (PEAD)
    - Stocks with large positive earnings surprises tend to drift upward in the following weeks.
    - Stocks with large negative surprises tend to drift downward.
    """

    def __init__(self, earnings_df: pd.DataFrame, price_df: pd.DataFrame):
        """
        Parameters:
        - earnings_df: DataFrame with ['Date', 'Ticker', 'Surprise']
        - price_df: DataFrame with ['Date', 'Ticker', 'Close']
        """
        self.earnings_df = earnings_df
        self.price_df = price_df

    def generate_signals(self, surprise_threshold=0.05):
        """
        Returns:
        - A DataFrame with columns: ['Date', 'Ticker', 'Surprise', 'Signal']
        Signal = +1 if Surprise > threshold (long drift)
        Signal = -1 if Surprise < -threshold (short drift)
        """
        signals = self.earnings_df.copy()
        signals['Signal'] = 0
        signals.loc[signals['Surprise'] > surprise_threshold, 'Signal'] = 1
        signals.loc[signals['Surprise'] < -surprise_threshold, 'Signal'] = -1
        return signals[['Date', 'Ticker', 'Surprise', 'Signal']]

    def simulate_drift_returns(self, holding_days=10):
        """
        Simulates returns following earnings surprise for a holding period.
        Assumes buying the stock at close on earnings day and selling after 'holding_days'.
        """
        signals = self.generate_signals()
        signals['DriftReturn'] = 0.0

        for i, row in signals.iterrows():
            ticker = row['Ticker']
            start_date = pd.to_datetime(row['Date'])

            ticker_prices = self.price_df[self.price_df['Ticker'] == ticker].copy()
            ticker_prices['Date'] = pd.to_datetime(ticker_prices['Date'])
            ticker_prices = ticker_prices.sort_values('Date')

            entry_row = ticker_prices[ticker_prices['Date'] == start_date]
            if entry_row.empty:
                continue

            entry_idx = entry_row.index[0]
            exit_idx = entry_idx + holding_days
            if exit_idx >= len(ticker_prices):
                continue

            entry_price = ticker_prices.loc[entry_idx, 'Close']
            exit_price = ticker_prices.loc[exit_idx, 'Close']
            drift_return = (exit_price - entry_price) / entry_price

            signals.at[i, 'DriftReturn'] = drift_return * row['Signal']

        return signals.sort_values('DriftReturn', ascending=False)

    def latest_signal(self):
        """Returns the most recent signal"""
        signals = self.generate_signals()
        return signals.sort_values('Date').iloc[-1]


# Example Usage
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    
    earnings_data = pd.DataFrame({
        'Date': np.random.choice(dates, size=10),
        'Ticker': np.random.choice(tickers, size=10),
        'Surprise': np.random.uniform(-0.2, 0.2, size=10)
    })

    price_data = pd.DataFrame()
    for ticker in tickers:
        temp = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Close': np.cumsum(np.random.randn(30)) + 100
        })
        price_data = pd.concat([price_data, temp], ignore_index=True)

    strategy = EarningsDriftPostReaction(earnings_data, price_data)
    print(strategy.simulate_drift_returns())