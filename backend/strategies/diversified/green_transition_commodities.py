import pandas as pd
import numpy as np

class GreenTransitionCommodities:
    """
    Alpha strategy to long critical green-transition metals (e.g., copper, lithium, nickel)
    when demand indicators are rising or global green policy momentum is strong.
    """

    def __init__(self, commodity_data: dict, policy_sentiment_data: pd.DataFrame):
        """
        Parameters:
        - commodity_data: dict of DataFrames, e.g. {"copper": df1, "lithium": df2}
                          each with ['Date', 'Price']
        - policy_sentiment_data: DataFrame with ['Date', 'SentimentScore'] from ESG/green headlines
        """
        self.commodity_data = commodity_data
        self.policy_sentiment_data = policy_sentiment_data
        self.signals = {}

    def generate_signals(self, sentiment_threshold: float = 0.1):
        """
        Generate a binary signal for each commodity: Long if sentiment > threshold
        """
        merged_sentiment = self.policy_sentiment_data.copy()
        merged_sentiment['Signal'] = np.where(merged_sentiment['SentimentScore'] > sentiment_threshold, 1, 0)

        for name, df in self.commodity_data.items():
            merged = pd.merge(df, merged_sentiment, on='Date', how='inner')
            merged = merged[['Date', 'Price', 'Signal']]
            self.signals[name] = merged

        return self.signals

    def backtest(self, holding_days: int = 5):
        """
        Backtest each metal's returns using the sentiment-driven signal
        """
        if not self.signals:
            self.generate_signals()

        all_trades = {}
        for name, data in self.signals.items():
            trades = []
            for i in range(len(data) - holding_days):
                if data.loc[i, 'Signal'] != 1:
                    continue
                entry_date = data.loc[i, 'Date']
                exit_date = data.loc[i + holding_days, 'Date']
                entry_price = data.loc[i, 'Price']
                exit_price = data.loc[i + holding_days, 'Price']
                pnl = (exit_price - entry_price) / entry_price

                trades.append({
                    'EntryDate': entry_date,
                    'ExitDate': exit_date,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL': pnl
                })

            all_trades[name] = pd.DataFrame(trades)

        return all_trades

    def latest_signal(self):
        """
        Return latest signals for all tracked commodities
        """
        latest = {}
        for name, df in self.signals.items():
            if not df.empty:
                latest[name] = df.iloc[-1].to_dict()
        return latest


# Example usage
if __name__ == "__main__":
    dates = pd.date_range(end=pd.Timestamp.today(), periods=250)
    commodity_data = {
        "copper": pd.DataFrame({
            "Date": dates,
            "Price": 8000 + np.cumsum(np.random.normal(0, 20, size=250))
        }),
        "lithium": pd.DataFrame({
            "Date": dates,
            "Price": 60000 + np.cumsum(np.random.normal(0, 200, size=250))
        }),
    }

    policy_sentiment_data = pd.DataFrame({
        "Date": dates,
        "SentimentScore": np.random.normal(0.05, 0.1, size=250)
    })

    strategy = GreenTransitionCommodities(commodity_data, policy_sentiment_data)
    signals = strategy.generate_signals()
    trades = strategy.backtest()
    print(strategy.latest_signal())
    print(trades['copper'].tail())