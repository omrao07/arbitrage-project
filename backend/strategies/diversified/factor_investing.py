import pandas as pd
import numpy as np

class FactorInvestingAlpha:
    """
    Factor Investing Strategy:
    - Ranks stocks by chosen factors: value (P/E), momentum (12M return), quality (ROE)
    - Constructs long-short portfolio by taking top and bottom decile of ranked scores
    """

    def __init__(self, factor_data: pd.DataFrame):
        """
        Parameters:
        - factor_data: DataFrame with columns ['Date', 'Ticker', 'PE', 'ROE', '12M_Return']
        """
        self.data = factor_data.copy()
        self.signals = None

    def compute_scores(self):
        """Normalize factors and create composite score"""
        df = self.data.copy()
        df['PE_rank'] = df.groupby('Date')['PE'].transform(lambda x: -x.rank())  # Lower PE is better
        df['ROE_rank'] = df.groupby('Date')['ROE'].transform(lambda x: x.rank())
        df['Momentum_rank'] = df.groupby('Date')['12M_Return'].transform(lambda x: x.rank())

        df['CompositeScore'] = (
            df['PE_rank'] + df['ROE_rank'] + df['Momentum_rank']
        ) / 3

        self.data = df
        return self.data

    def generate_signals(self):
        """
        Generate long/short signal based on top and bottom decile of scores
        Signal: +1 for long, -1 for short, 0 for neutral
        """
        if 'CompositeScore' not in self.data.columns:
            self.compute_scores()

        def assign_signal(x):
            decile = pd.qcut(x['CompositeScore'], 10, labels=False, duplicates='drop')
            return np.where(decile >= 9, 1, np.where(decile <= 0, -1, 0))

        self.data['Signal'] = self.data.groupby('Date').apply(assign_signal).reset_index(level=0, drop=True)
        self.signals = self.data[['Date', 'Ticker', 'Signal', 'CompositeScore']]
        return self.signals

    def simulate_pnl(self, price_data: pd.DataFrame, holding_days: int = 5):
        """
        Simulates PnL based on signals and price movement

        Parameters:
        - price_data: DataFrame with ['Date', 'Ticker', 'Close']
        """
        if self.signals is None:
            raise ValueError("Run generate_signals() before simulating PnL.")

        merged = pd.merge(self.signals, price_data, on=['Date', 'Ticker'], how='inner')
        merged.sort_values(['Ticker', 'Date'], inplace=True)

        results = []
        for ticker, df in merged.groupby('Ticker'):
            df = df.reset_index(drop=True)
            for i in range(len(df) - holding_days):
                signal = df.loc[i, 'Signal']
                if signal == 0:
                    continue

                entry_price = df.loc[i, 'Close']
                exit_price = df.loc[i + holding_days, 'Close']
                pnl = signal * ((exit_price - entry_price) / entry_price)

                results.append({
                    'Ticker': ticker,
                    'EntryDate': df.loc[i, 'Date'],
                    'ExitDate': df.loc[i + holding_days, 'Date'],
                    'Signal': signal,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL': pnl
                })

        return pd.DataFrame(results)

    def latest_signals(self):
        """Return latest factor-based signals"""
        if self.signals is None:
            self.generate_signals()
        return self.signals.groupby('Ticker').tail(1).reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN']
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    rows = []

    for date in dates:
        for ticker in tickers:
            rows.append({
                'Date': date,
                'Ticker': ticker,
                'PE': np.random.uniform(10, 50),
                'ROE': np.random.uniform(5, 30),
                '12M_Return': np.random.normal(0.1, 0.15)
            })

    factor_df = pd.DataFrame(rows)
    prices = factor_df[['Date', 'Ticker']].copy()
    prices['Close'] = np.random.uniform(100, 300, size=len(prices))

    strategy = FactorInvestingAlpha(factor_df)
    signals = strategy.generate_signals()
    pnl = strategy.simulate_pnl(prices)

    print(signals.tail())
    print(pnl.tail())