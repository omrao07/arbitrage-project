import pandas as pd
import numpy as np


class GeopoliticalHedging:
    """
    Strategy that allocates to hedging assets (Gold, Oil, Defense ETFs)
    when a geopolitical risk index spikes above a dynamic threshold.
    """

    def __init__(self, risk_index: pd.DataFrame, asset_prices: pd.DataFrame):
        """
        Parameters:
        - risk_index: DataFrame with ['Date', 'GeopoliticalRiskScore']
        - asset_prices: DataFrame with ['Date', 'Asset', 'Close']
        """
        self.risk_index = risk_index.copy()
        self.asset_prices = asset_prices.copy()
        self.signals = None

    def generate_signals(self, threshold_quantile: float = 0.85):
        """
        Generate long signals for gold, oil, defense when risk is high
        """
        threshold = self.risk_index['GeopoliticalRiskScore'].quantile(threshold_quantile)
        high_risk_dates = self.risk_index[self.risk_index['GeopoliticalRiskScore'] >= threshold]['Date']

        hedging_assets = ['GLD', 'XLE', 'XAR']  # Gold ETF, Energy ETF, Aerospace & Defense ETF

        signal_list = []
        for date in high_risk_dates:
            for asset in hedging_assets:
                signal_list.append({'Date': date, 'Asset': asset, 'Signal': 1})

        self.signals = pd.DataFrame(signal_list)
        return self.signals

    def simulate_pnl(self, holding_days: int = 5):
        """
        Backtest hedging asset returns post-signal

        Returns:
        - DataFrame with trade-level PnL
        """
        if self.signals is None:
            self.generate_signals()

        merged = pd.merge(self.signals, self.asset_prices, on=['Date', 'Asset'], how='inner')
        merged.sort_values(['Asset', 'Date'], inplace=True)

        results = []
        for asset, df in merged.groupby('Asset'):
            df = df.reset_index(drop=True)
            for i in range(len(df) - holding_days):
                signal = df.loc[i, 'Signal']
                if signal != 1:
                    continue

                entry_price = df.loc[i, 'Close']
                exit_price = df.loc[i + holding_days, 'Close']
                pnl = (exit_price - entry_price) / entry_price

                results.append({
                    'Asset': asset,
                    'EntryDate': df.loc[i, 'Date'],
                    'ExitDate': df.loc[i + holding_days, 'Date'],
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL': pnl
                })

        return pd.DataFrame(results)

    def latest_signals(self):
        """Return most recent hedge triggers"""
        if self.signals is None:
            self.generate_signals()
        return self.signals.groupby('Asset').tail(1).reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    dates = pd.date_range(end=pd.Timestamp.today(), periods=200)
    risk_scores = np.random.normal(50, 10, size=len(dates))
    risk_df = pd.DataFrame({'Date': dates, 'GeopoliticalRiskScore': risk_scores})

    assets = ['GLD', 'XLE', 'XAR']
    price_data = []
    for date in dates:
        for asset in assets:
            price_data.append({
                'Date': date,
                'Asset': asset,
                'Close': np.random.uniform(90, 150)
            })
    asset_df = pd.DataFrame(price_data)

    strat = GeopoliticalHedging(risk_df, asset_df)
    signals = strat.generate_signals()
    pnl = strat.simulate_pnl()

    print(signals.tail())
    print(pnl.tail())