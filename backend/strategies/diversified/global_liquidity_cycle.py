import pandas as pd
import numpy as np


class GlobalLiquidityCycle:
    """
    Strategy that goes long risk assets during QE periods
    and short or neutral during QT periods using central bank balance sheets.
    """

    def __init__(self, liquidity_data: pd.DataFrame, asset_prices: pd.DataFrame):
        """
        Parameters:
        - liquidity_data: DataFrame with ['Date', 'LiquidityIndex'] (composite global liquidity)
        - asset_prices: DataFrame with ['Date', 'Asset', 'Close']
        """
        self.liquidity_data = liquidity_data.copy()
        self.asset_prices = asset_prices.copy()
        self.signals = None

    def generate_signals(self, regime_sma: int = 30):
        """
        Determines QE/QT based on LiquidityIndex trend (simple moving average).
        Returns signals: 1 = Long risk asset (QE), 0 = Neutral (QT)
        """
        df = self.liquidity_data.copy()
        df['SMA'] = df['LiquidityIndex'].rolling(window=regime_sma).mean()
        df['Regime'] = np.where(df['LiquidityIndex'] > df['SMA'], 'QE', 'QT')

        signal_list = []
        for _, row in df.iterrows():
            signal = 1 if row['Regime'] == 'QE' else 0
            signal_list.append({'Date': row['Date'], 'Signal': signal})

        self.signals = pd.DataFrame(signal_list)
        return self.signals

    def apply_to_assets(self, risk_assets: list = ['SPY', 'EEM', 'QQQ'], holding_days: int = 5):
        """
        Executes simulated trades on selected risk assets during QE periods.
        Returns a DataFrame with trade-level PnL.
        """
        if self.signals is None:
            self.generate_signals()

        merged = pd.merge(self.signals, self.asset_prices, on='Date', how='inner')
        merged = merged[merged['Asset'].isin(risk_assets)].sort_values(['Asset', 'Date'])

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

    def latest_regime(self):
        """Returns the most recent QE/QT regime classification"""
        if self.signals is None:
            self.generate_signals()
        return self.signals.tail(1).reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    # Simulated data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=300)
    liquidity_index = np.cumsum(np.random.normal(0.1, 1, size=len(dates)))
    liquidity_df = pd.DataFrame({'Date': dates, 'LiquidityIndex': liquidity_index})

    asset_prices = []
    for date in dates:
        for asset in ['SPY', 'EEM', 'QQQ']:
            asset_prices.append({
                'Date': date,
                'Asset': asset,
                'Close': np.random.uniform(100, 200)
            })
    prices_df = pd.DataFrame(asset_prices)

    strat = GlobalLiquidityCycle(liquidity_df, prices_df)
    signals = strat.generate_signals()
    pnl = strat.apply_to_assets()

    print(signals.tail())
    print(pnl.tail())