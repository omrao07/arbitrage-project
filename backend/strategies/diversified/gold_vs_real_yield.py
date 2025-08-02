import pandas as pd
import numpy as np


class GoldVsRealYield:
    """
    Long gold when real yields are falling, short or neutral when rising.
    Uses TIPS yield vs. gold price correlation.
    """

    def __init__(self, gold_data: pd.DataFrame, real_yield_data: pd.DataFrame):
        """
        Parameters:
        - gold_data: DataFrame with ['Date', 'Close']
        - real_yield_data: DataFrame with ['Date', 'RealYield'] (e.g., 10Y TIPS yield)
        """
        self.gold_data = gold_data.copy()
        self.real_yield_data = real_yield_data.copy()
        self.signals = None

    def generate_signals(self, window: int = 10):
        """
        Signal: 1 if real yield falling (long gold), 0 otherwise.
        """
        ry = self.real_yield_data.copy()
        ry['RY_Change'] = ry['RealYield'].diff(window)
        ry['Signal'] = np.where(ry['RY_Change'] < 0, 1, 0)
        self.signals = ry[['Date', 'Signal']]
        return self.signals

    def backtest(self, holding_days: int = 5):
        """
        Applies signals to gold price data and calculates PnL.
        """
        if self.signals is None:
            self.generate_signals()

        merged = pd.merge(self.signals, self.gold_data, on='Date', how='inner')
        merged = merged.sort_values('Date').reset_index(drop=True)

        trades = []
        for i in range(len(merged) - holding_days):
            if merged.loc[i, 'Signal'] != 1:
                continue
            entry_date = merged.loc[i, 'Date']
            exit_date = merged.loc[i + holding_days, 'Date']
            entry_price = merged.loc[i, 'Close']
            exit_price = merged.loc[i + holding_days, 'Close']
            pnl = (exit_price - entry_price) / entry_price

            trades.append({
                'EntryDate': entry_date,
                'ExitDate': exit_date,
                'EntryPrice': entry_price,
                'ExitPrice': exit_price,
                'PnL': pnl
            })

        return pd.DataFrame(trades)

    def latest_signal(self):
        """Returns the most recent trade signal"""
        if self.signals is None:
            self.generate_signals()
        return self.signals.tail(1).reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    dates = pd.date_range(end=pd.Timestamp.today(), periods=300)
    gold_prices = pd.DataFrame({
        'Date': dates,
        'Close': 1900 + np.cumsum(np.random.normal(0, 1, size=300))  # simulate gold price
    })
    real_yields = pd.DataFrame({
        'Date': dates,
        'RealYield': 1.0 + np.cumsum(np.random.normal(0, 0.02, size=300))  # simulate 10Y real yield
    })

    strat = GoldVsRealYield(gold_prices, real_yields)
    signals = strat.generate_signals()
    pnl = strat.backtest()
    print(signals.tail())
    print(pnl.tail())