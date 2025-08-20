"""
Bond Future Basis Strategy
--------------------------
Captures the mispricing between a government bond's spot price and its futures contract.
The basis is the difference between the futures price and the theoretical fair value.

Typical Use:
- Arbitrage between bond futures and the underlying deliverable bond.
- Hedge fixed income exposure while capturing basis convergence.

Asset Class: Fixed Income
Category: Diversified / Relative Value
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class BondFutureBasisStrategy:
    def __init__(self, spot_data: pd.DataFrame, futures_data: pd.DataFrame, risk_free_rate: float = 0.03):
        """
        Initialize the strategy.

        :param spot_data: DataFrame with ['date', 'bond_price']
        :param futures_data: DataFrame with ['date', 'futures_price', 'expiry_date']
        :param risk_free_rate: Annualized risk-free rate (decimal)
        """
        self.spot_data = spot_data
        self.futures_data = futures_data
        self.risk_free_rate = risk_free_rate

    def compute_theoretical_futures(self):
        """
        Compute theoretical futures price using cost-of-carry model:
        F = Spot * (1 + r * (T/365))
        """
        merged = pd.merge(self.spot_data, self.futures_data, on="date", how="inner")
        merged['days_to_expiry'] = (pd.to_datetime(merged['expiry_date']) - pd.to_datetime(merged['date'])).dt.days
        merged['theoretical_price'] = merged['bond_price'] * (1 + self.risk_free_rate * merged['days_to_expiry'] / 365.0)
        return merged

    def generate_signals(self, threshold: float = 0.1):
        """
        Generate trading signals based on basis deviation.
        Positive basis => short futures, long bond.
        Negative basis => long futures, short bond.
        :param threshold: Minimum price deviation (%) to trigger trade.
        """
        data = self.compute_theoretical_futures()
        data['basis'] = data['futures_price'] - data['theoretical_price']
        data['basis_pct'] = data['basis'] / data['theoretical_price'] * 100

        data['signal'] = np.where(data['basis_pct'] > threshold, "SELL_FUTURES_BUY_BOND",
                           np.where(data['basis_pct'] < -threshold, "BUY_FUTURES_SELL_BOND", "HOLD"))
        return data[['date', 'bond_price', 'futures_price', 'theoretical_price', 'basis_pct', 'signal']]

    def backtest(self, threshold: float = 0.1):
        """
        Simple backtest that tracks strategy signal and returns.
        """
        signals = self.generate_signals(threshold=threshold)
        signals['return'] = np.where(
            signals['signal'] == "SELL_FUTURES_BUY_BOND",
            -signals['basis_pct'] / 100,
            np.where(signals['signal'] == "BUY_FUTURES_SELL_BOND", signals['basis_pct'] / 100, 0)
        )
        signals['cum_return'] = (1 + signals['return']).cumprod() - 1
        return signals

if __name__ == "__main__":
    # Example usage with mock data
    spot_df = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=10),
        'bond_price': np.linspace(99, 101, 10)
    })

    futures_df = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=10),
        'futures_price': np.linspace(100, 102, 10),
        'expiry_date': ["2023-03-01"] * 10
    })

    strategy = BondFutureBasisStrategy(spot_df, futures_df)
    results = strategy.backtest(threshold=0.1)
    print(results)