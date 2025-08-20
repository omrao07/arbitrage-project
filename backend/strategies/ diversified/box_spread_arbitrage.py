"""
Box Spread Arbitrage Strategy
-----------------------------
Captures risk-free profit from mispricing between a synthetic long and short in options.

Mechanics:
- Buy a bull call spread (long call at lower strike, short call at higher strike)
- Buy a bear put spread (long put at higher strike, short put at lower strike)
- Net payoff is fixed (difference in strikes) regardless of market direction.

Asset Class: Options
Category: Diversified / Arbitrage
"""

import pandas as pd
import numpy as np

class BoxSpreadArbitrageStrategy:
    def __init__(self, options_data: pd.DataFrame, risk_free_rate: float = 0.03):
        """
        :param options_data: DataFrame with ['date', 'strike', 'type', 'price', 'expiry']
        :param risk_free_rate: Annualized risk-free rate (decimal)
        """
        self.options_data = options_data
        self.risk_free_rate = risk_free_rate

    def find_box_spreads(self):
        """
        Identify possible box spreads and compute theoretical value vs market price.
        """
        calls = self.options_data[self.options_data['type'] == 'call']
        puts = self.options_data[self.options_data['type'] == 'put']

        results = []

        for expiry in self.options_data['expiry'].unique():
            call_chain = calls[calls['expiry'] == expiry]
            put_chain = puts[puts['expiry'] == expiry]
            strikes = sorted(call_chain['strike'].unique())

            for i in range(len(strikes) - 1):
                K1, K2 = strikes[i], strikes[i + 1]

                # Prices
                c1 = call_chain[call_chain['strike'] == K1]['price'].mean()
                c2 = call_chain[call_chain['strike'] == K2]['price'].mean()
                p1 = put_chain[put_chain['strike'] == K1]['price'].mean()
                p2 = put_chain[put_chain['strike'] == K2]['price'].mean()

                if pd.isna([c1, c2, p1, p2]).any():
                    continue

                # Market cost of box spread
                market_cost = (c1 - c2) + (p2 - p1)

                # Theoretical value: (K2 - K1) discounted
                days_to_expiry = (pd.to_datetime(expiry) - pd.to_datetime(self.options_data['date'].min())).days
                theoretical_value = (K2 - K1) / (1 + self.risk_free_rate * days_to_expiry / 365)

                # Arbitrage opportunity
                arb_profit = theoretical_value - market_cost

                results.append({
                    'expiry': expiry,
                    'K1': K1,
                    'K2': K2,
                    'market_cost': market_cost,
                    'theoretical_value': theoretical_value,
                    'arb_profit': arb_profit,
                    'signal': 'BUY_BOX' if arb_profit > 0 else 'SELL_BOX' if arb_profit < 0 else 'HOLD'
                })

        return pd.DataFrame(results)

    def backtest(self, min_profit: float = 0.1):
        """
        Backtest box spread arbitrage, filtering for opportunities above min_profit.
        """
        opportunities = self.find_box_spreads()
        return opportunities[opportunities['arb_profit'].abs() >= min_profit]

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'date': ['2023-01-01'] * 8,
        'strike': [100, 105, 100, 105, 100, 105, 100, 105],
        'type': ['call', 'call', 'put', 'put', 'call', 'call', 'put', 'put'],
        'price': [6, 3, 4, 1, 6, 3, 4, 1],
        'expiry': ['2023-03-01'] * 8
    })

    strategy = BoxSpreadArbitrageStrategy(data)
    print(strategy.backtest(min_profit=0.05))