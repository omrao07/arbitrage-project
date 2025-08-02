import pandas as pd
import numpy as np

class CurrencyCarryTrade:
    """
    Implements a basic currency carry trade strategy:
    Long high-yielding currencies, short low-yielding currencies.
    """

    def __init__(self, data: pd.DataFrame):
        """
        data: DataFrame with columns ['CurrencyPair', 'InterestRateLong', 'InterestRateShort', 'Volatility']
        - InterestRateLong: interest rate of currency to be long
        - InterestRateShort: interest rate of currency to be short
        - Volatility: estimated historical or implied vol
        """
        self.df = data

    def generate_signals(self, min_spread=1.0, max_vol=0.2):
        df = self.df.copy()

        # Calculate the interest rate differential (carry)
        df['Carry'] = df['InterestRateLong'] - df['InterestRateShort']

        # Apply filters for positive carry and low volatility
        df['signal'] = np.where(
            (df['Carry'] > min_spread) & (df['Volatility'] < max_vol),
            1,  # Long the pair
            0   # No position
        )

        return df[['CurrencyPair', 'Carry', 'Volatility', 'signal']]

    def latest_signals(self):
        return self.generate_signals()


# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({
        'CurrencyPair': ['AUD/JPY', 'USD/JPY', 'EUR/CHF', 'NZD/CHF', 'TRY/JPY'],
        'InterestRateLong': [4.35, 5.50, 3.75, 5.50, 30.00],
        'InterestRateShort': [0.10, -0.10, 1.00, 1.25, 0.10],
        'Volatility': [0.12, 0.09, 0.08, 0.15, 0.35]
    })

    strat = CurrencyCarryTrade(data)
    print(strat.latest_signals())