import pandas as pd
import numpy as np

class EnergyRollYield:
    """
    Strategy: Energy Roll Yield Capture
    - Exploits the pricing difference between front-month and next-month energy futures.
    - Long backwardated contracts (positive roll yield).
    - Short contangoed contracts (negative roll yield).
    """

    def __init__(self, futures_data: pd.DataFrame):
        """
        Parameters:
        - futures_data: DataFrame with ['Date', 'Commodity', 'FrontMonthPrice', 'NextMonthPrice']
        """
        self.data = futures_data.copy()

    def calculate_roll_yield(self):
        """Calculates roll yield and generates signals"""
        self.data['RollYield'] = (self.data['FrontMonthPrice'] - self.data['NextMonthPrice']) / self.data['NextMonthPrice']

        # Signal: +1 for backwardation (positive roll yield), -1 for contango
        self.data['Signal'] = np.where(self.data['RollYield'] > 0, 1, -1)
        return self.data[['Date', 'Commodity', 'RollYield', 'Signal']]

    def simulate_roll_return(self, days_held: int = 10):
        """
        Simulate PnL over a short holding period based on roll yield signal
        """
        results = []
        grouped = self.data.groupby('Commodity')

        for commodity, df in grouped:
            df = df.sort_values('Date').reset_index(drop=True)

            for i in range(len(df) - days_held):
                signal = df.loc[i, 'Signal']
                entry_price = df.loc[i, 'FrontMonthPrice']
                exit_price = df.loc[i + days_held, 'FrontMonthPrice']

                pnl = signal * ((exit_price - entry_price) / entry_price)

                results.append({
                    'Commodity': commodity,
                    'StartDate': df.loc[i, 'Date'],
                    'EndDate': df.loc[i + days_held, 'Date'],
                    'Signal': signal,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL': pnl
                })

        return pd.DataFrame(results)

    def latest_signals(self):
        """Return the latest signal per commodity"""
        return self.calculate_roll_yield().groupby('Commodity').tail(1).reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
    commodities = ['WTI', 'Brent', 'NaturalGas']
    futures_df = pd.DataFrame()

    for commodity in commodities:
        temp = pd.DataFrame({
            'Date': dates,
            'Commodity': commodity,
            'FrontMonthPrice': np.random.uniform(50, 100, size=len(dates)),
            'NextMonthPrice': np.random.uniform(50, 100, size=len(dates))
        })
        futures_df = pd.concat([futures_df, temp], ignore_index=True)

    strategy = EnergyRollYield(futures_df)
    signals = strategy.calculate_roll_yield()
    pnl = strategy.simulate_roll_return()

    print(signals.tail())
    print(pnl.tail())