import numpy as np
import pandas as pd

class DispersionTrades:
    """
    Strategy: Exploit volatility dispersion between index options and individual constituents.
    Long individual vol, short index vol when correlation breaks down (or vice versa).
    """

    def __init__(self, index_data: pd.DataFrame, component_data: dict):
        """
        Parameters:
        - index_data: DataFrame with columns ['Date', 'ImpliedVol']
        - component_data: dict of {ticker: DataFrame with ['Date', 'ImpliedVol']}
        """
        self.index_data = index_data
        self.component_data = component_data

    def compute_average_component_vol(self):
        """Calculates average implied vol across all components for each date"""
        all_vols = []

        for df in self.component_data.values():
            df = df[['Date', 'ImpliedVol']].copy()
            df.columns = ['Date', 'Vol']
            all_vols.append(df)

        merged = all_vols[0]
        for df in all_vols[1:]:
            merged = pd.merge(merged, df, on='Date', how='inner', suffixes=('', '_dup'))

        merged['AvgComponentVol'] = merged.drop(columns=['Date']).mean(axis=1)
        return merged[['Date', 'AvgComponentVol']]

    def generate_signals(self, threshold=0.2):
        """
        Signal:
        - Long dispersion: if AvgComponentVol - IndexVol > threshold
        - Short dispersion: if IndexVol - AvgComponentVol > threshold
        """
        avg_vol_df = self.compute_average_component_vol()
        combined = pd.merge(avg_vol_df, self.index_data, on='Date', how='inner')
        combined['VolDiff'] = combined['AvgComponentVol'] - combined['ImpliedVol']

        combined['signal'] = 0
        combined.loc[combined['VolDiff'] > threshold, 'signal'] = 1   # Long dispersion
        combined.loc[combined['VolDiff'] < -threshold, 'signal'] = -1  # Short dispersion

        return combined[['Date', 'AvgComponentVol', 'ImpliedVol', 'VolDiff', 'signal']]

    def latest_signal(self):
        """Returns the most recent signal"""
        df = self.generate_signals()
        return df.iloc[-1]


# Example usage
if __name__ == "__main__":
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)

    index_data = pd.DataFrame({
        'Date': dates,
        'ImpliedVol': np.random.uniform(0.15, 0.25, size=30)
    })

    component_data = {
        'AAPL': pd.DataFrame({'Date': dates, 'ImpliedVol': np.random.uniform(0.18, 0.28, size=30)}),
        'MSFT': pd.DataFrame({'Date': dates, 'ImpliedVol': np.random.uniform(0.16, 0.26, size=30)}),
        'GOOGL': pd.DataFrame({'Date': dates, 'ImpliedVol': np.random.uniform(0.17, 0.27, size=30)})
    }

    strategy = DispersionTrades(index_data, component_data)
    print(strategy.latest_signal())