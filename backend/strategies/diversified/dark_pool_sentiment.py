import pandas as pd
import numpy as np

class DarkPoolSentiment:
    """
    Strategy: Detect accumulation or distribution by institutions via dark pool volume spikes.
    Signal is generated when dark pool volume exceeds a moving average by a threshold.
    """

    def __init__(self, data: pd.DataFrame):
        """
        data: DataFrame with columns ['Ticker', 'Date', 'DarkPoolVolume', 'TotalVolume']
        """
        self.data = data

    def generate_signals(self, volume_window=10, threshold=1.5):
        """
        Parameters:
        - volume_window: rolling window to calculate average dark pool volume
        - threshold: multiplier to determine significant volume spike
        """
        df = self.data.copy()
        df['DarkPoolVolumeMA'] = df.groupby('Ticker')['DarkPoolVolume'].transform(
            lambda x: x.rolling(volume_window, min_periods=1).mean()
        )

        # Detect volume spike
        df['VolumeSpike'] = df['DarkPoolVolume'] / df['DarkPoolVolumeMA']

        # Signal: If spike > threshold and dark pool volume > 30% of total volume
        df['signal'] = np.where(
            (df['VolumeSpike'] > threshold) & (df['DarkPoolVolume'] / df['TotalVolume'] > 0.3),
            1,  # Long signal due to possible institutional buying
            0
        )

        return df[['Ticker', 'Date', 'DarkPoolVolume', 'TotalVolume', 'VolumeSpike', 'signal']]

    def latest_signals(self):
        """
        Returns the latest signals (most recent date per ticker)
        """
        signal_df = self.generate_signals()
        return signal_df.sort_values('Date').groupby('Ticker').tail(1)


# Example usage
if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'Ticker': ['AAPL'] * 15 + ['MSFT'] * 15,
        'Date': pd.date_range(end=pd.Timestamp.today(), periods=15).tolist() * 2,
        'DarkPoolVolume': np.random.randint(1000, 5000, size=30),
        'TotalVolume': np.random.randint(5000, 10000, size=30)
    })

    strat = DarkPoolSentiment(sample_data)
    print(strat.latest_signals())