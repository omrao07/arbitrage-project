import pandas as pd
import numpy as np

class ETFFlowAlpha:
    """
    Strategy: ETF Flow Alpha
    - Detects abnormal inflows/outflows into ETFs and generates signals based on demand surges.
    - Long ETFs with high inflows, short those with high outflows.
    """

    def __init__(self, etf_data: pd.DataFrame):
        """
        Parameters:
        - etf_data: DataFrame with ['Date', 'ETF', 'AUM', 'SharesOutstanding', 'NAV', 'Flow']
        """
        self.data = etf_data.copy()
        self.signals = None

    def calculate_flow_signal(self, lookback: int = 10, z_threshold: float = 1.5):
        """
        Generate z-score of flows and trade signal.

        Signal:
        +1 for large positive flows (buy),
        -1 for large negative flows (sell),
        0 otherwise.
        """
        self.data.sort_values(['ETF', 'Date'], inplace=True)
        self.data['FlowZ'] = self.data.groupby('ETF')['Flow'].transform(
            lambda x: (x - x.rolling(lookback).mean()) / x.rolling(lookback).std()
        )

        self.data['Signal'] = np.where(self.data['FlowZ'] > z_threshold, 1,
                               np.where(self.data['FlowZ'] < -z_threshold, -1, 0))

        self.signals = self.data[['Date', 'ETF', 'FlowZ', 'Signal']]
        return self.signals

    def simulate_pnl(self, price_data: pd.DataFrame, holding_days: int = 5):
        """
        Simulate trade PnL based on signals.

        Parameters:
        - price_data: DataFrame with ['Date', 'ETF', 'Close']
        """
        if self.signals is None:
            raise ValueError("Run calculate_flow_signal() before backtesting.")

        merged = pd.merge(self.signals, price_data, on=['Date', 'ETF'], how='inner')
        merged.sort_values(['ETF', 'Date'], inplace=True)

        results = []
        for etf, df in merged.groupby('ETF'):
            df = df.reset_index(drop=True)
            for i in range(len(df) - holding_days):
                signal = df.loc[i, 'Signal']
                if signal == 0:
                    continue

                entry_price = df.loc[i, 'Close']
                exit_price = df.loc[i + holding_days, 'Close']
                pnl = signal * ((exit_price - entry_price) / entry_price)

                results.append({
                    'ETF': etf,
                    'EntryDate': df.loc[i, 'Date'],
                    'ExitDate': df.loc[i + holding_days, 'Date'],
                    'Signal': signal,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL': pnl
                })

        return pd.DataFrame(results)

    def latest_signals(self):
        """Return latest flow signal per ETF"""
        if self.signals is None:
            self.calculate_flow_signal()
        return self.signals.groupby('ETF').tail(1).reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    # Simulate some ETF flow data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    etfs = ['SPY', 'QQQ', 'IWM']
    flow_df = pd.DataFrame()

    for etf in etfs:
        temp = pd.DataFrame({
            'Date': dates,
            'ETF': etf,
            'AUM': np.random.uniform(1e9, 1e10, size=len(dates)),
            'SharesOutstanding': np.random.uniform(1e6, 1e8, size=len(dates)),
            'NAV': np.random.uniform(100, 500, size=len(dates)),
            'Flow': np.random.normal(0, 1e8, size=len(dates))
        })
        flow_df = pd.concat([flow_df, temp], ignore_index=True)

    price_df = flow_df[['Date', 'ETF']].copy()
    price_df['Close'] = flow_df['NAV'] + np.random.normal(0, 2, size=len(price_df))

    strategy = ETFFlowAlpha(flow_df)
    signal_df = strategy.calculate_flow_signal()
    pnl_df = strategy.simulate_pnl(price_df)

    print(signal_df.tail())
    print(pnl_df.tail())