import pandas as pd
import numpy as np

class CrossAssetCorrelationBreakdown:
    """
    Cross-Asset Correlation Breakdown Strategy:
    Monitors correlation between two assets (e.g., SPY & TLT). If correlation deviates significantly
    from historical norm, take a long/short position to exploit mean reversion.
    """

    def __init__(self, asset1_df: pd.DataFrame, asset2_df: pd.DataFrame, window: int = 60):
        """
        asset1_df: DataFrame with ['Date', 'Price'] for asset 1
        asset2_df: DataFrame with ['Date', 'Price'] for asset 2
        window: rolling window for correlation computation
        """
        self.asset1 = asset1_df.set_index("Date").rename(columns={"Price": "Asset1"})
        self.asset2 = asset2_df.set_index("Date").rename(columns={"Price": "Asset2"})
        self.window = window
        self.data = self._prepare()

    def _prepare(self):
        df = self.asset1.join(self.asset2, how="inner")
        df["Asset1_Return"] = df["Asset1"].pct_change()
        df["Asset2_Return"] = df["Asset2"].pct_change()
        df["rolling_corr"] = df["Asset1_Return"].rolling(self.window).corr(df["Asset2_Return"])
        df["rolling_mean_corr"] = df["rolling_corr"].rolling(10).mean()
        df["rolling_std_corr"] = df["rolling_corr"].rolling(10).std()
        df["z_score"] = (df["rolling_corr"] - df["rolling_mean_corr"]) / df["rolling_std_corr"]
        return df.dropna()

    def generate_signals(self, z_threshold=1.5, z_exit=0.3):
        df = self.data.copy()
        df["signal"] = 0

        # Long correlation breakdown: Bet correlation will revert back
        df.loc[df["z_score"] > z_threshold, "signal"] = -1  # Correlation too high → mean reversion short
        df.loc[df["z_score"] < -z_threshold, "signal"] = 1  # Correlation too low → mean reversion long

        # Exit condition
        df.loc[df["z_score"].abs() < z_exit, "signal"] = 0

        return df

    def latest_signal(self):
        df = self.generate_signals()
        sig = df.iloc[-1]["signal"]
        if sig == 1:
            return "LONG CORRELATION BREAKDOWN TRADE"
        elif sig == -1:
            return "SHORT CORRELATION REVERSION TRADE"
        else:
            return "NO TRADE"

    def backtest(self, asset1_returns: pd.Series, asset2_returns: pd.Series):
        df = self.generate_signals()
        df["Asset1Ret"] = asset1_returns.pct_change()
        df["Asset2Ret"] = asset2_returns.pct_change()
        df = df.dropna()

        df["strategy_return"] = df["signal"].shift(1) * (df["Asset1Ret"] - df["Asset2Ret"])
        df["cumulative_returns"] = (1 + df["strategy_return"]).cumprod()

        return df

# Example usage
if __name__ == "__main__":
    dates = pd.date_range("2023-01-01", periods=150, freq="B")
    asset1 = pd.Series(100 + np.random.normal(0, 1, len(dates)).cumsum(), index=dates)
    asset2 = pd.Series(100 + np.random.normal(0, 1, len(dates)).cumsum(), index=dates)

    asset1_df = pd.DataFrame({"Date": dates, "Price": asset1.values})
    asset2_df = pd.DataFrame({"Date": dates, "Price": asset2.values})

    strat = CrossAssetCorrelationBreakdown(asset1_df, asset2_df)
    print("Latest Signal:", strat.latest_signal())

    bt = strat.backtest(asset1, asset2)
    print(bt.tail())