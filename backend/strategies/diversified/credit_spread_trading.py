import pandas as pd
import numpy as np

class CreditSpreadTrading:
    """
    Credit Spread Trading Strategy:
    Long IG (e.g., LQD), Short HY (e.g., HYG) when spreads are expected to compress, and vice versa.
    """

    def __init__(self, ig_data: pd.DataFrame, hy_data: pd.DataFrame):
        """
        ig_data: DataFrame with ['Date', 'IG_Price']
        hy_data: DataFrame with ['Date', 'HY_Price']
        """
        self.ig_df = ig_data.set_index("Date")
        self.hy_df = hy_data.set_index("Date")
        self.data = self._prepare()

    def _prepare(self):
        df = self.ig_df.join(self.hy_df, how="inner")
        df["Spread"] = df["HY_Price"] - df["IG_Price"]
        df["SpreadZ"] = (df["Spread"] - df["Spread"].rolling(20).mean()) / df["Spread"].rolling(20).std()
        return df.dropna()

    def generate_signals(self, z_entry=1.0, z_exit=0.3):
        df = self.data.copy()
        df["signal"] = 0

        # If spread is wide (HY over IG), expect compression → Long IG / Short HY
        df.loc[df["SpreadZ"] > z_entry, "signal"] = 1

        # If spread is narrow, expect widening → Short IG / Long HY
        df.loc[df["SpreadZ"] < -z_entry, "signal"] = -1

        # Exit condition
        df.loc[df["SpreadZ"].abs() < z_exit, "signal"] = 0

        return df

    def latest_signal(self):
        df = self.generate_signals()
        signal = df.iloc[-1]["signal"]
        if signal == 1:
            return "LONG IG / SHORT HY"
        elif signal == -1:
            return "SHORT IG / LONG HY"
        else:
            return "NO POSITION"

    def backtest(self, ig_returns: pd.Series, hy_returns: pd.Series, hedge_ratio=1.0):
        df = self.generate_signals()
        df["IG_Return"] = ig_returns.pct_change()
        df["HY_Return"] = hy_returns.pct_change()
        df = df.dropna()

        df["strategy_return"] = df["signal"].shift(1) * (
            df["IG_Return"] - hedge_ratio * df["HY_Return"]
        )
        df["cumulative_returns"] = (1 + df["strategy_return"]).cumprod()

        return df

# Sample usage for testing
if __name__ == "__main__":
    dates = pd.date_range(start="2023-01-01", periods=120, freq="B")
    ig_prices = 110 + np.random.normal(0, 0.3, 120).cumsum()
    hy_prices = 105 + np.random.normal(0, 0.4, 120).cumsum()

    ig_df = pd.DataFrame({"Date": dates, "IG_Price": ig_prices})
    hy_df = pd.DataFrame({"Date": dates, "HY_Price": hy_prices})

    strat = CreditSpreadTrading(ig_df, hy_df)
    print("Latest Signal:", strat.latest_signal())

    bt = strat.backtest(
        pd.Series(ig_prices, index=dates),
        pd.Series(hy_prices, index=dates)
    )
    print(bt.tail())