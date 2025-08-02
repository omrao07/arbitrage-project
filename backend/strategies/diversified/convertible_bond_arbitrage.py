import pandas as pd
import numpy as np

class ConvertibleBondArbitrage:
    """
    Convertible Bond Arbitrage:
    Long convertible bond, short underlying stock to hedge delta and extract mispricing.
    """

    def __init__(self, bond_data: pd.DataFrame, stock_data: pd.DataFrame):
        """
        bond_data: DataFrame with ['Date', 'BondPrice', 'ConversionRatio']
        stock_data: DataFrame with ['Date', 'StockPrice']
        """
        self.bond_df = bond_data.set_index("Date")
        self.stock_df = stock_data.set_index("Date")
        self.data = self._merge_data()

    def _merge_data(self):
        df = self.bond_df.join(self.stock_df, how="inner")
        df["ImpliedStockValue"] = df["ConversionRatio"] * df["StockPrice"]
        df["Spread"] = df["BondPrice"] - df["ImpliedStockValue"]
        df["SpreadZ"] = (df["Spread"] - df["Spread"].rolling(20).mean()) / df["Spread"].rolling(20).std()
        return df.dropna()

    def generate_signals(self, z_entry=1.0, z_exit=0.2):
        df = self.data.copy()
        df["signal"] = 0

        # Long bond / Short stock when spread is wide
        df.loc[df["SpreadZ"] > z_entry, "signal"] = 1
        # Exit position
        df.loc[df["SpreadZ"].abs() < z_exit, "signal"] = 0
        # Short bond / Long stock (rare case)
        df.loc[df["SpreadZ"] < -z_entry, "signal"] = -1

        return df

    def latest_signal(self):
        df = self.generate_signals()
        sig = df.iloc[-1]["signal"]
        if sig == 1:
            return "LONG BOND / SHORT STOCK"
        elif sig == -1:
            return "SHORT BOND / LONG STOCK"
        else:
            return "NO POSITION"

    def backtest(self, bond_returns: pd.Series, stock_returns: pd.Series, hedge_ratio=1.0):
        df = self.generate_signals()
        df["BondRet"] = bond_returns.pct_change()
        df["StockRet"] = stock_returns.pct_change()
        df = df.dropna()

        df["strategy_return"] = df["signal"].shift(1) * (
            df["BondRet"] - hedge_ratio * df["StockRet"]
        )
        df["cumulative_returns"] = (1 + df["strategy_return"]).cumprod()
        return df

# EXAMPLE USAGE
if __name__ == "__main__":
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    bond_price = 100 + np.random.normal(0, 1, 100).cumsum()
    stock_price = 50 + np.random.normal(0, 1, 100).cumsum()
    conversion_ratio = 2  # fixed

    bond_df = pd.DataFrame({"Date": dates, "BondPrice": bond_price, "ConversionRatio": conversion_ratio})
    stock_df = pd.DataFrame({"Date": dates, "StockPrice": stock_price})

    strategy = ConvertibleBondArbitrage(bond_df, stock_df)
    print("Latest Signal:", strategy.latest_signal())

    bond_returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
    stock_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

    bt_result = strategy.backtest(bond_returns, stock_returns)
    print(bt_result.tail())