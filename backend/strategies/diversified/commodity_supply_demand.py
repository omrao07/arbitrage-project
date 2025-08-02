import pandas as pd
import numpy as np

class CommoditySupplyDemandAlpha:
    """
    Commodity Supply-Demand Alpha:
    Trades long/short commodities based on inventory shocks (supply signals) and demand trends.
    Example: Long oil if inventories fall sharply and demand rises.
    """

    def __init__(self, inventory_df: pd.DataFrame, demand_df: pd.DataFrame, commodity: str):
        """
        inventory_df: DataFrame with columns = ["Date", "Inventory"]
        demand_df: DataFrame with columns = ["Date", "Demand"]
        commodity: e.g., "crude_oil"
        """
        self.inventory = inventory_df.set_index("Date").rename(columns={"Inventory": "inventory"})
        self.demand = demand_df.set_index("Date").rename(columns={"Demand": "demand"})
        self.commodity = commodity
        self.data = self._merge_and_clean()

    def _merge_and_clean(self):
        df = self.inventory.join(self.demand, how="inner")
        df["inventory_change"] = df["inventory"].diff()
        df["demand_change"] = df["demand"].diff()
        return df.dropna()

    def generate_signals(self, inventory_thresh=-5, demand_thresh=5):
        """
        inventory_thresh: % change threshold for inventory shock (e.g., -5%)
        demand_thresh: % change threshold for rising demand (e.g., +5%)
        """
        df = self.data.copy()
        df["signal"] = 0

        df["inv_pct"] = df["inventory_change"] / df["inventory"].shift(1) * 100
        df["dem_pct"] = df["demand_change"] / df["demand"].shift(1) * 100

        df.loc[(df["inv_pct"] <= inventory_thresh) & (df["dem_pct"] >= demand_thresh), "signal"] = 1
        df.loc[(df["inv_pct"] >= abs(inventory_thresh)) & (df["dem_pct"] <= -demand_thresh), "signal"] = -1
        return df

    def latest_signal(self):
        df = self.generate_signals()
        sig = df.iloc[-1]["signal"]
        if sig == 1:
            return f"LONG {self.commodity.upper()}"
        elif sig == -1:
            return f"SHORT {self.commodity.upper()}"
        else:
            return f"No signal for {self.commodity.upper()}"

    def backtest(self, commodity_returns: pd.Series):
        df = self.generate_signals()
        df = df.join(commodity_returns.rename("returns"), how="inner")
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
        df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
        return df

# EXAMPLE USAGE:
if __name__ == "__main__":
    dates = pd.date_range("2023-01-01", periods=100, freq="W")
    inventory = pd.DataFrame({"Date": dates, "Inventory": 1000 - np.cumsum(np.random.randn(100))})
    demand = pd.DataFrame({"Date": dates, "Demand": 500 + np.cumsum(np.random.randn(100))})
    returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)

    strat = CommoditySupplyDemandAlpha(inventory, demand, "crude_oil")
    print("Latest Signal:", strat.latest_signal())
    result = strat.backtest(returns)
    print(result.tail())