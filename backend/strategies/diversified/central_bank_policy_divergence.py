import pandas as pd
from datetime import datetime

class CentralBankPolicyDivergence:
    """
    Central Bank Policy Divergence Strategy:
    Compares the policy rates of two countries to detect divergence and trade FX accordingly.
    Example: If the US hikes and EU holds, go long USD/EUR.
    """

    def __init__(self, policy_data, base="US", quote="EU"):
        self.base = base
        self.quote = quote
        self.data = self._prepare_data(policy_data)

    def _prepare_data(self, df):
        df = df[[self.base, self.quote]].dropna()
        df["diff"] = df[self.base] - df[self.quote]
        df["diff_change"] = df["diff"].diff()
        return df

    def generate_signals(self, threshold=0.25):
        df = self.data.copy()
        df["signal"] = 0
        df.loc[df["diff_change"] > threshold, "signal"] = 1   # Long base currency
        df.loc[df["diff_change"] < -threshold, "signal"] = -1  # Long quote currency
        return df

    def latest_signal(self):
        df = self.generate_signals()
        signal = df.iloc[-1]["signal"]
        if signal == 1:
            return f"LONG {self.base}/{self.quote}"
        elif signal == -1:
            return f"SHORT {self.base}/{self.quote}"
        else:
            return "No signal"

    def backtest(self, fx_returns):
        df = self.generate_signals()
        df = df.join(fx_returns, how='inner')
        df["strategy_returns"] = df["signal"].shift(1) * df["fx_returns"]
        df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
        return df

# EXAMPLE USAGE:
if __name__ == "__main__":
    # Mock policy rate data
    data = {
        "Date": pd.date_range(start="2022-01-01", periods=100, freq="W"),
        "US": pd.Series([1.0 + 0.05*i for i in range(100)]),
        "EU": pd.Series([0.5 + 0.02*i for i in range(100)])
    }
    policy_df = pd.DataFrame(data).set_index("Date")

    # FX return proxy
    fx_returns = pd.DataFrame({
        "fx_returns": pd.Series([0.001 if i % 2 == 0 else -0.001 for i in range(100)])
    }, index=policy_df.index)

    strategy = CentralBankPolicyDivergence(policy_df, base="US", quote="EU")
    print("Latest Signal:", strategy.latest_signal())

    backtest_df = strategy.backtest(fx_returns)
    print(backtest_df.tail())