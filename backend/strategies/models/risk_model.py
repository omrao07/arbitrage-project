import numpy as np
import pandas as pd

class RiskModel:
    """
    RiskModel handles risk metrics like volatility, correlation, Value at Risk (VaR),
    and stress testing for strategy allocation and risk control.
    """

    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level

    def calculate_volatility(self, returns: pd.Series, window: int = 30):
        """
        Calculates rolling volatility over a specified window (in days).
        """
        return returns.rolling(window).std()

    def calculate_var(self, returns: pd.Series, portfolio_value: float):
        """
        Calculates 1-day parametric Value at Risk (VaR) at the given confidence level.
        """
        mean = returns.mean()
        std = returns.std()
        z_score = abs(np.percentile(np.random.normal(0, 1, 10000), (1 - self.confidence_level) * 100))
        var = portfolio_value * (mean - z_score * std)
        return round(var, 2)

    def calculate_correlation_matrix(self, returns_df: pd.DataFrame):
        """
        Returns the correlation matrix between different strategy returns.
        """
        return returns_df.corr()

    def detect_correlation_breakdown(self, returns_df: pd.DataFrame, threshold: float = 0.8):
        """
        Detects highly correlated pairs (> threshold) that could signal systemic risk.
        """
        correlation_matrix = self.calculate_correlation_matrix(returns_df)
        breakdowns = []
        for i in correlation_matrix.columns:
            for j in correlation_matrix.columns:
                if i != j and correlation_matrix.loc[i, j] > threshold:
                    breakdowns.append((i, j, correlation_matrix.loc[i, j]))
        return breakdowns

    def risk_parity_weights(self, returns_df: pd.DataFrame):
        """
        Allocates weights using inverse volatility (risk parity approximation).
        """
        volatilities = returns_df.std()
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights.round(4)

    def stress_test(self, returns_df: pd.DataFrame, drop_percent: float = 0.2):
        """
        Simulates a uniform market shock (e.g., -20%) and returns simulated portfolio return.
        """
        shocked_returns = returns_df.apply(lambda x: x - drop_percent)
        portfolio_return = shocked_returns.mean(axis=1).mean()
        return round(portfolio_return, 4)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dummy_returns = pd.DataFrame({
        'Strategy_A': np.random.normal(0.001, 0.01, 100),
        'Strategy_B': np.random.normal(0.0008, 0.012, 100),
        'Strategy_C': np.random.normal(0.0012, 0.009, 100),
    })

    rm = RiskModel()
    print("[Risk Parity Weights]")
    print(rm.risk_parity_weights(dummy_returns))

    print("\n[Correlation Breakdown]")
    print(rm.detect_correlation_breakdown(dummy_returns, threshold=0.7))

    print("\n[Stress Test Portfolio Return]")
    print(rm.stress_test(dummy_returns, drop_percent=0.15))