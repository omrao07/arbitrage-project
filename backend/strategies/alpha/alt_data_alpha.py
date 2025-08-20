# strategies/alpha/alt_data_alpha.py
from datetime import datetime
import numpy as np
import pandas as pd

class AltDataAlpha:
    """
    Alternative Data Alpha Model
    ----------------------------
    Combines multiple non-traditional datasets to generate predictive alpha signals.

    Data sources (examples):
      • Satellite imagery (e.g., store parking lot counts, oil storage tank shadows)
      • Credit/debit card spending aggregates
      • Web traffic & app usage trends
      • Social media sentiment
      • Job postings, hiring trends
      • Shipping & trade manifests
      • ESG / carbon footprint disclosures

    Signal logic:
      1. Normalize each feature to z-scores
      2. Weight features by correlation with target returns (rolling window)
      3. Aggregate into a composite alpha score
      4. Return score in range [-1, +1] for short/long bias
    """

    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self.feature_weights = {}
        self.last_fit = None

    def fit(self, df: pd.DataFrame, target_col: str = "returns"):
        """
        Fit feature weights based on correlation to target returns.

        Parameters:
            df : DataFrame with columns = [<features>..., target_col]
            target_col : string name of returns column
        """
        df = df.dropna()
        features = [c for c in df.columns if c != target_col]

        # Normalize features
        z = (df[features] - df[features].mean()) / df[features].std(ddof=0)

        # Rolling correlation weights
        corrs = {}
        for f in features:
            corrs[f] = df[f].corr(df[target_col])
        # Normalize weights to sum of absolute value = 1
        total_abs = sum(abs(c) for c in corrs.values())
        self.feature_weights = {f: (c / total_abs) if total_abs != 0 else 0 for f, c in corrs.items()}
        self.last_fit = datetime.utcnow()

    def score(self, latest_features: dict) -> float:
        """
        Score the latest observation.

        Parameters:
            latest_features : dict {feature_name: value}
        Returns:
            float in [-1, +1]
        """
        if not self.feature_weights:
            raise ValueError("Model not fit. Call fit() first.")

        score = 0.0
        for f, val in latest_features.items():
            if f in self.feature_weights:
                score += self.feature_weights[f] * val  # assumes z-score already
        # Clip to [-1, 1]
        return float(np.clip(score, -1.0, 1.0))

    def signal(self, score: float, long_threshold: float = 0.3, short_threshold: float = -0.3) -> str:
        """
        Convert score to discrete trading signal.
        """
        if score >= long_threshold:
            return "LONG"
        elif score <= short_threshold:
            return "SHORT"
        return "FLAT"


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120)
    df = pd.DataFrame({
        "satellite_cars": np.random.normal(0, 1, 120),
        "credit_spending": np.random.normal(0, 1, 120),
        "web_traffic": np.random.normal(0, 1, 120),
        "returns": np.random.normal(0, 0.02, 120)
    }, index=dates)

    model = AltDataAlpha()
    model.fit(df, target_col="returns")
    latest = {
        "satellite_cars": 0.8,
        "credit_spending": -0.5,
        "web_traffic": 0.2
    }
    score = model.score(latest)
    print(f"Score: {score:.3f}, Signal: {model.signal(score)}")