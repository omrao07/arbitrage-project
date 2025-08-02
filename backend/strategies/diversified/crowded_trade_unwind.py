import pandas as pd
import numpy as np

class CrowdedTradeUnwindAlpha:
    """
    Identifies over-owned, crowded trades (typically by hedge funds) that are at risk of sharp reversals.
    Based on HF ownership, recent inflows, sentiment, or performance vs. fundamentals.
    """

    def __init__(self, data: pd.DataFrame):
        """
        data: DataFrame containing columns:
        ['Ticker', 'HedgeFundOwnership', 'RecentPerformance', 'ValuationZScore', 'SentimentScore']
        """
        self.df = data

    def generate_signals(self, hf_threshold=0.9, valuation_thresh=1.5, perf_thresh=0.15, sentiment_thresh=0.2):
        df = self.df.copy()

        # Criteria for being overcrowded:
        # 1. High hedge fund ownership
        # 2. Strong recent performance
        # 3. Overvalued (positive valuation Z-score)
        # 4. Deteriorating sentiment

        df["crowded_flag"] = (
            (df["HedgeFundOwnership"] > hf_threshold) &
            (df["RecentPerformance"] > perf_thresh) &
            (df["ValuationZScore"] > valuation_thresh) &
            (df["SentimentScore"] < sentiment_thresh)
        )

        df["signal"] = np.where(df["crowded_flag"], -1, 0)  # Short the crowded trades
        return df[["Ticker", "signal"]]

    def latest_signals(self):
        return self.generate_signals()


# Example usage
if __name__ == "__main__":
    example_data = pd.DataFrame({
        "Ticker": ["TSLA", "NVDA", "AAPL", "META", "AMZN"],
        "HedgeFundOwnership": [0.92, 0.95, 0.88, 0.91, 0.97],
        "RecentPerformance": [0.2, 0.18, 0.05, 0.15, 0.25],
        "ValuationZScore": [1.8, 2.1, 1.2, 1.6, 2.3],
        "SentimentScore": [0.1, 0.25, 0.4, 0.15, 0.05]
    })

    strat = CrowdedTradeUnwindAlpha(example_data)
    print(strat.latest_signals())