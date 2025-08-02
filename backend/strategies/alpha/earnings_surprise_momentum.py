import yfinance as yf
from datetime import datetime
import numpy as np

class EarningsSurpriseMomentum:
    def __init__(self, ticker, surprise_threshold=0.05, lookback_days=30):
        self.ticker = ticker
        self.surprise_threshold = surprise_threshold
        self.lookback_days = lookback_days
        self.signal = None
        self.surprise = None

    def fetch_earnings_surprise(self):
        """
        Simulates pulling earnings surprise data. Replace with actual API (e.g., Zacks, FactSet) if needed.
        """
        # Placeholder surprise: (Actual - Estimate) / Estimate
        import random
        self.surprise = round(random.uniform(-0.2, 0.2), 3)
        return self.surprise

    def generate_signal(self):
        self.fetch_earnings_surprise()

        if self.surprise >= self.surprise_threshold:
            self.signal = 'BUY'
        elif self.surprise <= -self.surprise_threshold:
            self.signal = 'SELL'
        else:
            self.signal = 'HOLD'

        return {
            'ticker': self.ticker,
            'signal': self.signal,
            'surprise': self.surprise,
            'strategy': 'Earnings Surprise Momentum',
            'timestamp': datetime.utcnow().isoformat()
        }