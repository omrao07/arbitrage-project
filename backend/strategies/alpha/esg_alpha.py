from datetime import datetime
import random

class ESGAlpha:
    def __init__(self, ticker, improvement_threshold=10):
        self.ticker = ticker
        self.improvement_threshold = improvement_threshold
        self.signal = None
        self.esg_score_change = None

    def fetch_esg_score_change(self):
        """
        Simulate ESG score change. Replace with real data from MSCI, Refinitiv, Sustainalytics, etc.
        """
        self.esg_score_change = round(random.uniform(-20, 20), 2)
        return self.esg_score_change

    def generate_signal(self):
        change = self.fetch_esg_score_change()

        if change >= self.improvement_threshold:
            self.signal = "BUY"
        elif change <= -self.improvement_threshold:
            self.signal = "SELL"
        else:
            self.signal = "HOLD"

        return {
            'ticker': self.ticker,
            'signal': self.signal,
            'esg_score_change': self.esg_score_change,
            'strategy': 'ESG Alpha',
            'timestamp': datetime.utcnow().isoformat()
        }