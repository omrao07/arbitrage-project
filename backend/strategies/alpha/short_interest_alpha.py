import datetime
import random

class ShortInterestAlpha:
    def __init__(self, threshold_low=5.0, threshold_high=20.0):
        self.threshold_low = threshold_low  # Below this % = bullish
        self.threshold_high = threshold_high  # Above this % = bearish
        self.signal = "HOLD"

    def fetch_short_interest_data(self):
        """
        Simulate fetching short interest % of float.
        In production: Pull from APIs like Nasdaq, FINRA, or alternative data providers.
        """
        # Simulated short interest % for an example ticker
        return {
            "ticker": "XYZ",
            "short_interest_percent": round(random.uniform(1.0, 30.0), 2)
        }

    def generate_signal(self):
        data = self.fetch_short_interest_data()
        short_interest = data["short_interest_percent"]

        if short_interest < self.threshold_low:
            self.signal = "BUY"  # Low short interest → potential bullish bias
        elif short_interest > self.threshold_high:
            self.signal = "SELL"  # High short interest → risk of short squeeze OR bearish signal
        else:
            self.signal = "HOLD"

        return {
            "strategy": "Short Interest Alpha",
            "signal": self.signal,
            "ticker": data["ticker"],
            "short_interest_percent": short_interest,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }