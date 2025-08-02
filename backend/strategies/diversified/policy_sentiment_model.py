import pandas as pd
import numpy as np
from backend.utils.logger import log
from backend.models.nlp_model import PolicySentimentAnalyzer

class PolicySentimentAlpha:
    """
    Strategy:
    - Uses NLP to parse policy-related speeches (e.g., Fed, RBI, ECB).
    - Scores sentiment and goes LONG risk assets if policy tone is dovish,
      SHORT if hawkish.
    """

    def __init__(self, speech_json_path="backend/data/fed_speeches.json", threshold=0.2):
        self.speech_json_path = speech_json_path
        self.threshold = threshold
        self.signals = {}
        self.analyzer = PolicySentimentAnalyzer()

    def load_speeches(self):
        try:
            self.speeches = pd.read_json(self.speech_json_path)
            self.speeches["date"] = pd.to_datetime(self.speeches["date"])
        except Exception as e:
            log(f"[PolicySentimentAlpha] Failed to load speech data: {e}")
            self.speeches = pd.DataFrame()

    def compute_signals(self):
        self.signals = {}
        if self.speeches.empty:
            return self.signals

        for _, row in self.speeches.iterrows():
            date = row["date"].strftime("%Y-%m-%d")
            text = row["speech"]
            sentiment = self.analyzer.analyze(text)

            if sentiment >= self.threshold:
                signal = "LONG"
            elif sentiment <= -self.threshold:
                signal = "SHORT"
            else:
                signal = "NEUTRAL"

            self.signals[date] = {
                "SentimentScore": round(sentiment, 3),
                "Signal": signal
            }

        return self.signals

    def latest_signal(self):
        if not self.signals:
            return None
        latest_date = max(self.signals.keys())
        return {latest_date: self.signals[latest_date]}

# Example usage
if __name__ == "__main__":
    strat = PolicySentimentAlpha()
    strat.load_speeches()
    strat.compute_signals()
    print("Latest signal:", strat.latest_signal())