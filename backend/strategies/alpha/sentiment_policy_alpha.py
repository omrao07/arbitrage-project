import datetime
import random

class SentimentPolicyAlpha:
    def __init__(self):
        self.hawkish_keywords = [
            "tightening", "inflation", "rate hike", "overheating", "restrictive", "balance sheet reduction"
        ]
        self.dovish_keywords = [
            "easing", "accommodation", "rate cut", "stimulus", "quantitative easing", "supportive"
        ]
        self.signal = "HOLD"

    def fetch_latest_policy_text(self):
        """
        Placeholder method – in production this should use an NLP service/API that fetches
        Fed or ECB speeches, FOMC statements, press conferences, etc.
        """
        # Simulate a policy text for now (e.g., sample from database or news API)
        sample_texts = [
            "The Fed continues to monitor inflation closely and is prepared to raise rates if necessary.",
            "Central banks remain supportive and may continue asset purchases.",
            "Policy remains accommodative given uncertainty in growth outlook.",
            "We are moving toward a more neutral stance and may reduce the balance sheet."
        ]
        return random.choice(sample_texts)

    def analyze_sentiment(self, text):
        """
        Simple keyword matching sentiment score.
        """
        text_lower = text.lower()
        hawkish_score = sum(word in text_lower for word in self.hawkish_keywords)
        dovish_score = sum(word in text_lower for word in self.dovish_keywords)

        sentiment_score = hawkish_score - dovish_score
        return sentiment_score

    def generate_signal(self):
        policy_text = self.fetch_latest_policy_text()
        sentiment_score = self.analyze_sentiment(policy_text)

        if sentiment_score > 1:
            self.signal = "SELL"  # Hawkish policy expected to hurt risk assets
        elif sentiment_score < -1:
            self.signal = "BUY"   # Dovish policy supports markets
        else:
            self.signal = "HOLD"

        return {
            "strategy": "Sentiment Policy Alpha",
            "signal": self.signal,
            "sentiment_score": sentiment_score,
            "policy_excerpt": policy_text,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }