import random
from datetime import datetime

class NLPSocialAlpha:
    def __init__(self, ticker, threshold=0.2):
        self.ticker = ticker
        self.threshold = threshold
        self.tweets = []
        self.avg_sentiment = 0.0
        self.signal = None

    def fetch_social_data(self):
        """
        Simulates fetching social media posts mentioning the ticker.
        Replace this with real data from Twitter, Reddit (via Pushshift or Tweepy).
        """
        mock_posts = [
            f"{self.ticker} to the moon!",
            f"Bearish on {self.ticker}, weak results.",
            f"{self.ticker} just crushed earnings!",
            f"Too many headwinds for {self.ticker}",
            f"Holding {self.ticker} long term, strong fundamentals"
        ]
        self.tweets = random.sample(mock_posts, k=3)
        return self.tweets

    def analyze_sentiment(self, post):
        """
        Simulated NLP sentiment scoring (replace with real model like VADER, TextBlob, or FinBERT).
        Returns sentiment score between -1 and +1.
        """
        return round(random.uniform(-1, 1), 2)

    def generate_signal(self):
        self.fetch_social_data()
        sentiment_scores = [self.analyze_sentiment(post) for post in self.tweets]
        self.avg_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 2)

        if self.avg_sentiment > self.threshold:
            self.signal = "BUY"
        elif self.avg_sentiment < -self.threshold:
            self.signal = "SELL"
        else:
            self.signal = "HOLD"

        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "average_sentiment": self.avg_sentiment,
            "posts": self.tweets,
            "strategy": "NLP Social Alpha",
            "timestamp": datetime.utcnow().isoformat()
        }