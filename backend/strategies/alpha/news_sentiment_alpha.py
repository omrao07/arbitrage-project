import random
from datetime import datetime

class NewsSentimentAlpha:
    def __init__(self, ticker, sentiment_threshold=0.3):
        self.ticker = ticker
        self.sentiment_threshold = sentiment_threshold  # Positive/negative threshold
        self.headlines = []
        self.average_sentiment = 0.0
        self.signal = None

    def fetch_news_headlines(self):
        """
        Replace this with real-time scraping or news API like NewsAPI, Finnhub, or GNews.
        For now, it simulates headlines.
        """
        sample_headlines = [
            f"{self.ticker} surges on strong quarterly earnings",
            f"{self.ticker} faces regulatory investigation in Europe",
            f"{self.ticker} partners with major AI firm",
            f"{self.ticker} sees decline in smartphone shipments",
            f"{self.ticker} announces dividend hike amid strong growth"
        ]
        self.headlines = random.sample(sample_headlines, k=3)
        return self.headlines

    def analyze_sentiment(self, headline):
        """
        Simulated sentiment scoring (replace with real NLP model like VADER or BERT).
        Range: -1 (very negative) to +1 (very positive)
        """
        return round(random.uniform(-1, 1), 2)

    def generate_signal(self):
        self.fetch_news_headlines()
        sentiment_scores = [self.analyze_sentiment(h) for h in self.headlines]
        self.average_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 2)

        if self.average_sentiment > self.sentiment_threshold:
            self.signal = "BUY"
        elif self.average_sentiment < -self.sentiment_threshold:
            self.signal = "SELL"
        else:
            self.signal = "HOLD"

        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "average_sentiment": self.average_sentiment,
            "headlines": self.headlines,
            "strategy": "News Sentiment Alpha",
            "timestamp": datetime.utcnow().isoformat()
        }