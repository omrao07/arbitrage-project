import datetime
import random

class WebTrafficAlpha:
    def __init__(self, threshold=0.15):
        self.threshold = threshold  # +15% weekly web traffic increase
        self.signal = "HOLD"

    def fetch_web_traffic_data(self):
        """
        Simulated fetch of web/app traffic change for a stock.
        In production: Use APIs like SimilarWeb, Google Trends, or your own analytics feed.
        """
        return {
            "ticker": "AMZN",
            "traffic_change_pct": round(random.uniform(-0.20, 0.30), 4),  # -20% to +30%
            "date": str(datetime.date.today())
        }

    def generate_signal(self):
        data = self.fetch_web_traffic_data()
        change = data["traffic_change_pct"]

        if change > self.threshold:
            self.signal = "BUY"
        elif change < -self.threshold:
            self.signal = "SELL"
        else:
            self.signal = "HOLD"

        return {
            "strategy": "Web Traffic Alpha",
            "signal": self.signal,
            "ticker": data["ticker"],
            "traffic_change_pct": change,
            "date": data["date"],
            "timestamp": datetime.datetime.utcnow().isoformat()
        }