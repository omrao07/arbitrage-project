import yfinance as yf
from datetime import datetime, timedelta

class PostIPODriftAlpha:
    def __init__(self, ticker, ipo_date, window_days=30, momentum_threshold=0.10):
        self.ticker = ticker
        self.ipo_date = datetime.strptime(ipo_date, "%Y-%m-%d")
        self.window_days = window_days
        self.momentum_threshold = momentum_threshold
        self.signal = "HOLD"

    def fetch_price_data(self):
        """
        Fetch historical price data starting from IPO date to (IPO + window).
        """
        end_date = self.ipo_date + timedelta(days=self.window_days)
        df = yf.download(self.ticker, start=self.ipo_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
        return df["Close"] if "Close" in df.columns else None

    def generate_signal(self):
        prices = self.fetch_price_data()
        if prices is None or len(prices) < 2:
            return {
                "ticker": self.ticker,
                "signal": "NO DATA",
                "strategy": "Post-IPO Drift Alpha",
                "reason": "Insufficient price data"
            }

        pct_change = (prices[-1] - prices[0]) / prices[0]
        if pct_change > self.momentum_threshold:
            self.signal = "BUY"
        elif pct_change < -self.momentum_threshold:
            self.signal = "SELL"
        else:
            self.signal = "HOLD"

        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "return_since_ipo": round(pct_change * 100, 2),
            "days_tracked": self.window_days,
            "strategy": "Post-IPO Drift Alpha",
            "timestamp": datetime.utcnow().isoformat()
        }