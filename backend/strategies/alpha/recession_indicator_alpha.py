import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class RecessionIndicatorAlpha:
    def __init__(self, lookback_days=180, yield_threshold=0.0, unemployment_threshold=0.05):
        self.lookback_days = lookback_days
        self.yield_threshold = yield_threshold
        self.unemployment_threshold = unemployment_threshold
        self.signal = "HOLD"

    def fetch_yield_curve_data(self):
        """
        Fetch 10Y and 2Y US Treasury yields to check inversion.
        """
        today = datetime.today()
        start = (today - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        end = today.strftime('%Y-%m-%d')

        ten_year = yf.download("^TNX", start=start, end=end, progress=False)["Close"] / 100
        two_year = yf.download("^IRX", start=start, end=end, progress=False)["Close"] / 100

        return ten_year, two_year

    def fetch_unemployment_rate(self):
        """
        Get recent US unemployment rate (macro proxy using FRED ETF alternative).
        Replace with real macro feed in prod.
        """
        try:
            # Proxy using labor market ETF (e.g., "PSJ" tech employment-related)
            data = yf.download("PSJ", period="6mo", interval="1wk", progress=False)["Close"]
            pct_change = (data[-1] - data[0]) / data[0]
            proxy_unemployment_rate = 0.04 + (pct_change * -0.5)  # Inverse proxy
            return max(min(proxy_unemployment_rate, 0.15), 0.03)
        except:
            return None

    def generate_signal(self):
        ten_year, two_year = self.fetch_yield_curve_data()
        unemployment = self.fetch_unemployment_rate()

        if ten_year.empty or two_year.empty or unemployment is None:
            return {
                "strategy": "Recession Indicator Alpha",
                "signal": "NO DATA",
                "reason": "Missing yield curve or unemployment data"
            }

        latest_spread = ten_year.iloc[-1] - two_year.iloc[-1]
        recession_risk = (latest_spread < self.yield_threshold) and (unemployment > self.unemployment_threshold)

        if recession_risk:
            self.signal = "SELL"
        elif latest_spread > 0.5 and unemployment < 0.045:
            self.signal = "BUY"
        else:
            self.signal = "HOLD"

        return {
            "strategy": "Recession Indicator Alpha",
            "signal": self.signal,
            "yield_spread": round(latest_spread, 4),
            "unemployment_rate_est": round(unemployment, 4),
            "timestamp": datetime.utcnow().isoformat()
        }