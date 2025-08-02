from datetime import datetime
import random

class InsiderTransactionAlpha:
    def __init__(self, ticker, net_buy_threshold=500000):
        self.ticker = ticker
        self.net_buy_threshold = net_buy_threshold  # USD amount of net insider buying
        self.net_transaction = None
        self.signal = None

    def fetch_insider_transaction(self):
        """
        Simulates insider transaction net value.
        Replace this with real data using APIs like Finviz, OpenInsider, or SEC EDGAR scraping.
        """
        # Simulate net transaction amount in USD
        self.net_transaction = round(random.uniform(-2_000_000, 2_000_000), 2)
        return self.net_transaction

    def generate_signal(self):
        tx = self.fetch_insider_transaction()

        if tx >= self.net_buy_threshold:
            self.signal = "BUY"
        elif tx <= -self.net_buy_threshold:
            self.signal = "SELL"
        else:
            self.signal = "HOLD"

        return {
            'ticker': self.ticker,
            'signal': self.signal,
            'net_transaction_usd': self.net_transaction,
            'strategy': 'Insider Transaction Alpha',
            'timestamp': datetime.utcnow().isoformat()
        }