import yfinance as yf
import numpy as np
from backend.utils.logger import log

class SkewArbitrage:
    """
    Skew Arbitrage Strategy:
    Detects and trades mispricing between call and put implied volatilities.
    If the implied volatility skew exceeds a threshold, it signals overpricing in puts or calls.
    Typically executed using options spreads (not traded here directly).
    """

    def __init__(self, ticker="SPY", threshold=0.15):
        self.ticker = ticker
        self.threshold = threshold
        self.signal = {}

    def fetch_option_chain(self):
        try:
            stock = yf.Ticker(self.ticker)
            expirations = stock.options
            if not expirations:
                raise ValueError("No options data available")

            chain = stock.option_chain(expirations[0])
            return chain.calls, chain.puts
        except Exception as e:
            log(f"[SkewArbitrage] Error fetching options: {e}")
            return None, None

    def compute_skew(self, calls, puts):
        try:
            call_iv = calls["impliedVolatility"].dropna()
            put_iv = puts["impliedVolatility"].dropna()
            if call_iv.empty or put_iv.empty:
                return None
            skew = np.mean(put_iv) - np.mean(call_iv)
            return skew
        except Exception as e:
            log(f"[SkewArbitrage] Skew calculation error: {e}")
            return None

    def generate_signal(self):
        calls, puts = self.fetch_option_chain()
        if calls is None or puts is None:
            return {}