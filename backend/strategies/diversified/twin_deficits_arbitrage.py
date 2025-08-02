import yfinance as yf
import numpy as np
from backend.utils.logger import log

class SkewArbitrage:
    """
    Strategy:
    - Compares implied volatility skew between puts and calls.
    - If skew exceeds threshold, go long/short volatility.
    """

    def __init__(self, symbol="AAPL", threshold=0.15):
        self.symbol = symbol
        self.threshold = threshold
        self.signal = {}

    def fetch_option_skew(self):
        try:
            ticker = yf.Ticker(self.symbol)
            expirations = ticker.options
            if not expirations:
                raise ValueError("No option data available.")

            opt_chain = ticker.option_chain(expirations[0])
            calls = opt_chain.calls
            puts = opt_chain.puts

            atm_strike = ticker.history(period="1d")["Close"].iloc[-1]
            atm_strike = round(atm_strike / 5) * 5  # nearest strike

            call_iv = calls.loc[calls["strike"] == atm_strike, "impliedVolatility"].mean()
            put_iv = puts.loc[puts["strike"] == atm_strike, "impliedVolatility"].mean()

            skew = (put_iv - call_iv) / call_iv if call_iv else 0
            return skew
        except Exception as e:
            log(f"[SkewArbitrage] Error fetching skew: {e}")
            return None

    def generate_signal(self):
        skew = self.fetch_option_skew()
        if skew is None:
            return {}

        if skew > self.threshold:
            decision = "LONG PUT VOL / SHORT CALL VOL"
        elif skew < -self.threshold:
            decision = "LONG CALL VOL / SHORT PUT VOL"
        else:
            decision = "NEUTRAL"

        self.signal = {
            "symbol": self.symbol,
            "skew": round(skew, 4),
            "signal": decision
        }
        return self.signal