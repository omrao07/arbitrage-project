import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from backend.utils.logger import log

class VolatilityArbitrage:
    """
    Strategy:
    - Compares implied vs. historical (realized) volatility.
    - Long volatility if implied < historical.
    - Short volatility if implied > historical.
    """

    def __init__(self, symbol="AAPL", lookback_days=30, vol_threshold=0.15):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.vol_threshold = vol_threshold
        self.signal = {}

    def get_historical_volatility(self):
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=self.lookback_days)
            df = yf.download(self.symbol, start=start_date, end=end_date)
            if df.empty or len(df) < 10:
                raise ValueError("Not enough data for HV calculation")

            df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
            volatility = np.std(df["log_ret"].dropna()) * np.sqrt(252)
            return round(volatility, 4)
        except Exception as e:
            log(f"[VolArb] HV fetch error: {e}")
            return None

    def get_implied_volatility(self):
        try:
            ticker = yf.Ticker(self.symbol)
            expiries = ticker.options
            if not expiries:
                raise ValueError("No option data")

            option_chain = ticker.option_chain(expiries[0])
            calls = option_chain.calls
            puts = option_chain.puts

            atm_strike = round(ticker.history(period="1d")["Close"].iloc[-1] / 5) * 5
            atm_call_iv = calls[calls["strike"] == atm_strike]["impliedVolatility"].mean()
            atm_put_iv = puts[puts["strike"] == atm_strike]["impliedVolatility"].mean()

            iv = np.mean([atm_call_iv, atm_put_iv])
            return round(iv, 4) if not np.isnan(iv) else None
        except Exception as e:
            log(f"[VolArb] IV fetch error: {e}")
            return None

    def generate_signal(self):
        hv = self.get_historical_volatility()
        iv = self.get_implied_volatility()

        if hv is None or iv is None:
            self.signal = {"error": "Data unavailable"}
            return self.signal

        delta = iv - hv
        if delta > self.vol_threshold:
            decision = "SHORT VOL"
        elif delta < -self.vol_threshold:
            decision = "LONG VOL"
        else:
            decision = "NEUTRAL"

        self.signal = {
            "symbol": self.symbol,
            "implied_vol": iv,
            "realized_vol": hv,
            "vol_diff": round(delta, 4),
            "signal": decision
        }
        return self.signal