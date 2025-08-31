# backend/strategies/seasonality_intraday.py
from __future__ import annotations

import os
import time
import json
import math
from typing import Dict, Any, Optional, List

import redis
from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset

# ---------- Redis Setup ----------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class SeasonalityIntraday(Strategy):
    """
    Strategy that exploits intraday seasonality patterns:
      - Market often has predictable flows (open auction volatility, lunch lull, close ramp).
      - Uses historical average return patterns per minute/hour of day.
      - Generates long/short signals when current price deviates vs seasonality expectation.
    """

    def __init__(self, name="seasonality_intraday", region=None, symbol="SPY", lookback_days=60):
        super().__init__(name=name, region=region)
        self.symbol = symbol.upper()
        self.lookback_days = lookback_days

        # Cache of intraday seasonality {minute_of_day: avg_return}
        self.seasonality: Dict[int, float] = {}
        # Rolling state
        self.last_price: Optional[float] = None
        self.intraday_returns: Dict[int, List[float]] = {}

    # ---------------- Lifecycle ----------------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["alpha", "intraday", "seasonality"],
            "notes": "Exploits predictable intraday return seasonality (open, lunch lull, close ramp)."
        })

    # ---------------- Helpers ----------------
    def _minute_of_day(self, ts: int) -> int:
        """Convert unix timestamp (ms) to minute of trading day (0 = open)."""
        tm = time.gmtime(ts // 1000)  # use UTC unless you want local tz logic
        return tm.tm_hour * 60 + tm.tm_min

    def _update_seasonality(self, minute: int, ret: float) -> None:
        """Update rolling average return per minute."""
        if minute not in self.intraday_returns:
            self.intraday_returns[minute] = []
        self.intraday_returns[minute].append(ret)

        # keep last N days
        if len(self.intraday_returns[minute]) > self.lookback_days:
            self.intraday_returns[minute] = self.intraday_returns[minute][-self.lookback_days:]

        # update average
        self.seasonality[minute] = sum(self.intraday_returns[minute]) / len(self.intraday_returns[minute])

    def _expected_return(self, minute: int) -> float:
        """Expected return based on seasonality curve."""
        return self.seasonality.get(minute, 0.0)

    # ---------------- Core ----------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym != self.symbol:
            return

        price = tick.get("price") or tick.get("p") or tick.get("mid")
        if price is None:
            bid, ask = tick.get("bid"), tick.get("ask")
            if bid and ask:
                price = 0.5 * (float(bid) + float(ask))
        try:
            price = float(price) # type: ignore
        except Exception:
            return

        ts = int(tick.get("ts", time.time() * 1000))
        minute = self._minute_of_day(ts)

        # update rolling return
        if self.last_price:
            ret = (price / self.last_price - 1.0)
            self._update_seasonality(minute, ret)
        self.last_price = price

        # expected seasonal return
        exp_ret = self._expected_return(minute)

        # signal: go long if return is below seasonal expectation, short if above
        signal_score = exp_ret - (0.0 if self.last_price is None else 0.0)
        self.emit_signal(signal_score)

        # execution logic: buy if price under seasonal exp, sell if above
        if exp_ret > 0 and signal_score > 0.0001:
            self.order(self.symbol, "buy", qty=1, order_type="market", mark_price=price,
                       extra={"reason": "seasonality_long", "exp_ret": exp_ret})
        elif exp_ret < 0 and signal_score < -0.0001:
            self.order(self.symbol, "sell", qty=1, order_type="market", mark_price=price,
                       extra={"reason": "seasonality_short", "exp_ret": exp_ret})


# ---------------- Optional Runner ----------------
if __name__ == "__main__":
    """
    Example usage:
      export REDIS_HOST=localhost REDIS_PORT=6379
      python -m backend.strategies.seasonality_intraday
    """
    strat = SeasonalityIntraday()
    # strat.run(stream="ticks.equities.us")