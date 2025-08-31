from __future__ import annotations
import os, time, json
from typing import Optional, Dict
from dataclasses import dataclass
import redis
from backend.engine.strategy_base import Strategy

"""
Inflation Swap Arbitrage
------------------------
Compares:
  • Market 10y inflation swap rate (from swap quotes)
  • Breakeven inflation from nominal vs. inflation-linked bonds

If swap rate > breakeven + threshold → pay fixed / receive inflation
If swap rate < breakeven - threshold → receive fixed / pay inflation

Feeds:
  HSET infswap:rate <TENOR> <rate>
  HSET breakeven:<TENOR> <TENOR> <breakeven>
Rates in decimals (e.g., 0.025 for 2.5%).
"""

REDIS_HOST = os.getenv("INF_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("INF_REDIS_PORT", "6379"))
TENOR      = os.getenv("INF_TENOR", "10Y").upper()

ENTRY_BPS = float(os.getenv("INF_ENTRY_BPS", "15.0"))
EXIT_BPS  = float(os.getenv("INF_EXIT_BPS",  "5.0"))
USD_NOTIONAL = float(os.getenv("INF_USD_NOTIONAL", "1000000"))

RECHECK_SECS = int(os.getenv("INF_RECHECK_SECS", "60"))

SWAP_KEY_FMT = os.getenv("INF_SWAP_KEY", "infswap:rate")
BE_KEY_FMT   = os.getenv("INF_BE_KEY",   "breakeven")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _hgetf(hashkey: str, field: str) -> Optional[float]:
    val = r.hget(hashkey, field)
    if val is None:
        return None
    try: return float(val) # type: ignore
    except: return None

@dataclass
class OpenState:
    side: str
    entry_diff_bps: float
    ts: float

def _poskey(name: str) -> str:
    return f"inf_swap_arbitrage:open:{name}:{TENOR}"

class InflationSwapArbitrage(Strategy):
    def __init__(self, name: str = "inflation_swap_arbitrage", region: Optional[str] = None, default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    def _evaluate(self) -> None:
        swap_rate = _hgetf(SWAP_KEY_FMT, TENOR)
        breakeven = _hgetf(BE_KEY_FMT, TENOR)
        if swap_rate is None or breakeven is None:
            return

        diff_bps = (swap_rate - breakeven) * 1e4
        st = self._load_state()

        # Exit
        if st:
            if abs(diff_bps) <= EXIT_BPS:
                self._close(st)
            return

        # Entry
        if abs(diff_bps) >= ENTRY_BPS:
            if diff_bps > 0:
                # Swap rich → pay fixed, receive inflation
                self.order(f"INF_SWAP_{TENOR}", "sell", qty=USD_NOTIONAL, order_type="market", venue="SWAP")
                side = "pay_fixed_receive_infl"
            else:
                # Swap cheap → receive fixed, pay inflation
                self.order(f"INF_SWAP_{TENOR}", "buy", qty=USD_NOTIONAL, order_type="market", venue="SWAP")
                side = "receive_fixed_pay_infl"
            self._save_state(OpenState(side=side, entry_diff_bps=diff_bps, ts=time.time()))

    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw:
            return None
        try: return OpenState(**json.loads(raw)) # type: ignore
        except: return None

    def _save_state(self, st: OpenState) -> None:
        r.set(_poskey(self.ctx.name), json.dumps(st.__dict__))

    def _close(self, st: OpenState) -> None:
        if st.side == "pay_fixed_receive_infl":
            self.order(f"INF_SWAP_{TENOR}", "buy", qty=USD_NOTIONAL, order_type="market", venue="SWAP")
        else:
            self.order(f"INF_SWAP_{TENOR}", "sell", qty=USD_NOTIONAL, order_type="market", venue="SWAP")
        r.delete(_poskey(self.ctx.name))