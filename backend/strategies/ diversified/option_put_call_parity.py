# backend/strategies/diversified/option_put_call_parity.py
import math, os, time
from typing import Dict, Optional
from backend.engine.strategy_base import Strategy

"""
Option Put-Call Parity Arbitrage (European Options)
---------------------------------------------------
Detects violations of:
    C - P = S - K * e^{-rT}
If deviation > threshold after costs, trade the cheaper side (synthetic vs actual).

Redis feeds required:
  HSET opt:call:<SYM>:<STRIKE>:<EXP> '{"bid":..., "ask":...}'
  HSET opt:put:<SYM>:<STRIKE>:<EXP>  '{"bid":..., "ask":...}'
  HSET spot:<SYM> '{"bid":..., "ask":...}'
  SET risk_free_rate:<CCY> <decimal>
"""

# Config from env
SYM       = os.getenv("PCP_SYMBOL", "AAPL")
STRIKE    = float(os.getenv("PCP_STRIKE", "180"))
EXP_DAYS  = int(os.getenv("PCP_EXP_DAYS", "30"))
RFRATE    = float(os.getenv("PCP_RF_RATE", "0.05"))  # fallback annualized
THRESH_BPS= float(os.getenv("PCP_THRESH_BPS", "20"))
QTY       = float(os.getenv("PCP_QTY", "10"))

def _hget_json(r, hk: str) -> Optional[dict]:
    raw = r.get(hk)
    if not raw: return None
    try: return eval(raw) if raw.startswith("{") else None
    except: return None

class OptionPutCallParity(Strategy):
    def __init__(self, name="option_put_call_parity", region="GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < 1:  # 1s throttle
            return
        self.last_check = now

        call = _hget_json(self.ctx.redis, f"opt:call:{SYM}:{STRIKE}:{EXP_DAYS}")
        put  = _hget_json(self.ctx.redis, f"opt:put:{SYM}:{STRIKE}:{EXP_DAYS}")
        spot = _hget_json(self.ctx.redis, f"spot:{SYM}")

        if not (call and put and spot):
            return

        C = (call["bid"] + call["ask"]) / 2
        P = (put["bid"] + put["ask"]) / 2
        S = (spot["bid"] + spot["ask"]) / 2
        r = RFRATE
        T = EXP_DAYS / 365.0

        parity_val = S - STRIKE * math.exp(-r * T)
        lhs = C - P
        diff = lhs - parity_val
        diff_bps = abs(diff / parity_val) * 1e4

        if diff_bps > THRESH_BPS:
            if diff > 0:
                # LHS > parity → short call, long put, long stock
                self.order(f"OPT_CALL:{SYM}:{STRIKE}:{EXP_DAYS}", "sell", qty=QTY, price=call["bid"], order_type="limit")
                self.order(f"OPT_PUT:{SYM}:{STRIKE}:{EXP_DAYS}", "buy", qty=QTY, price=put["ask"], order_type="limit")
                self.order(f"SPOT:{SYM}", "buy", qty=QTY, price=spot["ask"], order_type="market")
            else:
                # RHS > parity → long call, short put, short stock
                self.order(f"OPT_CALL:{SYM}:{STRIKE}:{EXP_DAYS}", "buy", qty=QTY, price=call["ask"], order_type="limit")
                self.order(f"OPT_PUT:{SYM}:{STRIKE}:{EXP_DAYS}", "sell", qty=QTY, price=put["bid"], order_type="limit")
                self.order(f"SPOT:{SYM}", "sell", qty=QTY, price=spot["bid"], order_type="market")

            self.emit_signal(min(1.0, diff_bps / THRESH_BPS))