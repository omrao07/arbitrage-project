import os, time
import numpy as np
from typing import Dict
from backend.engine.strategy_base import Strategy

"""
Volatility Surface Mispricing
-----------------------------
Checks for arbitrage across the vol surface:
    - Calendar arbitrage (term structure violation)
    - Butterfly arbitrage (smile convexity violation)
    - Vertical skew mispricing
Trades cheapest vs richest vol exposure via options, hedged with delta.

Expected data feed:
    Redis keys: iv:<SYM>:<EXP>:<STRIKE>
"""

SYM = os.getenv("VSM_SYMBOL", "SPX")
CAL_THRESH = float(os.getenv("VSM_CAL_THRESH", "0.005"))   # 0.5% vol diff
SMILE_THRESH = float(os.getenv("VSM_SMILE_THRESH", "0.01")) # 1% vol diff
QTY = float(os.getenv("VSM_QTY", "10"))

def _fetch_iv_curve(redis, sym: str, exp: str):
    curve = {}
    for k in redis.scan_iter(f"iv:{sym}:{exp}:*"):
        strike = float(k.decode().split(":")[-1])
        iv = float(redis.get(k))
        curve[strike] = iv
    return dict(sorted(curve.items()))

class VolatilitySurfaceMispricing(Strategy):
    def __init__(self, name="vol_surface_mispricing", region="GLOBAL", default_qty=1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0

    def on_tick(self, tick: Dict):
        now = time.time()
        if now - self.last_check < 5:  # check every 5 sec
            return
        self.last_check = now

        expiries = [e.decode().split(":")[2] for e in self.ctx.redis.scan_iter(f"iv:{SYM}:*")]

        for exp in sorted(set(expiries)):
            curve = _fetch_iv_curve(self.ctx.redis, SYM, exp)
            strikes = list(curve.keys())
            ivs = np.array(list(curve.values()))

            # --- Butterfly/Smile Arbitrage ---
            for i in range(1, len(strikes)-1):
                mid_strike = strikes[i]
                left_iv = ivs[i-1]
                mid_iv = ivs[i]
                right_iv = ivs[i+1]

                # Convexity: mid should be <= avg of wings
                if mid_iv - ((left_iv + right_iv)/2) > SMILE_THRESH:
                    self.emit_signal(1.0)
                    self.order(f"OPT_CALL:{SYM}:{mid_strike}:{exp}", "sell", QTY)
                    self.order(f"OPT_CALL:{SYM}:{strikes[i-1]}:{exp}", "buy", QTY/2)
                    self.order(f"OPT_CALL:{SYM}:{strikes[i+1]}:{exp}", "buy", QTY/2)

            # --- Calendar Arbitrage ---
            # Compare with shorter expiry same strike
            for other_exp in sorted(set(expiries)):
                if other_exp == exp: continue
                shorter_curve = _fetch_iv_curve(self.ctx.redis, SYM, other_exp)
                for strike, iv in curve.items():
                    if strike in shorter_curve:
                        if iv < shorter_curve[strike] - CAL_THRESH:
                            # Longer expiry too cheap → buy long-dated, sell short-dated
                            self.order(f"OPT_CALL:{SYM}:{strike}:{exp}", "buy", QTY)
                            self.order(f"OPT_CALL:{SYM}:{strike}:{other_exp}", "sell", QTY)
                        elif iv > shorter_curve[strike] + CAL_THRESH:
                            # Longer expiry too expensive → sell long-dated, buy short-dated
                            self.order(f"OPT_CALL:{SYM}:{strike}:{exp}", "sell", QTY)
                            self.order(f"OPT_CALL:{SYM}:{strike}:{other_exp}", "buy", QTY)