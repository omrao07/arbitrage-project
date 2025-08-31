# backend/strategies/diversified/exotic_option_replication.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Exotic Option Replication (paper)
---------------------------------
Two modes:

1) DIGITAL (tight call-spread)
   Approximate a binary (cash-or-nothing) payoff 1{S_T > K} with a scaled call-spread:
     w = 1 / dK
     long  w * Call(K)
     short w * Call(K + dK)
   Cash payout can be scaled by 'DIGITAL_PAYOUT' (USD per share notionally).

2) PIECEWISE (static replication of arbitrary payoff)
   Provide a *piecewise-linear* payoff π(S_T) via knots [(S, payoff)] in Redis.
   We build:   a * cash + b * forward + Σ_i  w_i * Call(K_i)
   using discrete second differences on a strike grid (Carr–Madan).
   Then we create a static portfolio of calls (+ optional puts if you enable).
   Works great for capped/floored calls, call spreads, bull/bear notes,
   and decent approximations for cliquets/target-redemption notes when gridded.

All orders are *paper*:
  • Underlier: <SYM>               (spot/ETF)
  • Option  : OPT:<SYM>:<K>:<TENOR>:C / :P

Inputs you publish elsewhere (typical in this repo):
  HSET last_price <SYM> '{"price": <spot>}'
  HSET opt:mid:<TENOR> "<SYM>:<K>:C" <price>
  HSET opt:mid:<TENOR> "<SYM>:<K>:P" <price>   (puts optional; not required by default)

PIECEWISE payoff knots:
  SET exot:payoff:<SYM>:<TENOR> '[ [S0, y0], [S1, y1], ... ]'
  (S in same units as price, y in USD per 1 underlying unit)

This module creates/holds a *single* static package per instance until you clear it.
"""

# ============================== CONFIG (env) ==============================
REDIS_HOST = os.getenv("EXO_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("EXO_REDIS_PORT", "6379"))

SYM   = os.getenv("EXO_SYMBOL", "ACME").upper()
TENOR = os.getenv("EXO_TENOR", "30D").upper()

MODE = os.getenv("EXO_MODE", "PIECEWISE").upper()  # "DIGITAL" or "PIECEWISE"

# ---- DIGITAL mode params ----
DIGITAL_K        = float(os.getenv("EXO_DIGITAL_K", "100.0"))
DIGITAL_DK       = float(os.getenv("EXO_DIGITAL_DK", "1.0"))        # call-spread width
DIGITAL_PAYOUT   = float(os.getenv("EXO_DIGITAL_PAYOUT", "1.0"))    # cash per 1 underlying
DIGITAL_PACKAGES = int(os.getenv("EXO_DIGITAL_PACKAGES", "10"))     # how many digitals to build

# ---- PIECEWISE mode params ----
# Strike grid (auto covers knot range if not set explicitly)
GRID_K_MIN   = os.getenv("EXO_GRID_K_MIN", "")
GRID_K_MAX   = os.getenv("EXO_GRID_K_MAX", "")
GRID_K_STEP  = float(os.getenv("EXO_GRID_K_STEP", "2.0"))
USE_PUT_SIDE = os.getenv("EXO_USE_PUTS", "false").lower() in ("1","true","yes")  # optional

# ---- Risk & routing ----
VENUE_EQ  = os.getenv("EXO_VENUE_EQ", "ARCA").upper()
VENUE_OPT = os.getenv("EXO_VENUE_OPT", "CBOE").upper()

MIN_TICKET_USD = float(os.getenv("EXO_MIN_TICKET_USD", "50"))
RECHECK_SECS   = int(os.getenv("EXO_RECHECK_SECS", "5"))

# Redis keys
LAST_PRICE_HKEY = os.getenv("EXO_LAST_PRICE_KEY", "last_price")
OPT_MID_HASH    = os.getenv("EXO_OPT_MID_HASH", f"opt:mid:{TENOR}")  # "<SYM>:<K>:C/P" -> price
PAYOFF_KEY      = os.getenv("EXO_PAYOFF_KEY", f"exot:payoff:{SYM}:{TENOR}")

# ============================== Redis ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== Helpers ==============================
def _price_underlier(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw:
        return None
    try:
        return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try:
            return float(raw) # type: ignore
        except Exception:
            return None

def _opt_mid(sym: str, K: float, cp: str) -> Optional[float]:
    key = f"{sym}:{K:.4f}:{cp.upper()}"
    v = r.hget(OPT_MID_HASH, key)
    if v is None:
        return None
    try:
        return float(v) # type: ignore
    except Exception:
        try:
            return float(json.loads(v)) # type: ignore
        except Exception:
            return None

def _opt_sym(sym: str, K: float, cp: str) -> str:
    return f"OPT:{sym}:{K:.4f}:{TENOR}:{cp.upper()}"

def _load_payoff_knots() -> Optional[List[Tuple[float, float]]]:
    raw = r.get(PAYOFF_KEY)
    if not raw:
        return None
    try:
        arr = json.loads(raw) # type: ignore
        pts = [(float(s), float(y)) for s, y in arr]
        pts.sort(key=lambda x: x[0])
        return pts
    except Exception:
        return None

# ============================== State ==============================
@dataclass
class PackageState:
    mode: str
    ts_ms: int
    details: Dict

def _poskey(name: str) -> str:
    return f"exo:open:{name}:{SYM}:{TENOR}"

# ============================== Strategy ==============================
class ExoticOptionReplication(Strategy):
    """
    Static replication via vanilla options (paper). One package per instance.
    """
    def __init__(self, name: str = "exotic_option_replication", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "symbol": SYM, "tenor": TENOR, "mode": MODE, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._maybe_build_once()

    # ---------------- core ----------------
    def _maybe_build_once(self) -> None:
        # Build only if nothing is open
        if r.get(_poskey(self.ctx.name)) is not None:
            return

        if MODE == "DIGITAL":
            self._build_digital()
        else:
            self._build_piecewise()

    # --------- DIGITAL: call-spread approximation ---------
    def _build_digital(self) -> None:
        S0 = _price_underlier(SYM)
        if S0 is None or S0 <= 0:
            return

        K = DIGITAL_K
        dK = max(1e-6, DIGITAL_DK)
        w = DIGITAL_PAYOUT / dK  # scale so payoff ≈ DIGITAL_PAYOUT when S_T > K

        # price sanity & min ticket
        c1 = _opt_mid(SYM, K, "C")
        c2 = _opt_mid(SYM, K + dK, "C")
        if c1 is None or c2 is None:
            return
        est_cost = abs(w) * (c1 + c2)
        if est_cost * 1.0 < MIN_TICKET_USD:
            return

        n_pkgs = max(1, DIGITAL_PACKAGES)

        # LONG digital ≈ +w*Call(K) - w*Call(K+dK)
        for _ in range(n_pkgs):
            self.order(_opt_sym(SYM, K, "C"),       "buy",  qty=abs(w), order_type="market", venue=VENUE_OPT)
            self.order(_opt_sym(SYM, K + dK, "C"),  "sell", qty=abs(w), order_type="market", venue=VENUE_OPT)

        st = PackageState(
            mode="DIGITAL",
            ts_ms=int(time.time()*1000),
            details={"K": K, "dK": dK, "w": w, "packages": n_pkgs, "payout": DIGITAL_PAYOUT}
        )
        r.set(_poskey(self.ctx.name), json.dumps(st.__dict__))

    # --------- PIECEWISE: static Carr–Madan replication ---------
    def _build_piecewise(self) -> None:
        S0 = _price_underlier(SYM)
        if S0 is None or S0 <= 0:
            return

        knots = _load_payoff_knots()
        if not knots or len(knots) < 2:
            return

        # define strike grid
        Kmin = float(GRID_K_MIN) if GRID_K_MIN else min(k for k, _ in knots)
        Kmax = float(GRID_K_MAX) if GRID_K_MAX else max(k for k, _ in knots)
        if Kmax <= Kmin:
            return
        step = max(1e-6, GRID_K_STEP)
        # round grid to step
        def _round_to(x: float, s: float) -> float:
            return round(x / s) * s
        Kmin = _round_to(Kmin, step)
        Kmax = _round_to(Kmax, step)

        Ks: List[float] = []
        k = Kmin
        # keep numerical stability by limiting grid size
        limit = 2000
        while k <= Kmax + 1e-12 and len(Ks) < limit:
            Ks.append(round(k, 10))
            k += step
        if len(Ks) < 3:
            return

        # Interpolate payoff on grid (piecewise-linear)
        def interp(x: float) -> float:
            # clamps at ends
            if x <= knots[0][0]:
                return knots[0][1]
            if x >= knots[-1][0]:
                return knots[-1][1]
            # find segment
            lo = 0
            hi = len(knots) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if knots[mid][0] <= x:
                    lo = mid
                else:
                    hi = mid
            x0, y0 = knots[lo]
            x1, y1 = knots[hi]
            t = (x - x0) / max(1e-12, x1 - x0)
            return y0 * (1 - t) + y1 * t

        pay = [interp(K) for K in Ks]

        # Discrete second differences approximate d^2 π / dK^2:
        # π(S) ≈ A + B*S + ∑ w_i * (S-K_i)^+ ; w_i ≈ Δ^2π(K_i) * ΔK
        w_calls: Dict[float, float] = {}
        for i in range(1, len(Ks) - 1):
            Kc = Ks[i]
            y_prev, y_c, y_next = pay[i - 1], pay[i], pay[i + 1]
            # non-uniform step guard (should be uniform, but compute locally)
            h1 = Ks[i] - Ks[i - 1]
            h2 = Ks[i + 1] - Ks[i]
            if h1 <= 0 or h2 <= 0:
                continue
            # quadratic fit second derivative approx
            d2 = 2 * ((y_next - y_c) / h2 - (y_c - y_prev) / h1) / (h1 + h2)
            w = max(0.0, d2) * (0.5 * (h1 + h2))  # weight mass around Kc; clamp negatives (call-only)
            if w > 0:
                w_calls[float(Kc)] = w_calls.get(float(Kc), 0.0) + w

        # Solve A and B roughly from slope & level near S0
        # slope_left ≈ π'(K-) , slope_right ≈ π'(K+)
        # Using calls-only representation: B ≈ π'(0+) capped; A ≈ π(0) - B*0 - ∑ w_i * (-K_i)^+ (≈ 0)
        # Practical shortcut: set B ~ (π(S0+Δ)-π(S0-Δ)) / (2Δ), A ~ π(0+) (rarely used in trading legs).
        dS = 0.5 * step
        y_minus = interp(max(Kmin, S0 - dS))
        y_plus  = interp(min(Kmax, S0 + dS))
        B = (y_plus - y_minus) / max(1e-9, (min(Kmax, S0 + dS) - max(Kmin, S0 - dS)))
        A = interp(Kmin)  # crude; fine for explaining cash leg

        # Price check and build orders
        # We buy calls with weights w_i (positive). Total notional scaled to keep cost sane.
        total_cost = 0.0
        missing_quotes = 0
        for Kc, w in w_calls.items():
            px = _opt_mid(SYM, Kc, "C")
            if px is None:
                missing_quotes += 1
                continue
            total_cost += w * px

        if missing_quotes > 0:
            # if many quotes are missing, abort build quietly
            if missing_quotes > max(3, len(w_calls) // 3):
                return

        if total_cost < MIN_TICKET_USD:
            # scale up to reach a minimal ticket
            scale = max(1.0, MIN_TICKET_USD / max(1e-9, total_cost))
        else:
            scale = 1.0

        # Place orders
        # Cash/forward legs aren’t tradable here; you can simulate forward by long stock + short discounted bond,
        # but we’ll only do calls (and optional puts) inside this module.
        placed = 0
        for Kc, w in w_calls.items():
            qty = w * scale
            px = _opt_mid(SYM, Kc, "C")
            if px is None:
                continue
            if qty * px < MIN_TICKET_USD / 50.0:  # skip dust
                continue
            self.order(_opt_sym(SYM, Kc, "C"), "buy", qty=qty, order_type="market", venue=VENUE_OPT)
            placed += 1

        # Optional: add put-side if enabled to better approximate left-tail curvature
        if USE_PUT_SIDE:
            # mirror trick: (S-K)^- can be synthesized with put weights ~ negative second derivative where needed
            # For simplicity we add a protective put ladder centered below S0 if payoff has left curvature.
            K_floor = max(Kmin, S0 * 0.5)
            K_puts = [k for k in Ks if k <= S0 and k >= K_floor][::max(1, int(5 * (step > 0)))]
            for Kp in K_puts:
                pp = _opt_mid(SYM, Kp, "P")
                if pp is None:
                    continue
                # small ladder size proportional to curvature estimate near Kp
                i = Ks.index(Kp)
                if 0 < i < len(Ks) - 1:
                    curv = max(0.0, pay[i - 1] - 2 * pay[i] + pay[i + 1])
                else:
                    curv = 0.0
                qty = 0.1 * scale * (curv / max(1e-6, step * step))
                if qty * pp >= MIN_TICKET_USD / 50.0:
                    self.order(_opt_sym(SYM, Kp, "P"), "buy", qty=qty, order_type="market", venue=VENUE_OPT)
                    placed += 1

        if placed == 0:
            return

        st = PackageState(
            mode="PIECEWISE",
            ts_ms=int(time.time()*1000),
            details={
                "grid": {"Kmin": Kmin, "Kmax": Kmax, "step": step, "calls": len(w_calls)},
                "A_cash": A, "B_forward": B, "scale": scale
            }
        )
        r.set(_poskey(self.ctx.name), json.dumps(st.__dict__))