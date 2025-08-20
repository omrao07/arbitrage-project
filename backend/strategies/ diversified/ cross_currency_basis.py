# backend/strategies/diversified/cross_currency_basis.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

# ---------------- Env / knobs ----------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Pair + tenor you want to trade (can run multiple instances with different envs)
PAIR        = os.getenv("CCB_PAIR", "EURUSD").upper()    # e.g., EURUSD, USDJPY, GBPUSD
TENOR_DAYS  = int(os.getenv("CCB_TENOR_DAYS", "30"))     # 7, 30, 90
ENTRY_BPS   = float(os.getenv("CCB_ENTRY_BPS", "5.0"))   # enter when |basis| >= this
EXIT_BPS    = float(os.getenv("CCB_EXIT_BPS", "2.0"))    # exit when |basis| <= this
SIZE_BASE   = float(os.getenv("CCB_BASE_UNITS", "10000"))# trade size in base currency units (e.g., 10k EUR)
COOLDOWN_S  = int(os.getenv("CCB_COOLDOWN_S", "60"))     # min time between trades per pair
MAX_POS     = int(os.getenv("CCB_MAX_POS", "1"))         # max concurrent positions per pair

# Symbols (how we’ll see them in your feed & how OMS knows them)
SPOT_SYMBOL = os.getenv("CCB_SPOT_SYMBOL", f"{PAIR}.SPOT").upper()
FWD_SYMBOL  = os.getenv("CCB_FWD_SYMBOL",  f"{PAIR}.FWD{TENOR_DAYS}D").upper()
VENUE       = os.getenv("CCB_VENUE", "OANDA").upper()    # hint for region router (FX)

# Where short rates live (you can update these keys from your data jobs)
# Example:
#   HSET rates:USD '1D' 0.052 '30D' 0.053
#   HSET rates:EUR '1D' 0.036 '30D' 0.037
DOM_CCY = os.getenv("CCB_DOM_CCY", "USD").upper()  # quote currency for EURUSD is USD
FOR_CCY = os.getenv("CCB_FOR_CCY", "EUR").upper()  # base currency for EURUSD is EUR
RATES_DOM_KEY = f"rates:{DOM_CCY}"
RATES_FOR_KEY = f"rates:{FOR_CCY}"
RATES_FIELD   = f"{TENOR_DAYS}D"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# ---------------- Helpers ----------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _get_rate(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0.0, 0) else default

def _basis_bps(spot_px: float, fwd_px: float, r_dom: float, r_for: float, T_years: float) -> float:
    """
    Covered Interest Parity (CIP) says: F/S = (1+r_dom*T)/(1+r_for*T)
    Cross-currency basis (approx) = 10,000 * [ (F/S) / ((1+r_dom*T)/(1+r_for*T)) - 1 ]
    Positive basis -> forward rich vs CIP (USD scarcity in EURUSD convention).
    """
    if spot_px <= 0 or fwd_px <= 0:
        return 0.0
    lhs = fwd_px / spot_px
    rhs = (1.0 + r_dom * T_years) / (1.0 + r_for * T_years)
    return 1e4 * (lhs / rhs - 1.0)

def _state_key() -> str:
    return f"ccb:state:{PAIR}:{TENOR_DAYS}"

def _last_trade_key() -> str:
    return f"ccb:last_trade_ts:{PAIR}:{TENOR_DAYS}"

def _position_key() -> str:
    return f"ccb:open:{PAIR}:{TENOR_DAYS}"


@dataclass
class BasisSnapshot:
    spot: float
    fwd: float
    r_dom: float
    r_for: float
    bps: float


# ---------------- Strategy ----------------
class CrossCurrencyBasisStrategy(Strategy):
    """
    Cross-currency basis arbitrage (paper mode friendly).

    Trade logic (simplified, symmetric):
      - Compute basis_bps from spot, forward, and short rates.
      - If basis >= ENTRY_BPS: forward rich -> SHORT forward, LONG spot (buy base)
      - If basis <= -ENTRY_BPS: forward cheap -> LONG forward, SHORT spot (sell base)
      - Exit when |basis| <= EXIT_BPS (close both legs).

    We submit legs as two independent orders; your OMS/risk will handle fills + PnL.
    """

    def __init__(self,
        name: str = "cross_currency_basis",
        region: str | None = "FX",
        default_qty: float = SIZE_BASE,
    ):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.latest_spot: Optional[float] = None
        self.latest_fwd: Optional[float] = None
        self.cooldown_until: float = 0.0

        # book-keeping: store open side in Redis so restart-safe
        # format: {"side":"short_fwd_long_spot" | "long_fwd_short_spot", "entry_bps":..., "qty":..., "ts_ms":...}
        self.open_state = self._load_position()

    # --------- Persistence ---------
    def _load_position(self) -> Optional[Dict]:
        raw = r.get(_position_key())
        try:
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _save_position(self, state: Optional[Dict]) -> None:
        if state is None:
            r.delete(_position_key())
        else:
            r.set(_position_key(), json.dumps(state))

    def _cooldown_ok(self) -> bool:
        return time.time() >= self.cooldown_until

    def _bump_cooldown(self):
        self.cooldown_until = time.time() + COOLDOWN_S
        r.set(_last_trade_key(), str(int(self.cooldown_until)))

    # --------- Order helpers (two legs) ---------
    def _open_short_fwd_long_spot(self, qty_base: float) -> None:
        # Leg A: short forward -> 'sell' FWD (synthetic symbol)
        self.order(FWD_SYMBOL, "sell", qty=qty_base, order_type="market", venue=VENUE)
        # Leg B: long spot  -> 'buy' SPOT (buy base currency)
        self.order(SPOT_SYMBOL, "buy", qty=qty_base, order_type="market", venue=VENUE)

    def _open_long_fwd_short_spot(self, qty_base: float) -> None:
        # Leg A: long forward -> 'buy' FWD
        self.order(FWD_SYMBOL, "buy", qty=qty_base, order_type="market", venue=VENUE)
        # Leg B: short spot   -> 'sell' SPOT
        self.order(SPOT_SYMBOL, "sell", qty=qty_base, order_type="market", venue=VENUE)

    def _close_all(self) -> None:
        st = self._load_position()
        if not st:
            return
        qty = float(st.get("qty", 0.0))
        if qty <= 0:
            return
        side = st.get("side")
        # reverse both legs
        if side == "short_fwd_long_spot":
            self.order(FWD_SYMBOL, "buy",  qty=qty, order_type="market", venue=VENUE)
            self.order(SPOT_SYMBOL, "sell", qty=qty, order_type="market", venue=VENUE)
        elif side == "long_fwd_short_spot":
            self.order(FWD_SYMBOL, "sell", qty=qty, order_type="market", venue=VENUE)
            self.order(SPOT_SYMBOL, "buy",  qty=qty, order_type="market", venue=VENUE)
        self._save_position(None)
        self._bump_cooldown()

    # --------- Core tick processing ---------
    def _snap(self) -> Optional[BasisSnapshot]:
        r_dom = _get_rate(RATES_DOM_KEY, RATES_FIELD)
        r_for = _get_rate(RATES_FOR_KEY, RATES_FIELD)
        if self.latest_spot is None or self.latest_fwd is None or r_dom is None or r_for is None:
            return None
        T = TENOR_DAYS / 365.0
        bps = _basis_bps(self.latest_spot, self.latest_fwd, r_dom, r_for, T)
        return BasisSnapshot(self.latest_spot, self.latest_fwd, r_dom, r_for, bps)

    def on_tick(self, tick: Dict) -> None:
        """
        Expect ticks for SPOT_SYMBOL and FWD_SYMBOL:
          {"symbol":"EURUSD.SPOT","price":1.08342,"ts_ms":...}
          {"symbol":"EURUSD.FWD30D","price":1.08510,"ts_ms":...}
        Your data jobs should publish forward points or outright forwards as FWD price.
        """
        sym = str(tick.get("symbol") or tick.get("s") or "").upper()
        px  = float(tick.get("price") or tick.get("p") or 0.0)
        if not sym or px <= 0:
            return

        if sym == SPOT_SYMBOL:
            self.latest_spot = px
        elif sym == FWD_SYMBOL:
            self.latest_fwd = px
        else:
            return

        snap = self._snap()
        if snap is None:
            return

        # Emit signal in [-1,1] roughly scaled by bps/entry
        score = max(-1.0, min(1.0, snap.bps / max(1e-6, ENTRY_BPS)))
        self.emit_signal(score)

        # If position open, consider exit
        st = self._load_position()
        if st:
            if abs(snap.bps) <= EXIT_BPS and self._cooldown_ok():
                self._close_all()
            return

        # If no position open, consider entry
        if not self._cooldown_ok():
            return

        # Enforce max concurrent positions using a simple counter (per pair)
        open_cnt = int(r.get(f"{_position_key()}:count") or "0")
        if open_cnt >= MAX_POS:
            return

        if snap.bps >= ENTRY_BPS:
            # forward rich -> short fwd / long spot
            self._open_short_fwd_long_spot(self.ctx.default_qty)
            state = {"side": "short_fwd_long_spot", "qty": float(self.ctx.default_qty), "entry_bps": float(snap.bps), "ts_ms": _now_ms()}
            self._save_position(state)
            r.set(f"{_position_key()}:count", str(open_cnt + 1))
            self._bump_cooldown()

        elif snap.bps <= -ENTRY_BPS:
            # forward cheap -> long fwd / short spot
            self._open_long_fwd_short_spot(self.ctx.default_qty)
            state = {"side": "long_fwd_short_spot", "qty": float(self.ctx.default_qty), "entry_bps": float(snap.bps), "ts_ms": _now_ms()}
            self._save_position(state)
            r.set(f"{_position_key()}:count", str(open_cnt + 1))
            self._bump_cooldown()