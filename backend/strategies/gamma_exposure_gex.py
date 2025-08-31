# backend/engine/strategies/gamma_exposure_gex.py
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple, DefaultDict
from collections import defaultdict

import redis

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset

# ---------- Redis (optional, for precomputed GEX) ----------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
_r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# ---------- Config ----------
@dataclass
class GEXConfig:
    symbol: str = "SPY"                     # underlying to trade
    contract_multiplier: float = 100.0      # equity options default
    # If you have a separate chain worker that publishes GEX snapshots, set this True
    use_redis_gex: bool = True
    # Redis key pattern for precomputed aggregates
    gex_key_tpl: str = "options:gex:{sym}"  # HGETALL -> {"gex_total": x, "wall": k, "spot": s, "ts": ms}
    # If computing from chain ticks, we expect periodic messages shaped like:
    # {"type": "chain", "symbol": "SPY", "spot": 500.12, "rows": [{"k": 500, "cp":"C","oi": 1234,"gamma":0.0123}, ...]}
    # Missing gamma allowed; we will approximate gamma from iv/delta if provided (else skip).
    # ---- Trading knobs ----
    default_qty: float = 1.0
    min_wall_dist_bps: float = 10.0         # min distance from wall before acting (bps of spot)
    reversion_k: float = 0.5                # strength of fade when GEX>0
    breakout_k: float = 0.7                 # strength of momentum when GEX<0
    enter_thresh: float = 0.20              # |signal| to place orders
    cooldown_ms: int = 1000
    max_notional: float = 100_000.0
    # Safety / behavior
    hard_kill: bool = False
    venues: tuple[str, ...] = ("IBKR", "PAPER")   # optional, used for metadata only


# ---------- Strategy ----------
class GammaExposureGEX(Strategy):
    """
    Dealer Gamma Exposure (GEX) strategy on the underlying:
      - If GEX > 0 : liquidity providers are long gamma → they hedge against moves → mean-reversion regime.
      - If GEX < 0 : they are short gamma → hedging amplifies moves → momentum/breakout regime.

    Inputs:
      (A) Precomputed GEX snapshot in Redis (recommended for performance), or
      (B) Chain tick with rows = [{k, cp, oi, gamma, iv?, delta?}, ...] to compute a rough GEX & gamma wall.

    Action:
      - Emits signal in [-1,+1].
      - Places small market orders when |signal| >= enter_thresh and cooldown passes.
    """

    def __init__(self, name="alpha_gamma_gex", region=None, cfg: Optional[GEXConfig] = None):
        cfg = cfg or GEXConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty)
        self.cfg = cfg
        self.sym = cfg.symbol.upper()

        # State
        self._last_px: float = 0.0
        self._last_sig: float = 0.0
        self._last_ts_ms: int = 0

        # For momentum proxy (when GEX < 0)
        self._ewma_fast: float = 0.0
        self._ewma_slow: float = 0.0
        self._alpha_fast: float = 0.2
        self._alpha_slow: float = 0.05

    # -------- lifecycle --------
    def on_start(self) -> None:
        super().on_start()
        # Register metadata for allocators/UI
        hset("strategy:meta", self.ctx.name, {
            "tags": ["options", "gamma", "microstructure"],
            "region": self.ctx.region or "US",
            "underlying": self.sym,
            "notes": "Trades underlying using GEX regime (mean-revert if GEX>0, trend if GEX<0)."
        })

    # -------- helpers --------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _wall_dist_bps(self, spot: float, wall: Optional[float]) -> float:
        if wall is None or spot <= 0:
            return 0.0
        return abs((spot - wall) / spot) * 1e4

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _update_momo_filters(self, price: float) -> None:
        # Simple dual EWMA for momentum bias used when GEX<0
        if self._ewma_fast == 0.0:
            self._ewma_fast = price
            self._ewma_slow = price
        self._ewma_fast = (1 - self._alpha_fast) * self._ewma_fast + self._alpha_fast * price
        self._ewma_slow = (1 - self._alpha_slow) * self._ewma_slow + self._alpha_slow * price

    # ---- Redis precomputed path ----
    def _read_gex_from_redis(self) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
        key = self.cfg.gex_key_tpl.format(sym=self.sym)
        try:
            d = _r.hgetall(key) or {}
            if not d:
                return None
            gex_total = self._safe_float(d.get("gex_total"), 0.0) # type: ignore
            wall = d.get("wall") # type: ignore
            wall = self._safe_float(wall, None) if wall is not None else None # type: ignore
            spot = d.get("spot") # type: ignore
            spot = self._safe_float(spot, None) if spot is not None else None # type: ignore
            return gex_total, wall, spot
        except Exception:
            return None

    # ---- Chain-compute path ----
    def _compute_gex_and_wall(self, chain_rows: List[Dict[str, Any]], spot: float) -> Tuple[float, Optional[float]]:
        """
        Rough GEX:
          GEX = sum( gamma_i * OI_i * contract_multiplier * sign ),  sign = +1 for calls, -1 for puts
        Gamma wall:
          strike where cumulative signed GEX crosses zero (approx) → choose nearest crossing to spot.
        """
        cm = float(self.cfg.contract_multiplier)
        # Aggregate per-strike signed GEX
        gex_by_k: Dict[float, float] = defaultdict(float)
        for row in chain_rows:
            try:
                k = self._safe_float(row.get("k") or row.get("strike"))
                if k <= 0:
                    continue
                cp = str(row.get("cp") or row.get("type") or "").upper()[:1]  # "C"/"P"
                oi = max(0.0, self._safe_float(row.get("oi") or row.get("open_interest"), 0.0))
                gamma = self._safe_float(row.get("gamma"))
                if gamma == 0.0:
                    # if gamma missing, try a crude proxy from delta slope if provided; else skip
                    if "dgdp" in row:
                        gamma = self._safe_float(row.get("dgdp"))
                    else:
                        continue
                sign = +1.0 if cp == "C" else -1.0 if cp == "P" else 0.0
                gex_by_k[k] += sign * gamma * oi * cm
            except Exception:
                continue

        # Total GEX
        total_gex = sum(gex_by_k.values())

        # Find gamma wall ≈ strike nearest to spot where cumulative signed GEX flips sign.
        if not gex_by_k:
            return total_gex, None
        # Sort strikes
        ks = sorted(gex_by_k.keys())
        # Build cumulative from lowest to highest strike
        cum = 0.0
        wall_candidates: List[Tuple[float, float]] = []  # (strike, cum_after)
        for k in ks:
            cum += gex_by_k[k]
            wall_candidates.append((k, cum))
        # Find crossing around zero closest to spot
        wall = None
        best_dist = 1e18
        for i in range(1, len(wall_candidates)):
            k1, c1 = wall_candidates[i - 1]
            k2, c2 = wall_candidates[i]
            if c1 == 0.0:
                cand = k1
            elif c1 * c2 < 0.0:
                # linear interpolation between k1..k2 where cum crosses 0
                ratio = abs(c1) / (abs(c1) + abs(c2))
                cand = k1 + ratio * (k2 - k1)
            else:
                continue
            d = abs(cand - spot)
            if d < best_dist:
                best_dist = d
                wall = cand
        return total_gex, wall

    # ---- Translate regime → signal ----
    def _signal_from_gex(self, gex_total: float, wall: Optional[float], spot: float) -> float:
        """
        -> If GEX > 0: mean-reversion; signal is toward the wall (negative if above wall, positive if below).
        -> If GEX < 0: momentum;  signal follows short-term momentum bias from dual-EWMA.
        Returns in [-1, +1].
        """
        # Distance to wall (bps)
        dist_bps = self._wall_dist_bps(spot, wall)
        reversion = 0.0
        if wall is not None and dist_bps >= self.cfg.min_wall_dist_bps:
            # sign is toward wall
            toward = -1.0 if spot > wall else +1.0
            reversion = toward * min(1.0, dist_bps / 50.0)  # scale: 50 bps → full tilt

        # Momentum proxy
        momo = 0.0
        if self._ewma_slow > 0:
            slope = (self._ewma_fast - self._ewma_slow) / self._ewma_slow
            momo = max(-1.0, min(1.0, slope * 100.0))  # 1% gap → ~1.0

        if gex_total >= 0:
            sig = self.cfg.reversion_k * reversion
        else:
            sig = self.cfg.breakout_k * momo

        # Clamp
        return max(-1.0, min(1.0, sig))

    # -------- main --------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill:
            return

        # Accept either underlying ticks or chain ticks
        typ = (tick.get("type") or "").lower()
        sym = (tick.get("symbol") or tick.get("s") or "").upper()

        # Underlying price updates (used for momentum + execution)
        if sym == self.sym and typ in ("", "trade", "quote", "underlying"):
            px = tick.get("price") or tick.get("p") or tick.get("mid")
            if px is None:
                bid, ask = tick.get("bid"), tick.get("ask")
                if bid and ask and bid > 0 and ask > 0:
                    px = 0.5 * (float(bid) + float(ask))
            if px is None:
                return
            try:
                px = float(px)
            except Exception:
                return
            if px <= 0:
                return

            self._last_px = px
            self._update_momo_filters(px)

            # If we rely on Redis aggregates, read GEX & wall and act here
            if self.cfg.use_redis_gex:
                g = self._read_gex_from_redis()
                if not g:
                    return
                gex_total, wall, spot = g
                spot = float(spot or self._last_px)
                sig = self._signal_from_gex(gex_total, wall, spot)
                self._act_on_signal(sig, spot)
            return

        # Chain updates to compute GEX locally
        if typ == "chain":
            if sym != self.sym:
                return
            rows = tick.get("rows") or []
            spot = self._safe_float(tick.get("spot"), self._last_px)
            if spot <= 0:
                return
            gex_total, wall = self._compute_gex_and_wall(rows, spot)
            # keep momentum filters updated from last spot
            self._update_momo_filters(spot)
            sig = self._signal_from_gex(gex_total, wall, spot)
            self._act_on_signal(sig, spot)
            return

    # -------- execution --------
    def _act_on_signal(self, sig: float, spot: float) -> None:
        # Emit for allocators/UI
        self.emit_signal(sig)

        now = self._now_ms()
        if now - self._last_ts_ms < self.cfg.cooldown_ms:
            return
        if abs(sig) < self.cfg.enter_thresh:
            return

        # Notional guard
        if spot * (self.ctx.default_qty or self.cfg.default_qty) > self.cfg.max_notional:
            return

        side = "buy" if sig > 0 else "sell"
        self.order(
            symbol=self.sym,
            side=side,
            qty=self.ctx.default_qty or self.cfg.default_qty,
            order_type="market",
            mark_price=spot,
            extra={"reason": "gex_signal", "sig": sig}
        )
        self._last_ts_ms = now


# -------------- optional runner --------------
if __name__ == "__main__":
    """
    Examples:
      # Using Redis precomputed aggregates:
      export REDIS_HOST=localhost REDIS_PORT=6379
      HSET options:gex:SPY gex_total 1.2e9 wall 500 spot 503.2
      python -m backend.engine.strategies.gamma_exposure_gex

      # Using chain ticks:
      # XADD ticks.options * type chain symbol SPY spot 503.2 rows '[{"k":500,"cp":"C","oi":1200,"gamma":0.012}, ...]'
    """
    strat = GammaExposureGEX()
    # strat.run(stream="ticks.merged")  # attach in your runner