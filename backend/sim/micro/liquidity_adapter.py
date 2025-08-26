# backend/execution/liquidity_adapter.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional: hook your cost model (kept optional to avoid hard deps)
try:
    from backend.execution.cost_model import CostModel # type: ignore
except Exception:
    CostModel = None  # type: ignore


# =============================== Models ===============================

@dataclass
class ParentOrder:
    symbol: str
    side: str                  # "buy" | "sell"
    qty: float                 # parent quantity in units (shares/contracts)
    limit_price: Optional[float] = None
    tif: str = "day"           # "day" | "ioc" | "gtc"
    strategy: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChildOrder:
    symbol: str
    side: str
    qty: float
    order_type: str            # "market" | "limit"
    limit_price: Optional[float]
    tif: str                   # "ioc" | "day"
    slice_id: int
    parent_coid: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Book:
    bid_px: Optional[float] = None
    bid_sz: Optional[float] = None
    ask_px: Optional[float] = None
    ask_sz: Optional[float] = None
    last_px: Optional[float] = None

@dataclass
class LiquidityConfig:
    # Global caps
    max_participation: float = 0.15     # max fraction of traded volume (POV cap)
    max_of_visible_book: float = 0.25   # max fraction of top-of-book size per slice
    max_adv_participation: float = 0.05 # max fraction of ADV for the *whole* parent
    min_clip_qty: float = 1.0           # minimum child qty (post rounding/lot)
    clip_round_lot: float = 1.0         # round child to nearest lot
    cool_down_sec: float = 0.7          # wait between child submissions
    price_offset_ticks: int = 0         # +ticks for buy (aggression); negative to improve
    tick_size: float = 0.05             # default tick size (override per symbol)
    # Slicing
    mode: str = "POV"                   # "POV" | "TWAP" | "VWAPL"
    target_rate: float = 0.10           # for POV: target participation (<= max_participation)
    horizon_sec: int = 120              # for TWAP/VWAPL: slice across this window
    slices: int = 12                    # number of slices across horizon
    # Risk/venue
    allow_market: bool = True
    allow_limit: bool = True
    venue: Optional[str] = None
    # Optional cost guard (bps) — if CostModel present and est cost > cap -> de-aggress
    max_cost_bps_per_slice: Optional[float] = None

@dataclass
class PlanState:
    parent: ParentOrder
    cfg: LiquidityConfig
    created_ms: int
    due_ms: int
    total_qty: float
    filled_qty: float = 0.0
    last_slice_ts: float = 0.0
    slice_index: int = 0
    # running market stats
    traded_vol_est: float = 0.0  # cumulative traded vol since plan start (POV uses this)
    adv_shares: Optional[float] = None
    # bookkeeping
    done: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_qty - self.filled_qty)

    @property
    def time_left_sec(self) -> float:
        return max(0.0, (self.due_ms / 1000.0) - time.time())


# =============================== Adapter ===============================

class LiquidityAdapter:
    """
    Liquidity-aware slicer for parent orders.
    - Modes: POV (participation of volume), TWAP (even time), VWAP-lite (T+book)
    - Caps: ADV, participation, visible book fraction, min clip, cool-down
    - Optional cost gating using CostModel (if available)
    """

    def __init__(self, *, default_config: Optional[LiquidityConfig] = None, cost_model: Optional[Any] = None):
        self.cfg = default_config or LiquidityConfig()
        self.cm = cost_model  # e.g., CostModel("in_zerodha_cash") or None
        self._plans: Dict[str, PlanState] = {}  # key: parent_coid

    # ---------- planning ----------

    def start_plan(
        self,
        parent: ParentOrder,
        *,
        parent_coid: str,
        adv_shares: Optional[float] = None,
        horizon_sec: Optional[int] = None,
        now_ms: Optional[int] = None,
    ) -> PlanState:
        cfg = self.cfg
        now_ms = now_ms or int(time.time() * 1000)
        due_ms = now_ms + int(1000 * (horizon_sec if horizon_sec is not None else cfg.horizon_sec))
        total = float(parent.qty)

        # ADV cap
        if adv_shares and adv_shares > 0:
            max_adv = adv_shares * max(0.0, min(1.0, cfg.max_adv_participation))
            if total > max_adv:
                total = max_adv
        st = PlanState(
            parent=parent, cfg=cfg, created_ms=now_ms, due_ms=due_ms,
            total_qty=total, adv_shares=adv_shares
        )
        st.notes.append(f"plan start: mode={cfg.mode}, target_rate={cfg.target_rate}")
        self._plans[parent_coid] = st
        return st

    def get_plan(self, parent_coid: str) -> Optional[PlanState]:
        return self._plans.get(parent_coid)

    # ---------- real-time inputs ----------

    def on_trade(self, parent_coid: str, last_px: float, last_qty: float) -> None:
        """Feed prints/trades to update POV participation math."""
        st = self._plans.get(parent_coid)
        if not st:
            return
        st.traded_vol_est += max(0.0, float(last_qty))

    def on_fill(self, parent_coid: str, fill_qty: float) -> None:
        st = self._plans.get(parent_coid)
        if not st:
            return
        st.filled_qty += max(0.0, float(fill_qty))
        if st.remaining <= 0:
            st.done = True

    # ---------- slicing core ----------

    def _clip_bounds_from_book(self, st: PlanState, book: Optional[Book]) -> float:
        """
        Limit child by visible top-of-book size (max_of_visible_book).
        """
        if not book:
            return float("inf")
        max_frac = max(0.0, min(1.0, st.cfg.max_of_visible_book))
        if st.parent.side == "buy":
            if book.ask_sz and book.ask_sz > 0:
                return float(book.ask_sz) * max_frac
        else:
            if book.bid_sz and book.bid_sz > 0:
                return float(book.bid_sz) * max_frac
        return float("inf")

    def _determine_slice_qty(self, st: PlanState, book: Optional[Book]) -> float:
        cfg = st.cfg
        rem = st.remaining
        if rem <= 0:
            return 0.0

        # base wish size by mode
        if cfg.mode.upper() == "POV":
            # target participation of traded volume since last slice
            # if we lack per-interval volume, use traded_vol_est total and time pacing
            # Slice target = target_rate * (ΔVolume). As a safe proxy, scale by horizon.
            # If no volume, fall back to twap pacing.
            vol = max(0.0, st.traded_vol_est)
            wish = max(rem / max(st.slices_left(), 1), vol * max(0.0, min(cfg.target_rate, cfg.max_participation))) # type: ignore
        else:
            # TWAP / VWAP-lite -> even across remaining slices/time
            total_slices = max(1, int(cfg.slices))
            elapsed = max(0.0, (time.time() - (st.created_ms / 1000.0)))
            slice_len = max(0.1, (st.due_ms - st.created_ms) / 1000.0 / total_slices)
            already = int(elapsed // slice_len)
            left = max(1, total_slices - already - st.slice_index)
            wish = rem / left

        # Visible book guard
        book_cap = self._clip_bounds_from_book(st, book)

        # Round/clip
        qty = min(rem, wish, book_cap)
        qty = self._round_clip(qty, cfg)
        qty = max(0.0, qty)

        # Enforce min clip: if rem < min_clip, we still allow rem (to finish)
        if qty < cfg.min_clip_qty and rem >= cfg.min_clip_qty:
            qty = 0.0
        return qty

    def _round_clip(self, qty: float, cfg: LiquidityConfig) -> float:
        lot = max(1e-12, float(cfg.clip_round_lot))
        return math.floor(max(0.0, qty) / lot) * lot

    def _price_from_book(self, st: PlanState, book: Optional[Book]) -> Tuple[str, Optional[float]]:
        """
        Decide order type + price anchor given aggression config and book.
        """
        cfg = st.cfg
        side = st.parent.side
        ticks = int(cfg.price_offset_ticks or 0)
        tick = max(1e-9, float(cfg.tick_size))

        def _n(px: Optional[float]) -> Optional[float]:
            if px is None:
                return None
            return round(round(px / tick) * tick, 10)

        if side == "buy":
            if cfg.allow_market and (book is None or book.ask_px is None or ticks >= 0 and ticks >= 1_000_000):
                return "market", None
            if book and book.ask_px is not None:
                px = book.ask_px + ticks * tick
                typ = "limit" if cfg.allow_limit else "market"
                return typ, _n(px)
        else:  # sell
            if cfg.allow_market and (book is None or book.bid_px is None or ticks >= 1_000_000):
                return "market", None
            if book and book.bid_px is not None:
                px = book.bid_px - ticks * tick
                typ = "limit" if cfg.allow_limit else "market"
                return typ, _n(px)
        # fallback: last
        if book and book.last_px is not None:
            return ("limit" if cfg.allow_limit else "market"), _n(book.last_px)
        return "market", None

    def _cost_guard_ok(self, st: PlanState, qty: float, px_ref: Optional[float]) -> bool:
        """If a CostModel is present and a cap is set, veto overly expensive slices."""
        if not (self.cm and st.cfg.max_cost_bps_per_slice and qty > 0 and px_ref):
            return True
        try:
            venue = getattr(st.cfg, "venue", "us_ibkr")
            cm: Any = self.cm if not isinstance(self.cm, type) else self.cm()  # instance or class?
            symbol = st.parent.symbol
            side = st.parent.side
            br = cm.cost_for(symbol, side=side, qty=qty, price=float(px_ref), maker=False)  # type: ignore[attr-defined]
            return float(br.total_bps) <= float(st.cfg.max_cost_bps_per_slice)
        except Exception:
            # if cost model not initialized correctly, allow (fail-open) but you can tighten here
            return True

    def _cooldown_ok(self, st: PlanState) -> bool:
        return (time.time() - float(st.last_slice_ts or 0.0)) >= float(st.cfg.cool_down_sec)

    # ---------- public API ----------

    def next_child(
        self,
        parent_coid: str,
        *,
        book: Optional[Book] = None,
        now: Optional[float] = None,
    ) -> Optional[ChildOrder]:
        """
        Produce the next child order respecting all caps. Returns None if:
          - plan is done
          - cool-down not elapsed
          - no liquidity / below min clip
          - time horizon elapsed (will force finish with remaining if small)
        """
        st = self._plans.get(parent_coid)
        if not st or st.done:
            return None

        # Horizon expired? allow a final small clip to clean up.
        if st.time_left_sec <= 0.0 and st.remaining > 0:
            qty = self._round_clip(st.remaining, st.cfg)
            if qty <= 0.0:
                st.done = True
                return None
        else:
            if not self._cooldown_ok(st):
                return None
            qty = self._determine_slice_qty(st, book)
            if qty <= 0.0:
                return None

        order_type, px = self._price_from_book(st, book)
        if not self._cost_guard_ok(st, qty, px or (book.last_px if book else None)):
            # too costly right now; skip this interval
            st.notes.append("cost_guard_block")
            return None

        co = ChildOrder(
            symbol=st.parent.symbol,
            side=st.parent.side,
            qty=qty,
            order_type=order_type,
            limit_price=px if order_type == "limit" else None,
            tif="ioc" if order_type == "market" else "day",
            slice_id=st.slice_index,
            parent_coid=None,  # you can set the actual COID outside before send
            meta=dict(st.parent.meta),
        )
        st.slice_index += 1
        st.last_slice_ts = time.time()
        return co

    # ---------- utility ----------

    def cancel_plan(self, parent_coid: str) -> None:
        if parent_coid in self._plans:
            del self._plans[parent_coid]

    # Helper to compute remaining slices from current time
    def slices_left(self, parent_coid: str) -> int:
        st = self._plans.get(parent_coid)
        return st.slices_left() if st else 0 # type: ignore


# ------------------------- PlanState helpers --------------------------

def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def _safe_div(a: float, b: float, dflt: float = 0.0) -> float:
    return a / b if b not in (0.0, None) else dflt

def _ceil_to(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.ceil(x / step) * step

def _floor_to(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step

def _now_ms() -> int:
    return int(time.time() * 1000)

def _secs_left(st: PlanState) -> float:
    return st.time_left_sec

def _slice_length_sec(st: PlanState) -> float:
    return max(0.1, float(st.cfg.horizon_sec) / max(1, int(st.cfg.slices)))

def planstate_slices_left(st: PlanState) -> int:
    # based on elapsed time & configured slices
    elapsed = max(0.0, (time.time() - (st.created_ms / 1000.0)))
    slice_len = _slice_length_sec(st)
    already = int(elapsed // slice_len)
    return max(1, int(st.cfg.slices) - already - st.slice_index)

# attach as method dynamically (to keep dataclass clean)
def _slices_left(self: PlanState) -> int:
    return planstate_slices_left(self)
PlanState.slices_left = _slices_left  # type: ignore[attr-defined]