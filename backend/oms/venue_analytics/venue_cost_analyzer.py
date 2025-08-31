# backend/tca/venue_cost_analyzer.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, List


# ---------------------------- Config ----------------------------

@dataclass
class VenueFees:
    maker_rebate_bps: float = 0.0   # positive = you earn rebate
    taker_fee_bps: float = 0.0      # positive = you pay fee
    # Optional per-share cents if you prefer; we'll convert to bps using send_px
    maker_rebate_cents: float = 0.0
    taker_fee_cents: float = 0.0


@dataclass
class AnalyzerConfig:
    # Static fee table you can override per venue
    fees: Dict[str, VenueFees] = field(default_factory=dict)

    # Rolling EWMAs (0..1); bigger = faster react
    alpha_slip: float = 0.15      # slippage vs send-benchmark (mid)
    alpha_mo1s: float = 0.12      # 1s mark-out
    alpha_mo5s: float = 0.08      # 5s mark-out
    alpha_fill: float = 0.10      # fill-rate
    alpha_etf:  float = 0.10      # expected time to first fill (seconds)
    alpha_size: float = 0.10      # avg fill size

    # Weighting to form a single "cost" in bps (minimize for buy; maximize negative for sell)
    w_slip: float = 1.00          # implicit slippage at fill (bps)
    w_mo1s: float = 0.60          # adverse 1s mark-out (bps)
    w_mo5s: float = 0.40          # adverse 5s mark-out (bps)
    w_fee:  float = 1.00          # explicit fee/rebate (bps; positive increases cost)
    w_speed: float = 0.25         # speed penalty (bps-equivalent via mapping)
    w_reject: float = 0.30        # penalty per reject ratio
    w_cancel: float = 0.10        # penalty per cancel ratio (you canceled/no fill)

    # Speed → bps mapping (how much do we “value” 1s faster fill)
    speed_bps_per_sec: float = 0.6

    # Safety
    decay_on_idle: float = 0.995   # passive decay per minute without events
    max_history: int = 4096


# --------------------------- Internals ---------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _to_bps(x: float) -> float:
    return 1e4 * x

def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


@dataclass
class SendRecord:
    ts_ms: int
    side: str                # "buy" | "sell"
    qty: float
    limit_px: Optional[float]
    mid_at_send: Optional[float]


@dataclass
class VenueStats:
    # explicit accounting
    sends: int = 0
    cancels: int = 0
    rejects: int = 0
    sent_qty: float = 0.0
    filled_qty: float = 0.0

    # rolling metrics (EWMAs)
    slip_bps: float = 0.0     # (buy: + means worse than mid; sell: + means worse than mid)
    mo1s_bps: float = 0.0     # adverse markout after 1s
    mo5s_bps: float = 0.0     # adverse markout after 5s
    etf_sec: float = 3.0      # expected time to first fill
    avg_fill_sz: float = 0.0  # average per fill size

    # ratios
    fill_rate: float = 0.0    # filled_qty / sent_qty (0..1)
    reject_ratio: float = 0.0
    cancel_ratio: float = 0.0

    # last activity
    last_ts_ms: int = 0

    # temp per-send cache: send_id -> SendRecord
    pending: Dict[str, SendRecord] = field(default_factory=dict)


# -------------------------- Analyzer -----------------------------

class VenueCostAnalyzer:
    """
    Per-venue TCA with explicit + implicit costs, mark-outs, speed, and reliability.
    Wire these hooks from your OMS/market-data:
      - on_child_send(venue, side, qty, limit_px, mid_at_send, send_id)
      - on_fill(venue, qty, px, mid_at_fill, send_id, is_first_fill=False)
      - on_cancel(venue, send_id)
      - on_reject(venue, send_id)
      - on_mid_update(mid)  [optional if you pass mids in send/fill]
    Query:
      - total_cost_bps(venue, side)  → lower is better for BUY; more negative is better for SELL
      - recommend(side)              → best venue by current score
      - snapshot()                   → full metrics for dashboards
    """

    def __init__(self, cfg: Optional[AnalyzerConfig] = None):
        self.cfg = cfg or AnalyzerConfig()
        self.venues: Dict[str, VenueStats] = {}
        self._last_mid: float = 0.0

    # ---------- helpers ----------
    def _vs(self, venue: str) -> VenueStats:
        if venue not in self.venues:
            self.venues[venue] = VenueStats(last_ts_ms=_now_ms())
        return self.venues[venue]

    def _ewma(self, old: float, new: float, alpha: float) -> float:
        return (1 - alpha) * old + alpha * new

    def _fee_bps(self, venue: str, side: str, send_px: float) -> float:
        f: VenueFees = self.cfg.fees.get(venue, VenueFees())
        # Convert cents if set; cents per share / send_px -> bps
        maker_bps = f.maker_rebate_bps or (_to_bps((f.maker_rebate_cents / 100.0) / max(send_px, 1e-9)))
        taker_bps = f.taker_fee_bps or (_to_bps((f.taker_fee_cents / 100.0) / max(send_px, 1e-9)))
        # We don't know passivity here; treat as taker by default; strategies can override by passing “passive=True”.
        # Caller may adjust by using on_fill(..., mid_at_fill, ...), comparing to touch to infer passive/active.
        return taker_bps  # pessimistic default

    # ---------- ingest ----------
    def on_mid_update(self, mid: float) -> None:
        if mid and mid > 0:
            self._last_mid = float(mid)

    def on_child_send(
        self,
        venue: str,
        side: str,
        qty: float,
        limit_px: Optional[float],
        mid_at_send: Optional[float],
        send_id: str,
    ) -> None:
        vs = self._vs(venue)
        vs.sends += 1
        vs.sent_qty += max(0.0, float(qty))
        vs.last_ts_ms = _now_ms()
        if mid_at_send is None:
            mid_at_send = self._last_mid or 0.0
        vs.pending[send_id] = SendRecord(
            ts_ms=vs.last_ts_ms,
            side=side.lower(),
            qty=float(qty),
            limit_px=(float(limit_px) if (limit_px is not None) else None),
            mid_at_send=(float(mid_at_send) if (mid_at_send is not None) else None),
        )

    def on_fill(
        self,
        venue: str,
        qty: float,
        price: float,
        mid_at_fill: Optional[float],
        send_id: Optional[str] = None,
        is_first_fill: bool = False,
        passive: Optional[bool] = None,
    ) -> None:
        """
        Update implicit slippage (vs mid_at_send) and mark-outs (vs mid_at_fill forward).
        If 'passive' is provided, we’ll use fee/rebate accordingly; else fall back to taker fee.
        """
        now = _now_ms()
        vs = self._vs(venue)
        q = max(0.0, float(qty))
        px = float(price)
        if q <= 0 or px <= 0:
            return

        # Update fills / average fill size
        vs.filled_qty += q
        avg_fill = q if vs.avg_fill_sz <= 0 else self._ewma(vs.avg_fill_sz, q, self.cfg.alpha_size)
        vs.avg_fill_sz = avg_fill

        # Fill-rate
        vs.fill_rate = _safe_div(vs.filled_qty, vs.sent_qty)

        # Find the originating send to compute slippage vs mid_at_send
        sr: Optional[SendRecord] = vs.pending.get(send_id) if send_id else None
        bench_mid = (sr.mid_at_send if (sr and sr.mid_at_send) else (self._last_mid or mid_at_fill or 0.0))
        side = (sr.side if sr else "buy").lower()

        # implicit slippage sign:
        #  BUY: (px - bench) / bench  → + worse
        #  SELL: (bench - px) / bench → + worse
        slip = 0.0
        if bench_mid and bench_mid > 0:
            if side == "buy":
                slip = _to_bps((px - bench_mid) / bench_mid)
            else:
                slip = _to_bps((bench_mid - px) / bench_mid)
        vs.slip_bps = self._ewma(vs.slip_bps, slip, self.cfg.alpha_slip)

        # explicit fee/rebate estimate in bps
        fee_bps = 0.0
        if bench_mid and bench_mid > 0:
            f: VenueFees = self.cfg.fees.get(venue, VenueFees())
            maker_bps = f.maker_rebate_bps or _to_bps((f.maker_rebate_cents / 100.0) / bench_mid)
            taker_bps = f.taker_fee_bps or _to_bps((f.taker_fee_cents / 100.0) / bench_mid)
            if passive is True:
                fee_bps = -maker_bps  # rebate reduces cost (negative = good)
            elif passive is False:
                fee_bps = taker_bps
            else:
                fee_bps = taker_bps  # conservative default
        # We don’t store fee_bps directly; it’s folded into cost at query time via weights.

        # Mark-outs (adverse selection proxy)
        mid_now = (mid_at_fill if (mid_at_fill and mid_at_fill > 0) else (self._last_mid or px))
        if mid_now and mid_now > 0:
            # In practice, you’d compute these later with real future mids; as a streaming proxy we compare to current mid.
            # If you can schedule callbacks at +1s/+5s with the mid then call 'update_markout' helpers below.
            mo1 = 0.0
            mo5 = 0.0
            if side == "buy":
                mo1 = _to_bps((mid_now - px) / px)   # positive = mid moved up (bad for buy take)
                mo5 = mo1
            else:
                mo1 = _to_bps((px - mid_now) / px)   # positive = mid moved down (bad for sell take)
                mo5 = mo1
            vs.mo1s_bps = self._ewma(vs.mo1s_bps, max(0.0, mo1), self.cfg.alpha_mo1s)
            vs.mo5s_bps = self._ewma(vs.mo5s_bps, max(0.0, mo5), self.cfg.alpha_mo5s)

        # Expected time to first fill (seconds)
        if is_first_fill and sr:
            dt_sec = max(0.0, (now - sr.ts_ms) / 1000.0)
            vs.etf_sec = self._ewma(vs.etf_sec, dt_sec, self.cfg.alpha_etf)

        vs.last_ts_ms = now

    def on_cancel(self, venue: str, send_id: Optional[str]) -> None:
        vs = self._vs(venue)
        vs.cancels += 1
        vs.cancel_ratio = _safe_div(vs.cancels, max(1, vs.sends))
        if send_id and send_id in vs.pending:
            del vs.pending[send_id]
        vs.last_ts_ms = _now_ms()

    def on_reject(self, venue: str, send_id: Optional[str]) -> None:
        vs = self._vs(venue)
        vs.rejects += 1
        vs.reject_ratio = _safe_div(vs.rejects, max(1, vs.sends))
        if send_id and send_id in vs.pending:
            del vs.pending[send_id]
        vs.last_ts_ms = _now_ms()

    # Optional: if you can call these later with realized future mids, they’ll sharpen markout estimates.
    def update_markout(self, venue: str, side: str, fill_px: float, future_mid: float, horizon_s: float) -> None:
        vs = self._vs(venue)
        if fill_px <= 0 or future_mid <= 0:
            return
        # positive = adverse
        if side.lower() == "buy":
            mo = _to_bps((future_mid - fill_px) / fill_px)
        else:
            mo = _to_bps((fill_px - future_mid) / fill_px)
        if horizon_s <= 2.0:
            vs.mo1s_bps = self._ewma(vs.mo1s_bps, max(0.0, mo), self.cfg.alpha_mo1s)
        else:
            vs.mo5s_bps = self._ewma(vs.mo5s_bps, max(0.0, mo), self.cfg.alpha_mo5s)

    # ---------- scoring ----------
    def total_cost_bps(self, venue: str, side: str, assume_taker: bool = True) -> float:
        """
        Estimated *all-in* cost in bps for executing next child at this venue.
        Lower is better for BUY; more negative is better for SELL.
        """
        vs = self._vs(venue)
        # implicit
        slip = vs.slip_bps
        mo1  = vs.mo1s_bps
        mo5  = vs.mo5s_bps

        # explicit (assume taker unless you know you will post passively)
        fee = 0.0
        ref_px = max(1e-6, self._last_mid)
        f = self.cfg.fees.get(venue, VenueFees())
        taker_bps = f.taker_fee_bps or _to_bps((f.taker_fee_cents / 100.0) / ref_px)
        maker_bps = f.maker_rebate_bps or _to_bps((f.maker_rebate_cents / 100.0) / ref_px)
        fee = taker_bps if assume_taker else -maker_bps

        # speed penalty: map expected time to first fill to bps
        speed_pen = self.cfg.speed_bps_per_sec * max(0.0, vs.etf_sec)

        # reliability penalties
        rej_pen = 1e4 * vs.reject_ratio * 0.0001  # ~1 bps per 1% reject (tunable)
        cxl_pen = 0.5e4 * vs.cancel_ratio * 0.0001  # ~0.5 bps per 1% cancel (tunable)

        # assemble
        cost = (
            self.cfg.w_slip  * slip +
            self.cfg.w_mo1s  * mo1  +
            self.cfg.w_mo5s  * mo5  +
            self.cfg.w_fee   * fee  +
            self.cfg.w_speed * speed_pen +
            self.cfg.w_reject * rej_pen +
            self.cfg.w_cancel * cxl_pen
        )

        # For SELL, a negative cost is good; keep sign as “bps you pay” (non-negative)
        # Consumers can compare raw numbers; lower is always better.
        return float(cost)

    def recommend(self, side: str, venues: Optional[List[str]] = None, assume_taker: bool = True) -> Optional[Tuple[str, float]]:
        cand = venues or list(self.venues.keys())
        if not cand:
            return None
        best_v = None
        best_cost = float("inf")
        for v in cand:
            c = self.total_cost_bps(v, side, assume_taker=assume_taker)
            if c < best_cost:
                best_cost, best_v = c, v
        return (best_v, best_cost) if best_v else None

    # ---------- maintenance / export ----------
    def decay_idle(self) -> None:
        """Call every minute if you want slow decay without events."""
        for vs in self.venues.values():
            vs.slip_bps *= self.cfg.decay_on_idle
            vs.mo1s_bps *= self.cfg.decay_on_idle
            vs.mo5s_bps *= self.cfg.decay_on_idle
            vs.etf_sec   = self._ewma(vs.etf_sec, vs.etf_sec * self.cfg.decay_on_idle, self.cfg.alpha_etf)

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for v, vs in self.venues.items():
            out[v] = {
                "sends": vs.sends,
                "sent_qty": round(vs.sent_qty, 3),
                "filled_qty": round(vs.filled_qty, 3),
                "fill_rate": round(vs.fill_rate, 4),
                "reject_ratio": round(vs.reject_ratio, 4),
                "cancel_ratio": round(vs.cancel_ratio, 4),
                "slip_bps": round(vs.slip_bps, 3),
                "mo1s_bps": round(vs.mo1s_bps, 3),
                "mo5s_bps": round(vs.mo5s_bps, 3),
                "etf_sec": round(vs.etf_sec, 3),
                "avg_fill_sz": round(vs.avg_fill_sz, 3),
                "cost_bps_buy": round(self.total_cost_bps(v, "buy", assume_taker=True), 3),
                "cost_bps_sell": round(self.total_cost_bps(v, "sell", assume_taker=True), 3),
                "last_ts_ms": vs.last_ts_ms,
            }
        return out