# backend/microstructure/venue_toxicity_score.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Optional, Tuple, List, Literal

Side = Literal["buy", "sell"]


# --------------------------- Config ---------------------------

@dataclass
class VenueToxicityConfig:
    # Windows (ms)
    tape_window_ms: int = 3_000          # prints & cancels window
    l2_window_ms: int = 2_000            # book snapshots window
    markout_horizons_ms: Tuple[int, ...] = (500, 1500, 5000)

    # EWMAs
    alpha_fast: float = 0.15             # spread, variance
    alpha_slow: float = 0.06             # mark-out / impact

    # Normalizers (tune per asset class)
    spread_bps_norm: float = 8.0
    vol_z_norm: float = 3.0
    trade_through_norm: float = 0.06
    oddlot_ratio_norm: float = 0.35
    cancel_ratio_norm: float = 3.0
    depletion_norm: float = 10_000.0
    markout_bps_norm: float = 6.0
    reject_ratio_norm: float = 0.15

    # Weights for score composition
    w_spread: float = 1.0
    w_vol: float = 1.0
    w_through: float = 0.9
    w_oddlot: float = 0.5
    w_cancel: float = 0.7
    w_deplete: float = 0.9
    w_markout: float = 1.2
    w_reject: float = 0.8

    # Thresholds
    warn_threshold: float = 0.45
    block_threshold: float = 0.65

    # Misc
    max_ring: int = 4096


# --------------------------- State ---------------------------

@dataclass
class VenueState:
    # rolling mid/vol per-venue (uses NBBO/venue mid proxy)
    mid: float = 0.0
    last_mid: float = 0.0
    spread_bps_ewma: float = 2.0
    ret2_ewma: float = 0.0

    # L2 best snapshot (if available for venue)
    bid: float = 0.0
    ask: float = 0.0
    bid_sz: float = 0.0
    ask_sz: float = 0.0

    # Counters
    sends: int = 0
    cancels: int = 0
    rejects: int = 0

    # Derived / rings
    trades: Deque[Tuple[int, float, float, Side]] = field(default_factory=lambda: deque(maxlen=4096))
    canc_ring: Deque[Tuple[int, Side, float]] = field(default_factory=lambda: deque(maxlen=4096))
    l2_hist: Deque[Tuple[int, float, float, float, float]] = field(default_factory=lambda: deque(maxlen=4096))

    # depletion accumulators (touch removals)
    deplete_buy: float = 0.0
    deplete_sell: float = 0.0

    # markout EWMAs (adverse, bps)
    mo1s_bps: float = 0.0
    mo5s_bps: float = 0.0

    # last update
    last_ts_ms: int = 0


# ---------------------- Scorer (per symbol) -------------------

class VenueToxicityScorer:
    """
    Tracks microstructure toxicity PER VENUE for one symbol.
    Feed it venue-level L2 (if you have), tape prints (tagged by venue if possible),
    and OMS events (sends/rejects/cancels). Query per-venue scores or pick the
    best venue for taking liquidity.

    Minimal inputs still work:
      - If you only have NBBO + prints (no per-venue L2), it still scores using spread/vol/trade-through/mark-out.
    """

    def __init__(self, symbol: str, cfg: Optional[VenueToxicityConfig] = None):
        self.sym = symbol.upper()
        self.cfg = cfg or VenueToxicityConfig()
        self.venues: Dict[str, VenueState] = {}
        # Optional NBBO snapshot (used as fallback)
        self.nbbo_bid = 0.0
        self.nbbo_ask = 0.0

    # ---------- helpers ----------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _sf(x, d=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return d

    def _vs(self, venue: str) -> VenueState:
        if venue not in self.venues:
            self.venues[venue] = VenueState()
        return self.venues[venue]

    def _prune(self, buf: Deque, horizon_ms: int) -> None:
        cutoff = self._now_ms() - horizon_ms
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _mid_from(self, bid: float, ask: float, fallback: float) -> float:
        if bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
        return fallback

    # ---------- NBBO (optional) ----------
    def on_nbbo(self, bid: Optional[float], ask: Optional[float]) -> None:
        if bid is not None: self.nbbo_bid = self._sf(bid, self.nbbo_bid)
        if ask is not None: self.nbbo_ask = self._sf(ask, self.nbbo_ask)

    # ---------- L2 per-venue (optional but recommended) ----------
    def on_l2(self, venue: str, bid: Optional[float], bid_sz: Optional[float],
              ask: Optional[float], ask_sz: Optional[float]) -> None:
        vs = self._vs(venue)
        ts = self._now_ms()
        if bid is not None: vs.bid = self._sf(bid, vs.bid)
        if ask is not None: vs.ask = self._sf(ask, vs.ask)
        if bid_sz is not None: vs.bid_sz = self._sf(bid_sz, vs.bid_sz)
        if ask_sz is not None: vs.ask_sz = self._sf(ask_sz, vs.ask_sz)

        mid = self._mid_from(vs.bid, vs.ask, vs.mid or 0.5 * (self.nbbo_bid + self.nbbo_ask) if (self.nbbo_bid and self.nbbo_ask) else 0.0)
        if mid > 0:
            spread_bps = (max(vs.ask, self.nbbo_ask or vs.ask) - min(vs.bid, self.nbbo_bid or vs.bid)) / max(1e-9, mid) * 1e4
            a = self.cfg.alpha_fast
            vs.spread_bps_ewma = (1 - a) * vs.spread_bps_ewma + a * spread_bps
            if vs.last_mid > 0:
                r = (mid / vs.last_mid) - 1.0
                vs.ret2_ewma = 0.97 * vs.ret2_ewma + 0.03 * (r * r)
            vs.last_mid = vs.mid if vs.mid > 0 else mid
            vs.mid = mid

        # depletion (touch quantity drop)
        if vs.l2_hist:
            _, pb, qb, pa, qa = vs.l2_hist[-1]
            if vs.bid == pb and vs.bid_sz < qb:
                vs.deplete_buy += (qb - vs.bid_sz)
            if vs.ask == pa and vs.ask_sz < qa:
                vs.deplete_sell += (qa - vs.ask_sz)
        vs.l2_hist.append((ts, vs.bid, vs.bid_sz, vs.ask, vs.ask_sz))
        self._prune(vs.l2_hist, self.cfg.l2_window_ms)
        vs.last_ts_ms = ts

    # ---------- Tape prints (venue-tag if possible) ----------
    def on_trade(self, venue: str, price: float, size: float, aggressor_side: Optional[Side]) -> None:
        vs = self._vs(venue)
        if size is None or size <= 0:
            return
        side: Side = aggressor_side or ("buy" if (vs.ask and price >= vs.ask) else "sell")
        ts = self._now_ms()
        px = self._sf(price)
        sz = self._sf(size)
        vs.trades.append((ts, px, sz, side))
        self._prune(vs.trades, self.cfg.tape_window_ms)

        # mark-out proxy (instantaneous; sharper if you call update_markout later)
        if vs.mid > 0 and px > 0:
            if side == "buy":
                mo = (vs.mid - px) / px  # adverse if mid rises after a buy-take
            else:
                mo = (px - vs.mid) / px
            mo_bps = max(0.0, mo * 1e4)
            vs.mo1s_bps = (1 - self.cfg.alpha_slow) * vs.mo1s_bps + self.cfg.alpha_slow * mo_bps
            vs.mo5s_bps = (1 - self.cfg.alpha_slow) * vs.mo5s_bps + self.cfg.alpha_slow * mo_bps

    # Optional: explicit cancels and OMS accounting improve cancel/reject ratios
    def on_cancel(self, venue: str, side: Optional[Side] = None, size: Optional[float] = None) -> None:
        vs = self._vs(venue)
        ts = self._now_ms()
        vs.cancels += 1
        vs.canc_ring.append((ts, side or "buy", self._sf(size or 1.0)))
        self._prune(vs.canc_ring, self.cfg.tape_window_ms)
        vs.last_ts_ms = ts

    def on_send(self, venue: str) -> None:
        self._vs(venue).sends += 1

    def on_reject(self, venue: str) -> None:
        vs = self._vs(venue)
        vs.rejects += 1

    # Optional: sharper mark-out if you provide future mid snapshots
    def update_markout(self, venue: str, side: Side, fill_px: float, future_mid: float, horizon_s: float) -> None:
        vs = self._vs(venue)
        if fill_px <= 0 or future_mid <= 0:
            return
        if side == "buy":
            mo = (future_mid - fill_px) / fill_px
        else:
            mo = (fill_px - future_mid) / fill_px
        mo_bps = max(0.0, mo * 1e4)
        if horizon_s <= 2.0:
            vs.mo1s_bps = (1 - self.cfg.alpha_slow) * vs.mo1s_bps + self.cfg.alpha_slow * mo_bps
        else:
            vs.mo5s_bps = (1 - self.cfg.alpha_slow) * vs.mo5s_bps + self.cfg.alpha_slow * mo_bps

    # ---------- Components ----------
    def _trade_through_ratio(self, vs: VenueState) -> float:
        """% of trades at or beyond the passive touch (adverse)."""
        if vs.bid <= 0 or vs.ask <= 0 or not vs.trades:
            return 0.0
        now = self._now_ms()
        trades = [x for x in vs.trades if x[0] >= now - self.cfg.tape_window_ms]
        if not trades:
            return 0.0
        through = 0
        for _, px, _, s in trades:
            if s == "buy" and px >= vs.ask:
                through += 1
            elif s == "sell" and px <= vs.bid:
                through += 1
        return through / max(1, len(trades))

    def _oddlot_ratio(self, vs: VenueState) -> float:
        now = self._now_ms()
        trades = [x for x in vs.trades if x[0] >= now - self.cfg.tape_window_ms]
        if not trades:
            return 0.0
        odd = sum(1 for _, _, sz, _ in trades if sz < 100.0)
        return odd / max(1, len(trades))

    def _cancel_ratio(self, vs: VenueState) -> float:
        # cancels per send (simple proxy; can be improved with execs)
        return (vs.cancels / max(1, vs.sends)) if vs.sends > 0 else 0.0

    def _depletion(self, vs: VenueState) -> float:
        # max depletion at touch over window (shares)
        # decay a bit every call to avoid runaway accumulation
        vs.deplete_buy *= 0.9
        vs.deplete_sell *= 0.9
        return max(vs.deplete_buy, vs.deplete_sell)

    # ---------- Score & Regime ----------
    def score(self, venue: str) -> float:
        vs = self._vs(venue)
        # components normalized to 0..1
        spread_c = min(1.0, vs.spread_bps_ewma / max(1e-9, self.cfg.spread_bps_norm))
        vol_c    = min(1.0, (math.sqrt(max(0.0, vs.ret2_ewma)) * 100.0) / max(1e-9, self.cfg.vol_z_norm))
        thr_c    = min(1.0, self._trade_through_ratio(vs) / max(1e-9, self.cfg.trade_through_norm))
        odd_c    = min(1.0, self._oddlot_ratio(vs) / max(1e-9, self.cfg.oddlot_ratio_norm))
        cxl_c    = min(1.0, self._cancel_ratio(vs) / max(1e-9, self.cfg.cancel_ratio_norm))
        dep_c    = min(1.0, self._depletion(vs) / max(1.0, self.cfg.depletion_norm))
        mo_c     = min(1.0, max(vs.mo1s_bps, vs.mo5s_bps) / max(1e-9, self.cfg.markout_bps_norm))
        rej_c    = min(1.0, (vs.rejects / max(1, vs.sends)) / max(1e-9, self.cfg.reject_ratio_norm))

        w = self.cfg
        parts = [
            (spread_c, w.w_spread),
            (vol_c,    w.w_vol),
            (thr_c,    w.w_through),
            (odd_c,    w.w_oddlot),
            (cxl_c,    w.w_cancel),
            (dep_c,    w.w_deplete),
            (mo_c,     w.w_markout),
            (rej_c,    w.w_reject),
        ]
        wsum = sum(x[1] for x in parts) or 1.0
        s = sum(val * wt for val, wt in parts) / wsum
        return max(0.0, min(1.0, s))

    def regime(self, venue: str) -> str:
        s = self.score(venue)
        if s >= self.cfg.block_threshold:
            return "toxic"
        if s >= self.cfg.warn_threshold:
            return "caution"
        return "calm"

    # ---------- Recommendations ----------
    def recommend_for_take(self, side: Side, venues: Optional[List[str]] = None) -> Optional[Tuple[str, float]]:
        """
        Return the venue with the LOWEST toxicity score (safer to take).
        """
        cand = venues or list(self.venues.keys())
        if not cand:
            return None
        best_v, best_s = None, 1e9
        for v in cand:
            s = self.score(v)
            if s < best_s:
                best_v, best_s = v, s
        return (best_v, best_s) if best_v else None

    def should_block(self, venue: str, side: Side) -> bool:
        """
        True if this venue is currently in 'toxic' regime (block taking here).
        """
        return self.regime(venue) == "toxic"

    # ---------- Export ----------
    def snapshot(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for v, vs in self.venues.items():
            out[v] = {
                "score": round(self.score(v), 4),
                "regime": {"calm": 0, "caution": 1, "toxic": 2}[self.regime(v)],
                "spread_bps": round(vs.spread_bps_ewma, 3),
                "vol_z": round(math.sqrt(max(0.0, vs.ret2_ewma)) * 100.0, 3),
                "trade_through": round(self._trade_through_ratio(vs), 4),
                "oddlot_ratio": round(self._oddlot_ratio(vs), 4),
                "cancel_ratio": round(self._cancel_ratio(vs), 4),
                "depletion": round(self._depletion(vs), 2),
                "mo1s_bps": round(vs.mo1s_bps, 3),
                "mo5s_bps": round(vs.mo5s_bps, 3),
                "sends": vs.sends,
                "rejects": vs.rejects,
                "last_ts_ms": vs.last_ts_ms,
            }
        return out