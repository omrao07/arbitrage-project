# backend/microstructure/toxic_flow_filter.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Literal, Optional, Tuple
from collections import deque

Side = Literal["buy", "sell"]


@dataclass
class ToxicFlowConfig:
    # --- windows (milliseconds) ---
    tape_window_ms: int = 3_000          # for VPIN/OFI/markout calc
    book_window_ms: int = 2_000          # for depletion/imbalance calc
    markout_horizons_ms: Tuple[int, ...] = (500, 1500, 5000)

    # --- smoothing ---
    ewma_alpha_fast: float = 0.15        # microstructure (spread, ret^2)
    ewma_alpha_slow: float = 0.03        # impact/Kyle lambda

    # --- normalizers (rough scales; tune per venue/symbol) ---
    spread_bps_norm: float = 8.0         # 8 bps is "wide"
    ret_z_norm: float = 3.0              # 3σ short-horizon move
    vpin_norm: float = 0.15              # 15% VPIN considered high
    kyle_lambda_norm: float = 3e-5       # impact per share (heuristic)
    ofi_norm: float = 5_000.0            # order flow imbalance (shares)
    cancel_ratio_norm: float = 3.0       # cancels per add/exec
    trade_through_norm: float = 0.05     # 5% prints at through-price
    oddlot_ratio_norm: float = 0.35      # 35% odd-lot share
    depletion_norm: float = 10_000.0     # shares removed at touch
    markout_bps_norm: float = 6.0        # 6 bps adverse markout

    # --- scoring weights (sum doesn’t need to be 1; we re-normalize) ---
    w_spread: float = 1.0
    w_vol: float = 1.0
    w_vpin: float = 1.0
    w_lambda: float = 0.8
    w_ofi: float = 0.7
    w_cxl: float = 0.6
    w_through: float = 0.6
    w_oddlot: float = 0.4
    w_deplete: float = 0.8
    w_markout: float = 1.2

    # --- thresholds ---
    block_threshold: float = 0.65        # >= blocks aggressive actions
    warn_threshold: float = 0.45         # >= show caution / prefer passive

    # --- misc ---
    max_ring: int = 2048                 # safety on deques


class ToxicFlowFilter:
    """
    Streaming microstructure toxicity detector.
    Computes a 0..1 'toxicity' score from:
      • Spread (bps) and short-horizon variance (vol z)
      • VPIN-like buy/sell imbalance on the tape
      • Kyle's λ (impact per volume)
      • Order Flow Imbalance (OFI) at best
      • Cancel/replace pressure & trade-through frequency
      • Odd-lot ratio
      • Top-of-book depletion (touch removals)
      • Multi-horizon adverse mark-outs (bps)
    You feed:
      - on_l2(symbol, bid, bid_sz, ask, ask_sz)  [can be partial; pass what you have]
      - on_trade(symbol, price, size, aggressor_side)
      - on_cancel(symbol, side, size)            [optional; improves cancel ratio]
    Then query:
      - score() in [0,1]
      - regime(): "calm" | "caution" | "toxic"
      - should_block(side): True/False for taking liquidity
    """

    def __init__(self, cfg: Optional[ToxicFlowConfig] = None):
        self.cfg = cfg or ToxicFlowConfig()

        # rolling state
        self.mid: float = 0.0
        self.last_mid: float = 0.0
        self.bid: float = 0.0
        self.ask: float = 0.0
        self.bid_sz: float = 0.0
        self.ask_sz: float = 0.0

        # EWMA stats
        self.spread_bps_ewma: float = 2.0
        self.ret2_ewma: float = 0.0
        self.lambda_ewma: float = 0.0

        # rings
        self.trades: Deque[Tuple[int, float, float, Side]] = deque(maxlen=self.cfg.max_ring)   # (ts, px, sz, side)
        self.cancels: Deque[Tuple[int, Side, float]] = deque(maxlen=self.cfg.max_ring)         # (ts, side, sz)
        self.l2_hist: Deque[Tuple[int, float, float, float, float]] = deque(maxlen=self.cfg.max_ring)  # (ts, bid, bidSz, ask, askSz)

        # derived counters
        self._touch_depletions_buy: float = 0.0
        self._touch_depletions_sell: float = 0.0

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

    def _prune(self, buf: Deque, horizon_ms: int) -> None:
        cutoff = self._now_ms() - horizon_ms
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    # ---------- ingestors ----------
    def on_l2(self, bid: Optional[float], bid_sz: Optional[float], ask: Optional[float], ask_sz: Optional[float]) -> None:
        ts = self._now_ms()
        if bid is not None: self.bid = self._sf(bid, self.bid)
        if ask is not None: self.ask = self._sf(ask, self.ask)
        if bid_sz is not None: self.bid_sz = self._sf(bid_sz, self.bid_sz)
        if ask_sz is not None: self.ask_sz = self._sf(ask_sz, self.ask_sz)

        if self.bid > 0 and self.ask > 0:
            mid = 0.5 * (self.bid + self.ask)
            spread_bps = (self.ask - self.bid) / max(1e-9, mid) * 1e4
            a = self.cfg.ewma_alpha_fast
            self.spread_bps_ewma = (1 - a) * self.spread_bps_ewma + a * spread_bps
            if self.last_mid > 0:
                r = (mid / self.last_mid) - 1.0
                self.ret2_ewma = 0.97 * self.ret2_ewma + 0.03 * (r * r)
            self.last_mid = self.mid if self.mid > 0 else mid
            self.mid = mid

        # depletion proxy: track big drops at touch across snapshots
        if self.l2_hist:
            _, pb, qb, pa, qa = self.l2_hist[-1]
            if self.bid == pb and self.bid_sz < qb:
                self._touch_depletions_buy += (qb - self.bid_sz)
            if self.ask == pa and self.ask_sz < qa:
                self._touch_depletions_sell += (qa - self.ask_sz)

        self.l2_hist.append((ts, self.bid, self.bid_sz, self.ask, self.ask_sz))
        self._prune(self.l2_hist, self.cfg.book_window_ms)

    def on_trade(self, price: float, size: float, aggressor_side: Optional[Side]) -> None:
        if price is None or size is None or size <= 0:
            return
        side: Side = aggressor_side or ("buy" if (self.ask and price >= self.ask) else "sell")
        ts = self._now_ms()
        self.trades.append((ts, self._sf(price), self._sf(size), side))
        self._prune(self.trades, self.cfg.tape_window_ms)

        # Kyle's λ update: |ΔP| / Volume over short window
        # Estimate ΔP by last two mids; volume = sum sz in window
        if self.mid > 0 and self.last_mid > 0:
            dP = abs(self.mid - self.last_mid) / max(1e-9, self.last_mid)
            vol = sum(sz for t, _, sz, _ in self.trades if t >= ts - 800)  # 0.8s micro window
            if vol > 0:
                lam = dP / vol
                a = self.cfg.ewma_alpha_slow
                self.lambda_ewma = (1 - a) * self.lambda_ewma + a * lam

    def on_cancel(self, side: Side, size: float) -> None:
        ts = self._now_ms()
        self.cancels.append((ts, side, max(0.0, self._sf(size))))
        self._prune(self.cancels, self.cfg.tape_window_ms)

    # ---------- features ----------
    def _vpin(self) -> float:
        """
        Volume-synchronized probability of informed trading (simple proxy).
        """
        now = self._now_ms()
        self._prune(self.trades, self.cfg.tape_window_ms)
        buy_v = sum(sz for t, _, sz, s in self.trades if s == "buy" and t >= now - self.cfg.tape_window_ms)
        sell_v = sum(sz for t, _, sz, s in self.trades if s == "sell" and t >= now - self.cfg.tape_window_ms)
        tot = buy_v + sell_v
        if tot <= 0:
            return 0.0
        return abs(buy_v - sell_v) / tot

    def _ofi(self) -> float:
        """
        Order Flow Imbalance at touch using last two L2 points.
        """
        if len(self.l2_hist) < 2:
            return 0.0
        # last and prev
        _, b0, q0b, a0, q0a = self.l2_hist[-2]
        _, b1, q1b, a1, q1a = self.l2_hist[-1]
        # Cont et al. definition simplified at best level
        ofi = 0.0
        if b1 > b0: ofi += q1b
        elif b1 == b0: ofi += (q1b - q0b)
        if a1 < a0: ofi -= q1a
        elif a1 == a0: ofi -= (q1a - q0a)
        return ofi

    def _cancel_ratio(self) -> float:
        now = self._now_ms()
        self._prune(self.cancels, self.cfg.tape_window_ms)
        canc = sum(sz for t, _, sz in self.cancels if t >= now - self.cfg.tape_window_ms)
        # executions proxy = tape volume
        execs = sum(sz for t, _, sz, _ in self.trades if t >= now - self.cfg.tape_window_ms)
        if execs <= 0:
            return 0.0
        return canc / max(1.0, execs)

    def _trade_through_ratio(self) -> float:
        """
        Fraction of prints occurring at a price worse than current touch for passive side.
        Without NBBO feed, approximate: buy prints >= ask, sell prints <= bid.
        """
        now = self._now_ms()
        if self.bid <= 0 or self.ask <= 0:
            return 0.0
        trades = [x for x in self.trades if x[0] >= now - self.cfg.tape_window_ms]
        if not trades:
            return 0.0
        through = 0
        for _, px, _, s in trades:
            if s == "buy" and px >= self.ask:   # consumed ask or worse
                through += 1
            elif s == "sell" and px <= self.bid:
                through += 1
        return through / max(1, len(trades))

    def _oddlot_ratio(self) -> float:
        """
        If odd-lot flag not available, infer by tiny size < 1 standard lot (e.g., <100).
        """
        now = self._now_ms()
        trades = [x for x in self.trades if x[0] >= now - self.cfg.tape_window_ms]
        if not trades:
            return 0.0
        odd = sum(1 for _, _, sz, _ in trades if sz < 100.0)
        return odd / max(1, len(trades))

    def _depletion(self) -> Tuple[float, float]:
        # returns (buy_touch_depletion, sell_touch_depletion) over book_window
        now = self._now_ms()
        cutoff = now - self.cfg.book_window_ms
        # rough decay so it doesn’t keep growing
        self._touch_depletions_buy *= 0.9
        self._touch_depletions_sell *= 0.9
        # ensure history pruned already by on_l2
        return self._touch_depletions_buy, self._touch_depletions_sell

    def _markout_bps(self) -> float:
        """
        Worst (adverse) short-horizon mark-out in bps across configured horizons.
        Positive value means 'bad' for taking liquidity (adverse move).
        """
        if self.mid <= 0:
            return 0.0
        worst_bps = 0.0
        now = self._now_ms()
        for h in self.cfg.markout_horizons_ms:
            t0 = now - h
            # pick closest mid snapshot around t0
            ref_mid = None
            for ts, _, _, _, _ in reversed(self.l2_hist):
                if ts <= t0:
                    ref_mid = 0.5 * (self.bid + self.ask) if (self.bid > 0 and self.ask > 0) else self.mid
                    break
            if ref_mid is None:
                continue
            move = (self.mid - ref_mid) / max(1e-9, ref_mid)
            # adverse for buys = up, for sells = down; we take absolute 'badness'
            worst_bps = max(worst_bps, abs(move) * 1e4)
        return worst_bps

    # ---------- score ----------
    def score(self) -> float:
        """
        Returns toxicity score in [0,1].
        """
        # components normalized 0..1 and weighted
        spread_c = min(1.0, self.spread_bps_ewma / max(1e-9, self.cfg.spread_bps_norm))
        vol_c    = min(1.0, (math.sqrt(self.ret2_ewma) * 100.0) / max(1e-9, self.cfg.ret_z_norm))
        vpin_c   = min(1.0, self._vpin() / max(1e-9, self.cfg.vpin_norm))
        lam_c    = min(1.0, self.lambda_ewma / max(1e-12, self.cfg.kyle_lambda_norm))
        ofi_c    = min(1.0, abs(self._ofi()) / max(1e-9, self.cfg.ofi_norm))
        cx_c     = min(1.0, self._cancel_ratio() / max(1e-9, self.cfg.cancel_ratio_norm))
        thr_c    = min(1.0, self._trade_through_ratio() / max(1e-9, self.cfg.trade_through_norm))
        odd_c    = min(1.0, self._oddlot_ratio() / max(1e-9, self.cfg.oddlot_ratio_norm))
        dep_b, dep_s = self._depletion()
        dep_c   = min(1.0, max(dep_b, dep_s) / max(1.0, self.cfg.depletion_norm))
        mk_c    = min(1.0, self._markout_bps() / max(1e-9, self.cfg.markout_bps_norm))

        w = self.cfg
        parts = [
            (spread_c, w.w_spread),
            (vol_c,    w.w_vol),
            (vpin_c,   w.w_vpin),
            (lam_c,    w.w_lambda),
            (ofi_c,    w.w_ofi),
            (cx_c,     w.w_cxl),
            (thr_c,    w.w_through),
            (odd_c,    w.w_oddlot),
            (dep_c,    w.w_deplete),
            (mk_c,     w.w_markout),
        ]
        wsum = sum(weight for _, weight in parts) or 1.0
        score = sum(val * weight for val, weight in parts) / wsum
        return max(0.0, min(1.0, score))

    def regime(self) -> str:
        s = self.score()
        if s >= self.cfg.block_threshold:
            return "toxic"
        if s >= self.cfg.warn_threshold:
            return "caution"
        return "calm"

    def should_block(self, side: Side) -> bool:
        """
        Block aggressive action (market/cross) when toxicity is high.
        You can also bias by side with OFI direction (optional).
        """
        s = self.score()
        if s < self.cfg.block_threshold:
            return False
        # Optional directional bias: if OFI strongly against our side, be stricter
        ofi = self._ofi()
        if side == "buy" and ofi < -self.cfg.ofi_norm:
            return True
        if side == "sell" and ofi > self.cfg.ofi_norm:
            return True
        return True

    # ---------- convenience ----------
    def snapshot(self) -> Dict[str, float]:
        """Return components for dashboards/logging."""
        dep_b, dep_s = self._depletion()
        return {
            "score": self.score(),
            "regime": {"calm": 0, "caution": 1, "toxic": 2}[self.regime()],
            "spread_bps": self.spread_bps_ewma,
            "vol_z": math.sqrt(self.ret2_ewma) * 100.0,
            "vpin": self._vpin(),
            "kyle_lambda": self.lambda_ewma,
            "ofi": self._ofi(),
            "cancel_ratio": self._cancel_ratio(),
            "trade_through": self._trade_through_ratio(),
            "oddlot_ratio": self._oddlot_ratio(),
            "depletion_buy": dep_b,
            "depletion_sell": dep_s,
            "markout_bps": self._markout_bps(),
        }