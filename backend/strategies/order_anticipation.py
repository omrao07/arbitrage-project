# backend/engine/strategies/order_anticipation.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, DefaultDict, Deque, List
from collections import defaultdict, deque

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


@dataclass
class AnticipationConfig:
    symbols: tuple[str, ...] = ("AAPL",)
    venues:  tuple[str, ...] = ("IBKR", "ZERODHA", "PAPER")

    # --- Feature windows / params ---
    ofi_window: int = 20                  # ticks
    trade_window_ms: int = 2000           # recent trades horizon
    vpin_bucket_vol: float = 5_000.0      # bucket size in shares/contracts
    vpin_max_buckets: int = 30            # rolling VPIN buckets
    vpin_min_buckets: int = 6

    # --- Execution knobs ---
    default_qty: float = 1.0
    max_notional: float = 75_000.0
    cool_ms_symbol: int = 800
    cool_ms_venue: int = 350
    peg_improve_bps: float = 0.15         # improve inside spread by this bps of mid
    enter_thresh: float = 0.25            # |signal| to act
    flip_thresh: float = 0.45             # stronger conviction → market-take

    # --- Behavior / safety ---
    allow_aggression: bool = True
    hard_kill: bool = False


class OrderAnticipation(Strategy):
    """
    Order anticipation micro-alpha:
      Fuses three footprints of meta-orders:
        1) OFI (Order-Flow Imbalance) at the top of book (Easley/O'Hara style).
        2) Aggressor trade imbalance in a short window.
        3) VPIN-style volume bucket toxicity (absolute buy/sell imbalance / total).
      Builds a score in [-1,+1]; places predictive pegs, occasionally takes when conviction high.

    Tick tolerance:
      - Quote: {symbol|s, venue|v, bid, ask, bid_size|bs, ask_size|as}
      - Trade: {symbol|s, venue|v, type:"trade", price|p, size, side?("buy"/"sell" aggressor)}
    """

    def __init__(self, name="alpha_order_anticipation", region=None, cfg: Optional[AnticipationConfig] = None):
        cfg = cfg or AnticipationConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty)
        self.cfg = cfg

        # L1 memory per symbol/venue: (bid, bs, ask, asz)
        self.l1: DefaultDict[str, Dict[str, Tuple[float, float, float, float]]] = defaultdict(dict)

        # OFI ring buffers per symbol
        self.ofi_buf: DefaultDict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))

        # Trade tape for imbalance (ts, side(+1/-1), price, size)
        self.tape: DefaultDict[str, Deque[Tuple[int, int, float, float]]] = defaultdict(lambda: deque(maxlen=1024))

        # VPIN buckets per symbol: list of (buy_vol, sell_vol); fill sequentially
        self.vpin_buckets: DefaultDict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=120))
        self.cur_bucket: DefaultDict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
        self.cur_bucket_fill: DefaultDict[str, float] = defaultdict(float)

        # Cooldowns
        self.last_sym_ms: DefaultDict[str, int] = defaultdict(lambda: 0)
        self.last_ven_ms: DefaultDict[str, int] = defaultdict(lambda: 0)

    # ---------------- lifecycle ----------------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["microstructure", "anticipation", "toxicity"],
            "region": self.ctx.region or "US",
            "notes": "OFI + trade imbalance + VPIN buckets; predictive pegs + occasional taker."
        })

    # ---------------- helpers ------------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _mid(bid: float, ask: float) -> float:
        if bid > 0 and ask > 0: return 0.5 * (bid + ask)
        return max(bid, ask, 0.0)

    def _ofi_increment(self, prev: Tuple[float,float,float,float], cur: Tuple[float,float,float,float]) -> float:
        """
        OFI (Cont et al.): ΔBidSize if bid up or same; -prevBidSize if bid down; analogous for ask.
        Simplified robust formulation at top of book.
        """
        pb, pbs, pa, pas = prev
        cb, cbs, ca, cas = cur
        inc = 0.0
        if cb > pb: inc += cbs
        elif cb < pb: inc -= pbs
        else: inc += (cbs - pbs)

        if ca < pa: inc -= cas
        elif ca > pa: inc += pas
        else: inc -= (cas - pas)
        return inc

    def _ofi_score(self, sym: str) -> float:
        buf = self.ofi_buf[sym]
        if not buf: return 0.0
        s = sum(buf[-self.cfg.ofi_window:]) if len(buf) >= self.cfg.ofi_window else sum(buf) # type: ignore
        # tanh normalization to [-1,1]
        return math.tanh(s / (1e4))  # scale safe; tune by symbol lot sizes

    def _trade_imbalance(self, sym: str, now: int) -> float:
        dq = self.tape[sym]
        # drop old
        horizon = self.cfg.trade_window_ms
        while dq and (now - dq[0][0]) > horizon:
            dq.popleft()
        if not dq: return 0.0
        buys = sum(1 for _, sgn, *_ in dq if sgn > 0)
        sells = len(dq) - buys
        tot = max(1, len(dq))
        return (buys - sells) / tot  # [-1,1]

    def _vpin(self, sym: str) -> float:
        """
        Volume-Synchronized Probability of Informed Trading proxy:
        mean(|B_i - S_i| / (B_i + S_i)) over recent buckets, mapped to [0,1].
        We also derive a signed tilt using last bucket (B-S)/(B+S).
        """
        buckets = self.vpin_buckets[sym]
        if len(buckets) < self.cfg.vpin_min_buckets:
            return 0.0
        fracs = []
        for b, s in buckets[-self.cfg.vpin_max_buckets:]: # type: ignore
            tot = max(1e-9, b + s)
            fracs.append(abs(b - s) / tot)
        return sum(fracs) / len(fracs)  # 0..1

    def _vpin_tilt(self, sym: str) -> float:
        buckets = self.vpin_buckets[sym]
        if not buckets: return 0.0
        b, s = buckets[-1]
        tot = max(1e-9, b + s)
        return (b - s) / tot  # [-1,1]

    # ---------------- main --------------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill:
            return

        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym not in self.cfg.symbols:
            return
        ven = (tick.get("venue") or tick.get("v") or "").upper()
        if ven and ven not in self.cfg.venues:
            return

        typ = (tick.get("type") or "").lower()
        now = self._now_ms()

        # --- Trades: fill trade tape + VPIN buckets ---
        if typ == "trade":
            px = float(tick.get("price") or tick.get("p") or 0.0)
            sz = float(tick.get("size") or 0.0)
            side_txt = (tick.get("side") or "").lower()
            sgn = +1 if side_txt == "buy" else -1 if side_txt == "sell" else 0
            if px > 0 and sz > 0:
                self.tape[sym].append((now, sgn, px, sz))
                # VPIN: allocate to current bucket by side
                b, s = self.cur_bucket[sym]
                b += sz if sgn > 0 else 0.0
                s += sz if sgn < 0 else 0.0
                self.cur_bucket[sym] = (b, s)
                self.cur_bucket_fill[sym] += sz
                if self.cur_bucket_fill[sym] >= self.cfg.vpin_bucket_vol:
                    self.vpin_buckets[sym].append(self.cur_bucket[sym])
                    self.cur_bucket[sym] = (0.0, 0.0)
                    self.cur_bucket_fill[sym] = 0.0
            return

        # --- Quotes: compute OFI and act ---
        bid = tick.get("bid"); ask = tick.get("ask")
        try:
            bid = float(bid or 0.0); ask = float(ask or 0.0)
        except Exception:
            return
        if bid <= 0 and ask <= 0:
            return
        bs = float(tick.get("bid_size") or tick.get("bs") or 0.0)
        asz = float(tick.get("ask_size") or tick.get("as") or 0.0)

        prev = self.l1[sym].get(ven)
        cur = (bid, bs, ask, asz)
        if prev:
            ofi_inc = self._ofi_increment(prev, cur)
            self.ofi_buf[sym].append(ofi_inc)
        self.l1[sym][ven] = cur  # update after

        # Aggregate best across venues
        best_bid = max((v for v in self.l1[sym].items()), key=lambda x: x[1][0], default=None)
        best_ask = min((v for v in self.l1[sym].items()), key=lambda x: x[1][2], default=None)
        if not best_bid or not best_ask:
            return
        bven, (bpx, bsz, _, _) = best_bid
        aven, (_, _, apx, asz_top) = best_ask
        mid = self._mid(bpx, apx)
        if mid <= 0: return

        # ---------- Build signal ----------
        ofi = self._ofi_score(sym)            # [-1,1]
        ti  = self._trade_imbalance(sym, now) # [-1,1]
        vpin = self._vpin(sym)                # [0,1] = toxicity level
        tilt = self._vpin_tilt(sym)           # [-1,1] = direction of toxicity

        # combine: direction = weighted average; magnitude scaled by toxicity
        dir_score = 0.5 * ofi + 0.3 * ti + 0.2 * tilt
        sig = max(-1.0, min(1.0, dir_score)) * max(0.0, min(1.0, vpin))
        self.emit_signal(sig)

        # ---------- Risk / cooldown ----------
        if now - self.last_sym_ms[sym] < self.cfg.cool_ms_symbol:
            return
        use_ven = bven if sig > 0 else aven
        if now - self.last_ven_ms[use_ven] < self.cfg.cool_ms_venue:
            return
        qty = self.ctx.default_qty or self.cfg.default_qty
        if mid * qty > self.cfg.max_notional:
            return

        # ---------- Execution ----------
        # Low/medium conviction → place improving limit (join/step)
        if abs(sig) >= self.cfg.enter_thresh and abs(sig) < self.cfg.flip_thresh:
            improve = mid * (self.cfg.peg_improve_bps / 1e4)
            if sig > 0:
                # bullish → work the bid slightly higher (but stay inside spread)
                limit_px = min(apx - 0.01, bpx + improve) if apx > 0 else bpx
                self.order(sym, "buy", qty=qty, order_type="limit", limit_price=limit_px, venue=bven,
                           extra={"reason": "anticipation_peg_bid", "ofi": ofi, "ti": ti, "vpin": vpin})
            else:
                # bearish → work the ask slightly lower
                limit_px = max(bpx + 0.01, apx - improve) if bpx > 0 else apx
                self.order(sym, "sell", qty=qty, order_type="limit", limit_price=limit_px, venue=aven,
                           extra={"reason": "anticipation_peg_ask", "ofi": ofi, "ti": ti, "vpin": vpin})
            self.last_sym_ms[sym] = now
            self.last_ven_ms[use_ven] = now
            return

        # High conviction + toxicity (optionally flip to taker)
        if self.cfg.allow_aggression and abs(sig) >= self.cfg.flip_thresh and vpin >= 0.4:
            side = "buy" if sig > 0 else "sell"
            self.order(sym, side, qty=qty, order_type="market", mark_price=mid,
                       extra={"reason": "anticipation_take", "ofi": ofi, "ti": ti, "vpin": vpin})
            self.last_sym_ms[sym] = now
            self.last_ven_ms[use_ven] = now


# ---------------------- optional runner ----------------------
if __name__ == "__main__":
    """
    Attach elsewhere in your runner, e.g.:
      strat = OrderAnticipation()
      # strat.run(stream="ticks.equities.us")
    """
    strat = OrderAnticipation()