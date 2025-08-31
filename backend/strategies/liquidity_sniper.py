# backend/engine/strategies/liquidity_sniper.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, DefaultDict, List
from collections import defaultdict, deque

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


# ----------------------- Config -----------------------
@dataclass
class SniperConfig:
    symbols: tuple[str, ...] = ("AAPL",)
    venues: tuple[str, ...]  = ("IBKR", "ZERODHA", "PAPER")

    # Microstructure thresholds
    min_spread_bps: float = 0.5          # only act if spread is at least this wide
    min_imbalance: float  = 0.65         # bid/(bid+ask) to go long; <(1-x) to go short
    thin_thresh_qty: float = 0.25        # if top-of-book qty fraction is below → “thin”
    gap_levels: int = 2                  # levels to check for gaps (L2 if provided)

    # Order sizing
    default_qty: float = 1.0
    max_notional: float = 50_000.0

    # Cooldowns (ms)
    symbol_cooldown_ms: int = 600
    venue_cooldown_ms: int  = 300

    # Behavior
    allow_aggression: bool = True        # allow occasional market sweep on detected burst
    burst_window_ms: int = 400
    burst_min_trades: int = 5            # trades in window to qualify as burst
    peg_improve_bps: float = 0.2         # if stepping inside spread, how much to improve (bps of mid)
    cancel_replace_ms: int = 1500        # after this, re-evaluate/replace resting order
    hard_kill: bool = False


# ----------------------- Strategy -----------------------
class LiquiditySniper(Strategy):
    """
    Microstructure 'liquidity sniper':
      - Watches L1/L2 (if present) for one-sided imbalance, thin quotes, and gaps
      - Places pegged/improved LIMIT orders to capture spread when favorable
      - On bursts (tape speed-up) can flip to taker for momentum continuation
      - Emits signal in [-1,+1]; OMS/risk enforce limits

    Expected tick shapes (tolerant):
      {symbol|s, venue|v, bid, ask, bid_size?, ask_size?, bids?[], asks?[], type?("quote"|"trade"), size?}
      For L2, 'bids'/'asks' are lists of [price, size] or dicts {"p":..,"q":..}
    """

    def __init__(self, name="alpha_liquidity_sniper", region=None, cfg: Optional[SniperConfig] = None):
        cfg = cfg or SniperConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty)
        self.cfg = cfg

        # per-venue best quotes: book[sym][venue] = (bid, bid_sz, ask, ask_sz, ts_ms, bids_l2, asks_l2)
        self.book: DefaultDict[str, Dict[str, Tuple[float, float, float, float, int, list, list]]] = defaultdict(dict)

        # quick tape buffer for burst detection: trades deque of (ts_ms, side, px, qty)
        self.tape: DefaultDict[str, deque] = defaultdict(lambda: deque(maxlen=512))

        # cooldown trackers
        self._last_trade_symbol_ms: Dict[str, int] = defaultdict(lambda: 0)
        self._last_trade_venue_ms: Dict[str, int] = defaultdict(lambda: 0)

        # resting order memory (very lightweight hinting; real OMS holds truth)
        self._last_action_ms: Dict[str, int] = defaultdict(lambda: 0)
        self._last_side: Dict[str, Optional[str]] = defaultdict(lambda: None)

    # ------------- lifecycle ----------------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["microstructure", "liquidity", "maker-taker"],
            "region": self.ctx.region or "US",
            "notes": "Snipes thin quotes/imbalance; pegs limit orders and sweeps on bursts."
        })

    # ------------- helpers ------------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _mid(bid: float, ask: float) -> float:
        if bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
        return max(bid, ask, 0.0)

    def _cooldown_hit(self, sym: str, ven: str, now: int) -> bool:
        if now - self._last_trade_symbol_ms[sym] < self.cfg.symbol_cooldown_ms:
            return True
        if now - self._last_trade_venue_ms[ven] < self.cfg.venue_cooldown_ms:
            return True
        return False

    @staticmethod
    def _norm_levels(levels: Optional[List[Any]]) -> list:
        """
        Accept [ [px,qty], ... ] or [ {"p":..,"q":..}, ... ] -> return [(px,qty), ...]
        """
        out = []
        if not levels:
            return out
        for row in levels:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                try:
                    out.append((float(row[0]), float(row[1])))
                except Exception:
                    continue
            elif isinstance(row, dict):
                try:
                    out.append((float(row.get("p") or row.get("price")), float(row.get("q") or row.get("size")))) # type: ignore
                except Exception:
                    continue
        return out

    def _gap_ratio(self, side_levels: list, up_to_n: int) -> float:
        """
        Return a crude 'gap ratio' for top N levels: bigger = thinner.
        """
        if len(side_levels) < 2:
            return 1.0
        n = min(up_to_n, len(side_levels))
        qtys = [max(0.0, q) for _, q in side_levels[:n]]
        total = sum(qtys) or 1.0
        top = qtys[0]
        return 1.0 - (top / total)

    def _imbalance(self, bid_sz: float, ask_sz: float) -> float:
        tot = max(1e-9, bid_sz + ask_sz)
        return bid_sz / tot  # in [0,1]; >0.5 bull; <0.5 bear

    def _bursting(self, sym: str, now: int) -> Tuple[bool, float]:
        """
        Tape-speed detector: count recent trades within window; return (is_burst, drift_sign)
        drift_sign: +1 if mostly upticks, -1 if downticks.
        """
        dq = self.tape[sym]
        # drop old
        window = self.cfg.burst_window_ms
        while dq and (now - dq[0][0]) > window:
            dq.popleft()
        if len(dq) >= self.cfg.burst_min_trades:
            ups = sum(1 for _, side, *_ in dq if side == "buy")
            downs = len(dq) - ups
            sign = 1.0 if ups > downs else -1.0 if downs > ups else 0.0
            return True, sign
        return False, 0.0

    # ------------- main ---------------------
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

        # Track trades for burst detection
        if typ == "trade":
            side = (tick.get("side") or "").lower()  # "buy"/"sell" aggressor if provided
            px = float(tick.get("price") or tick.get("p") or 0.0)
            qty = float(tick.get("size") or 0.0)
            if px > 0 and qty >= 0:
                self.tape[sym].append((now, side, px, qty))
            return  # keep trades separate; quotes drive decision

        # Normalize quotes
        bid = tick.get("bid"); ask = tick.get("ask")
        if bid is None or ask is None:
            # attempt to reconstruct from mid
            mid = tick.get("mid")
            if mid:
                try:
                    mid = float(mid)
                    bid = bid or mid
                    ask = ask or mid
                except Exception:
                    return
        try:
            bid = float(bid or 0.0)
            ask = float(ask or 0.0)
        except Exception:
            return
        if bid <= 0 and ask <= 0:
            return
        if ask > 0 and bid > ask:
            bid = ask = max(bid, ask)

        bid_sz = float(tick.get("bid_size") or tick.get("bs") or 0.0)
        ask_sz = float(tick.get("ask_size") or tick.get("as") or 0.0)
        bids_l2 = self._norm_levels(tick.get("bids"))
        asks_l2 = self._norm_levels(tick.get("asks"))

        self.book[sym][ven] = (bid, bid_sz, ask, ask_sz, now, bids_l2, asks_l2)

        # Construct aggregated L1 across venues
        best_bid: Optional[Tuple[str, float, float]] = None  # (venue, price, size)
        best_ask: Optional[Tuple[str, float, float]] = None

        for v, (b, bs, a, asz, ts, *_levels) in self.book[sym].items():
            if b > 0 and (best_bid is None or b > best_bid[1]):
                best_bid = (v, b, bs)
            if a > 0 and (best_ask is None or a < best_ask[1]):
                best_ask = (v, a, asz)

        if not (best_bid and best_ask):
            return

        bven, bpx, bsz = best_bid
        aven, apx, asz = best_ask
        mid = self._mid(bpx, apx)
        if mid <= 0:
            return

        spread_bps = (apx - bpx) / mid * 1e4
        imb = self._imbalance(bsz, asz)
        # Side L2 thinness/gap signals (optional, robust if L2 present on best venues)
        bl2 = self.book[sym][bven][5]
        al2 = self.book[sym][aven][6]
        gap_bid = self._gap_ratio(bl2, self.cfg.gap_levels)
        gap_ask = self._gap_ratio(al2, self.cfg.gap_levels)

        # Build attractiveness score:
        #  + long_score grows when spread wide, imbalance to bid, ask side thin/gappy
        #  + short_score grows when spread wide, imbalance to ask, bid side thin/gappy
        long_score = 0.0
        short_score = 0.0

        if spread_bps >= self.cfg.min_spread_bps:
            # imbalance contribution
            if imb >= self.cfg.min_imbalance:
                long_score += (imb - self.cfg.min_imbalance) * 2.0
            if imb <= (1.0 - self.cfg.min_imbalance):
                short_score += ((1.0 - self.cfg.min_imbalance) - imb) * 2.0

            # thin/gap contribution
            if asz > 0:
                thin_ask = min(1.0, (self.cfg.thin_thresh_qty / max(asz, 1e-9)))
                long_score += 0.25 * thin_ask + 0.25 * gap_ask
            if bsz > 0:
                thin_bid = min(1.0, (self.cfg.thin_thresh_qty / max(bsz, 1e-9)))
                short_score += 0.25 * thin_bid + 0.25 * gap_bid

        # Normalize to [-1, +1]
        sig = long_score - short_score
        sig = max(-1.0, min(1.0, sig))
        self.emit_signal(sig)

        # Cooldown / notional guard
        if self._cooldown_hit(sym, bven if sig > 0 else aven, now):
            return
        qty = self.ctx.default_qty or self.cfg.default_qty
        if mid * qty > self.cfg.max_notional:
            return

        # Decide action
        if sig > 0.15:
            # Bullish: prefer to BID (maker) slightly inside the spread
            improve = mid * (self.cfg.peg_improve_bps / 1e4)
            my_bid = min(apx - 0.01, bpx + improve) if apx > 0 else bpx  # penny/cent tick guard
            self.order(
                symbol=sym, side="buy", qty=qty, order_type="limit",
                limit_price=my_bid, venue=bven,
                extra={"reason": "sniper_join_bid", "mid": mid, "spread_bps": spread_bps, "imb": imb}
            )
            self._last_trade_symbol_ms[sym] = now
            self._last_trade_venue_ms[bven] = now
            self._last_action_ms[sym] = now
            self._last_side[sym] = "buy"

        elif sig < -0.15:
            # Bearish: prefer to OFFER slightly inside spread
            improve = mid * (self.cfg.peg_improve_bps / 1e4)
            my_ask = max(bpx + 0.01, apx - improve) if bpx > 0 else apx
            self.order(
                symbol=sym, side="sell", qty=qty, order_type="limit",
                limit_price=my_ask, venue=aven,
                extra={"reason": "sniper_join_ask", "mid": mid, "spread_bps": spread_bps, "imb": imb}
            )
            self._last_trade_symbol_ms[sym] = now
            self._last_trade_venue_ms[aven] = now
            self._last_action_ms[sym] = now
            self._last_side[sym] = "sell"

        # Optional aggression on burst
        is_burst, drift = self._bursting(sym, now)
        if self.cfg.allow_aggression and is_burst and abs(sig) >= 0.25 and drift != 0.0:
            side = "buy" if drift > 0 else "sell"
            self.order(
                symbol=sym, side=side, qty=qty, order_type="market", mark_price=mid,
                extra={"reason": "sniper_burst_follow", "drift": drift, "sig": sig}
            )
            self._last_trade_symbol_ms[sym] = now
            self._last_trade_venue_ms[aven if side == "buy" else bven] = now
            self._last_action_ms[sym] = now
            self._last_side[sym] = side

        # Cancel/replace hint (soft): if we placed last order long ago and signal flipped
        last_side = self._last_side[sym]
        if last_side and (now - self._last_action_ms[sym] > self.cfg.cancel_replace_ms):
            if (last_side == "buy" and sig < 0) or (last_side == "sell" and sig > 0):
                # fire small opposite to nudge OMS toward cancel/replace in your pipeline
                self.order(
                    symbol=sym, side=("sell" if last_side == "buy" else "buy"), qty=0.0001,
                    order_type="market", mark_price=mid,
                    extra={"reason": "sniper_flip_hint"}
                )
                self._last_action_ms[sym] = now
                self._last_side[sym] = None


# ---------------------- optional runner ----------------------
if __name__ == "__main__":
    """
    Run (wired elsewhere normally):
      python -m backend.engine.strategies.liquidity_sniper
    Attach with: strat.run(stream="ticks.equities.us")
    """
    strat = LiquiditySniper()
    # strat.run(stream="ticks.equities.us")