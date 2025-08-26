# backend/execution/dark_pool.py
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Quotes / price utils (soft import to play nice with your codebase)
# -----------------------------------------------------------------------------
try:
    from backend.execution.pricer import Quote # type: ignore
except Exception:
    @dataclass
    class Quote:
        symbol: str
        bid: Optional[float] = None
        ask: Optional[float] = None
        last: Optional[float] = None
        def mid(self) -> Optional[float]:
            if self.bid and self.ask:
                return (self.bid + self.ask) / 2.0
            return self.last

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

OrderID = str
TradeID = str

@dataclass(order=True)
class _Resting:
    """
    Internal resting-liquidity record.
    Uses a min-heap by arrival (price is always midpoint-pegged in this venue).
    """
    arrival: float
    qty: float
    side: str     # 'buy' or 'sell'
    order_id: OrderID = field(compare=False)
    owner: str = field(default="", compare=False)    # strategy/account tag
    min_qty: float = field(default=0.0, compare=False)
    peg: str = field(default="mid", compare=False)   # 'mid' only for now
    meta: Dict[str, Any] = field(default_factory=dict, compare=False)

@dataclass
class Order:
    symbol: str
    side: str                       # 'buy' or 'sell'
    qty: float
    tif: str = "IOC"                # 'IOC', 'FOK', 'DAY' (DAY = remain hidden)
    min_qty: float = 0.0            # minimum acceptable execution size
    peg: str = "mid"                # 'mid' (midpoint-peg)
    owner: str = ""                 # strategy/account tag
    order_id: Optional[OrderID] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    trade_id: TradeID
    ts: float
    symbol: str
    price: float
    qty: float
    side: str               # aggressor side
    maker_order_id: OrderID
    taker_order_id: OrderID
    venue: str
    fees: float             # signed (taker negative)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VenueConfig:
    name: str = "DPX"                   # venue code
    taker_fee_bps: float = 0.0          # e.g., 0.2 bps
    maker_rebate_bps: float = 0.0       # e.g., 0.1 bps
    min_exec_qty: float = 0.0           # venue-wide minimum
    midpoint_round: Optional[float] = None  # e.g., $0.005 to round to half-penny
    allow_cross_when_locked: bool = True
    allow_cross_when_wide_only: bool = False   # execute only if spread >= X (see 'min_spread')
    min_spread: float = 0.0             # if >0 and allow_cross_when_wide_only=True
    latency_ms: float = 0.0             # synthetic venue latency (one-way) for fills
    max_child_pct_adv: float = 0.2      # liquidity protection per child vs ADV (0..1); 0 disables
    adv_by_symbol: Dict[str, float] = field(default_factory=dict)  # ADV map for caps
    record_exec_paths: bool = True

# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------

class DarkPool:
    """
    Minimal dark-pool crossing engine (midpoint peg, hidden book).
    - Resting orders are hidden; price is always midpoint of reference NBBO.
    - Supports IOC/FOK/DAY, min_qty, venue-wide min exec size, taker fee & maker rebate.
    - Optional liquidity caps (per child notional vs ADV).
    - External NBBO must be supplied on each call (Quote with bid/ask).
    """
    def __init__(self, cfg: VenueConfig):
        self.cfg = cfg
        self._resting: Dict[str, List[_Resting]] = {}   # symbol -> heap by arrival
        self._id_seq = itertools.count(1)
        self._trade_seq = itertools.count(1)

    # ---------- Public API ----------
    def submit(self, order: Order, nbbo: Quote) -> Tuple[List[Trade], float]:
        """
        Submit an order; returns (trades, leaves_qty).
        For DAY orders, any unfilled quantity rests hidden.
        For IOC, unfilled is canceled.
        For FOK, all-or-none at once or cancel (min_qty is enforced too).
        """
        assert order.side in ("buy", "sell"), "side must be 'buy' or 'sell'"
        assert order.peg == "mid", "only midpoint-peg supported"
        order.order_id = order.order_id or f"DPX{next(self._id_seq)}"
        now = time.time()

        mid = self._mid(nbbo)
        if mid is None:
            return ([], order.qty)  # no reference price

        # Spread guard
        if self.cfg.allow_cross_when_wide_only and nbbo.bid and nbbo.ask:
            if (nbbo.ask - nbbo.bid) < max(0.0, self.cfg.min_spread):
                return ([], order.qty)

        # compute max eligible qty due to ADV cap
        cap_qty = self._cap_qty(order.symbol, mid)
        target_qty = min(order.qty, cap_qty) if cap_qty > 0 else order.qty

        if order.tif.upper() == "FOK":
            # Check if enough contra is available *right now* to satisfy min(all, min_qty)
            avail = self._peek_available(order.symbol, contra=order.side, min_clip=max(order.min_qty, self.cfg.min_exec_qty))
            if avail < max(order.min_qty, target_qty):
                return ([], order.qty)  # cancel all

        trades, leaves = self._cross(order, nbbo, now, target_qty, mid)

        # Rest leftover (DAY only) if any qty remains and meets min resting size
        if leaves > 0 and order.tif.upper() == "DAY":
            self._rest(order, leaves, now)

        return trades, leaves

    def cancel_all(self, symbol: Optional[str] = None, owner: Optional[str] = None) -> int:
        """
        Cancel resting liquidity. Returns number of canceled clips.
        """
        n = 0
        keys = list(self._resting.keys()) if symbol is None else [symbol]
        for sym in keys:
            heap = self._resting.get(sym, [])
            keep = []
            while heap:
                r = heapq.heappop(heap)
                if owner and r.owner != owner:
                    keep.append(r)
                else:
                    n += 1
            for r in keep:
                heapq.heappush(heap, r)
        return n

    def snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Hidden book statistics (does not reveal counterparties).
        """
        bu, su = 0.0, 0.0
        for r in self._resting.get(symbol, []):
            if r.side == "buy":
                bu += r.qty
            else:
                su += r.qty
        return {"symbol": symbol, "resting_buy": bu, "resting_sell": su, "clips": len(self._resting.get(symbol, []))}

    # ---------- Core mechanics ----------
    def _cross(self, order: Order, nbbo: Quote, now: float, qty_cap: float, mid: float) -> Tuple[List[Trade], float]:
        """
        Match against hidden contra at midpoint. Oldest liquidity fills first.
        """
        trades: List[Trade] = []
        need = max(0.0, qty_cap)
        if need <= 0:
            return trades, order.qty

        heap = self._resting.get(order.symbol, [])
        if not heap:
            return trades, order.qty

        contra_side = "sell" if order.side == "buy" else "buy"
        min_clip = max(order.min_qty, self.cfg.min_exec_qty)

        while need > 0 and heap:
            r = heap[0]  # peek oldest
            if r.side != contra_side:
                # Skip same-side liquidity; rotate heap item to scan others (cheap)
                rr = heapq.heappop(heap)
                heapq.heappush(heap, rr)
                # If we made a full loop without finding contra, break
                # (since we only store both sides mixed, if head isn't contra after rotation, no contra exists)
                same = all(x.side != contra_side for x in heap)
                if same:
                    break
                continue

            # Determine executable size
            clip = min(need, r.qty)
            if clip <= 0:
                heapq.heappop(heap)
                continue

            # Enforce per-clip minimums (both taker min and maker min)
            if clip < max(min_clip, r.min_qty):
                # if maker has larger min, try to consume larger chunk if available
                if need >= max(min_clip, r.min_qty) and r.qty >= max(min_clip, r.min_qty):
                    clip = max(min_clip, r.min_qty)
                else:
                    # skip this maker (can't satisfy min sizes)
                    rr = heapq.heappop(heap)
                    heapq.heappush(heap, rr)
                    # detect no progress
                    break

            px = self._round_mid(mid)
            if px <= 0:
                break

            # Fees (taker pays, maker may get rebate)
            taker_fee = -(self.cfg.taker_fee_bps / 1e4) * clip * px
            maker_rebate = +(self.cfg.maker_rebate_bps / 1e4) * clip * px

            tr = Trade(
                trade_id=f"T{next(self._trade_seq)}",
                ts=now + self.cfg.latency_ms / 1000.0,
                symbol=order.symbol,
                price=px,
                qty=clip,
                side=order.side,
                maker_order_id=r.order_id,
                taker_order_id=order.order_id or "",
                venue=self.cfg.name,
                fees=taker_fee,
                meta={"maker_rebate": maker_rebate} if self.cfg.maker_rebate_bps else {},
            )
            trades.append(tr)

            # Update book
            need -= clip
            r.qty -= clip
            if r.qty <= 1e-9:
                heapq.heappop(heap)  # remove exhausted
            else:
                # Update head
                heapq.heapreplace(heap, r)

            # IOC semantics: continue until need==0 or no contra
            # FOK semantics handled earlier (availability check)

        leaves = max(0.0, order.qty - sum(t.qty for t in trades))
        return trades, leaves

    def _rest(self, order: Order, qty: float, now: float) -> None:
        sym = order.symbol
        heap = self._resting.setdefault(sym, [])
        rec = _Resting(
            arrival=now,
            qty=float(qty),
            side=order.side,
            order_id=order.order_id or f"DPX{next(self._id_seq)}",
            owner=order.owner or "",
            min_qty=max(0.0, order.min_qty),
            peg=order.peg,
            meta=order.meta or {},
        )
        heapq.heappush(heap, rec)

    def _mid(self, nbbo: Quote) -> Optional[float]:
        if nbbo and nbbo.bid and nbbo.ask:
            m = (nbbo.bid + nbbo.ask) / 2.0
            return self._round_mid(m)
        return nbbo.mid() if nbbo else None

    def _round_mid(self, px: float) -> float:
        if self.cfg.midpoint_round and self.cfg.midpoint_round > 0:
            step = self.cfg.midpoint_round
            return round(px / step) * step
        return px

    def _cap_qty(self, symbol: str, mid: float) -> float:
        """
        Limit child size by max_child_pct_adv of ADV (if configured).
        """
        if self.cfg.max_child_pct_adv <= 0:
            return float("inf")
        adv = self.cfg.adv_by_symbol.get(symbol)
        if not adv or adv <= 0:
            return float("inf")
        cap_notional = adv * float(self.cfg.max_child_pct_adv)
        return cap_notional / max(1e-12, mid)

    def _peek_available(self, symbol: str, contra: str, min_clip: float) -> float:
        """
        Sum immediately executable contra qty above min clip thresholds.
        """
        tot = 0.0
        for r in self._resting.get(symbol, []):
            if r.side == contra and r.qty >= max(min_clip, r.min_qty):
                tot += r.qty
        return tot


# -----------------------------------------------------------------------------
# Venue adapter (router-facing)
# -----------------------------------------------------------------------------

class DarkPoolVenueAdapter:
    """
    Simple adapter that your router can treat like a broker/venue.
    Methods mirror a tiny subset of your broker interface.
    """
    def __init__(self, engine: DarkPool):
        self.eng = engine

    def place_ioc(self, symbol: str, side: str, qty: float, nbbo: Quote, *, min_qty: float = 0.0, owner: str = "", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        o = Order(symbol=symbol, side=side, qty=qty, tif="IOC", min_qty=min_qty, owner=owner, meta=meta or {})
        trades, leaves = self.eng.submit(o, nbbo)
        return {"trades": [asdict(t) for t in trades], "leaves_qty": leaves}

    def place_fok(self, symbol: str, side: str, qty: float, nbbo: Quote, *, min_qty: float = 0.0, owner: str = "", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        o = Order(symbol=symbol, side=side, qty=qty, tif="FOK", min_qty=min_qty, owner=owner, meta=meta or {})
        trades, leaves = self.eng.submit(o, nbbo)
        return {"trades": [asdict(t) for t in trades], "leaves_qty": leaves}

    def place_day(self, symbol: str, side: str, qty: float, nbbo: Quote, *, min_qty: float = 0.0, owner: str = "", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        o = Order(symbol=symbol, side=side, qty=qty, tif="DAY", min_qty=min_qty, owner=owner, meta=meta or {})
        trades, leaves = self.eng.submit(o, nbbo)
        return {"trades": [asdict(t) for t in trades], "leaves_qty": leaves}

    def cancel_all(self, symbol: Optional[str] = None, owner: Optional[str] = None) -> int:
        return self.eng.cancel_all(symbol=symbol, owner=owner)

    def book_snapshot(self, symbol: str) -> Dict[str, Any]:
        return self.eng.snapshot(symbol)


# -----------------------------------------------------------------------------
# Tiny smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Venue config (half-penny midpoint, tiny fees, ADV cap)
    cfg = VenueConfig(
        name="DPX",
        taker_fee_bps=0.2,
        maker_rebate_bps=0.1,
        midpoint_round=0.005,
        allow_cross_when_wide_only=True,
        min_spread=0.01,
        max_child_pct_adv=0.05,
        adv_by_symbol={"AAPL": 2.5e10},  # $25bn adv
    )
    dp = DarkPool(cfg)
    venue = DarkPoolVenueAdapter(dp)

    # Reference NBBO
    q = Quote(symbol="AAPL", bid=192.00, ask=192.04)

    # Rest hidden DAY liquidity
    venue.place_day("AAPL", "sell", 25_000, q, min_qty=500)
    venue.place_day("AAPL", "sell", 10_000, q, min_qty=250)

    # IOC taker tries to cross at midpoint
    res = venue.place_ioc("AAPL", "buy", 12_000, q, min_qty=500, owner="alpha1")
    print("IOC result:", res)

    # Snapshot hidden book after partial fills
    print("Book:", venue.book_snapshot("AAPL"))

    # FOK (all-or-none)
    res2 = venue.place_fok("AAPL", "buy", 50_000, q, min_qty=5_000)
    print("FOK result:", res2)