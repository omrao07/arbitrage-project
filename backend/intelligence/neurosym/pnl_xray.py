# backend/analytics/pnl_xray.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------- Soft deps / hooks -----------------------------------
try:
    # optional: your cost model (returns fees given trade dict)
    from backend.execution.cost_model import estimate_fees  # type: ignore
except Exception:
    def estimate_fees(trade: Dict[str, Any]) -> float:
        # fallback: 1 bps + 0.5 fixed per trade
        notional = abs(float(trade.get("qty", 0.0)) * float(trade.get("price", 0.0)))
        return 0.0001 * notional + 0.5

try:
    # optional: sector/industry map, beta exposures, etc.
    from backend.data.classifiers import classify_symbol  # type: ignore
except Exception:
    def classify_symbol(symbol: str) -> Dict[str, str]:
        return {"sector": "UNKNOWN", "industry": "UNKNOWN"}

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# ----------------------- Data models -----------------------------------------

@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0  # VWAP of open lots
    realized_pnl: float = 0.0
    fees: float = 0.0

@dataclass
class Mark:
    price: float
    ts: float

@dataclass
class SliceKey:
    # dimensions to attribute on (any can be "")
    strategy: str = ""
    symbol: str = ""
    venue: str = ""
    region: str = ""
    sector: str = ""
    book: str = ""       # e.g., "alpha", "arb", "hedge"
    side: str = ""       # "long" / "short"

    def as_tuple(self) -> Tuple[str, ...]:
        return (self.strategy, self.symbol, self.venue, self.region, self.sector, self.book, self.side)

@dataclass
class SliceStats:
    realized: float = 0.0
    unrealized: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    fees: float = 0.0
    turnover: float = 0.0
    trades: int = 0
    max_dd: float = 0.0
    peak: float = 0.0
    trough: float = 0.0
    exposure: float = 0.0  # sum |qty*price|
    last_ts: float = 0.0

    # rolling equity to compute drawdown at this slice granularity
    equity_curve: List[Tuple[float, float]] = field(default_factory=list)  # (ts, net_pnl_cum)

# ----------------------- Core engine -----------------------------------------

class PnLXray:
    """
    Real-time PnL attribution by arbitrary dimensions (strategy/symbol/venue/sector/region/book/side).
    - Keeps positions & marks
    - Computes realized/unrealized, fees, turnover, exposure
    - Tracks drawdown per-slice
    - Optional alpha/beta split if you feed benchmark returns/betas
    """

    def __init__(self, *, base_ccy: str = "USD"):
        self.base_ccy = base_ccy
        self._pos: Dict[Tuple[str, str], Position] = {}           # (strategy, symbol) -> Position
        self._mark: Dict[str, Mark] = {}                           # symbol -> Mark
        self._slices: Dict[Tuple[str, ...], SliceStats] = {}       # SliceKey.tuple -> stats
        self._beta: Dict[str, float] = {}                          # symbol -> beta to benchmark (optional)
        self._bench_ret: float = 0.0                               # latest benchmark return (period)
        self._tags_cb = None                                       # optional callback(trade)->dict extra dims

    # ---------------- Tags / factors ----------------

    def set_beta(self, symbol: str, beta: float) -> None:
        self._beta[symbol.upper()] = float(beta)

    def set_benchmark_return(self, r: float) -> None:
        """Call per bar/period before snapshotting to enable alpha/beta split in snapshots."""
        self._bench_ret = float(r)

    def set_trade_tags_callback(self, cb) -> None:
        """
        cb(trade_dict) -> dict of extra tags to add to SliceKey (e.g., {"book":"alpha"}).
        """
        self._tags_cb = cb

    # ---------------- Ingestion ---------------------

    def ingest_trade(self, trade: Dict[str, Any]) -> None:
        """
        trade must include:
          ts (float or int seconds), strategy, symbol, side ('buy'/'sell'), qty, price
        optional: venue, region, book
        """
        ts = float(trade.get("ts") or time.time())
        strat = str(trade["strategy"])
        sym = str(trade["symbol"]).upper()
        side = str(trade.get("side","")).lower()
        qty = float(trade["qty"])
        px = float(trade["price"])
        venue = str(trade.get("venue",""))
        region = str(trade.get("region",""))
        book = str(trade.get("book",""))
        sector = classify_symbol(sym).get("sector","UNKNOWN")
        side_tag = "long" if (side == "buy") else "short"

        # optional tags from user callback
        extra = self._tags_cb(trade) if self._tags_cb else {}
        book = extra.get("book", book)
        region = extra.get("region", region)

        pos_key = (strat, sym)
        pos = self._pos.setdefault(pos_key, Position())
        fees = float(trade.get("fees", estimate_fees(trade)))
        signed = qty if side == "buy" else -qty

        # Realized PnL on closing part
        realized = 0.0
        if signed * pos.qty < 0:  # crossing the zero or reducing existing
            close_qty = min(abs(signed), abs(pos.qty)) * (1 if pos.qty > 0 else -1)
            # PnL = (sell - buy) * closed_qty
            entry = pos.avg_price
            realized = (entry - px) * close_qty if pos.qty > 0 else (px - entry) * (-close_qty)

        # Update position (FIFO approximated by running average)
        new_qty = pos.qty + signed
        if (pos.qty == 0) or (pos.qty * signed > 0):
            # adding to same side -> update vwap
            pos.avg_price = ((pos.avg_price * abs(pos.qty)) + (px * abs(signed))) / max(1e-9, (abs(pos.qty) + abs(signed)))
        elif pos.qty * signed < 0 and abs(signed) > abs(pos.qty):
            # flipped side: new avg at residual
            residual = new_qty
            pos.avg_price = px  # start new lot at trade price
        pos.qty = new_qty

        pos.realized_pnl += realized
        pos.fees += fees

        # slice accounting
        notional = abs(qty * px)
        k = SliceKey(strategy=strat, symbol=sym, venue=venue, region=region, sector=sector, book=book, side=side_tag).as_tuple()
        s = self._slices.setdefault(k, SliceStats())
        s.realized += realized
        s.fees += fees
        s.turnover += notional
        s.trades += 1
        s.exposure += notional
        s.last_ts = ts

        # rolling equity (gross -> net)
        s.gross_pnl = s.realized + s.unrealized  # unrealized may be updated by marks later
        s.net_pnl = s.gross_pnl - s.fees
        self._update_dd(s, ts)

    def update_mark(self, symbol: str, price: float, ts: Optional[float] = None) -> None:
        """Update mark; recompute unrealized PnL for all strategies holding the symbol."""
        ts = float(ts or time.time())
        sym = symbol.upper()
        self._mark[sym] = Mark(price=float(price), ts=ts)
        # recompute unrealized per (strategy, symbol) and propagate into slice stats
        for (strat, s) in [k for k in self._pos.keys() if k[1] == sym]:
            pos = self._pos[(strat, sym)]
            u = (price - pos.avg_price) * pos.qty
            # update all slices that match this (strategy, symbol)
            for key, st in self._slices.items():
                if key[0] == strat and key[1] == sym:
                    st.unrealized = u
                    st.gross_pnl = st.realized + st.unrealized
                    st.net_pnl = st.gross_pnl - st.fees
                    st.last_ts = ts
                    self._update_dd(st, ts)

    # ---------------- Attribution / snapshots --------------

    def snapshot(self, *, group_by: Iterable[str] = ("strategy","symbol")) -> Dict[str, Any]:
        """
        Roll up current stats by the requested dimensions. Valid dims:
          strategy, symbol, venue, region, sector, book, side
        Includes optional alpha/beta split if benchmark return and beta are set.
        """
        dims = tuple(group_by)
        out: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        for key, st in self._slices.items():
            dmap = {"strategy": key[0], "symbol": key[1], "venue": key[2], "region": key[3], "sector": key[4], "book": key[5], "side": key[6]}
            gid = tuple(dmap[d] for d in dims)
            agg = out.setdefault(gid, {"realized":0.0,"unrealized":0.0,"gross":0.0,"net":0.0,"fees":0.0,"turnover":0.0,"trades":0,"exposure":0.0,"max_dd":0.0})
            agg["realized"] += st.realized
            agg["unrealized"] += st.unrealized
            agg["gross"] += st.gross_pnl
            agg["net"] += st.net_pnl
            agg["fees"] += st.fees
            agg["turnover"] += st.turnover
            agg["trades"] += st.trades
            agg["exposure"] += st.exposure
            agg["max_dd"] = max(agg["max_dd"], st.max_dd)

            # alpha/beta split if possible (only meaningful when grouping includes symbol)
            if "symbol" in dims and self._bench_ret != 0.0:
                sym = dmap["symbol"]
                beta = self._beta.get(sym, 0.0)
                # Approximate beta PnL = beta * bench_ret * |position_value| (use latest mark)
                mk = self._mark.get(sym)
                if mk:
                    pos = self._pos.get((dmap.get("strategy",""), sym))
                    if pos and pos.qty != 0:
                        pos_val = pos.qty * mk.price
                        beta_pnl = beta * self._bench_ret * pos_val
                        agg.setdefault("beta_pnl", 0.0)
                        agg.setdefault("alpha_pnl", 0.0)
                        agg["beta_pnl"] += beta_pnl
                        agg["alpha_pnl"] += (st.realized + st.unrealized) - beta_pnl

        # prettify keys
        pretty = []
        for gid, vals in out.items():
            row = {dims[i]: gid[i] for i in range(len(dims))}
            row.update({k: round(float(v), 6) for k, v in vals.items()}) # type: ignore
            pretty.append(row)
        pretty.sort(key=lambda r: (-r.get("net", 0.0), tuple(str(r[d]) for d in dims)))
        return {"group_by": dims, "rows": pretty, "ts": int(time.time()*1000)}

    # ---------------- I/O ----------------------------

    def to_json(self) -> str:
        payload = {
            "base_ccy": self.base_ccy,
            "positions": {f"{k[0]}|{k[1]}": asdict(v) for k, v in self._pos.items()},
            "marks": {s: asdict(m) for s, m in self._mark.items()},
            "slices": {"|".join(k): self._slice_to_dict(v) for k, v in self._slices.items()},
            "beta": self._beta,
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def from_json(s: str) -> "PnLXray":
        o = json.loads(s)
        xr = PnLXray(base_ccy=o.get("base_ccy","USD"))
        # positions
        for k, v in o.get("positions", {}).items():
            strat, sym = k.split("|", 1)
            xr._pos[(strat, sym)] = Position(**v)
        # marks
        for sym, m in o.get("marks", {}).items():
            xr._mark[sym] = Mark(**m)
        # slices
        for k, v in o.get("slices", {}).items():
            tup = tuple(k.split("|"))
            xr._slices[tup] = xr._slice_from_dict(v)
        xr._beta = {k: float(v) for k, v in o.get("beta", {}).items()}
        return xr

    def to_dataframe(self, *, group_by=("strategy","symbol")):
        if pd is None:
            raise RuntimeError("pandas not installed")
        snap = self.snapshot(group_by=group_by)["rows"]
        return pd.DataFrame(snap)

    # ---------------- Internals ----------------------

    def _slice_to_dict(self, s: SliceStats) -> Dict[str, Any]:
        d = asdict(s)
        # avoid dumping full equity curve for huge runs unless needed
        return {k: v for k, v in d.items() if k != "equity_curve"}

    def _slice_from_dict(self, d: Dict[str, Any]) -> SliceStats:
        s = SliceStats(**{k: v for k, v in d.items() if k in SliceStats.__annotations__})
        return s

    def _update_dd(self, s: SliceStats, ts: float) -> None:
        """
        Update equity curve and max drawdown for the slice (net PnL cumulative).
        """
        prev = s.equity_curve[-1][1] if s.equity_curve else 0.0
        cur = prev + (s.net_pnl - (s.equity_curve[-1][1] if s.equity_curve else 0.0))
        # actually we want cumulative net; easier: store instantaneous and derive peak/trough
        cur = s.net_pnl
        s.equity_curve.append((ts, cur))
        if not s.equity_curve:
            s.peak = s.trough = cur
            s.max_dd = 0.0
            return
        s.peak = max(s.peak, cur) if s.peak or s.peak == 0.0 else cur
        s.trough = min(s.trough, cur) if s.trough or s.trough == 0.0 else cur
        s.max_dd = max(s.max_dd, max(0.0, s.peak - cur))

# ----------------------- Tiny demo -------------------------------------------

if __name__ == "__main__":
    xr = PnLXray()
    # trades
    xr.ingest_trade({"ts": time.time(), "strategy": "alpha.meanrev", "symbol":"AAPL", "side":"buy", "qty":100, "price":190.0, "venue":"NASDAQ", "region":"US"})
    xr.update_mark("AAPL", 191.0)
    xr.ingest_trade({"ts": time.time(), "strategy": "alpha.meanrev", "symbol":"AAPL", "side":"sell", "qty":50, "price":191.5, "venue":"NASDAQ", "region":"US"})
    xr.update_mark("AAPL", 190.8)
    xr.set_beta("AAPL", 1.2)
    xr.set_benchmark_return(0.003)  # +30 bps period return

    # snapshot examples
    print(json.dumps(xr.snapshot(group_by=("strategy","symbol")), indent=2))
    print(json.dumps(xr.snapshot(group_by=("strategy","sector")), indent=2))
    if pd:
        print(xr.to_dataframe(group_by=("strategy","symbol")).head())