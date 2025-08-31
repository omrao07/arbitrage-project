# backend/analytics/pnl_xray.py
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple, Any, List, Literal

Side = Literal["buy", "sell"]


# -------------------------- helpers --------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _sgn(side: Side) -> int:
    return +1 if side == "buy" else -1

def _safe(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN guard
            return d
        return v
    except Exception:
        return d


# -------------------------- data model -----------------------

@dataclass
class PositionState:
    """FIFO position with realized PnL accounting."""
    qty: float = 0.0               # signed (+ long / - short)
    costs: float = 0.0             # cumulative cash spent (positive cash outlay for buys)
    realized: float = 0.0          # realized PnL in quote CCY
    fees: float = 0.0              # fees/levies (signed; negative reduces PnL)
    carry: float = 0.0             # funding/borrow/financing accrued (negative if you pay)
    fx_pnl: float = 0.0            # FX translation PnL to reporting CCY
    greeks_pnl: float = 0.0        # delta-hedge pnl / theta/vanna/vega if you feed it
    slip_impact: float = 0.0       # slippage + impact attribution from execution
    idio: float = 0.0              # residual/idiosyncratic
    last_px: float = 0.0           # last mark price
    last_fx: float = 1.0           # last FX rate (local→reporting)
    lots: List[Tuple[float, float]] = field(default_factory=list)  # [(qty, price)], qty>0 long lots, qty<0 short lots

    def avg_price(self) -> float:
        if not self.lots:
            return 0.0
        q = sum(q for q, _ in self.lots)
        if abs(q) < 1e-12:
            return 0.0
        # weighted average with sign
        v = sum(q * px for q, px in self.lots)
        return v / q

    def market_value(self) -> float:
        return self.qty * self.last_px

    def unrealized(self) -> float:
        return (self.qty * self.last_px) - self.qty * self.avg_price()


@dataclass
class Attribution:
    price: float = 0.0
    fees: float = 0.0
    carry: float = 0.0
    fx: float = 0.0
    slip_impact: float = 0.0
    greeks: float = 0.0
    idio: float = 0.0

    def total(self) -> float:
        return self.price + self.fees + self.carry + self.fx + self.slip_impact + self.greeks + self.idio


@dataclass
class Key:
    strategy: str
    symbol: str
    venue: Optional[str] = None

    def tup(self) -> Tuple[str, str, Optional[str]]:
        return (self.strategy, self.symbol, self.venue)


@dataclass
class Row:
    ts_ms: int
    key: Key
    kind: str                     # "fill" | "mark" | "fee" | "carry" | "fx" | "greek" | "note"
    qty: float = 0.0              # signed trade qty for fills
    px: float = 0.0               # trade or mark price
    info: Dict[str, Any] = field(default_factory=dict)
    attrib: Attribution = field(default_factory=Attribution)


# -------------------------- main engine ----------------------

class PnLXray:
    """
    Real-time PnL tracker + attribution:
      - FIFO positions per (strategy, symbol, venue)
      - Realized/unrealized & MV
      - Multi-channel attribution: {price, fees, carry, fx, slip_impact, greeks, idio}
      - Roll-ups: per key + strategy/symbol/venue/time buckets

    Usage (minimal):
        xr = PnLXray(report_ccy="USD")
        xr.on_fill("alpha1", "AAPL", "NASDAQ", side="buy", qty=1000, px=190.10, fee=-1.8, slip_bps=3.5)
        xr.on_mark("alpha1", "AAPL", mark_px=191.00)     # updates unrealized (price attribution)
        snap = xr.snapshot()
    """

    def __init__(self, report_ccy: str = "USD"):
        self.report_ccy = report_ccy
        self.pos: Dict[Tuple[str, str, Optional[str]], PositionState] = {}
        self.ledger: List[Row] = []  # append-only event log (for dashboards)
        # aggregates
        self.by_strategy: Dict[str, Attribution] = {}
        self.by_symbol: Dict[str, Attribution] = {}
        self.by_venue: Dict[str, Attribution] = {}
        self._clock_ms = _now_ms()

    # --------------- internals ---------------
    def _ps(self, key: Key) -> PositionState:
        k = key.tup()
        if k not in self.pos:
            self.pos[k] = PositionState()
        return self.pos[k]

    def _bump(self, agg: Dict[str, Attribution], k: str, delta: Attribution) -> None:
        if k not in agg:
            agg[k] = Attribution()
        A = agg[k]
        A.price       += delta.price
        A.fees        += delta.fees
        A.carry       += delta.carry
        A.fx          += delta.fx
        A.slip_impact += delta.slip_impact
        A.greeks      += delta.greeks
        A.idio        += delta.idio

    def _log(self, row: Row) -> None:
        self.ledger.append(row)
        # soft cap
        if len(self.ledger) > 200_000:
            self.ledger = self.ledger[-120_000:]

    # --------------- public API ---------------

    # Fills & execution attribution
    def on_fill(
        self,
        strategy: str,
        symbol: str,
        venue: Optional[str],
        *,
        side: Side,
        qty: float,
        px: float,
        fee: float = 0.0,
        slip_bps: Optional[float] = None,
        impact_bps: Optional[float] = None,
        bench_px: Optional[float] = None,
        fx_rate: Optional[float] = None,  # local→reporting
    ) -> None:
        """
        Register a fill.
        - Realized PnL increases when a trade reduces/inverts position (FIFO).
        - fee (signed): negative lowers PnL.
        - slip_bps/impact_bps (optional): attribute execution shortfall separately.
        - bench_px (optional): execution benchmark (mid/open/VWAP) to improve price vs slip split.
        - fx_rate (optional): if instrument CCY != report CCY.
        """
        key = Key(strategy, symbol, venue)
        ps = self._ps(key)
        q = _safe(qty)
        p = _safe(px)
        f = _safe(fee)
        fx = _safe(fx_rate, ps.last_fx or 1.0)

        # Execution shortfall attribution (optional)
        slip_cash = 0.0
        if bench_px and slip_bps is None:
            # derive slip from bench if not provided (sign = cost to execute)
            if side == "buy":
                slip_cash = (p - bench_px) * q
            else:
                slip_cash = (bench_px - p) * q
        elif slip_bps is not None:
            slip_cash = (slip_bps / 1e4) * (p * q)

        impact_cash = 0.0
        if impact_bps is not None:
            impact_cash = (impact_bps / 1e4) * (p * q)

        # FIFO realized PnL
        signed_q = q * _sgn(side)  # buy +q, sell -q
        realized = 0.0

        if ps.qty == 0 or (ps.qty > 0 and signed_q > 0) or (ps.qty < 0 and signed_q < 0):
            # increasing exposure → add lot
            ps.lots.append((signed_q, p))
            ps.qty += signed_q
            ps.costs += p * signed_q
        else:
            # reducing / closing or flipping → realize against FIFO lots
            remaining = abs(signed_q)
            side_sign = 1 if signed_q < 0 else -1  # if sell reducing a long, side_sign=+1 for realized calc
            new_lots: List[Tuple[float, float]] = []
            for lot_q, lot_px in ps.lots:
                if remaining <= 0:
                    new_lots.append((lot_q, lot_px))
                    continue
                available = min(abs(lot_q), remaining)
                # PnL per matched slice:
                # If lot_q>0 (long) and we SELL available: realized = (trade_px - lot_px)*available
                # If lot_q<0 (short) and we BUY available: realized = (lot_px - trade_px)*available
                if lot_q > 0 and signed_q < 0:  # closing long with sell
                    realized += (p - lot_px) * available
                    lot_q -= available
                    remaining -= available
                elif lot_q < 0 and signed_q > 0:  # closing short with buy
                    realized += (lot_px - p) * available
                    lot_q += available
                    remaining -= available
                # keep residual lot if not fully consumed
                if abs(lot_q) > 1e-12:
                    new_lots.append((lot_q, lot_px))
            ps.lots = new_lots
            # update position qty and cash costs
            ps.qty += signed_q
            ps.costs += p * signed_q
            ps.realized += realized

        # fees, slip/impact attribution
        ps.fees += f
        ps.slip_impact += (slip_cash + impact_cash)

        # update last marks
        ps.last_px = p
        ps.last_fx = fx

        # bookkeeping / aggregates
        attrib = Attribution(
            price=realized,  # realized component from closing trades
            fees=f,
            slip_impact=(slip_cash + impact_cash),
            # carry/fx/greeks/idio are zero for the fill unless passed separately
        )
        row = Row(ts_ms=_now_ms(), key=key, kind="fill", qty=signed_q, px=p,
                  info={"bench_px": bench_px, "fx": fx}, attrib=attrib)
        self._log(row)
        self._bump(self.by_strategy, strategy, attrib)
        self._bump(self.by_symbol, symbol, attrib)
        if venue:
            self._bump(self.by_venue, venue, attrib)

    # Marks & price attribution (unrealized)
    def on_mark(self, strategy: str, symbol: str, mark_px: float, *, venue: Optional[str] = None) -> None:
        key = Key(strategy, symbol, venue)
        ps = self._ps(key)
        old_mv = ps.market_value()
        ps.last_px = _safe(mark_px, ps.last_px)
        new_mv = ps.market_value()
        delta = new_mv - old_mv
        # attribute to price
        attrib = Attribution(price=delta)
        self._log(Row(ts_ms=_now_ms(), key=key, kind="mark", px=ps.last_px, attrib=attrib))
        self._bump(self.by_strategy, strategy, attrib)
        self._bump(self.by_symbol, symbol, attrib)
        if venue:
            self._bump(self.by_venue, venue, attrib)

    # Fees/levies (SEC, stamp, commissions) not tied to a single fill
    def on_fee(self, strategy: str, symbol: str, amount: float, *, venue: Optional[str] = None) -> None:
        key = Key(strategy, symbol, venue)
        ps = self._ps(key)
        ps.fees += _safe(amount)
        attrib = Attribution(fees=_safe(amount))
        self._log(Row(ts_ms=_now_ms(), key=key, kind="fee", attrib=attrib))
        self._bump(self.by_strategy, strategy, attrib)
        self._bump(self.by_symbol, symbol, attrib)
        if venue:
            self._bump(self.by_venue, venue, attrib)

    # Funding/borrow carry accrual (per slice)
    def on_carry(self, strategy: str, symbol: str, amount: float, *, venue: Optional[str] = None) -> None:
        key = Key(strategy, symbol, venue)
        ps = self._ps(key)
        ps.carry += _safe(amount)
        attrib = Attribution(carry=_safe(amount))
        self._log(Row(ts_ms=_now_ms(), key=key, kind="carry", attrib=attrib))
        self._bump(self.by_strategy, strategy, attrib)
        self._bump(self.by_symbol, symbol, attrib)
        if venue:
            self._bump(self.by_venue, venue, attrib)

    # FX translation (if instrument CCY ≠ reporting CCY)
    def on_fx(self, strategy: str, symbol: str, local_mv: float, fx_old: float, fx_new: float,
              *, venue: Optional[str] = None) -> None:
        """
        local_mv: current market value in instrument's CCY
        fx_old/fx_new: old/new FX rates (local→reporting)
        """
        key = Key(strategy, symbol, venue)
        delta = _safe(local_mv) * (_safe(fx_new) - _safe(fx_old))
        ps = self._ps(key)
        ps.fx_pnl += delta
        ps.last_fx = _safe(fx_new, ps.last_fx)
        attrib = Attribution(fx=delta)
        self._log(Row(ts_ms=_now_ms(), key=key, kind="fx", attrib=attrib, info={"fx_old": fx_old, "fx_new": fx_new}))
        self._bump(self.by_strategy, strategy, attrib)
        self._bump(self.by_symbol, symbol, attrib)
        if venue:
            self._bump(self.by_venue, venue, attrib)

    # Greeks/hedging attribution (optional)
    def on_greeks(self, strategy: str, symbol: str, amount: float, *, venue: Optional[str] = None) -> None:
        key = Key(strategy, symbol, venue)
        ps = self._ps(key)
        ps.greeks_pnl += _safe(amount)
        attrib = Attribution(greeks=_safe(amount))
        self._log(Row(ts_ms=_now_ms(), key=key, kind="greek", attrib=attrib))
        self._bump(self.by_strategy, strategy, attrib)
        self._bump(self.by_symbol, symbol, attrib)
        if venue:
            self._bump(self.by_venue, venue, attrib)

    # Idiosyncratic/manual adjustment
    def on_idio(self, strategy: str, symbol: str, amount: float, *, venue: Optional[str] = None, note: str = "") -> None:
        key = Key(strategy, symbol, venue)
        ps = self._ps(key)
        ps.idio += _safe(amount)
        attrib = Attribution(idio=_safe(amount))
        self._log(Row(ts_ms=_now_ms(), key=key, kind="note", attrib=attrib, info={"note": note}))
        self._bump(self.by_strategy, strategy, attrib)
        self._bump(self.by_symbol, symbol, attrib)
        if venue:
            self._bump(self.by_venue, venue, attrib)

    # --------------- queries / exports ---------------

    def position_snapshot(self) -> Dict[str, Any]:
        """Flat snapshot per key with positions and running components."""
        out: Dict[str, Any] = {}
        for (strategy, symbol, venue), ps in self.pos.items():
            k = f"{strategy}|{symbol}|{venue or 'ALL'}"
            out[k] = {
                "qty": ps.qty,
                "avg_px": ps.avg_price(),
                "last_px": ps.last_px,
                "mv": ps.market_value(),
                "unrealized": ps.unrealized(),
                "realized": ps.realized,
                "fees": ps.fees,
                "carry": ps.carry,
                "fx": ps.fx_pnl,
                "greeks": ps.greeks_pnl,
                "slip_impact": ps.slip_impact,
                "idio": ps.idio,
            }
        return out

    def attribution_rollup(self) -> Dict[str, Dict[str, float]]:
        """Roll-ups by dimension for dashboards."""
        def pack(A: Attribution) -> Dict[str, float]:
            return {
                "price": A.price,
                "fees": A.fees,
                "carry": A.carry,
                "fx": A.fx,
                "slip_impact": A.slip_impact,
                "greeks": A.greeks,
                "idio": A.idio,
                "total": A.total(),
            }
        return {
            "by_strategy": {k: pack(v) for k, v in self.by_strategy.items()},
            "by_symbol":   {k: pack(v) for k, v in self.by_symbol.items()},
            "by_venue":    {k: pack(v) for k, v in self.by_venue.items()},
        } # type: ignore

    def ledger_tail(self, n: int = 2000) -> List[Dict[str, Any]]:
        """Recent rows (append-only event log) for UI tables."""
        return [self._row_to_dict(r) for r in self.ledger[-max(1, n):]]

    @staticmethod
    def _row_to_dict(r: Row) -> Dict[str, Any]:
        d = asdict(r)
        d["key"] = {"strategy": r.key.strategy, "symbol": r.key.symbol, "venue": r.key.venue}
        d["attrib"]["total"] = r.attrib.total()
        return d

    def snapshot(self) -> Dict[str, Any]:
        """Full snapshot: positions + rollups + last N events."""
        return {
            "positions": self.position_snapshot(),
            "attribution": self.attribution_rollup(),
            "events": self.ledger_tail(1000),
            "report_ccy": self.report_ccy,
            "ts_ms": _now_ms(),
        }


# ------------------------------ demo ------------------------------
if __name__ == "__main__":
    xr = PnLXray()

    # Buy → mark up → sell with fees & slip attribution
    xr.on_fill("alpha1", "AAPL", "NASDAQ", side="buy", qty=1000, px=100.00, fee=-2.5, slip_bps=3.0, bench_px=99.98)
    xr.on_mark("alpha1", "AAPL", mark_px=100.40)  # +unrealized(price)
    xr.on_carry("alpha1", "AAPL", amount=-0.8)
    xr.on_fee("alpha1", "AAPL", amount=-0.5)
    xr.on_fill("alpha1", "AAPL", "NASDAQ", side="sell", qty=600, px=100.30, fee=-1.6, slip_bps=1.2)
    xr.on_greeks("alpha1", "AAPL", amount=+0.3)
    xr.on_fx("alpha1", "AAPL", local_mv=1000*100.30, fx_old=1.00, fx_new=1.00)

    from pprint import pprint
    pprint(xr.snapshot())