# backend/execution/reconciler.py
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("reconciler")
if not log.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# Optional Redis helpers (your bus wrappers). If absent, Redis path is skipped.
try:
    from backend.bus.streams import hgetall, hset
except Exception:
    def hgetall(*_a, **_k):  # type: ignore
        return {}
    def hset(*_a, **_k):     # type: ignore
        return None

# Internal storage & broker APIs
try:
    from backend.execution.order_store import OrderStore # type: ignore
except Exception:
    OrderStore = None  # type: ignore

from backend.execution.broker_base import BrokerBase, PaperBroker, Position as BrokerPosition, Account as BrokerAccount # type: ignore


# ============================ Data Models ============================

@dataclass
class PosSnapshot:
    symbol: str
    qty: float
    avg_price: float
    source: str  # 'broker' | 'sqlite' | 'redis'

@dataclass
class CashSnapshot:
    equity: float
    cash: float
    buying_power: float
    currency: str
    source: str  # 'broker' | 'sqlite' | 'redis'

@dataclass
class PosDiff:
    symbol: str
    qty_broker: float
    qty_internal: float
    dq: float
    avg_broker: float
    avg_internal: float
    d_avg: float
    status: str  # 'ok' | 'warn' | 'mismatch'

@dataclass
class CashDiff:
    equity_broker: float
    equity_internal: Optional[float]
    cash_broker: float
    cash_internal: Optional[float]
    dequity: Optional[float]
    dcash: Optional[float]
    currency: str
    status: str  # 'ok' | 'warn' | 'mismatch'


# ============================ Reconciler ============================

class Reconciler:
    """
    Compares broker positions/cash with internal state (SQLite OrderStore and/or Redis).
    """

    def __init__(
        self,
        *,
        broker: Optional[BrokerBase] = None,
        db_path: str = "runtime/order_store.db",
        use_sqlite: bool = True,
        use_redis: bool = True,
        qty_tol: float = 1e-6,
        avg_tol_bps: float = 5.0,   # average cost tolerance in bps
        cash_tol: float = 1.00,     # absolute currency tolerance
        equity_tol: float = 5.00,   # absolute currency tolerance
        base_ccy: str = "USD",
    ):
        self.broker = broker or PaperBroker(currency=base_ccy)
        self.db_path = db_path
        self.use_sqlite = use_sqlite and (OrderStore is not None)
        self.use_redis = use_redis
        self.qty_tol = float(qty_tol)
        self.avg_tol_bps = float(avg_tol_bps)
        self.cash_tol = float(cash_tol)
        self.equity_tol = float(equity_tol)
        self.base_ccy = base_ccy

        self._store: Optional[OrderStore] = None # type: ignore
        if self.use_sqlite and OrderStore is not None:
            self._store = OrderStore(db_path=db_path)

    # ---- loading state ----

    def _load_broker_positions(self) -> Dict[str, PosSnapshot]:
        out: Dict[str, PosSnapshot] = {}
        try:
            self.broker.connect()
        except Exception:
            # assume already connected
            pass
        for p in self.broker.get_positions():
            sym = p.symbol.upper()
            out[sym] = PosSnapshot(symbol=sym, qty=float(p.qty), avg_price=float(p.avg_price), source="broker")
        return out

    def _load_broker_cash(self) -> CashSnapshot:
        acct = self.broker.get_account()
        return CashSnapshot(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            currency=str(acct.currency or self.base_ccy),
            source="broker",
        )

    def _load_sqlite_positions(self) -> Dict[str, PosSnapshot]:
        out: Dict[str, PosSnapshot] = {}
        if not self._store:
            return out
        for row in self._store.get_positions():
            sym = str(row["symbol"]).upper()
            out[sym] = PosSnapshot(symbol=sym, qty=float(row["qty"]), avg_price=float(row["avg_price"]), source="sqlite")
        return out

    def _load_redis_positions(self, key: str = os.getenv("REDIS_POS_KEY", "pos:live")) -> Dict[str, PosSnapshot]:
        out: Dict[str, PosSnapshot] = {}
        try:
            h = hgetall(key) or {}
            for sym, qty in h.items():
                out[sym.upper()] = PosSnapshot(symbol=sym.upper(), qty=float(qty), avg_price=float("nan"), source="redis")
        except Exception:
            pass
        return out

    # ---- compare helpers ----

    @staticmethod
    def _bps(x: float) -> float:
        return float(x) * 1e4

    def _avg_diff_bps(self, a: float, b: float) -> float:
        if a <= 0 or b <= 0:
            return math.inf if (a != b) else 0.0
        return self._bps((a - b) / b)

    # ---- reconcile ----

    def reconcile_positions(self) -> Dict[str, Any]:
        broker_pos = self._load_broker_positions()

        # merge internal (sqlite + redis). sqlite has avg_price; redis may not.
        internal_pos: Dict[str, PosSnapshot] = {}
        if self.use_sqlite:
            internal_pos.update(self._load_sqlite_positions())
        if self.use_redis:
            # do not overwrite sqlite avg_price if already present
            for sym, snap in self._load_redis_positions().items():
                if sym in internal_pos:
                    continue
                internal_pos[sym] = snap

        # union of symbols
        symbols = sorted(set(broker_pos.keys()) | set(internal_pos.keys()))
        diffs: List[PosDiff] = []
        n_ok = n_warn = n_mis = 0

        for sym in symbols:
            bp = broker_pos.get(sym, PosSnapshot(sym, 0.0, float("nan"), "broker"))
            ip = internal_pos.get(sym, PosSnapshot(sym, 0.0, float("nan"), "internal"))

            dq = float(bp.qty - ip.qty)
            d_avg_bps = self._avg_diff_bps(bp.avg_price, ip.avg_price) if (not math.isnan(bp.avg_price) and not math.isnan(ip.avg_price)) else math.inf

            status = "ok"
            if abs(dq) > self.qty_tol:
                status = "mismatch"
            elif not math.isinf(d_avg_bps) and abs(d_avg_bps) > self.avg_tol_bps:
                status = "warn"

            if status == "ok":
                n_ok += 1
            elif status == "warn":
                n_warn += 1
            else:
                n_mis += 1

            diffs.append(PosDiff(
                symbol=sym,
                qty_broker=bp.qty,
                qty_internal=ip.qty,
                dq=dq,
                avg_broker=bp.avg_price if not math.isnan(bp.avg_price) else 0.0,
                avg_internal=ip.avg_price if not math.isnan(ip.avg_price) else 0.0,
                d_avg=d_avg_bps if not math.isinf(d_avg_bps) else float("nan"),
                status=status,
            ))

        summary = {"symbols": len(symbols), "ok": n_ok, "warn": n_warn, "mismatch": n_mis}
        return {"summary": summary, "diffs": [asdict(d) for d in diffs]}

    def reconcile_cash(self) -> Dict[str, Any]:
        bro = self._load_broker_cash()

        # Try to infer internal cash from SQLite day pnl (if available)
        internal_equity = None
        internal_cash = None
        if self._store:
            try:
                pnl = self._store.get_pnl_day()
                # If you store starting equity/cash elsewhere, wire it here.
                # For now we only surface today's realized/fees; equity must be checked at broker.
                internal_cash = None
                internal_equity = None
            except Exception:
                pass

        deq = (bro.equity - internal_equity) if (internal_equity is not None) else None
        dca = (bro.cash - internal_cash) if (internal_cash is not None) else None

        status = "ok"
        if (deq is not None and abs(deq) > self.equity_tol) or (dca is not None and abs(dca) > self.cash_tol):
            status = "mismatch"

        diff = CashDiff(
            equity_broker=bro.equity,
            equity_internal=internal_equity,
            cash_broker=bro.cash,
            cash_internal=internal_cash,
            dequity=deq,
            dcash=dca,
            currency=bro.currency,
            status=status,
        )
        return {"cash": asdict(diff)}

    # ---- auto-fix hooks (optional) ----

    def write_redis_positions(self, key: str = os.getenv("REDIS_POS_KEY", "pos:live")) -> None:
        """
        Overwrite Redis pos:live with broker positions (use with care).
        """
        broker_pos = self._load_broker_positions()
        for sym, snap in broker_pos.items():
            try:
                hset(key, sym, snap.qty)
            except Exception as e:
                log.warning("hset(%s,%s) failed: %s", key, sym, e)

    def close(self) -> None:
        try:
            if self._store:
                self._store.close()
        except Exception:
            pass


# ============================ CLI ============================

def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Positions & cash reconciler")
    ap.add_argument("--db", default="runtime/order_store.db", help="SQLite path")
    ap.add_argument("--no-sqlite", action="store_true", help="disable SQLite comparison")
    ap.add_argument("--no-redis", action="store_true", help="disable Redis comparison")
    ap.add_argument("--qty-tol", type=float, default=1e-6, help="quantity tolerance")
    ap.add_argument("--avg-tol-bps", type=float, default=5.0, help="avg price tolerance in bps")
    ap.add_argument("--cash-tol", type=float, default=1.0, help="cash tolerance (abs)")
    ap.add_argument("--equity-tol", type=float, default=5.0, help="equity tolerance (abs)")
    ap.add_argument("--base-ccy", default="USD")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    ap.add_argument("--write-redis", action="store_true", help="overwrite Redis pos:live with broker positions")
    return ap.parse_args()

def main():
    args = _parse_args()
    rec = Reconciler(
        db_path=args.db,
        use_sqlite=not args.no_sqlite,
        use_redis=not args.no_redis,
        qty_tol=args.qty_tol,
        avg_tol_bps=args.avg_tol_bps,
        cash_tol=args.cash_tol,
        equity_tol=args.equity_tol,
        base_ccy=args.base_ccy,
    )

    try:
        pos_report = rec.reconcile_positions()
        cash_report = rec.reconcile_cash()
        out = {"positions": pos_report, "cash": cash_report}

        if args.json:
            print(json.dumps(out, indent=None, separators=(",", ":")))
        else:
            print(json.dumps(out, indent=2))

        if args.write_redis:
            rec.write_redis_positions()
            log.info("Redis pos:live updated from broker state.")

    finally:
        rec.close()

if __name__ == "__main__":
    main()