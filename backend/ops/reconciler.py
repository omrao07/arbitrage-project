# backend/oms/reconciler.py
from __future__ import annotations
"""
Reconciler — broker ↔ ledger state
----------------------------------
Compares broker snapshots against the local ledger (SQLite), finds deltas, and
optionally proposes/applies fixes (append-only corrections in the ledger).

Scope:
- Orders (presence/status/qty)
- Fills (presence/qty/price)
- Positions (qty/avg price)
- Cash balances (by currency)

Optional integrations:
- backend.ledger.ledger.Ledger   (for local store)
- backend.bus.streams.publish_stream  (for notifications)  [optional]

Broker adapter contract (duck-typed; implement these on your adapter):
    broker.list_orders() -> list[dict]        # id, ts_ms, account_id, symbol, side, qty, status, order_type, limit_price?
    broker.list_fills()  -> list[dict]        # id, ts_ms, order_id, account_id, symbol, price, qty, venue?
    broker.positions()   -> list[dict]        # account_id, symbol, qty, avg_px
    broker.cash()        -> dict[str,float]   # {ccy: balance}
    broker.name          -> str               # adapter name (for report)

CLI:
    python -m backend.oms.reconciler run \
        --db data/ledger.db \
        --account ACC1 \
        --out_json recon.json \
        --out_csv recon.csv \
        --apply_fixes false

By default runs in dry mode (no mutations). Set --apply_fixes true to write
missing orders/fills and correction events into the ledger (append-only).
"""

import argparse, json, math, os, csv, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------- optional modules ----------
try:
    from backend.ledger.ledger import Ledger # type: ignore
except Exception as e:
    Ledger = None  # type: ignore

try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:  # type: ignore
        pass

RECON_EVENTS_STREAM = os.getenv("RECON_EVENTS_STREAM", "ops.recon.events")

# ---------- tolerances ----------
EPS_QTY   = float(os.getenv("RECON_EPS_QTY", "1e-8"))
EPS_PX    = float(os.getenv("RECON_EPS_PX",  "1e-6"))
EPS_CASH  = float(os.getenv("RECON_EPS_CASH","1e-6"))

# ---------- helpers ----------
def _almost(a: float, b: float, eps: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= eps
    except Exception:
        return False

def _ts_ms() -> int:
    return int(time.time() * 1000)

def _canon_row(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common fields for stable matching."""
    x = dict(d)
    if "symbol" in x and x["symbol"] is not None:
        x["symbol"] = str(x["symbol"]).upper()
    if "side" in x and x["side"] is not None:
        x["side"] = str(x["side"]).lower()
    if "order_type" in x and x["order_type"] is not None:
        x["order_type"] = str(x["order_type"]).lower()
    return x

# ---------- core types ----------
@dataclass
class ReconDelta:
    domain: str                 # orders|fills|positions|cash
    kind: str                   # missing|extra|mismatch
    key: Dict[str, Any]         # matching key (e.g., id, symbol)
    broker: Optional[Dict[str, Any]]
    ledger: Optional[Dict[str, Any]]
    detail: Optional[str] = None
    fix: Optional[Dict[str, Any]] = None      # proposed fix (append-only)

@dataclass
class ReconReport:
    ts_ms: int
    account_id: Optional[str]
    broker_name: str
    summary: Dict[str, int]
    deltas: List[ReconDelta]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "account_id": self.account_id,
            "broker_name": self.broker_name,
            "summary": self.summary,
            "deltas": [asdict(d) for d in self.deltas],
        }

# ---------- reconciler ----------
class Reconciler:
    def __init__(self, *, ledger: Ledger, broker: Any, account_id: Optional[str] = None): # type: ignore
        if ledger is None:
            raise RuntimeError("Ledger is required. Make sure backend.ledger.ledger is importable.")
        self.ledger = ledger
        self.broker = broker
        self.account_id = account_id
        self.broker_name = getattr(broker, "name", "broker")

    # ----- snapshots -----
    def snapshot_broker(self) -> Dict[str, Any]:
        orders = [ _canon_row(o) for o in (self.broker.list_orders() or []) ]
        fills  = [ _canon_row(f) for f in (self.broker.list_fills()  or []) ]
        pos    = [ _canon_row(p) for p in (self.broker.positions()   or []) ]
        cash   = dict(self.broker.cash() or {})
        if self.account_id:
            orders = [o for o in orders if o.get("account_id") == self.account_id]
            fills  = [f for f in fills  if f.get("account_id")  == self.account_id]
            pos    = [p for p in pos    if p.get("account_id")  == self.account_id]
        return {"orders": orders, "fills": fills, "positions": pos, "cash": cash}

    def snapshot_ledger(self) -> Dict[str, Any]:
        # Orders
        orders = self.ledger.get_open_orders(self.account_id)
        # Fills (raw table)
        cur = self.ledger.conn.cursor()
        if self.account_id:
            rows = cur.execute(
                "SELECT id, ts_ms, order_id, account_id, symbol, price, qty, venue, liquidity FROM fills WHERE account_id=?",
                (self.account_id,)
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT id, ts_ms, order_id, account_id, symbol, price, qty, venue, liquidity FROM fills"
            ).fetchall()
        fills = []
        for r in rows:
            fills.append(_canon_row({
                "id": r[0], "ts_ms": r[1], "order_id": r[2], "account_id": r[3],
                "symbol": r[4], "price": float(r[5]), "qty": float(r[6]),
                "venue": r[7], "liquidity": r[8]
            }))

        # Positions
        pos = self.ledger.get_positions(self.account_id)

        # Cash (by ccy)
        cash = self.ledger.get_cash_balance(self.account_id or orders[0]["account_id"] if orders else (self.account_id or "ACC1"))

        return {"orders": orders, "fills": fills, "positions": pos, "cash": cash}

    # ----- diffs -----
    def reconcile(self) -> ReconReport:
        b = self.snapshot_broker()
        l = self.snapshot_ledger()

        deltas: List[ReconDelta] = []
        # Orders: match by id
        b_orders = {o["id"]: o for o in b["orders"] if "id" in o}
        l_orders = {o["id"]: o for o in l["orders"] if "id" in o}
        # Missing in ledger
        for oid, bo in b_orders.items():
            if oid not in l_orders:
                fix = {"action": "record_order", "order": {
                    "id": bo["id"], "ts_ms": bo.get("ts_ms") or _ts_ms(),
                    "account_id": bo["account_id"], "symbol": bo["symbol"],
                    "side": bo["side"], "qty": float(bo["qty"]),
                    "order_type": bo.get("order_type", "market"),
                    "limit_price": bo.get("limit_price"),
                    "status": bo.get("status","accepted")
                }}
                deltas.append(ReconDelta("orders", "missing", {"id": oid}, broker=bo, ledger=None, detail="order missing in ledger", fix=fix))
        # Extra in ledger (not at broker)
        for oid, lo in l_orders.items():
            if oid not in b_orders:
                deltas.append(ReconDelta("orders", "extra", {"id": oid}, broker=None, ledger=lo, detail="order not found at broker"))

        # Fills: try by id; fallback fuzzy (order_id+qty+price within eps)
        b_fills = b["fills"]
        l_fills = l["fills"]
        l_by_id = {f["id"]: f for f in l_fills if f.get("id")}
        seen_l_ids: set[str] = set()

        for bf in b_fills:
            bf_id = bf.get("id")
            if bf_id and bf_id in l_by_id:
                seen_l_ids.add(bf_id)
                # check qty/price close
                lf = l_by_id[bf_id]
                if not _almost(bf["qty"], lf["qty"], EPS_QTY) or not _almost(bf["price"], lf["price"], EPS_PX):
                    deltas.append(ReconDelta("fills", "mismatch", {"id": bf_id}, broker=bf, ledger=lf,
                                             detail="fill qty/price mismatch"))
                continue
            # fuzzy match
            match = None
            for lf in l_fills:
                lid = lf.get("id")
                if lid in seen_l_ids:
                    continue
                if lf.get("order_id") == bf.get("order_id") and lf.get("symbol") == bf.get("symbol") and _almost(lf["qty"], bf["qty"], EPS_QTY) and _almost(lf["price"], bf["price"], EPS_PX):
                    match = lf; break
            if match:
                if not _almost(bf["price"], match["price"], EPS_PX):
                    deltas.append(ReconDelta("fills","mismatch",{"order_id": bf.get("order_id"), "symbol": bf.get("symbol")},
                                             broker=bf, ledger=match, detail="price differs"))
                seen_l_ids.add(match.get("id",""))
            else:
                # missing fill in ledger
                fix = {"action": "record_fill", "fill": {
                    "id": bf_id or f"BFILL-{int(time.time()*1e6)}",
                    "ts_ms": bf.get("ts_ms") or _ts_ms(),
                    "order_id": bf.get("order_id"),
                    "account_id": bf.get("account_id"),
                    "symbol": bf.get("symbol"),
                    "price": float(bf.get("price")),
                    "qty": float(bf.get("qty")),
                    "venue": bf.get("venue"),
                    "liquidity": bf.get("liquidity")
                }}
                deltas.append(ReconDelta("fills","missing",{"id": fix["fill"]["id"]}, broker=bf, ledger=None, detail="fill missing in ledger", fix=fix))

        # Extra fills in ledger
        b_ids = set([f["id"] for f in b_fills if f.get("id")])
        for lf in l_fills:
            lid = lf.get("id")
            if lid and lid not in b_ids:
                deltas.append(ReconDelta("fills","extra",{"id": lid}, broker=None, ledger=lf, detail="fill not found at broker"))

        # Positions: match by symbol
        b_pos = { (p["account_id"], p["symbol"]): p for p in b["positions"] }
        l_pos = { (p["account_id"], p["symbol"]): p for p in l["positions"] }
        keys = set(b_pos.keys()) | set(l_pos.keys())
        for k in keys:
            bp = b_pos.get(k); lp = l_pos.get(k)
            if bp and not lp:
                deltas.append(ReconDelta("positions","missing",{"account_id":k[0],"symbol":k[1]}, broker=bp, ledger=None, detail="position missing in ledger"))
            elif lp and not bp:
                deltas.append(ReconDelta("positions","extra",{"account_id":k[0],"symbol":k[1]}, broker=None, ledger=lp, detail="position not on broker"))
            else:
                if not _almost(bp["qty"], lp["qty"], EPS_QTY): # type: ignore
                    deltas.append(ReconDelta("positions","mismatch",{"account_id":k[0],"symbol":k[1]}, broker=bp, ledger=lp, detail="qty differs"))
                elif ("avg_px" in bp or "avg_px" in lp) and not _almost(float(bp.get("avg_px", lp.get("avg_px", 0.0))), float(lp.get("avg_px", bp.get("avg_px", 0.0))), EPS_PX): # type: ignore
                    deltas.append(ReconDelta("positions","mismatch",{"account_id":k[0],"symbol":k[1]}, broker=bp, ledger=lp, detail="avg_px differs"))

        # Cash (by ccy)
        for ccy, b_bal in (b["cash"] or {}).items():
            l_bal = (l["cash"] or {}).get(ccy, 0.0)
            if not _almost(b_bal, l_bal, EPS_CASH):
                deltas.append(ReconDelta("cash","mismatch",{"ccy": ccy}, broker={"balance": float(b_bal)}, ledger={"balance": float(l_bal)}, detail="cash balance differs"))

        summary = {
            "orders_missing": sum(1 for d in deltas if d.domain=="orders" and d.kind=="missing"),
            "orders_extra":   sum(1 for d in deltas if d.domain=="orders" and d.kind=="extra"),
            "orders_mismatch":sum(1 for d in deltas if d.domain=="orders" and d.kind=="mismatch"),
            "fills_missing":  sum(1 for d in deltas if d.domain=="fills" and d.kind=="missing"),
            "fills_extra":    sum(1 for d in deltas if d.domain=="fills" and d.kind=="extra"),
            "fills_mismatch": sum(1 for d in deltas if d.domain=="fills" and d.kind=="mismatch"),
            "positions_delta":sum(1 for d in deltas if d.domain=="positions"),
            "cash_delta":     sum(1 for d in deltas if d.domain=="cash"),
            "total":          len(deltas),
        }

        report = ReconReport(ts_ms=_ts_ms(), account_id=self.account_id, broker_name=self.broker_name, summary=summary, deltas=deltas)
        # emit lightweight event
        publish_stream(RECON_EVENTS_STREAM, {"ts_ms": report.ts_ms, "broker": self.broker_name, "account": self.account_id, "summary": summary})
        return report

    # ----- apply fixes (append-only) -----
    def apply_fixes(self, report: ReconReport, *, dry_run: bool = True) -> List[Dict[str, Any]]:
        """
        Applies only safe fixes:
          - Missing orders -> record_order
          - Missing fills  -> record_fill
        Mismatches and positions/cash are logged for manual review.
        """
        results: List[Dict[str, Any]] = []
        for d in report.deltas:
            if d.fix and d.domain in ("orders","fills") and d.kind == "missing":
                act = d.fix.get("action")
                try:
                    if dry_run:
                        results.append({"ok": True, "dry_run": True, "action": act, "key": d.key})
                        continue
                    if act == "record_order":
                        self.ledger.record_order(d.fix["order"])
                    elif act == "record_fill":
                        self.ledger.record_fill(d.fix["fill"])
                    else:
                        results.append({"ok": False, "action": act, "err": "unknown action"}); continue
                    results.append({"ok": True, "action": act, "key": d.key})
                except Exception as e:
                    results.append({"ok": False, "action": act, "key": d.key, "err": str(e)})
        return results

# ---------- CSV/JSON writers ----------
def write_json(path: str, report: ReconReport) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

def write_csv(path: str, report: ReconReport) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["domain","kind","key","detail","broker","ledger","fix"])
        for d in report.deltas:
            wr.writerow([
                d.domain, d.kind, json.dumps(d.key),
                d.detail or "",
                json.dumps(d.broker, separators=(",", ":")) if d.broker else "",
                json.dumps(d.ledger, separators=(",", ":")) if d.ledger else "",
                json.dumps(d.fix, separators=(",", ":")) if d.fix else "",
            ])

# ---------- CLI ----------
def _load_broker_adapter(path: str) -> Any:
    """
    Dynamically load a broker adapter given a dotted path, e.g.:
      backend.brokers.ibkr:Adapter
    The class is instantiated with no args.
    """
    if ":" not in path:
        raise SystemExit("broker adapter must be in form 'module.path:ClassName'")
    mod, cls = path.split(":", 1)
    import importlib
    m = importlib.import_module(mod)
    C = getattr(m, cls)
    return C()

def main():
    ap = argparse.ArgumentParser(description="Reconcile broker vs ledger")
    ap.add_argument("--db", required=True, help="Path to SQLite ledger db (e.g., data/ledger.db)")
    ap.add_argument("--broker_adapter", required=True, help="Dotted path to broker adapter class (e.g., backend.brokers.paperbroker:Adapter)")
    ap.add_argument("--account", default=None, help="Restrict reconciliation to this account id")
    ap.add_argument("--out_json", default=None, help="Write JSON report")
    ap.add_argument("--out_csv", default=None, help="Write CSV deltas")
    ap.add_argument("--apply_fixes", default="false", help="true/false: write missing orders/fills to ledger")
    args = ap.parse_args()

    if Ledger is None:
        raise SystemExit("backend.ledger.ledger.Ledger not found. Please add it to your project.")

    ledger = Ledger(args.db)
    broker = _load_broker_adapter(args.broker_adapter)
    recon = Reconciler(ledger=ledger, broker=broker, account_id=args.account)

    report = recon.reconcile()

    if args.out_json:
        write_json(args.out_json, report)
    if args.out_csv:
        write_csv(args.out_csv, report)

    apply_fixes = str(args.apply_fixes).lower() in {"1","true","yes","y"}
    if apply_fixes:
        res = recon.apply_fixes(report, dry_run=False)
        # append summary event
        publish_stream(RECON_EVENTS_STREAM, {"ts_ms": int(time.time()*1000), "broker": recon.broker_name, "account": args.account, "applied": res})

    print(json.dumps(report.to_dict(), indent=2))

if __name__ == "__main__":  # pragma: no cover
    main()