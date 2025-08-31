# backend/reporting/report_generator.py
from __future__ import annotations
"""
Report Generator — PnL / TCA / Risk snapshot from the SQLite ledger
-------------------------------------------------------------------
Inputs:
  - backend/ledger/ledger.py (SQLite DB)
  - Optional: latest marks via --marks (JSON file: {"AAPL": 195.1, ...})
  - Optional: Redis metrics (vol/drawdown) if available

Outputs:
  - Markdown (.md) or HTML (.html) report with KPIs + tables
  - JSON (.json) with the same structured data

CLI:
  python -m backend.reporting.report_generator \
      --db data/ledger.db \
      --account ACC1 \
      --start 2025-08-01 --end 2025-08-30 \
      --out reports/aug_report.md --fmt md
"""

import argparse, contextlib, datetime as dt, json, os, sqlite3, time
from typing import Any, Dict, List, Optional

# -------- optional redis (vol/drawdown metrics) ----------
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

# -------- helpers ----------
def _to_epoch_ms(d: dt.datetime) -> int:
    return int(d.timestamp() * 1000)

def _parse_date(s: str) -> dt.datetime:
    s = str(s).strip()
    if s.isdigit():
        return dt.datetime.utcfromtimestamp(int(s) / 1000.0)
    return dt.datetime.strptime(s, "%Y-%m-%d")

def _fmt_ccy(x: float | None) -> str:
    try:
        return f"${float(x):,.2f}" # type: ignore
    except Exception:
        return "$0.00"

def _safe_float(x: Any, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([line1, line2, body]) if rows else "\n".join([line1, line2])

# -------- core ----------
class ReportGenerator:
    def __init__(self, db_path: str, account_id: Optional[str] = None, *, marks: Optional[Dict[str, float]] = None,
                 redis_url: Optional[str] = None):
        self.db_path = db_path
        self.account_id = account_id
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.marks = marks or {}
        self.r = None
        if redis_url and redis:
            with contextlib.suppress(Exception):
                self.r = redis.Redis.from_url(redis_url, socket_timeout=0.3)

    # ----- data pulls -----
    def _fills_in_range(self, t0_ms: int, t1_ms: int) -> List[sqlite3.Row]:
        q = """
            SELECT f.id, f.ts_ms, f.order_id, f.account_id, f.symbol, f.price, f.qty, f.venue, f.liquidity,
                   o.side, o.order_type
            FROM fills f
            LEFT JOIN orders o ON o.id = f.order_id
            WHERE f.ts_ms >= ? AND f.ts_ms < ?
        """
        args = [t0_ms, t1_ms]
        if self.account_id:
            q += " AND f.account_id=?"; args.append(self.account_id) # type: ignore
        q += " ORDER BY f.ts_ms ASC"
        return list(self.conn.execute(q, args).fetchall())

    def _positions_snapshot(self) -> List[sqlite3.Row]:
        q = "SELECT account_id, symbol, qty, avg_px, realized_pnl FROM positions"
        if self.account_id:
            q += " WHERE account_id=?"
            return list(self.conn.execute(q, [self.account_id]).fetchall())
        return list(self.conn.execute(q).fetchall())

    def _cash_deltas(self, t0_ms: int, t1_ms: int) -> List[sqlite3.Row]:
        q = """
            SELECT ts_ms, account_id, ccy, delta, reason, ref
            FROM cash_ledger
            WHERE ts_ms >= ? AND ts_ms < ?
        """
        args = [t0_ms, t1_ms]
        if self.account_id:
            q += " AND account_id=?"; args.append(self.account_id) # type: ignore
        q += " ORDER BY ts_ms"
        return list(self.conn.execute(q, args).fetchall())

    def _orders_in_range(self, t0_ms: int, t1_ms: int) -> List[sqlite3.Row]:
        q = """
            SELECT id, ts_ms, account_id, symbol, side, qty, order_type, limit_price, status
            FROM orders
            WHERE ts_ms >= ? AND ts_ms < ?
        """
        args = [t0_ms, t1_ms]
        if self.account_id:
            q += " AND account_id=?"; args.append(self.account_id) # type: ignore
        q += " ORDER BY ts_ms"
        return list(self.conn.execute(q, args).fetchall())

    # ----- metrics -----
    def _compute_vwap(self, fills: List[sqlite3.Row]) -> Dict[str, float]:
        by_sym: Dict[str, Dict[str, float]] = {}
        for f in fills:
            s = f["symbol"]
            notional = _safe_float(f["price"]) * abs(_safe_float(f["qty"]))
            d = by_sym.setdefault(s, {"notional": 0.0, "qty": 0.0})
            d["notional"] += notional; d["qty"] += abs(_safe_float(f["qty"]))
        return {s: (v["notional"] / v["qty"] if v["qty"] > 0 else 0.0) for s, v in by_sym.items()}

    def _estimate_fees(self, fills: List[sqlite3.Row], cash_rows: List[sqlite3.Row]) -> float:
        # Estimate fee as residual between theoretical cash and actual cash per fill.
        cash_by_ref: Dict[str, float] = {}
        for c in cash_rows:
            if c["reason"] == "fill" and c["ref"]:
                cash_by_ref[c["ref"]] = cash_by_ref.get(c["ref"], 0.0) + _safe_float(c["delta"])
        fee_est_total = 0.0
        for f in fills:
            ref = f["id"]
            if ref not in cash_by_ref: 
                continue
            theoretical = -_safe_float(f["price"]) * abs(_safe_float(f["qty"])) if (f["side"] == "buy") else _safe_float(f["price"]) * abs(_safe_float(f["qty"]))
            actual = cash_by_ref[ref]
            fee_est_total += abs(actual - theoretical)
        return fee_est_total

    def _pnl_unrealized(self, marks: Dict[str, float]) -> float:
        unreal = 0.0
        for p in self._positions_snapshot():
            s = p["symbol"]; px = marks.get(s)
            if px is None: 
                continue
            unreal += (float(px) - float(p["avg_px"])) * float(p["qty"])
        return unreal

    def build_report(self, start: dt.datetime, end: dt.datetime) -> Dict[str, Any]:
        t0, t1 = _to_epoch_ms(start), _to_epoch_ms(end)
        fills = self._fills_in_range(t0, t1)
        orders = self._orders_in_range(t0, t1)
        cash = self._cash_deltas(t0, t1)

        trades = len(fills)
        symbols = sorted(set(f["symbol"] for f in fills))
        notional = sum(_safe_float(f["price"]) * abs(_safe_float(f["qty"])) for f in fills)
        vwap = self._compute_vwap(fills)
        cash_sum = sum(_safe_float(c["delta"]) for c in cash if c["reason"] == "fill")
        fees_est = self._estimate_fees(fills, cash)
        avg_trade = (notional / trades) if trades else 0.0

        # Risk metrics (best-effort from Redis)
        vol_metric = None; dd_metric = None
        if self.r:
            with contextlib.suppress(Exception):
                vol_metric = self.r.hgetall("strategy:vol")
            with contextlib.suppress(Exception):
                dd_metric = self.r.hgetall("strategy:drawdown")

        unreal = self._pnl_unrealized(self.marks) if self.marks else None

        rep: Dict[str, Any] = {
            "as_of": dt.datetime.utcnow().isoformat() + "Z",
            "account": self.account_id,
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "kpis": {
                "trades": trades,
                "symbols": len(symbols),
                "turnover": notional,
                "avg_trade_notional": avg_trade,
                "cash_delta_est": cash_sum,
                "fees_est": fees_est,
                "unrealized_pnl_mark_to_market": unreal,
            },
            "by_symbol": [
                {
                    "symbol": s,
                    "volume_shares": float(sum(abs(_safe_float(f["qty"])) for f in fills if f["symbol"] == s)),
                    "notional": float(sum(_safe_float(f["price"]) * abs(_safe_float(f["qty"])) for f in fills if f["symbol"] == s)),
                    "vwap": float(vwap.get(s, 0.0))
                } for s in symbols
            ],
            "risk_metrics": {"vol": vol_metric, "drawdown": dd_metric},
            "orders": [dict(row) for row in orders],
            "fills": [dict(row) for row in fills],
        }
        return rep

    # ----- rendering -----
    def render_markdown(self, rep: Dict[str, Any]) -> str:
        k = rep["kpis"]; rows = []
        rows.append(["Trades", k["trades"]])
        rows.append(["Symbols", sum(1 for _ in rep["by_symbol"])])
        rows.append(["Turnover", _fmt_ccy(k["turnover"])])
        rows.append(["Avg Trade Notional", _fmt_ccy(k["avg_trade_notional"])])
        rows.append(["Cash Δ (est.)", _fmt_ccy(k["cash_delta_est"])])
        rows.append(["Fees (est.)", _fmt_ccy(k["fees_est"])])
        if k.get("unrealized_pnl_mark_to_market") is not None:
            rows.append(["Unrealized PnL (marks)", _fmt_ccy(k["unrealized_pnl_mark_to_market"])])

        lines = []
        lines.append(f"# Trading Report ({rep['window']['start']} → {rep['window']['end']})")
        if rep.get("account"):
            lines.append(f"**Account:** `{rep['account']}`  ")
        lines.append(f"**As of:** {rep['as_of']}")
        lines.append("")
        lines.append("## KPIs")
        lines.append(_md_table(["Metric", "Value"], rows))
        lines.append("")
        lines.append("## By Symbol")
        sym_rows = []
        for srow in rep["by_symbol"]:
            sym_rows.append([srow["symbol"],
                             f"{srow['volume_shares']:.4f}",
                             _fmt_ccy(srow["notional"]),
                             f"{srow['vwap']:.4f}"])
        lines.append(_md_table(["Symbol", "Volume", "Notional", "VWAP"], sym_rows))
        if rep.get("risk_metrics"):
            v = rep["risk_metrics"].get("vol") or {}
            d = rep["risk_metrics"].get("drawdown") or {}
            if v or d:
                lines.append("")
                lines.append("## Risk Metrics (from Redis, best-effort)")
                metrics_rows = []
                keys = sorted(set(list(v.keys()) + list(d.keys())))
                for name in keys:
                    vol_val, dd_val = v.get(name), d.get(name)
                    try:
                        # Stored values might be bytes/json strings
                        import json as _json
                        if isinstance(vol_val, (bytes, str)):
                            try: vol_val = _json.loads(vol_val)["vol"]
                            except Exception: pass
                        if isinstance(dd_val, (bytes, str)):
                            try: dd_val = _json.loads(dd_val)["dd"]
                            except Exception: pass
                    except Exception:
                        pass
                    try: vol_num = float(vol_val) # type: ignore
                    except Exception: vol_num = 0.0
                    try: dd_num = float(dd_val) # type: ignore
                    except Exception: dd_num = 0.0
                    metrics_rows.append([name, f"{vol_num:.6f}", f"{dd_num:.6f}"])
                lines.append(_md_table(["Strategy", "Vol", "Drawdown"], metrics_rows))
        lines.append("")
        lines.append("> Generated by `backend/reporting/report_generator.py`")
        return "\n".join(lines)

    def render_html(self, rep: Dict[str, Any]) -> str:
        md = self.render_markdown(rep)
        esc = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Trading Report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.35;padding:24px;}}
table{{border-collapse:collapse;margin:12px 0;width:100%;}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px;}}
th{{background:#f5f5f5;}}
code{{background:#f0f0f0;padding:2px 4px;border-radius:4px;}}
h1,h2{{margin:18px 0 6px;}}
</style></head><body><pre style="white-space:pre-wrap">{esc}</pre></body></html>"""

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Generate trading reports from the SQLite ledger")
    ap.add_argument("--db", required=True, help="Path to SQLite DB (e.g., data/ledger.db)")
    ap.add_argument("--account", default=None, help="Account id (optional)")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD or epoch ms)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD or epoch ms, exclusive)")
    ap.add_argument("--out", required=True, help="Output file (.md | .html | .json)")
    ap.add_argument("--fmt", choices=["md","html","json"], default=None)
    ap.add_argument("--marks", default=None, help="Path to JSON of symbol->price for unrealized PnL")
    ap.add_argument("--redis_url", default=None, help="redis://host:port for risk metrics (optional)")
    args = ap.parse_args()

    fmt = args.fmt or (args.out.split(".")[-1].lower())

    marks = None
    if args.marks and os.path.exists(args.marks):
        with open(args.marks, "r", encoding="utf-8") as f:
            marks = json.load(f)

    gen = ReportGenerator(args.db, account_id=args.account, marks=marks, redis_url=args.redis_url)
    rep = gen.build_report(_parse_date(args.start), _parse_date(args.end))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if fmt == "json":
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    elif fmt == "md":
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(gen.render_markdown(rep))
    elif fmt == "html":
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(gen.render_html(rep))
    else:
        raise SystemExit("Unknown fmt; use --fmt md|html|json")

if __name__ == "__main__":  # pragma: no cover
    main()