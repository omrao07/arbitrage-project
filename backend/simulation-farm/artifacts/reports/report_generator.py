# simulation-farm/artifacts/reports/report_generator.py
"""
ReportGenerator
---------------

Renders Simulation Farm HTML reports (summary, equity curve, risk report) using Jinja2
and exports them via Local/GCS/S3 exporters.

Install (recommended):
    pip install jinja2 pandas numpy

Usage:
    from artifacts.reports.report_generator import ReportGenerator, ReportInputs
    rg = ReportGenerator(
        run_id="run_2025_09_16_2130",
        output_prefix="runs/run_2025_09_16_2130/",   # used by exporters
        template_dir="simulation-farm/artifacts/reports/templates",
        exporter=("local", {"root": "artifacts/reports/out"})  # or ("s3", {...}), ("gcs", {...})
    )
    out = rg.generate_all(inputs)
    print(out)  # dict of URLs

You can also call rg.render_equity(), rg.render_risk(), rg.render_summary() individually.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Optional heavy deps
try:
    import pandas as pd  # type: ignore
    import numpy as np   # type: ignore
    _HAVE_PD = True
except Exception:
    _HAVE_PD = False

from jinja2 import Environment, FileSystemLoader  # type: ignore


# ----------------------- Data containers -----------------------

@dataclass
class ReportInputs:
    # Core series (aligned)
    dates: Sequence[str]
    equity: Sequence[float]
    benchmark: Optional[Sequence[float]] = None
    drawdown: Optional[Sequence[float]] = None     # in [-1, 0]
    pnl: Optional[Sequence[float]] = None          # daily return series (e.g., % as fraction)

    # Metadata
    title: str = "Simulation Report"
    strategy: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tz: Optional[str] = "UTC"

    # Summary blocks
    kpis: Optional[Dict] = None
    config: Optional[Dict] = None
    params: Optional[Dict] = None
    notes: Optional[List[str]] = None
    contributors: Optional[List[Dict]] = None
    positions: Optional[List[Dict]] = None
    trades: Optional[List[Dict]] = None
    diag: Optional[Dict] = None

    # Risk blocks
    var_hist: Optional[Dict[str, Sequence[float]]] = None  # {"bins": [...], "counts": [...]}
    factors: Optional[Dict[str, float]] = None
    sectors: Optional[Dict[str, float]] = None
    drawdowns_table: Optional[List[Dict]] = None
    stress: Optional[List[Dict]] = None


# ----------------------- Exporter loader -----------------------

def _make_exporter(kind: str, kwargs: Dict):
    kind = kind.lower()
    if kind == "local":
        from .exporters.local_exporter import LocalExporter # type: ignore
        return LocalExporter(**kwargs)
    if kind == "s3":
        from .exporters.s3_exporter import S3Exporter # type: ignore
        return S3Exporter(**kwargs)
    if kind == "gcs":
        from .exporters.gcs_exporter import GCSExporter # type: ignore
        return GCSExporter(**kwargs)
    raise ValueError("Exporter kind must be one of: local, s3, gcs")


# ----------------------- Stats helpers -----------------------

def _safe_len(xs) -> int:
    try:
        return len(xs)
    except Exception:
        return 0

def _compute_drawdown(equity: Sequence[float]) -> List[float]:
    peak = -float("inf")
    dd = []
    for v in equity:
        peak = max(peak, v)
        dd.append((v / peak - 1.0) if peak > 0 else 0.0)
    return dd

def _basic_kpis(dates: Sequence[str], equity: Sequence[float], pnl: Optional[Sequence[float]]) -> Dict:
    if not equity:
        return {}
    first, last = equity[0], equity[-1]
    years = max(1e-9, _approx_years(dates))
    cagr = (last / first) ** (1 / years) - 1 if first > 0 else 0.0
    # returns (daily %) for sharpe/sortino if we have pnl; otherwise infer from equity
    rets = list(pnl) if pnl is not None else _equity_to_returns(equity)
    if not rets:
        sharpe = sortino = vol_annual = 0.0
    else:
        mu = _nanmean(rets)
        sd = _nanstd(rets)
        downside = _nanstd([min(0.0, r) for r in rets])
        vol_annual = sd * math.sqrt(252)
        sharpe = (mu * 252) / (sd + 1e-12)
        sortino = (mu * 252) / (downside + 1e-12)
    max_dd = min(_compute_drawdown(equity)) if equity else 0.0
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "vol_annual": vol_annual,
        "trades": None,  # fill at caller if known
    }

def _equity_to_returns(equity: Sequence[float]) -> List[float]:
    out = []
    prev = None
    for v in equity:
        if prev is not None and prev > 0:
            out.append(v / prev - 1.0)
        prev = v
    return out

def _approx_years(dates: Sequence[str]) -> float:
    # crude estimate: trading days/252
    return max(1/252, _safe_len(dates) / 252.0)

def _nanmean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(vals) / max(1, len(vals))

def _nanstd(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return math.sqrt(var)


# ----------------------- Report generator -----------------------

class ReportGenerator:
    def __init__(
        self,
        run_id: str,
        output_prefix: str,
        template_dir: str = "simulation-farm/artifacts/reports/templates",
        exporter: tuple[str, Dict] = ("local", {"root": "artifacts/reports/out"}),
        footer: str = "© Simulation Farm",
    ):
        self.run_id = run_id
        self.output_prefix = output_prefix.strip("/").rstrip("/") + "/"
        self.footer = footer

        # Templating
        tdir = Path(template_dir)
        if not tdir.exists():
            raise FileNotFoundError(f"Template dir not found: {tdir}")
        self.env = Environment(loader=FileSystemLoader(str(tdir)))
        # Exporter
        kind, kwargs = exporter
        self.exporter = _make_exporter(kind, kwargs | {"prefix": self.output_prefix})

    # ---------- high-level API ----------

    def generate_all(self, inputs: ReportInputs) -> Dict[str, str]:
        """
        Render summary, equity curve and risk report (+ CSV/JSON assets).
        Returns dict: {"summary": url, "equity": url, "risk": url, "assets": {...}}
        """
        assets = self._emit_assets(inputs)
        urls = {
            "summary": self.render_summary(inputs, assets),
            "equity": self.render_equity(inputs, assets),
            "risk": self.render_risk(inputs, assets),
            "assets": assets,
        }
        return urls

    # ---------- individual renderers ----------

    def render_summary(self, inputs: ReportInputs, assets: Optional[Dict[str, str]] = None) -> str:
        kpis = inputs.kpis or _basic_kpis(inputs.dates, inputs.equity, inputs.pnl)
        ctx = {
            "title": inputs.title,
            "run_id": self.run_id,
            "strategy": inputs.strategy,
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "tz": inputs.tz,
            "kpis": kpis,
            "config": inputs.config or {},
            "params": inputs.params or {},
            "notes": inputs.notes or [],
            "links": {
                "equity": "equity.html",
                "risk": "risk.html",
                "raw": assets.get("results_json") if assets else None,
                "logs": "../../artifacts/logs/",
            },
            "dates": inputs.dates,
            "equity": inputs.equity,
            "benchmark": inputs.benchmark,
            "pnl": inputs.pnl,
            "contributors": inputs.contributors or [],
            "positions": inputs.positions or [],
            "trades": inputs.trades or [],
            "diag": inputs.diag or {},
            "footer": self.footer,
        }
        html = self.env.get_template("summary.html").render(**ctx)
        return self._upload_bytes(html.encode("utf-8"), "index.html", content_type="text/html")

    def render_equity(self, inputs: ReportInputs, assets: Optional[Dict[str, str]] = None) -> str:
        drawdown = inputs.drawdown or _compute_drawdown(inputs.equity)
        # compute trades markers y if missing is handled in template
        ctx = {
            "title": f"Equity Curve — {inputs.strategy or ''}".strip(),
            "run_id": self.run_id,
            "strategy": inputs.strategy,
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "tz": inputs.tz,
            "dates": inputs.dates,
            "equity": inputs.equity,
            "benchmark": inputs.benchmark,
            "drawdown": drawdown,
            "trades": inputs.trades or [],
            "stats": (inputs.kpis or _basic_kpis(inputs.dates, inputs.equity, inputs.pnl)) | {
                "win_rate": (inputs.kpis or {}).get("win_rate"),
                "trades": (inputs.kpis or {}).get("trades"),
            },
            "footer": self.footer,
        }
        html = self.env.get_template("equity_curve.html").render(**ctx)
        return self._upload_bytes(html.encode("utf-8"), "equity.html", content_type="text/html")

    def render_risk(self, inputs: ReportInputs, assets: Optional[Dict[str, str]] = None) -> str:
        ctx = {
            "title": f"Risk Report — {inputs.strategy or ''}".strip(),
            "run_id": self.run_id,
            "strategy": inputs.strategy,
            "start_date": inputs.start_date,
            "end_date": inputs.end_date,
            "tz": inputs.tz,
            "dates": inputs.dates,
            "equity": inputs.equity,
            "benchmark": inputs.benchmark,
            "drawdown": inputs.drawdown or _compute_drawdown(inputs.equity),
            "var_hist": inputs.var_hist or {"bins": [], "counts": []},
            "factors": inputs.factors or {},
            "sectors": inputs.sectors or {},
            "drawdowns": inputs.drawdowns_table or [],
            "stress": inputs.stress or [],
            "turnover": (inputs.diag or {}).get("turnover"),
        }
        html = self.env.get_template("risk_report.html").render(**ctx)
        return self._upload_bytes(html.encode("utf-8"), "risk.html", content_type="text/html")

    # ---------- assets (CSV/JSON) ----------

    def _emit_assets(self, inputs: ReportInputs) -> Dict[str, str]:
        """
        Write standard JSON/CSV outputs next to the HTML.
        Returns map of logical name -> URL.
        """
        urls: Dict[str, str] = {}

        # Raw results JSON (lightweight)
        raw = {
            "run_id": self.run_id,
            "title": inputs.title,
            "strategy": inputs.strategy,
            "period": {"start": inputs.start_date, "end": inputs.end_date, "tz": inputs.tz},
            "kpis": inputs.kpis or _basic_kpis(inputs.dates, inputs.equity, inputs.pnl),
            "config": inputs.config or {},
            "params": inputs.params or {},
        }
        urls["results_json"] = self._upload_bytes(json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8"),
                                                  "results.json", content_type="application/json")

        # Equity CSV (+ optional benchmark & drawdown)
        eq_lines = ["date,equity" + (",benchmark" if inputs.benchmark else "") + (",drawdown_pct" if inputs.drawdown else "")]
        dd = inputs.drawdown or _compute_drawdown(inputs.equity)
        for i, d in enumerate(inputs.dates):
            row = [d, _fmt(inputs.equity[i])]
            if inputs.benchmark:
                row.append(_fmt(inputs.benchmark[i]))
            if dd:
                row.append(_fmt(dd[i] * 100))
            eq_lines.append(",".join(row))
        urls["equity_csv"] = self._upload_bytes("\n".join(eq_lines).encode("utf-8"), "equity.csv", content_type="text/csv")

        # Trades CSV (if present)
        if inputs.trades:
            hdr = ["ts", "symbol", "side", "qty", "px", "fee"]
            lines = [",".join(hdr)]
            for t in inputs.trades:
                lines.append(",".join([
                    str(t.get("ts", "")),
                    str(t.get("symbol", "")),
                    str(t.get("side", "")),
                    _fmt(t.get("qty")),
                    _fmt(t.get("px")),
                    _fmt(t.get("fee", 0)),
                ]))
            urls["trades_csv"] = self._upload_bytes("\n".join(lines).encode("utf-8"), "trades.csv", content_type="text/csv")

        # Positions CSV (if present)
        if inputs.positions:
            hdr = ["symbol", "qty", "px", "value", "weight"]
            lines = [",".join(hdr)]
            for p in inputs.positions:
                lines.append(",".join([
                    str(p.get("symbol", "")),
                    _fmt(p.get("qty")),
                    _fmt(p.get("px")),
                    _fmt(p.get("value")),
                    _fmt(p.get("weight")),
                ]))
            urls["positions_csv"] = self._upload_bytes("\n".join(lines).encode("utf-8"), "positions.csv", content_type="text/csv")

        return urls

    # ---------- low-level upload ----------

    def _upload_bytes(self, data: bytes, dest: str, *, content_type: str) -> str:
        # All exporters implement upload_bytes/upload_file with dest relative to prefix.
        return self.exporter.upload_bytes(data, dest, content_type=content_type)


# ----------------------- small utils -----------------------

def _fmt(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float):
            # avoid scientific notation; 6 dp default
            return f"{x:.6f}".rstrip("0").rstrip(".")
        return str(x)
    except Exception:
        return str(x)


# ----------------------- CLI (optional) -----------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render Simulation Farm reports")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-prefix", required=True, help="Exporter prefix, e.g., runs/run_2025_09_16_2130/")
    ap.add_argument("--templates", default="simulation-farm/artifacts/reports/templates")
    ap.add_argument("--exporter", default="local", choices=["local", "s3", "gcs"])
    ap.add_argument("--export-arg", action="append", default=[], help="key=value pairs for exporter (e.g., root=artifacts/reports/out)")
    ap.add_argument("--title", default="Simulation Report")
    args = ap.parse_args()

    kwargs = {}
    for kv in args.export_arg:
        if "=" in kv:
            k, v = kv.split("=", 1)
            kwargs[k] = v

    rg = ReportGenerator(
        run_id=args.run_id,
        output_prefix=args.out_prefix,
        template_dir=args.templates,
        exporter=(args.exporter, kwargs or ({"root": "artifacts/reports/out"} if args.exporter == "local" else {})),
    )

    # Minimal demo inputs from CSV files (if present), else a toy series
    eq_csv = Path("equity.csv")
    if eq_csv.exists():
        # Expect columns: date,equity[,benchmark,drawdown_pct]
        rows = [r.strip().split(",") for r in eq_csv.read_text(encoding="utf-8").splitlines()]
        hdr, data = rows[0], rows[1:]
        idx_date = hdr.index("date"); idx_eq = hdr.index("equity")
        idx_bench = hdr.index("benchmark") if "benchmark" in hdr else None
        idx_dd = hdr.index("drawdown_pct") if "drawdown_pct" in hdr else None
        dates = [r[idx_date] for r in data]
        equity = [float(r[idx_eq]) for r in data]
        benchmark = [float(r[idx_bench]) for r in data] if idx_bench is not None else None
        drawdown = [float(r[idx_dd]) / 100.0 for r in data] if idx_dd is not None else None
    else:
        # toy
        dates = [f"2024-01-{i:02d}" for i in range(1, 31)]
        equity = [100_000 * (1 + 0.002 * i + 0.01 * math.sin(i/3)) for i in range(len(dates))]
        benchmark = None
        drawdown = None

    inputs = ReportInputs(
        dates=dates,
        equity=equity,
        benchmark=benchmark,
        drawdown=drawdown,
        pnl=None,
        title=args.title,
        strategy="demo",
        start_date=dates[0] if dates else None,
        end_date=dates[-1] if dates else None,
    )
    urls = rg.generate_all(inputs)
    print(json.dumps(urls, indent=2))