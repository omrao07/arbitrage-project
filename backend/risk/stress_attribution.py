# backend/analytics/stress_attribution.py
"""
Stress Attribution: attribute portfolio PnL to shock components (Rates / Credit / Equity / FX)
and roll it up by any dimensions (strategy, sector, region, book, symbol, tenor, …).

Design notes
------------
- Rates/credit use DV01-style sensitivities (currency P&L per 1bp). By convention:
    price change ≈ - DV01 * Δy_bps       (so a +10bp move loses DV01*10)
    spread PnL   ≈ - SpreadDV01 * Δs_bps
- Equity uses beta * index_return * notional (direction embedded in notional sign).
- FX uses foreign_notional * fx_return_to_base (direction embedded in notional sign).
- Tenor-level DV01 can be given as a dict, e.g. {"2y": 12.3, "5y": 25.1} (units: base CCY per bp).
- Everything is dependency-light; pandas is optional for pretty DataFrames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------- Optional deps ---------------------------------
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# Soft link to your policy & curve shocks (if available)
try:
    from backend.macro.central_bank_ai import RateShock  # type: ignore
except Exception:
    @dataclass
    class RateShock:  # fallback shape-compatible
        parallel_bp: float = 0.0
        steepen_bp: float = 0.0
        butterfly_bp: float = 0.0
        twist_pivot_yrs: float = 5.0
        custom_tenor_bp: Dict[float, float] = field(default_factory=dict)  # tenor_yrs -> bps


# =============================================================================
# Data models
# =============================================================================

@dataclass
class Exposure:
    """
    One position or bucketed position.
    Notional sign convention: + for long, - for short.
    """
    symbol: str
    strategy: str = ""
    sector: str = ""
    region: str = ""
    book: str = ""
    currency: str = "USD"         # base CCY for PnL
    notional: float = 0.0         # base CCY exposure used for equity beta calc
    # Rates/credit greeks:
    dv01_tenor: Dict[str, float] = field(default_factory=dict)    # e.g. {"2y": 12.3, "5y": 25.1}  ($/bp)
    credit_dv01: float = 0.0                                      # spread DV01 ($/bp)
    # Equity factor:
    beta: float = 0.0                                              # to chosen equity factor/index
    # FX:
    fx_ccy: Optional[str] = None                                   # currency vs base (e.g., "EUR" if base is USD)
    fx_notional_foreign: float = 0.0                               # notional in foreign currency units (signed)


@dataclass
class Shock:
    """
    Unified shock container. All fields optional; zeros mean no shock.
    - rates_by_tenor: map "2y","5y","10y",... -> Δbps
      (you can also pass a RateShock via from_rate_shock)
    - credit_spread_bps: Δbps for credit spreads (positive = widening)
    - eq_factor_rets: map factor/index name -> return (e.g., {"SPX": -0.04})
      The engine will use 'beta' from exposures; you can choose any factor key you like.
    - fx_rets_to_base: map CCY -> return vs base CCY (e.g., {"EUR": -0.02} means EURUSD -2%)
    """
    rates_by_tenor: Dict[str, float] = field(default_factory=dict)     # tenor -> Δbps
    credit_spread_bps: float = 0.0
    eq_factor_rets: Dict[str, float] = field(default_factory=dict)     # use key that matches 'factor_key' at run()
    fx_rets_to_base: Dict[str, float] = field(default_factory=dict)    # CCY -> return vs base

    @staticmethod
    def from_rate_shock(rs: RateShock, *, tenors: Iterable[float] = (0.25, 0.5, 1, 2, 3, 5, 7, 10)) -> "Shock":
        m: Dict[str, float] = {}
        # Start with custom
        for T, bp in (rs.custom_tenor_bp or {}).items():
            m[_tenor_key(float(T))] = float(bp)
        # Add parametric components (very simple mapping)
        for T in tenors:
            k = _tenor_key(float(T))
            base = m.get(k, 0.0)
            # parallel
            base += rs.parallel_bp
            # steepen: +bps*(T - pivot>?) sign across curve; assume pivot at 5y
            base += rs.steepen_bp * _signed_slope(float(T), pivot=getattr(rs, "twist_pivot_yrs", 5.0))
            # butterfly: bump wings, fade belly (toy shape)
            base += rs.butterfly_bp * _butterfly_shape(float(T))
            m[k] = base
        return Shock(rates_by_tenor=m)


@dataclass
class Row:
    """Aggregated attribution row."""
    pnl_total: float = 0.0
    pnl_rates: float = 0.0
    pnl_credit: float = 0.0
    pnl_equity: float = 0.0
    pnl_fx: float = 0.0
    # Optional breakdowns
    rates_by_tenor: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Return object of a stress run."""
    group_by: Tuple[str, ...]
    rows: List[Dict[str, Any]]
    totals: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"group_by": self.group_by, "rows": self.rows, "totals": self.totals, "details": self.details}, indent=2)

    def to_dataframe(self):
        if pd is None:
            raise RuntimeError("pandas not installed")
        return pd.DataFrame(self.rows)


# =============================================================================
# Engine
# =============================================================================

class StressAttributor:
    """
    Add exposures, then run shocks and get PnL attribution rolled up by any dimensions.

    Example:
        sa = StressAttributor(base_ccy="USD")
        sa.add(Exposure("AAPL", strategy="alpha", notional=1_000_000, beta=1.1))
        sa.add(Exposure("UST10", strategy="hedge", dv01_tenor={"10y": 85.0}))
        shock = Shock(rates_by_tenor={"10y": +25}, eq_factor_rets={"SPX": -0.05})
        res = sa.run(shock, group_by=("strategy","sector"), factor_key="SPX", detail_rates_by_tenor=True)
    """
    def __init__(self, *, base_ccy: str = "USD"):
        self.base_ccy = base_ccy.upper()
        self._exposures: List[Exposure] = []

    # ---------------- API ----------------

    def add(self, exp: Exposure) -> None:
        self._exposures.append(exp)

    def add_many(self, exps: Iterable[Exposure]) -> None:
        for e in exps:
            self.add(e)

    def run(
        self,
        shock: Shock,
        *,
        group_by: Iterable[str] = ("strategy", "sector"),
        factor_key: Optional[str] = None,
        detail_rates_by_tenor: bool = False,
    ) -> Result:
        """
        Compute component PnL per exposure, aggregate by group_by, and return a table.
        """
        dims = tuple(group_by)
        table: Dict[Tuple[str, ...], Row] = {}
        totals = Row()

        for e in self._exposures:
            # ---- Rates ----
            pr_rates, per_tenor = _pnl_rates(e.dv01_tenor, shock.rates_by_tenor)

            # ---- Credit ----
            pr_credit = - float(e.credit_dv01) * float(shock.credit_spread_bps)

            # ---- Equity factor ----
            r_eq = 0.0
            if factor_key is not None:
                r_eq = float(shock.eq_factor_rets.get(factor_key, 0.0))
            pr_equity = float(e.beta) * r_eq * float(e.notional)

            # ---- FX ----
            pr_fx = 0.0
            if e.fx_ccy:
                r_fx = float(shock.fx_rets_to_base.get(e.fx_ccy.upper(), 0.0))
                pr_fx = float(e.fx_notional_foreign) * r_fx

            # ---- Aggregate for this exposure ----
            pnl_total = pr_rates + pr_credit + pr_equity + pr_fx

            key = _make_gid(e, dims)
            row = table.setdefault(key, Row())
            row.pnl_rates += pr_rates
            row.pnl_credit += pr_credit
            row.pnl_equity += pr_equity
            row.pnl_fx += pr_fx
            row.pnl_total += pnl_total
            if detail_rates_by_tenor and per_tenor:
                for t, v in per_tenor.items():
                    row.rates_by_tenor[t] = row.rates_by_tenor.get(t, 0.0) + v

            # update totals
            totals.pnl_rates += pr_rates
            totals.pnl_credit += pr_credit
            totals.pnl_equity += pr_equity
            totals.pnl_fx += pr_fx
            totals.pnl_total += pnl_total
            if detail_rates_by_tenor and per_tenor:
                for t, v in per_tenor.items():
                    totals.rates_by_tenor[t] = totals.rates_by_tenor.get(t, 0.0) + v

        # Pretty rows
        rows = []
        for gid, r in table.items():
            row_dict = {dims[i]: gid[i] for i in range(len(dims))}
            row_dict.update({
                "pnl_total": _round(r.pnl_total),
                "pnl_rates": _round(r.pnl_rates),
                "pnl_credit": _round(r.pnl_credit),
                "pnl_equity": _round(r.pnl_equity),
                "pnl_fx": _round(r.pnl_fx),
            }) # type: ignore
            if detail_rates_by_tenor and r.rates_by_tenor:
                # include as nested dict
                row_dict["rates_by_tenor"] = {k: _round(v) for k, v in sorted(r.rates_by_tenor.items(), key=lambda kv: _tenor_sort_key(kv[0]))} # type: ignore
            rows.append(row_dict)

        rows.sort(key=lambda d: -d["pnl_total"])
        totals_dict = {
            "pnl_total": _round(totals.pnl_total),
            "pnl_rates": _round(totals.pnl_rates),
            "pnl_credit": _round(totals.pnl_credit),
            "pnl_equity": _round(totals.pnl_equity),
            "pnl_fx": _round(totals.pnl_fx),
        }
        if detail_rates_by_tenor and totals.rates_by_tenor:
            totals_dict["rates_by_tenor"] = {k: _round(v) for k, v in sorted(totals.rates_by_tenor.items(), key=lambda kv: _tenor_sort_key(kv[0]))} # type: ignore

        details = {"base_ccy": self.base_ccy}
        return Result(group_by=dims, rows=rows, totals=totals_dict, details=details)

    # ---------------- Convenience ----------------

    @staticmethod
    def from_positions(
        positions: Iterable[Dict[str, Any]],
        *,
        default_ccy: str = "USD",
        rates_tenor_key: str = "dv01_tenor",
    ) -> "StressAttributor":
        """
        Build a StressAttributor from a list of dict-like positions.
        Keys recognized per item (missing default to harmless zeros):
            symbol, strategy, sector, region, book, currency, notional,
            beta, credit_dv01, fx_ccy, fx_notional_foreign, dv01_tenor (dict)
        """
        sa = StressAttributor(base_ccy=default_ccy)
        for p in positions:
            sa.add(Exposure(
                symbol=str(p.get("symbol","")),
                strategy=str(p.get("strategy","")),
                sector=str(p.get("sector","")),
                region=str(p.get("region","")),
                book=str(p.get("book","")),
                currency=str(p.get("currency", default_ccy)).upper(),
                notional=float(p.get("notional", 0.0)),
                beta=float(p.get("beta", 0.0)),
                credit_dv01=float(p.get("credit_dv01", 0.0)),
                fx_ccy=(str(p["fx_ccy"]).upper() if p.get("fx_ccy") else None),
                fx_notional_foreign=float(p.get("fx_notional_foreign", 0.0)),
                dv01_tenor=dict(p.get(rates_tenor_key, {})),
            ))
        return sa


# =============================================================================
# Internals
# =============================================================================

def _pnl_rates(dv01_tenor: Dict[str, float], shock_tenor: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Combine tenor DV01 map with tenor shock map (bps) → PnL and per-tenor breakdown.
    """
    if not dv01_tenor or not shock_tenor:
        return 0.0, {}
    pnl = 0.0
    per: Dict[str, float] = {}
    for k, dv01 in dv01_tenor.items():
        dk = _normalize_tenor_key(k)
        # match by normalized key if present, else try exact
        bp = shock_tenor.get(dk, shock_tenor.get(k, 0.0))
        # price change ≈ - DV01 * Δy_bps
        contrib = - float(dv01) * float(bp)
        pnl += contrib
        if abs(contrib) > 0:
            per[dk] = per.get(dk, 0.0) + contrib
    return pnl, per


def _tenor_key(T: float) -> str:
    if T < 1.0:
        # quarters
        if abs(T - 0.25) < 1e-9: return "3m"
        if abs(T - 0.5) < 1e-9:  return "6m"
        return f"{int(round(T*12))}m"
    return f"{int(round(T))}y"


def _normalize_tenor_key(k: str) -> str:
    k = k.strip().lower()
    if k.endswith("y"):
        return f"{int(round(float(k[:-1])))}y"
    if k.endswith("m"):
        return f"{int(round(float(k[:-1])))}m"
    # try numeric years
    try:
        v = float(k)
        return _tenor_key(v)
    except Exception:
        return k


def _tenor_sort_key(k: str) -> float:
    k = _normalize_tenor_key(k)
    if k.endswith("y"):
        return float(k[:-1])
    if k.endswith("m"):
        return float(k[:-1]) / 12.0
    return 999.0


def _signed_slope(T: float, pivot: float = 5.0) -> float:
    # negative before pivot, positive after
    return (T - pivot) / max(1e-9, pivot)


def _butterfly_shape(T: float, belly: float = 5.0, wings: Tuple[float, float] = (2.0, 10.0)) -> float:
    # +1 on wings, -1 in belly (very simple)
    if abs(T - belly) < 1e-6:
        return -1.0
    if T <= wings[0] or T >= wings[1]:
        return +1.0
    # linear between
    if T < belly:
        return (T - wings[0]) / max(1e-9, (belly - wings[0]))  # from +1 to -1
    return (wings[1] - T) / max(1e-9, (wings[1] - belly))      # from -1 to +1


def _make_gid(e: Exposure, dims: Tuple[str, ...]) -> Tuple[str, ...]:
    # map dimension name -> attribute on Exposure
    lookup = {
        "strategy": e.strategy, "sector": e.sector, "region": e.region,
        "book": e.book, "symbol": e.symbol, "currency": e.currency,
    }
    return tuple(str(lookup.get(d, "")) for d in dims)


def _round(x: float) -> float:
    return float(round(float(x), 6))


# =============================================================================
# Tiny demo
# =============================================================================

if __name__ == "__main__":
    # Build sample exposures
    exps = [
        # Equity book (USD base)
        {"symbol":"AAPL", "strategy":"alpha.eqt", "sector":"Tech", "region":"US", "notional": 2_000_000, "beta": 1.2},
        {"symbol":"RELIANCE.NS", "strategy":"alpha.eqt", "sector":"Energy", "region":"IN", "notional": 1_000_000, "beta": 0.9},
        # Rates hedge (DV01 in $/bp)
        {"symbol":"UST_5Y_HEDGE", "strategy":"hedge.rates", "sector":"Rates", "dv01_tenor":{"5y": 250.0}},
        {"symbol":"UST_10Y_HEDGE", "strategy":"hedge.rates", "sector":"Rates", "dv01_tenor":{"10y": 420.0}},
        # Credit book
        {"symbol":"IG_CDS", "strategy":"credit", "sector":"Credit", "credit_dv01": 300.0},
        # FX: long EUR assets (1,000,000 EUR)
        {"symbol":"EUR_ASSETS", "strategy":"alpha.fx", "sector":"FX", "fx_ccy":"EUR", "fx_notional_foreign": 1_000_000},
    ]
    sa = StressAttributor.from_positions(exps)

    # Build a shock: +25bp at 10y, +10bp at 5y, SPX -5%, EURUSD -2%, credit +40bp
    shock = Shock(
        rates_by_tenor={"5y": +10.0, "10y": +25.0},
        credit_spread_bps=+40.0,
        eq_factor_rets={"SPX": -0.05},
        fx_rets_to_base={"EUR": -0.02},
    )

    res = sa.run(shock, group_by=("strategy","sector"), factor_key="SPX", detail_rates_by_tenor=True)
    print(res.to_json())
    if pd:
        print(res.to_dataframe())