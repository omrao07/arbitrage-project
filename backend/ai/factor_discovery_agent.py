# backend/ai/agents/factor_discovery_agent.py
from __future__ import annotations

"""
Factor Discovery Agent
----------------------
- Candidate generation: price/volume/microstructure transforms
- Evaluation: CS-IC, IC std/IR/t, IC-decay, quantile L/S, turnover, capacity proxy
- Neutralization/orthogonalization vs known factors (optional)
- Regime/stability splits
- CLI & JSON report; optional bus publish

CLI:
  python -m backend.ai.agents.factor_discovery_agent \
      --csv bars.csv --date date --symbol symbol --close close --ret fwd_ret_1d \
      --out factors_report.json

bars.csv needs at least: date,symbol,close,fwd_ret_1d,(volume optional)
"""

from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import json
import math
import os
import time
import warnings

# ---- Optional deps (graceful fallbacks) --------------------------------
try:
    import pandas as pd
except Exception:  # very minimal fallback container
    pd = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    from scipy import stats as _scistats  # Spearman, pearsonr
except Exception:
    _scistats = None

# ---- Optional bus (no-op if unavailable) --------------------------------
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

DISCOVERY_OUT_STREAM = os.getenv("FACTOR_DISCOVERY_STREAM", "research.factors")

# ============================ Data Models ================================

@dataclass
class FactorSpec:
    name: str
    func: Callable[[Any], Any]  # expects pandas DataFrame view for one symbol OR xsec frame per date
    params: Dict[str, Any] = field(default_factory=dict)
    cross_sectional: bool = True  # True: compute per-date across symbols; False: purely TS

@dataclass
class FactorStats:
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_t: float
    ic_decay: List[float]                  # horizon IC 1..H
    ls_ann_ret: float                      # annualized long-short (Q5-Q1)
    ls_ann_sharpe: float
    turnover: float
    capacity_corr_adv: Optional[float]     # + = more capacity with bigger ADV
    redundancy_r2: Optional[float]         # R^2 vs known factors (high = redundant)
    stability_halves_diff: float           # |IC_H1 - IC_H2|
    coverage_days: int
    coverage_names: int

@dataclass
class FactorCandidate:
    spec: FactorSpec
    stats: FactorStats
    passed: bool
    notes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiscoveryReport:
    ts_ms: int
    universe: Dict[str, Any]
    top: List[FactorCandidate]
    rejected: List[FactorCandidate]
    config: Dict[str, Any]

# ============================ Utilities ==================================

def _is_ok():
    if pd is None or np is None:
        raise RuntimeError("This module needs pandas & numpy installed for best results.")

def _cs_rank(z: Any) -> Any:
    # cross-sectional rank [-0.5,0.5]
    r = z.rank(method="average", pct=True)
    return r - 0.5

def _zscore(x: Any) -> Any:
    mu = x.mean()
    sd = x.std(ddof=1) if getattr(x, "std", None) else None
    if sd is None or (isinstance(sd, (int,float)) and sd == 0) or (hasattr(sd, "__float__") and float(sd) == 0.0):
        return x*0
    return (x - mu) / (sd + 1e-9)

def _nan_to_num(a, v=0.0):
    return a.fillna(v) if hasattr(a, "fillna") else v

def _annualize_mean(m: float, periods_per_year: int) -> float:
    return float(m * periods_per_year)

def _annualize_sharpe(mu: float, sigma: float, periods_per_year: int) -> float:
    if sigma <= 1e-12: return 0.0
    return float((mu / sigma) * (periods_per_year ** 0.5))

def _spearman(x, y) -> float:
    if _scistats is not None:
        s, _ = _scistats.spearmanr(x, y, nan_policy="omit")
        return float(s if not math.isnan(s) else 0.0) # type: ignore
    # crude fallback: Pearson on ranks
    xr = pd.Series(x).rank(pct=True) if pd is not None else x
    yr = pd.Series(y).rank(pct=True) if pd is not None else y
    if pd is not None:
        return float(pd.Series(xr).corr(pd.Series(yr)))
    return 0.0

def _pearson(x, y) -> float:
    if _scistats is not None:
        r, _ = _scistats.pearsonr(x, y)
        return float(r if not math.isnan(r) else 0.0) # type: ignore
    if pd is not None:
        return float(pd.Series(x).corr(pd.Series(y)))
    return 0.0

def _ols_residual(y: Any, X: Any) -> Tuple[Any, Optional[float]]:
    """Return residual series and R^2 (if available)."""
    try:
        import numpy as _np
        X1 = _np.column_stack([_np.ones(len(y))] + [X[c].values for c in X.columns])
        beta = _np.linalg.lstsq(X1, y.values, rcond=None)[0]
        yhat = X1 @ beta
        resid = y.values - yhat
        ssr = float(((yhat - y.values.mean())**2).sum())
        sst = float(((y.values - y.values.mean())**2).sum()) + 1e-12
        r2 = min(1.0, max(0.0, ssr / sst))
        return pd.Series(resid, index=y.index), r2 # type: ignore
    except Exception:
        return y - X.sum(axis=1), None

# ===================== Built-in Factor Recipes ===========================

def recipe_price_mom(window: int = 10) -> FactorSpec:
    def f(df: pd.DataFrame) -> pd.Series: # type: ignore
        # expects OHLCV per symbol time-series; we compute last/lag window return per date (xsec panel)
        return df["close"].groupby(df["date"]).apply(lambda s: _cs_rank(s.pct_change(periods=window)))
    return FactorSpec(name=f"mom_{window}", func=f, params={"window": window}, cross_sectional=True)

def recipe_intraday_reversal() -> FactorSpec:
    def f(df: pd.DataFrame) -> pd.Series: # type: ignore
        # close->open gap reversal (ranked)
        gap = (df["open"] - df["prev_close"]) / df["prev_close"]
        return gap.groupby(df["date"]).apply(_cs_rank)
    return FactorSpec(name="intraday_reversal", func=f, cross_sectional=True)

def recipe_volatility(window: int = 20) -> FactorSpec:
    def f(df: pd.DataFrame) -> pd.Series: # type: ignore
        hv = df.groupby("symbol")["ret1"].rolling(window).std().reset_index(level=0, drop=True)
        return hv.groupby(df["date"]).apply(lambda x: -_cs_rank(x))  # low vol preference
    return FactorSpec(name=f"low_vol_{window}", func=f, params={"window": window}, cross_sectional=True)

def recipe_range() -> FactorSpec:
    def f(df: pd.DataFrame) -> pd.Series: # type: ignore
        rg = (df["high"] - df["low"]) / (df["close"] + 1e-9)
        return rg.groupby(df["date"]).apply(lambda x: -_cs_rank(x))  # favor smaller range
    return FactorSpec(name="range_compaction", func=f, cross_sectional=True)

def recipe_volume_pressure(window: int = 10) -> FactorSpec:
    def f(df: pd.DataFrame) -> pd.Series: # type: ignore
        zvol = df.groupby("symbol")["volume"].transform(lambda v: _zscore(v))
        mom = df.groupby("symbol")["close"].pct_change(periods=window)
        sig = zvol * mom
        return sig.groupby(df["date"]).apply(_cs_rank)
    return FactorSpec(name=f"vol_pressure_{window}", func=f, params={"window": window}, cross_sectional=True)

BUILT_INS: List[FactorSpec] = [
    recipe_price_mom(5),
    recipe_price_mom(10),
    recipe_price_mom(20),
    recipe_intraday_reversal(),
    recipe_volatility(20),
    recipe_range(),
    recipe_volume_pressure(10),
]

# ========================== Core Agent ===================================

class FactorDiscoveryAgent:
    def __init__(
        self,
        *,
        periods_per_year: int = 252,
        horizons_ic_decay: int = 10,
        quantiles: int = 5,
        neutralize: bool = True,
        neutralize_controls: Tuple[str, ...] = ("beta", "size"),
        out_stream: str = DISCOVERY_OUT_STREAM,
        redundancy_r2_threshold: float = 0.70,
        min_coverage_days: int = 120,
        min_ic_ir: float = 0.2,
        max_stability_halves_diff: float = 0.10,
    ):
        self.PY = periods_per_year
        self.H = horizons_ic_decay
        self.Q = quantiles
        self.neutralize = neutralize
        self.controls = neutralize_controls
        self.out_stream = out_stream
        self.r2_thr = redundancy_r2_threshold
        self.min_days = min_coverage_days
        self.min_ir = min_ic_ir
        self.max_halves = max_stability_halves_diff
        self.known_factors: Optional[pd.DataFrame] = None  # type: ignore # index=[date,symbol], columns=factors

    def register_known_factors(self, df_factors: pd.DataFrame) -> None: # type: ignore
        """Pass a MultiIndex [date,symbol] DataFrame of known factor exposures (e.g., beta,size,quality, etc.)."""
        self.known_factors = df_factors.copy()

    # ---- Public: discover over candidates ----
    def discover(
        self,
        df: pd.DataFrame, # type: ignore
        *,
        target_col: str = "fwd_ret_1d",
        adv_col: Optional[str] = None,
        extra_controls: Optional[List[str]] = None,
        candidates: Optional[List[FactorSpec]] = None,
        regimes: Optional[pd.Series] = None,  # date -> label # type: ignore
        top_k: int = 10,
    ) -> DiscoveryReport:
        _is_ok()
        if candidates is None:
            candidates = BUILT_INS

        df = df.copy()
        # sanity helpers commonly used by recipes
        if "ret1" not in df.columns and "close" in df.columns:
            df["ret1"] = df.groupby("symbol")["close"].pct_change()
        if "prev_close" not in df.columns and "close" in df.columns:
            df["prev_close"] = df.groupby("symbol")["close"].shift(1)

        # ensure MultiIndex
        if not isinstance(df.index, pd.MultiIndex): # type: ignore
            df["date"] = pd.to_datetime(df["date"]) # type: ignore
            df = df.set_index(["date", "symbol"]).sort_index()

        results: List[FactorCandidate] = []
        for spec in candidates:
            try:
                series = self._compute_series(spec, df)
                stats = self._evaluate(series, df[target_col], adv=df[adv_col] if adv_col else None, regimes=regimes)
                passed, notes = self._gate(stats)
                cand = FactorCandidate(spec=spec, stats=stats, passed=passed, notes=notes)
                results.append(cand)
            except Exception as e:
                results.append(FactorCandidate(spec=spec, stats=self._empty_stats(), passed=False,
                                               notes={"error": str(e)}))

        # sort by IC-IR then LS Sharpe
        good = [c for c in results if c.passed]
        good.sort(key=lambda c: (c.stats.ic_ir, c.stats.ls_ann_sharpe, c.stats.ls_ann_ret), reverse=True)
        bad = [c for c in results if not c.passed]

        report = DiscoveryReport(
            ts_ms=int(time.time() * 1000),
            universe={"n_days": int(df.index.get_level_values(0).nunique()),
                      "n_names": int(df.index.get_level_values(1).nunique())},
            top=good[:top_k],
            rejected=bad,
            config={
                "periods_per_year": self.PY,
                "horizons_ic_decay": self.H,
                "quantiles": self.Q,
                "neutralize": self.neutralize,
                "controls": self.controls,
                "redundancy_r2_threshold": self.r2_thr,
                "gates": {"min_coverage_days": self.min_days, "min_ic_ir": self.min_ir,
                          "max_stability_halves_diff": self.max_halves}
            }
        )

        # publish for dashboards (optional)
        try:
            payload = self.to_json(report)
            publish_stream(self.out_stream, json.loads(payload))
        except Exception:
            pass

        return report

    # ---- Compute factor series for a spec ----
    def _compute_series(self, spec: FactorSpec, df: pd.DataFrame) -> pd.Series: # type: ignore
        s = spec.func(df)
        s.name = spec.name
        # Expect index aligned as [date,symbol]
        if not isinstance(s.index, pd.MultiIndex): # type: ignore
            # allow returns grouped by date; try to align
            if "date" in df.columns and "symbol" in df.columns:
                s.index = df.set_index(["date","symbol"]).index[:len(s)]
            else:
                raise ValueError(f"Factor {spec.name} must return MultiIndex [date,symbol]")
        # neutralize controls if set and available
        if self.neutralize and isinstance(self.known_factors, pd.DataFrame): # type: ignore
            join = pd.concat([s.rename("f"), self.known_factors], axis=1).dropna() # type: ignore
            if join.shape[0] > 0 and join.shape[1] > 1:
                X = join.drop(columns=["f"])
                resid, _ = _ols_residual(join["f"], X)
                s = resid.rename(spec.name)
        return s

    # ---- Evaluate factor series vs next-returns ----
    def _evaluate(self, factor: pd.Series, fwd_ret: pd.Series, *, adv: Optional[pd.Series], regimes: Optional[pd.Series]) -> FactorStats: # type: ignore
        df = pd.concat({"f": factor, "r": fwd_ret}, axis=1).dropna().copy() # type: ignore
        if df.empty:
            return self._empty_stats()

        # Per-date cross-sectional IC
        ics = []
        dates = df.index.get_level_values(0).unique()
        for d in dates:
            sub = df.xs(d, level=0)
            if len(sub) < 5:
                continue
            ics.append(_spearman(sub["f"], sub["r"]))
        ic_mean = float(np.mean(ics)) if ics else 0.0 # type: ignore
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0 # type: ignore
        ic_ir = float(ic_mean / (ic_std + 1e-12))
        ic_t = float(ic_mean / (ic_std / math.sqrt(max(1, len(ics)))) ) if ic_std > 0 else 0.0

        # IC-decay across horizons 1..H (shift returns by k)
        ic_decay: List[float] = []
        if self.H and self.H > 0:
            for k in range(1, self.H + 1):
                r_k = fwd_ret.shift(-(k-1))  # naive: assumes fwd_ret_1d was 1d ahead; align quickly
                dfk = pd.concat({"f": factor, "r": r_k}, axis=1).dropna() # type: ignore
                vals = []
                for d in dfk.index.get_level_values(0).unique():
                    sub = dfk.xs(d, level=0)
                    if len(sub) < 5: continue
                    vals.append(_spearman(sub["f"], sub["r"]))
                ic_decay.append(float(np.mean(vals)) if (vals and np is not None) else (sum(vals)/len(vals) if vals else 0.0))

        # Quantile L/S performance per date (equal-weight within quantiles)
        q = self.Q
        ls_daily = []
        prev_top, prev_bot = set(), set()
        chg_count, overlap = 0, 0
        for d in dates:
            sub = df.xs(d, level=0)
            if len(sub) < q * 2:  # need enough names
                continue
            ranks = sub["f"].rank(pct=True)
            top = set(sub.index[ranks >= 1 - 1/q])
            bot = set(sub.index[ranks <= 1/q])
            # equal-weight returns
            rt = float(sub.loc[list(top), "r"].mean())
            rb = float(sub.loc[list(bot), "r"].mean())
            ls_daily.append(rt - rb)
            # turnover (overlap of membership)
            if prev_top:
                overlap += len(top & prev_top) + len(bot & prev_bot)
                chg_count += len(top) + len(bot)
            prev_top, prev_bot = top, bot

        if ls_daily:
            m = float(np.mean(ls_daily)) if np is not None else sum(ls_daily)/len(ls_daily)
            s = float(np.std(ls_daily, ddof=1)) if (np is not None and len(ls_daily)>1) else 0.0
            ls_ann_ret = _annualize_mean(m, self.PY)
            ls_ann_sharpe = _annualize_sharpe(m, s, self.PY)
        else:
            ls_ann_ret = 0.0
            ls_ann_sharpe = 0.0

        turnover = 0.0
        if chg_count > 0:
            # 1 - overlap ratio (higher = more churn)
            turnover = 1.0 - (overlap / float(chg_count))

        # Capacity proxy: correlation of factor ranks with ADV (bigger ADV better capacity)
        capacity_corr = None
        if adv is not None:
            j = pd.concat({"f": factor, "adv": adv}, axis=1).dropna() # type: ignore
            if not j.empty:
                corr = _spearman(j.groupby(level=0)["f"].apply(lambda s: s.rank(pct=True)),
                                 j.groupby(level=0)["adv"].apply(lambda s: s.rank(pct=True)))
                capacity_corr = float(corr)

        # Redundancy vs known factors (R^2)
        redundancy_r2 = None
        if self.known_factors is not None:
            join = pd.concat([factor.rename("f"), self.known_factors], axis=1).dropna() # type: ignore
            if not join.empty and join.shape[1] > 1:
                X = join.drop(columns=["f"])
                _, r2 = _ols_residual(join["f"], X)
                redundancy_r2 = r2

        # Stability across halves
        n_dates = len(dates)
        ic_h1 = float(np.mean(ics[: n_dates//2])) if (ics and np is not None) else (sum(ics[: n_dates//2])/max(1,len(ics[: n_dates//2])))
        ic_h2 = float(np.mean(ics[n_dates//2:])) if (ics and np is not None) else (sum(ics[n_dates//2:])/max(1,len(ics[n_dates//2:])))
        stability_halves = abs(ic_h1 - ic_h2)

        return FactorStats(
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_t=ic_t,
            ic_decay=ic_decay,
            ls_ann_ret=ls_ann_ret,
            ls_ann_sharpe=ls_ann_sharpe,
            turnover=float(turnover),
            capacity_corr_adv=capacity_corr,
            redundancy_r2=redundancy_r2,
            stability_halves_diff=float(stability_halves),
            coverage_days=int(len(ics)),
            coverage_names=int(df.index.get_level_values(1).nunique()),
        )

    # ---- Gates & defaults ----
    def _gate(self, st: FactorStats) -> Tuple[bool, Dict[str, Any]]:
        notes = {}
        if st.coverage_days < self.min_days:
            notes["fail_coverage_days"] = st.coverage_days
        if st.ic_ir < self.min_ir:
            notes["fail_ic_ir"] = st.ic_ir
        if st.stability_halves_diff > self.max_halves:
            notes["fail_stability_halves_diff"] = st.stability_halves_diff
        if st.redundancy_r2 is not None and st.redundancy_r2 >= self.r2_thr:
            notes["warn_redundant_r2"] = st.redundancy_r2
        return (len([k for k in notes.keys() if k.startswith("fail_")]) == 0), notes

    def _empty_stats(self) -> FactorStats:
        return FactorStats(
            ic_mean=0.0, ic_std=0.0, ic_ir=0.0, ic_t=0.0, ic_decay=[],
            ls_ann_ret=0.0, ls_ann_sharpe=0.0, turnover=0.0,
            capacity_corr_adv=None, redundancy_r2=None, stability_halves_diff=0.0,
            coverage_days=0, coverage_names=0
        )

    # ---- Export ----
    def to_json(self, rep: DiscoveryReport) -> str:
        def _cand(c: FactorCandidate) -> Dict[str, Any]:
            return {
                "spec": {"name": c.spec.name, "params": c.spec.params, "cross_sectional": c.spec.cross_sectional},
                "stats": asdict(c.stats),
                "passed": c.passed,
                "notes": c.notes
            }
        payload = {
            "ts_ms": rep.ts_ms,
            "universe": rep.universe,
            "config": rep.config,
            "top": [_cand(c) for c in rep.top],
            "rejected": [_cand(c) for c in rep.rejected],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

# ============================== CLI ======================================

def _read_csv(path: str, date_col: str, symbol_col: str) -> pd.DataFrame: # type: ignore
    df = pd.read_csv(path) # type: ignore
    df[date_col] = pd.to_datetime(df[date_col]) # type: ignore
    return df

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Factor Discovery Agent")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--date", type=str, default="date")
    p.add_argument("--symbol", type=str, default="symbol")
    p.add_argument("--close", type=str, default="close")
    p.add_argument("--ret", type=str, default="fwd_ret_1d")
    p.add_argument("--adv", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if pd is None or np is None:
        raise SystemExit("Please install pandas and numpy to use the CLI: pip install pandas numpy")

    raw = _read_csv(args.csv, args.date, args.symbol)
    # set required columns
    need = {args.date, args.symbol, args.close, args.ret}
    miss = need - set(raw.columns)
    if miss:
        raise SystemExit(f"Missing required columns: {sorted(miss)}")

    df = raw.rename(columns={args.date: "date", args.symbol: "symbol", args.close: "close", args.ret: "fwd_ret_1d"})
    if args.adv and args.adv in df.columns:
        df = df.rename(columns={args.adv: "adv"})

    agent = FactorDiscoveryAgent()
    rep = agent.discover(df, target_col="fwd_ret_1d", adv_col=("adv" if "adv" in df.columns else None))
    out = agent.to_json(rep)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)

if __name__ == "__main__":  # pragma: no cover
    _main()