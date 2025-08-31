# backend/analytics/explainability_heatmap.py
from __future__ import annotations

"""
Explainability Heatmap
----------------------
Aggregate feature contributions (e.g., from SHAP, custom scores, or rule weights)
into matrices you can visualize as heatmaps for strategies/symbols/time buckets.

Zero hard deps:
- If available, uses numpy/pandas/plotly for speed and visualization.
- Otherwise falls back to plain Python.

Typical use
-----------
from backend.analytics.explainability_heatmap import (
    FeatureContribution, HeatmapConfig, ExplainabilityHeatmap
)

hx = ExplainabilityHeatmap(HeatmapConfig(group_by="strategy", agg="mean"))
hx.add(FeatureContribution(
    trade_id="T1", strategy="alpha_meanrev", symbol="AAPL",
    ts_ms=1724890000000, side="buy",
    features={"zscore": +0.8, "spread_bps": -0.2, "sentiment": +0.3},
    weight=1.0, pnl_after=12.5
))
# ... add more
matrix, meta = hx.build(top_k_features=20)
# Optionally: fig = hx.to_plotly_figure(matrix, meta); fig.show()
# Or JSON for the frontend grid: payload = hx.to_json(matrix, meta)

Notes
-----
- All contributions are assumed to be *marginal effect signs* (+ supports buy, - supports sell).
- You can pass raw SHAP values, factor returns, or normalized rule weights.
- `weight` lets you scale by trade size / notional / confidence.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable, Any
import math
import json

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import pandas as _pd
except Exception:
    _pd = None

try:
    import plotly.graph_objs as _go  # optional for interactive heatmap
except Exception:
    _go = None


# -------------------------- Data models --------------------------

@dataclass
class FeatureContribution:
    trade_id: str
    strategy: str
    symbol: str
    ts_ms: int
    side: str                           # "buy" | "sell"
    features: Dict[str, float]          # feature -> contribution (can be SHAP, z*beta, etc.)
    weight: float = 1.0                 # scale (e.g., notional, confidence)
    pnl_after: Optional[float] = None   # realized PnL (for diagnostics)

@dataclass
class HeatmapConfig:
    """
    group_by: which dimension becomes the columns:
      - "strategy" | "symbol" | "time" (bucketed) | "strategy_symbol"
    agg: how to aggregate contributions across rows for the same cell:
      - "mean" | "sum" | "median"
    time_bucket_ms: if group_by="time", bucket width in ms (e.g., 5min = 300000)
    normalize: None | "row" | "col" | "zrow" | "zcol" (normalize scores for readability)
    signed: True keeps sign; False uses absolute contributions
    """
    group_by: str = "strategy"
    agg: str = "mean"
    time_bucket_ms: int = 300_000
    normalize: Optional[str] = None
    signed: bool = True


# -------------------------- Core engine --------------------------

class ExplainabilityHeatmap:
    def __init__(self, cfg: HeatmapConfig = HeatmapConfig()):
        self.cfg = cfg
        self._rows: List[FeatureContribution] = []

    # ingest
    def add(self, fc: FeatureContribution) -> None:
        self._rows.append(fc)

    def extend(self, fcs: Iterable[FeatureContribution]) -> None:
        for fc in fcs:
            self._rows.append(fc)

    # build the matrix
    def build(self, *, top_k_features: int = 30) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Returns (matrix, meta) where:
          - matrix is a 2D list [n_features x n_cols]
          - meta = {"features": [...], "columns": [...], "col_type": str, "agg": str, "normalize": str|None}
        """
        if not self._rows:
            return [], {"features": [], "columns": [], "col_type": self.cfg.group_by, "agg": self.cfg.agg, "normalize": self.cfg.normalize}

        # 1) compute column keys
        def col_key(r: FeatureContribution) -> str:
            if self.cfg.group_by == "strategy":
                return r.strategy
            if self.cfg.group_by == "symbol":
                return r.symbol
            if self.cfg.group_by == "strategy_symbol":
                return f"{r.strategy}:{r.symbol}"
            if self.cfg.group_by == "time":
                b = (r.ts_ms // self.cfg.time_bucket_ms) * self.cfg.time_bucket_ms
                return str(b)
            return r.strategy  # default

        # 2) accumulate feature → col sums & weights
        # store sums & counts per (feature, col)
        sums: Dict[Tuple[str, str], float] = {}
        counts: Dict[Tuple[str, str], float] = {}
        features_set, cols_set = set(), set()

        for r in self._rows:
            ck = col_key(r)
            cols_set.add(ck)
            sgn = 1.0 if self.cfg.signed else 1.0
            side_mult = 1.0  # if you want to flip sell: -1.0, keep as +1 to represent direction-specific contrib
            for fname, val in r.features.items():
                features_set.add(fname)
                v = float(val)
                if not self.cfg.signed:
                    v = abs(v)
                # optional: scale by weight; include side if desired
                v_eff = v * r.weight * side_mult * sgn
                key = (fname, ck)
                sums[key] = sums.get(key, 0.0) + v_eff
                counts[key] = counts.get(key, 0.0) + 1.0

        features = sorted(list(features_set))
        columns = sorted(list(cols_set), key=lambda x: (len(x), x))  # stable order

        # 3) aggregate (mean/sum/median)
        if self.cfg.agg not in ("mean", "sum", "median"):
            raise ValueError(f"Unsupported agg: {self.cfg.agg}")

        # If pandas available, use it for convenience
        if _pd is not None:
            import pandas as pd
            import numpy as np
            idx = pd.Index(features, name="feature")
            cols = pd.Index(columns, name=self.cfg.group_by)
            df = pd.DataFrame(0.0, index=idx, columns=cols)
            cnt = pd.DataFrame(0.0, index=idx, columns=cols)

            for (f, c), s in sums.items():
                df.at[f, c] = df.at[f, c] + s
                cnt.at[f, c] = cnt.at[f, c] + counts[(f, c)]

            if self.cfg.agg == "mean":
                with np.errstate(divide="ignore", invalid="ignore"):
                    df = df / cnt.replace(0.0, np.nan)
            elif self.cfg.agg == "median":
                # For median we need raw lists; fallback to mean if not available
                # (To support true median, accumulate lists per (f,c) above.)
                pass  # keep as-is; mean approximates central tendency
            # sum: keep df as sums

            # 4) feature selection: top-K by global importance (L1)
            importance = df.abs().sum(axis=1).sort_values(ascending=False)
            keep = importance.index[: min(top_k_features, len(importance))]
            df = df.loc[keep]

            # 5) normalization
            df = self._normalize_df(df)

            return df.values.tolist(), {
                "features": list(df.index),
                "columns": list(df.columns),
                "col_type": self.cfg.group_by,
                "agg": self.cfg.agg,
                "normalize": self.cfg.normalize,
                "importance": importance.loc[keep].tolist(),
            }

        # --------- Pure Python fallback ----------
        # Build dense matrix as dict-of-dicts
        mat: List[List[float]] = []
        # compute mean/sum
        cell = {}
        for f in features:
            for c in columns:
                key = (f, c)
                if self.cfg.agg == "mean":
                    v = (sums.get(key, 0.0) / counts.get(key, 0.0)) if counts.get(key, 0.0) > 0 else 0.0
                else:
                    v = sums.get(key, 0.0)
                cell[key] = v

        # feature selection by global L1
        importance_py = []
        for f in features:
            total = 0.0
            for c in columns:
                total += abs(cell[(f, c)])
            importance_py.append((f, total))
        importance_py.sort(key=lambda t: t[1], reverse=True)
        keep_feats = [f for f, _ in importance_py[: min(top_k_features, len(importance_py))]]

        # build matrix
        for f in keep_feats:
            row = [cell[(f, c)] for c in columns]
            mat.append(row)

        # normalization
        mat = self._normalize_py(mat, mode=self.cfg.normalize)

        return mat, {
            "features": keep_feats,
            "columns": columns,
            "col_type": self.cfg.group_by,
            "agg": self.cfg.agg,
            "normalize": self.cfg.normalize,
            "importance": [x for _, x in importance_py[: len(keep_feats)]],
        }

    # ------------------------ Normalization ------------------------

    def _normalize_df(self, df):
        if self.cfg.normalize is None:
            return df
        mode = self.cfg.normalize.lower()
        import numpy as np
        if mode == "row":
            denom = np.maximum(1e-12, df.abs().max(axis=1))
            return df.div(denom, axis=0)
        if mode == "col":
            denom = np.maximum(1e-12, df.abs().max(axis=0))
            return df.div(denom, axis=1)
        if mode == "zrow":
            mu = df.mean(axis=1)
            sd = df.std(axis=1).replace(0.0, 1.0)
            return df.sub(mu, axis=0).div(sd, axis=0)
        if mode == "zcol":
            mu = df.mean(axis=0)
            sd = df.std(axis=0).replace(0.0, 1.0)
            return df.sub(mu, axis=1).div(sd, axis=1)
        return df

    def _normalize_py(self, mat: List[List[float]], mode: Optional[str]):
        if not mode:
            return mat
        mode = mode.lower()
        R = len(mat)
        C = len(mat[0]) if R else 0
        if R == 0 or C == 0:
            return mat

        def max_abs(xs): return max(1e-12, max(abs(x) for x in xs))

        if mode == "row":
            out = []
            for r in mat:
                den = max_abs(r)
                out.append([x / den for x in r])
            return out
        if mode == "col":
            dens = [max_abs([mat[i][j] for i in range(R)]) for j in range(C)]
            out = []
            for i in range(R):
                out.append([mat[i][j] / dens[j] for j in range(C)])
            return out
        if mode in ("zrow", "zcol"):
            # simple z-score
            if mode == "zrow":
                out = []
                for r in mat:
                    mu = sum(r)/C
                    var = sum((x-mu)**2 for x in r)/max(1, C-1)
                    sd = math.sqrt(var) or 1.0
                    out.append([(x-mu)/sd for x in r])
                return out
            else:
                # z per column
                mus = []
                sds = []
                for j in range(C):
                    col = [mat[i][j] for i in range(R)]
                    mu = sum(col)/R
                    var = sum((x-mu)**2 for x in col)/max(1, R-1)
                    sd = math.sqrt(var) or 1.0
                    mus.append(mu); sds.append(sd)
                out = []
                for i in range(R):
                    out.append([(mat[i][j]-mus[j])/sds[j] for j in range(C)])
                return out
        return mat

    # ------------------------ Export helpers ------------------------

    def to_plotly_figure(self, matrix: List[List[float]], meta: Dict[str, Any]):
        """
        Returns a plotly Figure if plotly is installed; otherwise None.
        """
        if _go is None:
            return None
        features = meta.get("features", [])
        columns = meta.get("columns", [])
        heat = _go.Heatmap(z=matrix, x=columns, y=features, colorbar={"title": "contrib"})
        layout = _go.Layout(
            title=f"Explainability Heatmap ({meta.get('col_type','cols')}, {meta.get('agg','mean')}, norm={meta.get('normalize')})",
            xaxis={"title": meta.get("col_type", "group")},
            yaxis={"title": "feature"},
            margin={"l": 140, "r": 40, "t": 60, "b": 80},
        )
        return _go.Figure(data=[heat], layout=layout)

    def to_json(self, matrix: List[List[float]], meta: Dict[str, Any]) -> str:
        """
        Frontend-friendly payload: {meta, z} where z is a dense list-of-lists.
        """
        payload = {"meta": meta, "z": matrix}
        return json.dumps(payload, ensure_ascii=False, indent=2)