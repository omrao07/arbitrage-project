# backend/analytics/correlations.py
"""
Correlations & Collinearity Monitor
-----------------------------------
Compute correlation matrices (static & rolling), cluster by similarity,
flag high collinearity, and emit diversification metrics for dashboards.

Inputs (flexible):
  returns: Dict[name, List[{"ts": int, "ret": float}]]   # percentage/decimal returns
           or Dict[name, List[float]]                    # equally spaced
Options:
  align='union'|'intersect'   -> how to align on timestamps
  min_overlap: int            -> required overlapping points for a valid corr
  roll_window: int            -> if >0, compute rolling correlations
  alert_threshold: float      -> |corr| >= threshold triggers alert
  publish_insight: bool       -> publish compact summary to ai.insight (if bus present)

Outputs:
  {
    "asof": ts_ms,
    "names": [...],
    "corr": [[...]],              # Pearson correlations
    "diversification_score": 0..1 # 1 = very diversified (low average |corr|)
    "clusters": [[idx,...], ...], # rough agglomerative clusters
    "rolling": { "window": W, "series": { "A|B": [{"ts":..,"corr":..}, ...], ... } }  # optional
    "alerts": [ {"pair":"A|B","corr":0.93}, ... ]
  }

CLI:
  python -m backend.analytics.correlations --probe
  python -m backend.analytics.correlations --in data/returns.json --out runtime/corr.json --roll 60
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional

# Optional accelerators
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None

# Optional clustering
try:
    from scipy.cluster.hierarchy import linkage, fcluster  # type: ignore
    _SCIPY = True
except Exception:
    _SCIPY = False

# Optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore


# --------------------------- utils ---------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _is_ts_row(x: Any) -> bool:
    return isinstance(x, dict) and ("ts" in x) and ("ret" in x)

def _align_series(returns: Dict[str, Any], align: str = "union") -> Tuple[List[str], List[int], Dict[str, List[float]]]:
    """
    Align series by timestamp if dicts with ts are provided; else index by position.
    Returns (names, ts_list, series_map[name]->list[float]).
    """
    names = list(returns.keys())
    # Case 1: already lists of floats
    if names and returns[names[0]] and not _is_ts_row(returns[names[0]][0]):
        L = max(len(v) for v in returns.values())
        ts = list(range(L))
        series = {k: [float(x) for x in returns[k]] + [None]*(L-len(returns[k])) for k in names}
        return names, ts, series # type: ignore

    # Case 2: list of {"ts","ret"} dicts
    index_set = set()
    per = {}
    for k, rows in returns.items():
        per[k] = {int(r["ts"]): float(r["ret"]) for r in rows if r and "ts" in r and "ret" in r}
        index_set |= set(per[k].keys())

    if align == "intersect":
        idx = None
        for k in names:
            idx = set(per[k].keys()) if idx is None else (idx & set(per[k].keys()))
        idx = sorted(idx or [])
    else:
        idx = sorted(index_set)

    series = {k: [per[k].get(t, None) for t in idx] for k in names}
    return names, idx, series

def _pearson(x: List[Optional[float]], y: List[Optional[float]]) -> float:
    # Drop Nones pairwise
    xs, ys = [], []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        xs.append(float(a)); ys.append(float(b))
    n = len(xs)
    if n < 2:
        return float("nan")
    if _np is not None:
        return float(_np.corrcoef(_np.asarray(xs), _np.asarray(ys))[0,1])
    # stdlib fallback
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    denx = math.sqrt(sum((a-mx)**2 for a in xs))
    deny = math.sqrt(sum((b-my)**2 for b in ys))
    return num/(denx*deny) if denx>0 and deny>0 else float("nan")

def _diversification_score(C: List[List[float]]) -> float:
    """
    1 - average absolute correlation (off-diagonal).
    """
    n = len(C)
    if n <= 1: return 1.0
    s = 0.0; m = 0
    for i in range(n):
        for j in range(i+1, n):
            cij = C[i][j]
            if not (cij is None or math.isnan(cij)):
                s += abs(cij); m += 1
    if m == 0: return 1.0
    return max(0.0, min(1.0, 1.0 - s/m))

def _distance_from_corr(C: List[List[float]]) -> List[List[float]]:
    """
    Convert correlation to distance: d = sqrt(0.5 * (1 - corr)), safe for NaNs.
    """
    n = len(C)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            c = C[i][j]
            if i == j:
                D[i][j] = 0.0
            else:
                if c is None or math.isnan(c):
                    D[i][j] = 1.0
                else:
                    D[i][j] = math.sqrt(max(0.0, 0.5*(1.0 - c)))
    return D

def _cluster(D: List[List[float]], names: List[str], k: Optional[int] = None) -> List[List[int]]:
    """
    Rough clustering. If SciPy is available, use hierarchical clustering; else
    do a greedy linkage-ish grouping.
    Returns a list of clusters, each as a list of indices.
    """
    n = len(names)
    if n == 0: return []
    if _SCIPY and _np is not None:
        # Convert square to condensed form for linkage
        arr = _np.asarray(D)
        iu = _np.triu_indices(n, 1)
        condensed = arr[iu]
        Z = linkage(condensed, method="average")
        # choose number of clusters ~ sqrt(n) if not given
        k = k or max(1, int(round(math.sqrt(n))))
        labels = fcluster(Z, k, criterion="maxclust")
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            clusters[int(lab)].append(i)
        return list(clusters.values())

    # Fallback: greedy grouping by nearest neighbor
    remaining = set(range(n))
    clusters = [] # type: ignore
    while remaining:
        i = remaining.pop()
        # find closest neighbors under a soft radius
        radius = sorted(D[i])[min(2, n-1)] if n > 1 else 0.0  # 2nd-nearest distance
        group = [i]
        to_add = [j for j in list(remaining) if D[i][j] <= radius]
        for j in to_add:
            remaining.remove(j); group.append(j)
        clusters.append(sorted(group)) # type: ignore
    return clusters # type: ignore


# --------------------------- core API ---------------------------

def compute_correlations(
    returns: Dict[str, Any],
    *,
    align: str = "union",
    min_overlap: int = 20,
    roll_window: int = 0,
    alert_threshold: float = 0.9,
    publish_insight_flag: bool = False
) -> Dict[str, Any]:
    """
    Main entrypoint: compute static matrix, (optional) rolling, clusters, alerts.
    """
    names, ts, series = _align_series(returns, align=align)
    n = len(names)
    # Static matrix
    C = [[float("nan")]*n for _ in range(n)]
    for i in range(n):
        C[i][i] = 1.0
        for j in range(i+1, n):
            # enforce overlap
            if min_overlap > 0:
                overlap = 0
                for a, b in zip(series[names[i]], series[names[j]]):
                    if a is not None and b is not None:
                        overlap += 1
                if overlap < min_overlap:
                    cij = float("nan")
                else:
                    cij = _pearson(series[names[i]], series[names[j]]) # type: ignore
            else:
                cij = _pearson(series[names[i]], series[names[j]]) # type: ignore
            C[i][j] = C[j][i] = cij

    div_score = _diversification_score(C)
    D = _distance_from_corr(C)
    clusters = _cluster(D, names)

    # Alerts for high |corr|
    alerts: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i+1, n):
            cij = C[i][j]
            if not (cij is None or math.isnan(cij)) and abs(cij) >= alert_threshold:
                alerts.append({"pair": f"{names[i]}|{names[j]}", "corr": float(cij)})

    # Rolling correlations (pairwise)
    rolling = None
    if roll_window and roll_window > 3:
        roll_series: Dict[str, List[Dict[str, Any]]] = {}
        L = len(ts)
        for i in range(n):
            xi = series[names[i]]
            for j in range(i+1, n):
                key = f"{names[i]}|{names[j]}"
                xj = series[names[j]]
                pts = []
                for k in range(roll_window, L+1):
                    win_x = xi[k-roll_window:k]
                    win_y = xj[k-roll_window:k]
                    c = _pearson(win_x, win_y) # type: ignore
                    pts.append({"ts": ts[k-1] if ts else (k-1), "corr": float(c)})
                roll_series[key] = pts
        rolling = {"window": roll_window, "series": roll_series}

    out = {
        "asof": _utc_ms(),
        "names": names,
        "corr": C,
        "diversification_score": float(div_score),
        "clusters": clusters,
        "rolling": rolling,
        "alerts": alerts,
    }

    if publish_insight_flag:
        _publish_insight(names, C, div_score, alerts)

    return out


# --------------------------- insight publisher ---------------------------

def _publish_insight(names: List[str], C: List[List[float]], div: float, alerts: List[Dict[str, Any]]):
    if not publish_stream:
        return
    # summarize hottest pair and average |corr|
    n = len(names)
    hot_pair = None; hot_val = -1.0
    s = 0.0; m = 0
    for i in range(n):
        for j in range(i+1, n):
            cij = C[i][j]
            if cij is None or math.isnan(cij): continue
            if abs(cij) > hot_val:
                hot_val = abs(cij); hot_pair = (names[i], names[j], cij)
            s += abs(cij); m += 1
    avg_abs = (s/m) if m else 0.0
    summary = f"Correlation monitor — avg|ρ|={avg_abs:.2f}, divScore={div:.2f}" \
              + (f"; hottest: {hot_pair[0]}↔{hot_pair[1]} (ρ={hot_pair[2]:.2f})" if hot_pair else "")
    publish_stream("ai.insight", {
        "ts_ms": _utc_ms(),
        "kind": "correlation",
        "summary": summary[:240],
        "details": [f"Alerts: {len(alerts)} pairs ≥ threshold."],
        "tags": ["risk","correlation","diversification"],
        "refs": {}
    })


# --------------------------- convenience I/O ---------------------------

def load_returns(path: str) -> Dict[str, Any]:
    """
    Load returns JSON. Expected format:
      {"alpha.momo":[{"ts":1690000000000,"ret":0.003}, ...],
       "statarb.pairs":[...]}
    or lists of floats of equal/unequal length.
    """
    with open(path, "r") as f:
        return json.load(f)

def save_report(obj: Dict[str, Any], path: str) -> None:
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception:
        pass


# --------------------------- CLI / demo ---------------------------

def _synthetic_returns(n: int = 6, L: int = 250, seed: int = 7) -> Dict[str, List[Dict[str, Any]]]:
    import random
    rng = random.Random(seed)
    base = [rng.gauss(0, 0.01) for _ in range(L)]
    out: Dict[str, List[Dict[str, Any]]] = {}
    ts0 = _utc_ms() - L*86_400_000
    for i in range(n):
        # mix base + idiosyncratic
        series = [0.6*base[t] + 0.4*rng.gauss(0, 0.01) for t in range(L)]
        # add some pairs with deliberate correlation
        if i % 3 == 1:
            series = [series[t] + 0.3*base[t] for t in range(L)]
        if i % 5 == 2:
            series = [series[t] - 0.3*base[t] for t in range(L)]
        name = f"alpha_{i}"
        out[name] = [{"ts": ts0 + t*86_400_000, "ret": series[t]} for t in range(L)]
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Correlations & collinearity monitor")
    ap.add_argument("--in", dest="inp", type=str, help="Path to returns JSON")
    ap.add_argument("--out", dest="out", type=str, default="runtime/corr.json")
    ap.add_argument("--align", type=str, default="union", choices=["union","intersect"])
    ap.add_argument("--min-overlap", type=int, default=20)
    ap.add_argument("--roll", type=int, default=0, help="Rolling window length")
    ap.add_argument("--alert", type=float, default=0.9, help="|corr| alert threshold")
    ap.add_argument("--publish", action="store_true", help="Publish a short insight")
    ap.add_argument("--probe", action="store_true", help="Run a synthetic demo")
    args = ap.parse_args()

    if args.probe:
        rets = _synthetic_returns()
        rep = compute_correlations(rets, align=args.align, min_overlap=args.min_overlap,
                                   roll_window=args.roll, alert_threshold=args.alert,
                                   publish_insight_flag=args.publish)
        os.makedirs(os.path.dirname(args.out) or "runtime", exist_ok=True)
        save_report(rep, args.out)
        print(json.dumps(rep, indent=2)[:2000] + "\n... (truncated)")
        return

    if not args.inp:
        print("Provide --in <returns.json> or use --probe for a demo.")
        return

    rets = load_returns(args.inp)
    rep = compute_correlations(rets, align=args.align, min_overlap=args.min_overlap,
                               roll_window=args.roll, alert_threshold=args.alert,
                               publish_insight_flag=args.publish)
    os.makedirs(os.path.dirname(args.out) or "runtime", exist_ok=True)
    save_report(rep, args.out)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()