# backend/risk/liquidity_surface.py
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional deps kept soft
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore

# =============================================================================
# Data models
# =============================================================================

@dataclass
class LiquidityPoint:
    """
    One empirical observation of liquidity/impact.
    participation = child_notional / ADV_notional   (0..1+)
    'impact_bps' should be signed vs mid before the trade (taker perspective).
    """
    ts: datetime
    symbol: str
    venue: str
    side: str                    # 'buy' or 'sell' (may be used for asymmetry later)
    child_qty: float             # shares/contracts
    child_notional: float        # child_qty * execution_price in base CCY
    adv_notional: float          # ADV notional for symbol on that day
    vol_daily: float             # daily sigma (decimal), e.g., 0.02 = 2%
    spread_bps: float            # prevailing quoted spread (bps of mid)
    impact_bps: float            # execution impact vs mid (bps, taker signed)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def participation(self) -> float:
        if self.adv_notional <= 0:
            return 0.0
        return max(0.0, self.child_notional / self.adv_notional)


@dataclass
class BinStat:
    n: int = 0
    med_spread_bps: float = 0.0
    med_impact_bps: float = 0.0
    p90_impact_bps: float = 0.0
    p10_impact_bps: float = 0.0

@dataclass
class LiquiditySurface:
    """
    2D binned surface of expected impact/spread as a function of:
      x = participation (child_notional / ADV)
      y = volatility (daily sigma)
    """
    x_edges: List[float]                    # participation bin edges (e.g., [0, 0.02, 0.05, ...])
    y_edges: List[float]                    # vol bin edges (e.g., [0.01, 0.02, 0.03, ...])
    grid: List[List[BinStat]]               # shape [len(y_bins)-1][len(x_bins)-1]
    k_sqrt: float = 0.0                     # fitted square-root impact coefficient, in bps / sqrt(participation)
    k_vol: float = 1.0                      # multiplicative vol scaling (unitless)
    asof: datetime = field(default_factory=lambda: datetime.utcnow())
    points_used: int = 0

    # ------------------------- Construction -------------------------
    @staticmethod
    def empty(x_edges: Sequence[float], y_edges: Sequence[float]) -> "LiquiditySurface":
        xe = sorted(set(float(x) for x in x_edges))
        ye = sorted(set(float(y) for y in y_edges))
        grid = [[BinStat() for _ in range(len(xe)-1)] for __ in range(len(ye)-1)]
        return LiquiditySurface(x_edges=list(xe), y_edges=list(ye), grid=grid)

    @staticmethod
    def from_points(points: Iterable[LiquidityPoint],
                    *, x_edges: Sequence[float] = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
                    y_edges: Sequence[float] = (0.01, 0.015, 0.02, 0.03, 0.05)) -> "LiquiditySurface":
        surf = LiquiditySurface.empty(x_edges, y_edges)
        # bin collection
        buckets: List[List[List[LiquidityPoint]]] = [
            [ [] for _ in range(len(surf.x_edges)-1) ] for __ in range(len(surf.y_edges)-1)
        ]
        npts = 0
        for p in points:
            xi = _bin_index(p.participation, surf.x_edges)
            yi = _bin_index(p.vol_daily, surf.y_edges)
            if xi is None or yi is None:
                continue
            buckets[yi][xi].append(p)
            npts += 1

        # aggregate stats
        for yi in range(len(surf.y_edges)-1):
            for xi in range(len(surf.x_edges)-1):
                arr = buckets[yi][xi]
                bs = surf.grid[yi][xi]
                if not arr:
                    continue
                bs.n = len(arr)
                bs.med_spread_bps = _median([a.spread_bps for a in arr])
                impacts = [a.impact_bps for a in arr]
                bs.med_impact_bps = _median(impacts)
                bs.p90_impact_bps = _percentile(impacts, 0.90)
                bs.p10_impact_bps = _percentile(impacts, 0.10)

        # fit square-root model with vol scaling: E[impact_bps] ≈ k_sqrt * sqrt(participation) * (1 + k_vol_adj*(vol-μ))
        surf.k_sqrt, surf.k_vol = _fit_sqrt_model(points)
        surf.asof = datetime.utcnow()
        surf.points_used = npts
        return surf

    # ------------------------- Queries -------------------------
    def expected_impact_bps(self, participation: float, vol_daily: float) -> float:
        """
        Bilinear interpolate med_impact surface; fall back to analytic sqrt model if empty.
        """
        val = self._interp("med_impact_bps", participation, vol_daily)
        if val is not None:
            return float(val)
        # analytic fallback
        return max(0.0, self.k_sqrt * math.sqrt(max(0.0, participation)) * (1.0 + self._vol_scale(vol_daily)))

    def p90_impact_bps(self, participation: float, vol_daily: float) -> float:
        val = self._interp("p90_impact_bps", participation, vol_daily)
        if val is not None:
            return float(val)
        # crude safety multiplier above expectation
        return 1.5 * self.expected_impact_bps(participation, vol_daily)

    def spread_bps(self, participation: float, vol_daily: float) -> float:
        val = self._interp("med_spread_bps", participation, vol_daily)
        if val is not None:
            return float(val)
        # spread baseline ~ 2 * vol_tick where vol_tick ~ vol_daily / sqrt(6.5h*3600/1s)
        return 5.0 + 10000.0 * (vol_daily / 100.0)  # conservative fallback

    # ------------------------- Sizing helpers -------------------------
    def max_child_notional_for_cap(self, *, adv_notional: float, vol_daily: float, cap_bps: float) -> float:
        """
        Solve for child notional such that expected impact <= cap_bps.
        Uses analytic sqrt model for robustness (doesn't depend on grid sparsity).
        """
        if adv_notional <= 0:
            return 0.0
        # cap_bps >= k * sqrt(pi) * (1 + vol_scale)  =>  sqrt(pi) <= cap / denom
        denom = max(1e-9, self.k_sqrt * (1.0 + self._vol_scale(vol_daily)))
        root_pi = max(0.0, float(cap_bps) / denom)
        participation = max(0.0, root_pi * root_pi)
        return participation * adv_notional

    def twap_schedule(self, *, total_notional: float, adv_notional: float, vol_daily: float,
                      horizon_minutes: int = 30, bars_per_minute: float = 1.0,
                      per_child_cap_bps: Optional[float] = None) -> List[Dict[str, float]]:
        """
        Build a TWAP child schedule across fixed bars.
        If per_child_cap_bps is set, clip each child to satisfy the cap.
        Returns list of {t_min_from_start, child_notional, participation}.
        """
        steps = max(1, int(horizon_minutes * bars_per_minute))
        target = total_notional / steps
        out: List[Dict[str, float]] = []
        cap_child = None
        if per_child_cap_bps is not None:
            cap_child = self.max_child_notional_for_cap(
                adv_notional=adv_notional, vol_daily=vol_daily, cap_bps=per_child_cap_bps
            )
        for k in range(steps):
            child = min(target, cap_child) if cap_child else target
            part = child / max(1e-9, adv_notional)
            out.append({"t_min": k / bars_per_minute, "child_notional": child, "participation": part})
        return out

    # ------------------------- Serialization -------------------------
    def to_json(self) -> str:
        d = {
            "x_edges": self.x_edges,
            "y_edges": self.y_edges,
            "grid": [[asdict(c) for c in row] for row in self.grid],
            "k_sqrt": self.k_sqrt,
            "k_vol": self.k_vol,
            "asof": self.asof.isoformat(),
            "points_used": self.points_used,
        }
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "LiquiditySurface":
        o = json.loads(s)
        grid = [[BinStat(**cell) for cell in row] for row in o["grid"]]
        surf = LiquiditySurface(
            x_edges=list(map(float, o["x_edges"])),
            y_edges=list(map(float, o["y_edges"])),
            grid=grid,
            k_sqrt=float(o.get("k_sqrt", 0.0)),
            k_vol=float(o.get("k_vol", 1.0)),
            asof=datetime.fromisoformat(o.get("asof")),
            points_used=int(o.get("points_used", 0)),
        )
        return surf

    # ------------------------- Internals -------------------------
    def _vol_scale(self, v: float) -> float:
        """
        Linear scaling around median vol; k_vol is slope around anchor.
        """
        # anchor at mid of y_edges if available
        if self.y_edges and len(self.y_edges) > 1:
            anchor = 0.5 * (self.y_edges[0] + self.y_edges[-1])
        else:
            anchor = v
        return self.k_vol * (max(0.0, v) - anchor)

    def _interp(self, attr: str, x: float, y: float) -> Optional[float]:
        xi = _bin_floor_index(x, self.x_edges)
        yi = _bin_floor_index(y, self.y_edges)
        if xi is None or yi is None:
            return None
        # gather four surrounding cells for bilinear
        x0, x1 = self.x_edges[xi], self.x_edges[min(xi+1, len(self.x_edges)-1)]
        y0, y1 = self.y_edges[yi], self.y_edges[min(yi+1, len(self.y_edges)-1)]
        # if at the right/top edge, clamp to previous cell
        if x1 == x0 and xi > 0: x0, x1, xi = self.x_edges[xi-1], self.x_edges[xi], xi-1
        if y1 == y0 and yi > 0: y0, y1, yi = self.y_edges[yi-1], self.y_edges[yi], yi-1
        cells: List[Tuple[float, float, Optional[float]]] = []
        for dy in (0, 1):
            for dx in (0, 1):
                c = self.grid[min(yi+dy, len(self.grid)-1)][min(xi+dx, len(self.grid[0])-1)]
                val = getattr(c, attr, None)
                # empty bin -> None
                if not c.n:
                    val = None
                cells.append((dx, dy, val))
        # if any None, fall back to nearest non-empty neighbor
        if any(v is None for _, _, v in cells):
            neigh = _nearest_non_empty(self.grid, xi, yi, attr)
            return neigh
        # bilinear weights
        if x1 == x0 or y1 == y0:
            return cells[0][2]  # degenerate
        tx = (x - x0) / (x1 - x0)
        ty = (y - y0) / (y1 - y0)
        v00 = cells[0][2]; v10 = cells[1][2]; v01 = cells[2][2]; v11 = cells[3][2]
        return (
            (1-tx)*(1-ty)*v00 + tx*(1-ty)*v10 + (1-tx)*ty*v01 + tx*ty*v11  # type: ignore[operator]
        )

# =============================================================================
# Fitting / helpers
# =============================================================================

def _fit_sqrt_model(points: Iterable[LiquidityPoint]) -> Tuple[float, float]:
    """
    Fit E[impact_bps] ~ k_sqrt * sqrt(participation) * (1 + k_vol*(vol - median_vol))
    Robust: uses median-based regressors if numpy is absent / data is sparse.
    Returns (k_sqrt, k_vol).
    """
    xs: List[float] = []
    ys: List[float] = []
    vs: List[float] = []

    for p in points:
        pi = p.participation
        if pi <= 0:
            continue
        xs.append(math.sqrt(pi))
        ys.append(float(p.impact_bps))
        vs.append(float(p.vol_daily))

    if not xs or not ys:
        return (0.0, 0.0)

    v_med = statistics.median(vs)

    if _np is None or len(xs) < 24:
        # crude robust slope using median ratios
        # k ≈ median( impact / sqrt(pi) )
        base = statistics.median([y / max(1e-9, x) for x, y in zip(xs, ys)])
        # vol slope: regression on residual vs (v - v_med)
        resid = [y - base * x for x, y in zip(xs, ys)]
        dv = [v - v_med for v in vs]
        denom = sum(d* d for d in dv) or 1e-9
        kv = sum(r*d for r, d in zip(resid, dv)) / denom
        return (float(max(0.0, base)), float(kv / max(1.0, base)))  # normalize slope to be relative
    else:
        # numpy least squares on [sqrt(pi), sqrt(pi)*(v - v_med)]
        X = _np.column_stack([_np.array(xs), _np.array(xs) * (_np.array(vs) - v_med)])
        beta, *_ = _np.linalg.lstsq(X, _np.array(ys), rcond=None)
        k = float(max(0.0, beta[0]))
        kv_abs = float(beta[1])
        # convert kv_abs (per unit vol deviation) to relative k_vol (unitless multiplier slope)
        k_vol = 0.0 if k == 0 else (kv_abs / k)
        return (k, k_vol)

def _bin_index(x: float, edges: Sequence[float]) -> Optional[int]:
    if not edges or len(edges) < 2:
        return None
    for i in range(len(edges)-1):
        if edges[i] <= x < edges[i+1]:
            return i
    # include right edge
    if math.isclose(x, edges[-1]):
        return len(edges)-2
    return None

def _bin_floor_index(x: float, edges: Sequence[float]) -> Optional[int]:
    if not edges or len(edges) < 2:
        return None
    if x < edges[0]:
        return None
    for i in range(len(edges)-1):
        if edges[i] <= x <= edges[i+1]:
            return i
    return len(edges)-2 if x > edges[-1] else None

def _median(arr: Sequence[float]) -> float:
    try:
        return float(statistics.median(list(map(float, arr))))
    except Exception:
        a = sorted(float(x) for x in arr)
        n = len(a)
        if n == 0: return 0.0
        if n % 2: return a[n//2]
        return 0.5*(a[n//2 - 1] + a[n//2])

def _percentile(arr: Sequence[float], q: float) -> float:
    if not arr:
        return 0.0
    a = sorted(map(float, arr))
    if _np is not None:
        return float(_np.percentile(a, 100*q, interpolation="linear" if hasattr(_np, "percentile") else "linear")) # type: ignore
    # pure python
    k = q * (len(a) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return a[int(k)]
    return a[f] * (c - k) + a[c] * (k - f)

def _nearest_non_empty(grid: List[List[BinStat]], xi: int, yi: int, attr: str) -> Optional[float]:
    """
    Search outward Manhattan rings for the closest non-empty bin and return its attr.
    """
    H = len(grid); W = len(grid[0]) if H else 0
    for r in range(0, max(H, W)):
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if abs(dx) + abs(dy) != r:
                    continue
                x = xi + dx; y = yi + dy
                if x < 0 or y < 0 or x >= W or y >= H:
                    continue
                bs = grid[y][x]
                if bs.n > 0:
                    return float(getattr(bs, attr))
    return None

# =============================================================================
# Convenience: building points from fills/quotes
# =============================================================================

def points_from_execs_and_nbbo(
    fills: Iterable[Dict[str, Any]],
    nbbo_getter,  # callable(symbol, ts) -> dict with 'mid', 'spread_bps'
    adv_notional_getter,  # callable(symbol, ts_date) -> float
    vol_getter,  # callable(symbol, ts_date) -> float (daily sigma)
) -> List[LiquidityPoint]:
    """
    Convert your execution logs and quote snapshots to LiquidityPoints.
    Expected fill keys: ts (datetime/iso), symbol, side, qty, price, venue
    nbbo_getter(symbol, ts) -> {'mid': float, 'spread_bps': float}
    """
    out: List[LiquidityPoint] = []
    for f in fills:
        ts = f["ts"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        sym = f["symbol"]; side = f.get("side", "buy").lower()
        px = float(f["price"]); qty = float(f["qty"])
        nb = nbbo_getter(sym, ts) or {}
        mid = float(nb.get("mid") or px)
        spread_bps = float(nb.get("spread_bps") or 0.0)
        impact_bps = 1e4 * (px - mid) / max(1e-9, mid) * (1 if side == "buy" else -1)
        adv = float(adv_notional_getter(sym, ts.date()) or 0.0)
        vol = float(vol_getter(sym, ts.date()) or 0.0)
        out.append(LiquidityPoint(
            ts=ts, symbol=sym, venue=f.get("venue",""), side=side, child_qty=qty,
            child_notional=qty * px, adv_notional=adv, vol_daily=vol,
            spread_bps=spread_bps, impact_bps=impact_bps, meta={"mid": mid}
        ))
    return out

# =============================================================================
# Tiny demo
# =============================================================================

if __name__ == "__main__":
    # Fabricate a few observations for a symbol
    import random
    random.seed(42)

    def fake_nbbo(_sym: str, _ts: datetime):
        return {"mid": 100.0, "spread_bps": 3.0}

    def fake_adv(_sym: str, _d):
        return 50_000_000.0  # $50m ADV

    def fake_vol(_sym: str, _d):
        return 0.02  # 2% daily vol

    fills = []
    base_mid = 100.0
    for k in range(200):
        ts = datetime.utcnow() - timedelta(minutes=60-k)
        part = random.choice([0.005, 0.01, 0.02, 0.05, 0.1])  # 0.5% to 10% ADV child
        notional = part * fake_adv("SYM", ts.date())
        side = random.choice(["buy", "sell"])
        # synthetic square-root impact ~ 12 bps * sqrt(part)
        imp_bps = (12.0 * math.sqrt(part)) * (1 if side == "buy" else 1)
        # add noise
        imp_bps *= random.uniform(0.7, 1.3)
        px = base_mid * (1 + (imp_bps/1e4) * (1 if side == "buy" else -1))
        fills.append({"ts": ts, "symbol": "SYM", "side": side, "qty": notional/base_mid, "price": px, "venue": "SIM"})

    pts = points_from_execs_and_nbbo(fills, fake_nbbo, fake_adv, fake_vol)
    surf = LiquiditySurface.from_points(pts)

    print("k_sqrt (bps/sqrt(part)):", round(surf.k_sqrt, 2), "k_vol:", round(surf.k_vol, 4))
    ask_parts = [0.005, 0.02, 0.05, 0.1]
    for pi in ask_parts:
        print(f"Expected impact @ part={pi:.3%}:", round(surf.expected_impact_bps(pi, 0.02), 2), "bps")

    cap = 8.0  # cap each child at 8 bps expected impact
    child = surf.max_child_notional_for_cap(adv_notional=fake_adv("SYM", datetime.utcnow().date()), vol_daily=0.02, cap_bps=cap)
    print("Max child notional for", cap, "bps cap:", round(child, 0))

    sched = surf.twap_schedule(total_notional=10_000_000.0, adv_notional=fake_adv("SYM", datetime.utcnow().date()),
                               vol_daily=0.02, horizon_minutes=30, bars_per_minute=2, per_child_cap_bps=cap)
    print("TWAP bars:", len(sched), "first bar:", sched[0])