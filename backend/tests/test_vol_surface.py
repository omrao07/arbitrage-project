# tests/test_vol_surface.py
import importlib
import math
import numpy as np
import pytest # type: ignore
from typing import Any, Dict, List, Optional, Tuple

"""
Accepted public APIs (any one is fine)

Class-style:
-----------
class VolSurface:
    def fit(self, quotes: list[dict], S0: float, r: float = 0.0, q: float = 0.0, **kw) -> "VolSurface": ...
    def iv(self, T: float, K: float) -> float                                   # alias: vol(), get_vol()
    def price(self, kind: str, T: float, K: float, S0: float | None = None,
              r: float | None = None, q: float | None = None) -> float          # optional, uses surface IV
    def grid(self, expiries: List[float], strikes: List[float]) -> np.ndarray    # [len(T), len(K)] IVs (optional)
    # Optional helpers:
    def no_arb_checks(self) -> dict                                             # {"calendar_ok":bool, "butterfly_ok":bool, ...}
    def params(self) -> dict                                                    # model params (SABR/etc.)
    def export_json(self) -> dict | str
    def import_json(self, blob: dict | str) -> None

Function-style:
---------------
- build_surface(quotes, S0, r=0, q=0) -> handle
- iv(handle, T, K) / get_vol(...)
- price(handle, kind, T, K, S0=None, r=None, q=None)
- grid(handle, expiries, strikes)
- (optional) no_arb_checks(), params(), export_json(), import_json()

This test auto-skips optional features you don't expose.
"""

# ----------------------- Import resolver -----------------------

IMPORT_CANDIDATES = [
    "backend.quant.vol_surface",
    "backend.research.vol_surface",
    "backend.options.vol_surface",
    "quant.vol_surface",
    "vol_surface",
]

def _load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import vol_surface from {IMPORT_CANDIDATES} ({last})")

class API:
    def __init__(self, mod):
        self.mod = mod
        self.obj = None
        if hasattr(mod, "VolSurface"):
            Cls = getattr(mod, "VolSurface")
            try:
                self.obj = Cls()
            except TypeError:
                self.obj = Cls
        else:
            self.build = getattr(mod, "build_surface", None)
            if not self.build:
                pytest.skip("No VolSurface class and no build_surface() factory.")

    def fit(self, quotes, S0, r=0.0, q=0.0, **kw):
        if self.obj:
            if hasattr(self.obj, "fit"):
                return self.obj.fit(quotes, S0=S0, r=r, q=q, **kw)
            else:
                pytest.skip("VolSurface class lacks fit().")
        else:
            self.obj = self.build(quotes, S0=S0, r=r, q=q, **kw) # type: ignore
            return self.obj

    def iv(self, T, K):
        for nm in ("iv", "vol", "get_vol"):
            if hasattr(self.obj, nm) or hasattr(self.mod, nm):
                target = self.obj if hasattr(self.obj, nm) else self.mod
                return getattr(target, nm)(self.obj if target is self.mod else None, T, K) if target is self.mod else getattr(target, nm)(T, K)
        pytest.skip("No iv()/vol()/get_vol() exposed")

    def price(self, kind, T, K, S0=None, r=None, q=None):
        if hasattr(self.obj, "price"):
            return self.obj.price(kind, T, K, S0=S0, r=r, q=q) # type: ignore
        if hasattr(self.mod, "price"):
            return self.mod.price(self.obj, kind, T, K, S0=S0, r=r, q=q)
        pytest.skip("No price() exposed")

    def grid(self, expiries, strikes):
        if hasattr(self.obj, "grid"):
            return self.obj.grid(expiries, strikes) # type: ignore
        if hasattr(self.mod, "grid"):
            return self.mod.grid(self.obj, expiries, strikes)
        pytest.skip("No grid() exposed")

    def has(self, name):
        return hasattr(self.obj, name) or hasattr(self.mod, name)

# ----------------------- Black–Scholes helpers -----------------------

def _N(x):
    # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(kind: str, S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(0.0, (S0 - K) if kind == "call" else (K - S0))
    if sigma <= 0:
        fwd = S0 * math.exp((r - q) * T)
        disc = math.exp(-r * T)
        intrinsic = max(0.0, (fwd - K)) if kind == "call" else max(0.0, (K - fwd))
        return disc * intrinsic
    fwd = S0 * math.exp((r - q) * T)
    vol = sigma * math.sqrt(T)
    d1 = (math.log(fwd / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    df = math.exp(-r * T)
    if kind == "call":
        return df * (fwd * _N(d1) - K * _N(d2))
    else:
        # put via parity
        return df * (K * _N(-d2) - fwd * _N(-d1))

# ----------------------- Synthetic quotes -----------------------

def make_synthetic_quotes(S0=100.0, r=0.01, q=0.00):
    """
    Build a mild smile across strikes and a gentle term-structure:
      ATM vol ~ 20% at 3M, 19% at 6M, 18% at 1Y; wings +5 vols.
    Quotes are in IV (not prices), to avoid model mismatch.
    """
    expiries = [0.25, 0.5, 1.0]   # years
    strikes  = [60, 70, 80, 90, 95, 100, 105, 110, 120, 140]
    quotes = []
    for T in expiries:
        base = 0.20 - 0.02 * (T / 1.0)   # 0.20 -> 0.18
        for K in strikes:
            m = K / S0
            wing = 0.05 * (abs(math.log(m)) / math.log(1.4) if m != 1 else 0.0)  # up to +5 vols at the far wings
            iv = max(0.05, base + wing)
            quotes.append({"T": T, "K": float(K), "kind": "call", "iv": float(iv), "S0": S0, "r": r, "q": q})
            quotes.append({"T": T, "K": float(K), "kind": "put",  "iv": float(iv), "S0": S0, "r": r, "q": q})
    return quotes

# ----------------------- Fixtures -----------------------

@pytest.fixture(scope="module")
def api():
    return API(_load_mod())

@pytest.fixture()
def market():
    return {"S0": 100.0, "r": 0.01, "q": 0.00}

@pytest.fixture()
def fitted(api, market):
    q = make_synthetic_quotes(**market)
    api.fit(q, **market)
    return {"quotes": q}

# ----------------------- Tests -----------------------

def test_fit_and_basic_iv_queries(api, fitted):
    # Probe a few points on and off the grid
    iv_atm_6m = api.iv(0.5, 100.0)
    iv_otm_6m = api.iv(0.5, 120.0)
    assert 0.05 <= iv_atm_6m <= 1.0
    assert iv_otm_6m >= iv_atm_6m - 1e-6  # synthetic smile is wingier

def test_grid_interpolation_if_supported(api):
    try:
        G = api.grid([0.25, 0.5, 1.0], [80, 100, 120])
    except pytest.skip.Exception:
        pytest.skip("grid() not supported")
    arr = np.asarray(G)
    assert arr.shape == (3, 3)
    assert np.isfinite(arr).all()

def test_price_consistency_with_iv(api, market):
    S0, r, q = market["S0"], market["r"], market["q"]
    T, K = 0.5, 100.0
    sigma = api.iv(T, K)
    # If module exposes price(), verify it aligns with Black–Scholes using the same sigma
    try:
        p_mod = api.price("call", T, K, S0=S0, r=r, q=q)
    except pytest.skip.Exception:
        pytest.skip("price() not supported")
    p_bs = bs_price("call", S0, K, T, r, q, sigma)
    assert p_mod == pytest.approx(p_bs, rel=0.05, abs=1e-4)

def test_put_call_parity_if_pricer_exposed(api, market):
    if not api.has("price"):
        pytest.skip("price() not exposed")
    S0, r, q = market["S0"], market["r"], market["q"]
    T, K = 0.5, 110.0
    C = api.price("call", T, K, S0=S0, r=r, q=q)
    P = api.price("put",  T, K, S0=S0, r=r, q=q)
    lhs = C - P
    rhs = S0 * math.exp(-q*T) - K * math.exp(-r*T)
    assert lhs == pytest.approx(rhs, rel=1e-3, abs=1e-3)

def test_calendar_monotonic_call_prices(api, market):
    """Call price must be non-decreasing in maturity (calendar no-arb)."""
    if not api.has("price"):
        pytest.skip("price() not exposed")
    S0, r, q = market["S0"], market["r"], market["q"]
    K = 100.0
    p1 = api.price("call", 0.25, K, S0, r, q)
    p2 = api.price("call", 0.50, K, S0, r, q)
    p3 = api.price("call", 1.00, K, S0, r, q)
    assert p1 <= p2 + 1e-8 and p2 <= p3 + 1e-8

def test_butterfly_convexity_approx(api, market):
    """
    C(K) should be convex in strike (approx). We check a discrete second difference ≥ -ε.
    """
    if not api.has("price"):
        pytest.skip("price() not exposed")
    S0, r, q = market["S0"], market["r"], market["q"]
    T = 0.5
    strikes = np.array([80, 90, 95, 100, 105, 110, 120], dtype=float)
    prices = np.array([api.price("call", T, float(K), S0, r, q) for K in strikes])
    d2 = prices[:-2] - 2*prices[1:-1] + prices[2:]
    # Allow tiny numerical slack
    assert np.min(d2) >= -1e-3

def test_total_variance_non_decreasing_in_T_atm(api):
    """
    T * sigma(T, K_ATM)^2 should be non-decreasing in T (rough calendar sanity).
    """
    K = 100.0
    vols = [api.iv(T, K) for T in (0.25, 0.5, 1.0)]
    tw = [T * (v**2) for T, v in zip((0.25, 0.5, 1.0), vols)]
    assert tw[0] <= tw[1] + 1e-9 and tw[1] <= tw[2] + 1e-9

def test_symmetric_smile_roughly_around_fwd_if_surface_is_symmetric(api, market):
    """
    Many parametric surfaces (e.g., SABR beta≈1) yield 'rough' symmetry in log-moneyness.
    We only assert the two points equally distant in log-moneyness are close in IV.
    """
    S0, r, q = market["S0"], market["r"], market["q"]
    T = 0.5
    F = S0 * math.exp((r - q) * T)
    K1 = F * math.exp(-0.1)
    K2 = F * math.exp(+0.1)
    v1 = api.iv(T, K1)
    v2 = api.iv(T, K2)
    # Loose tolerance; synthetic quotes have wing term, so don't over-tighten
    assert abs(v1 - v2) <= 0.05

def test_export_import_optional(api, fitted):
    if not (api.has("export_json") and api.has("import_json")):
        pytest.skip("No export/import helpers")
    blob = api.obj.export_json() if hasattr(api.obj, "export_json") else None
    assert blob is not None
    if hasattr(api.obj, "import_json"):
        api.obj.import_json(blob)

def test_params_optional(api):
    if not api.has("params"):
        pytest.skip("No params()")
    p = api.obj.params() if hasattr(api.obj, "params") else None
    assert isinstance(p, dict)

def test_no_arb_checker_optional(api):
    if not api.has("no_arb_checks"):
        pytest.skip("no_arb_checks() not exposed")
    out = api.obj.no_arb_checks() if hasattr(api.obj, "no_arb_checks") else {}
    assert isinstance(out, dict)
    # If keys present, they should be booleans
    for k in ("calendar_ok", "butterfly_ok"):
        if k in out: assert isinstance(out[k], (bool, np.bool_))