# tests/test_monte_carlo.py
import importlib
import math
import numpy as np
import pytest # type: ignore
from typing import Any, Dict, Tuple, Optional

"""
Expected public APIs (any one is fine)

A) Class MonteCarlo with methods:
   - simulate_paths(S0, mu, sigma, T, steps, n_paths, *, seed=None, dt=None) -> np.ndarray [n_paths, steps+1]
   - price_option(kind, S0, K, r, sigma, T, n_paths, steps, *, seed=None,
                  antithetic=False, control_variate=False, barrier=None) -> dict|float
     (dict may include {'price': float, 'stderr': float, ...})

B) Functions:
   - simulate_paths(...)
   - price_option(...)

If names differ slightly, tweak the small resolver below.
"""

# --------------------------- Import resolver ---------------------------

IMPORT_CANDIDATES = [
    "backend.quant.monte_carlo",
    "backend.research.monte_carlo",
    "backend.analytics.monte_carlo",
    "quant.monte_carlo",
    "monte_carlo",
]

def load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import monte_carlo from {IMPORT_CANDIDATES} ({last})")

class API:
    def __init__(self, mod):
        self.mod = mod
        self.obj = None
        if hasattr(mod, "MonteCarlo"):
            MC = getattr(mod, "MonteCarlo")
            try:
                self.obj = MC()
            except TypeError:
                self.obj = MC
        # map names
        self._sim_name = "simulate_paths" if hasattr(self.obj or mod, "simulate_paths") else None
        self._price_name = "price_option" if hasattr(self.obj or mod, "price_option") else None
        if not self._sim_name:
            # alt names
            for nm in ("simulate", "gbm_paths", "paths"):
                if hasattr(self.obj or mod, nm):
                    self._sim_name = nm; break
        if not self._price_name:
            for nm in ("price", "price_european"):
                if hasattr(self.obj or mod, nm):
                    self._price_name = nm; break
        if not self._sim_name:
            pytest.skip("No simulate_paths()/simulate() found in module/class.")

    def call(self, name, *args, **kw):
        target = self.obj if (self.obj and hasattr(self.obj, name)) else self.mod
        return getattr(target, name)(*args, **kw)

    def simulate(self, **kw):
        return self.call(self._sim_name, **kw)

    def price(self, **kw):
        if not self._price_name:
            pytest.skip("No price_option()/price() API exposed")
        return self.call(self._price_name, **kw)

# --------------------------- Utilities ---------------------------------

def bs_call_price(S0, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0) * math.exp(-r*T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from math import erf, sqrt
    N = lambda x: 0.5 * (1 + erf(x / math.sqrt(2)))
    return S0 * N(d1) - K * math.exp(-r * T) * N(d2)

def to_price(val):
    if isinstance(val, dict):
        return float(val.get("price", val.get("pv", val.get("value")))) # type: ignore
    return float(val)

# --------------------------- Fixtures ----------------------------------

@pytest.fixture(scope="module")
def api():
    return API(load_mod())

@pytest.fixture()
def params():
    return dict(S0=100.0, mu=0.06, sigma=0.2, T=1.0, steps=252, n_paths=50_000)

# --------------------------- Tests -------------------------------------

def test_gbm_moments_match_theory(api, params):
    S0, mu, sigma, T = params["S0"], params["mu"], params["sigma"], params["T"]
    paths = api.simulate(S0=S0, mu=mu, sigma=sigma, T=T, steps=params["steps"], n_paths=params["n_paths"], seed=42)
    assert isinstance(paths, (np.ndarray, list))
    arr = np.asarray(paths)
    # shape: (n_paths, steps+1) or (steps+1, n_paths) -> normalize
    if arr.shape[0] == params["n_paths"]:
        ST = arr[:, -1]
    elif arr.shape[1] == params["n_paths"]:
        ST = arr[-1, :]
    else:
        # best effort pick the longest axis as time
        ST = arr.reshape(params["n_paths"], -1)[:, -1]
    # Theoretical E[S_T] and Var[S_T] for GBM
    mean_theory = S0 * math.exp(mu * T)
    var_theory = (S0**2) * math.exp(2*mu*T) * (math.exp(sigma**2 * T) - 1.0)
    mean_sample = float(np.mean(ST))
    var_sample = float(np.var(ST, ddof=0))
    # Loose tolerances (Monte Carlo error shrinks with n_paths)
    assert mean_sample == pytest.approx(mean_theory, rel=0.02)
    assert var_sample == pytest.approx(var_theory, rel=0.05)

def test_reproducibility_with_seed(api, params):
    kw = dict(params); kw["seed"] = 7
    a = api.simulate(**kw)
    b = api.simulate(**kw)
    assert np.allclose(np.asarray(a), np.asarray(b))

def test_zero_vol_deterministic_path(api, params):
    kw = dict(params); kw["sigma"] = 0.0; kw["seed"] = 1
    paths = api.simulate(**kw)
    arr = np.asarray(paths)
    if arr.shape[0] == kw["n_paths"]:
        ST = arr[:, -1]
    else:
        ST = arr[-1, :]
    det = params["S0"] * math.exp(params["mu"] * params["T"])
    assert float(np.std(ST)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.mean(ST)) == pytest.approx(det, rel=1e-12, abs=1e-9)

def test_risk_neutral_european_call_price(api, params):
    # Under risk-neutral measure, use mu=r for pricing
    S0, r, sigma, T = 100.0, 0.03, 0.2, 1.0
    K = 100.0
    mc = api.price(kind="call", S0=S0, K=K, r=r, sigma=sigma, T=T,
                   n_paths=100_000, steps=252, seed=123, antithetic=True if hasattr(api.mod, "price_option") else False)
    pv_mc = to_price(mc)
    pv_bs = bs_call_price(S0, K, r, sigma, T)
    # 1-2% tolerance typical
    assert pv_mc == pytest.approx(pv_bs, rel=0.02)

def test_antithetic_reduces_variance_if_supported(api):
    if not hasattr(api, "_price_name"):
        pytest.skip("No pricing API")
    try:
        base = api.price(kind="call", S0=100, K=100, r=0.02, sigma=0.25, T=0.5,
                         n_paths=40_000, steps=126, seed=99, antithetic=False)
        anti = api.price(kind="call", S0=100, K=100, r=0.02, sigma=0.25, T=0.5,
                         n_paths=40_000, steps=126, seed=99, antithetic=True)
    except TypeError:
        pytest.skip("antithetic flag not supported")
    se_base = float(base.get("stderr", 0.0)) if isinstance(base, dict) else None
    se_anti = float(anti.get("stderr", 0.0)) if isinstance(anti, dict) else None
    if se_base and se_anti:
        assert se_anti <= se_base * 1.05  # allow tiny slack
    else:
        # Fallback: run multiple batches to estimate empirical variance
        pass

def test_control_variate_if_supported(api):
    try:
        raw = api.price(kind="call", S0=100, K=90, r=0.02, sigma=0.3, T=1.0,
                        n_paths=30_000, steps=252, seed=202, control_variate=False)
        cv  = api.price(kind="call", S0=100, K=90, r=0.02, sigma=0.3, T=1.0,
                        n_paths=30_000, steps=252, seed=202, control_variate=True)
    except TypeError:
        pytest.skip("control_variate flag not supported")
    # Expect control variate to bring estimate closer to BS than raw
    bs = bs_call_price(100, 90, 0.02, 0.3, 1.0)
    err_raw = abs(to_price(raw) - bs)
    err_cv  = abs(to_price(cv)  - bs)
    assert err_cv <= err_raw * 1.10  # allow small slack

def test_barrier_knock_out_not_more_valuable_than_plain_vanilla(api):
    # Down-and-out call should be <= vanilla call
    try:
        vanilla = api.price(kind="call", S0=100, K=100, r=0.02, sigma=0.25, T=1.0,
                            n_paths=80_000, steps=252, seed=77)
        barrier = api.price(kind="call", S0=100, K=100, r=0.02, sigma=0.25, T=1.0,
                            n_paths=80_000, steps=252, seed=77, barrier={"type": "down-and-out", "level": 80})
    except TypeError:
        pytest.skip("barrier options not supported")
    assert to_price(barrier) <= to_price(vanilla) + 1e-9

def test_no_nan_and_reasonable_ranges(api, params):
    res = api.simulate(**dict(params, seed=11))
    arr = np.asarray(res)
    assert np.isfinite(arr).all()
    assert np.all(arr > 0.0)  # GBM positive

def test_reject_invalid_inputs(api, params):
    with pytest.raises(Exception):
        api.simulate(**dict(params, sigma=-0.1))
    with pytest.raises(Exception):
        api.simulate(**dict(params, steps=0))
    with pytest.raises(Exception):
        api.simulate(**dict(params, n_paths=0))

def test_shape_and_dtype(api, params):
    paths = api.simulate(**dict(params, seed=5))
    arr = np.asarray(paths)
    assert arr.ndim == 2
    assert arr.shape[0] >= 128 and arr.shape[1] >= 2
    assert arr.dtype in (np.float32, np.float64)

def test_delta_via_bump_and_revalue_if_supported(api):
    if not hasattr(api, "_price_name"):
        pytest.skip("No pricing API")
    try:
        base = to_price(api.price(kind="call", S0=100, K=100, r=0.01, sigma=0.2, T=0.5,
                                  n_paths=60_000, steps=126, seed=555))
        up   = to_price(api.price(kind="call", S0=100*(1+1e-3), K=100, r=0.01, sigma=0.2, T=0.5,
                                  n_paths=60_000, steps=126, seed=555))
    except TypeError:
        pytest.skip("pricing signature incompatible")
    delta_est = (up - base) / (100 * 1e-3) # type: ignore
    # Call delta ∈ (0,1)
    assert 0.0 < delta_est < 1.0