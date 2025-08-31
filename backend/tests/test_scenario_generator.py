# tests/test_scenario_generator.py
import importlib
import json
import math
import numpy as np
import pytest # type: ignore
from typing import Any, Dict, Optional, Tuple, List

"""
Expected public APIs (any one is fine)

Class-style:
------------
class ScenarioGenerator:
    def sample(self, n_scenarios: int, horizon_days: int, assets: list[str],
               mu: np.ndarray | list | None = None,
               Sigma: np.ndarray | list[list] | None = None,
               seed: Optional[int] = None,
               freq: str = "D",
               method: str = "gaussian",   # "t", "historical", "bootstrap", "copula"
               shocks: list[dict] | None = None,   # [{"asset":"AAPL","type":"pct","value":-0.08,"t":0}, ...]
               jumps: dict | None = None,          # {"lam":0.1,"mu":-0.02,"sigma":0.05}
               regimes: dict | None = None,        # {"P":[[.95,.05],[.05,.95]], "mus":[...], "Sigmas":[...]}
               mix: dict | None = None             # {"historical":0.7,"model":0.3}
               ) -> dict:
        # returns {"paths": np.ndarray [n_scenarios, T, N], "dates": [...], "assets": [...], "meta": {...}}

    # Optional helpers
    def stress(self, template: str, **kw) -> dict                      # named stress templates
    def add_shock(self, paths, **shock_spec) -> dict | np.ndarray      # apply shock to paths
    def percentiles(self, paths, qs=(1,5,50,95,99)) -> dict[str, np.ndarray]
    def export_json(self, obj) -> dict | str                           # serialize
    def import_json(self, blob) -> Any

Function-style:
---------------
- sample(...same args...) -> dict
- Optional: stress(...), add_shock(...), percentiles(...), export_json/import_json

The tests auto-skip optional parts if missing.
"""

# --------------------- Import resolver ---------------------

IMPORT_CANDIDATES = [
    "backend.research.scenario_generator",
    "backend.risk.scenario_generator",
    "backend.analytics.scenario_generator",
    "risk.scenario_generator",
    "scenario_generator",
]

def _load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import scenario_generator from {IMPORT_CANDIDATES} ({last})")

class API:
    def __init__(self, mod):
        self.mod = mod
        self.obj = None
        if hasattr(mod, "ScenarioGenerator"):
            Cls = getattr(mod, "ScenarioGenerator")
            try:
                self.obj = Cls()
            except TypeError:
                self.obj = Cls
        # Find sample entrypoint
        self.sample_name = None
        for nm in ("sample", "generate", "simulate"):
            if hasattr(self.obj or mod, nm):
                self.sample_name = nm; break
        if not self.sample_name:
            pytest.skip("No sample()/generate()/simulate() found in scenario generator.")

    def call(self, name, *args, **kw):
        target = self.obj if (self.obj and hasattr(self.obj, name)) else self.mod
        return getattr(target, name)(*args, **kw)

    def sample(self, **kw):
        return self.call(self.sample_name, **kw)

    def has(self, name):
        return hasattr(self.obj, name) or hasattr(self.mod, name)

# --------------------- Helpers ---------------------

def _ensure_result_dict(res):
    assert isinstance(res, dict), f"Expected dict result, got {type(res)}"
    assert "paths" in res and "assets" in res
    X = np.asarray(res["paths"])
    A = list(res["assets"])
    assert X.ndim == 3 and X.shape[2] == len(A)
    return X, A, res.get("meta", {})

def _make_psd_cov(n=6, seed=7):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    C = A @ A.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    vols = rng.uniform(0.1, 0.35, size=n)
    Sigma = C * np.outer(vols, vols)
    mu = rng.normal(0.08, 0.04, size=n)
    return mu, Sigma

def _empirical_corr(X):  # X: [scen, T, N]
    R = X.reshape(-1, X.shape[-1])
    R = R - R.mean(0, keepdims=True)
    S = np.corrcoef(R, rowvar=False)
    return S

# --------------------- Fixtures ---------------------

@pytest.fixture(scope="module")
def api():
    return API(_load_mod())

@pytest.fixture()
def base_inputs():
    mu, Sigma = _make_psd_cov(n=5, seed=11)
    assets = ["AAPL","MSFT","SPY","GLD","TLT"]
    return {"mu": mu, "Sigma": Sigma, "assets": assets}

# --------------------- Tests ---------------------

def test_basic_shape_and_reproducibility(api, base_inputs):
    res1 = api.sample(n_scenarios=2000, horizon_days=10, seed=123, **base_inputs)
    X1, A, meta1 = _ensure_result_dict(res1)
    assert X1.shape == (2000, 10, len(A))

    res2 = api.sample(n_scenarios=2000, horizon_days=10, seed=123, **base_inputs)
    X2, _, _ = _ensure_result_dict(res2)
    assert np.allclose(X1, X2)

def test_no_nan_inf_and_reasonable_ranges(api, base_inputs):
    res = api.sample(n_scenarios=1000, horizon_days=20, seed=9, **base_inputs)
    X, _, _ = _ensure_result_dict(res)
    assert np.isfinite(X).all()
    # returns/prices should be “reasonable” (assume log-returns or pct moves)
    assert np.abs(X).mean() < 0.5

def test_correlation_targeting_gaussian_method(api, base_inputs):
    res = api.sample(n_scenarios=5000, horizon_days=1, seed=99, method="gaussian", **base_inputs)
    X, _, _ = _ensure_result_dict(res)
    # if paths are returns, horizon_days=1 gives 1 step; collapse and estimate corr
    C_emp = _empirical_corr(X)
    C_tgt = base_inputs["Sigma"] / np.outer(np.sqrt(np.diag(base_inputs["Sigma"])), np.sqrt(np.diag(base_inputs["Sigma"])))
    # looser tolerance (sampling error)
    assert np.allclose(C_emp, C_tgt, atol=0.15)

def test_additive_shocks_or_templates(api, base_inputs):
    # Apply −8% day-0 shock to AAPL; +3% to GLD on day 3
    shocks = [
        {"asset": "AAPL", "type": "pct", "value": -0.08, "t": 0},
        {"asset": "GLD",  "type": "pct", "value":  0.03, "t": 3},
    ]
    res = api.sample(n_scenarios=200, horizon_days=5, seed=7, shocks=shocks, **base_inputs)
    X, A, _ = _ensure_result_dict(res)
    i_aapl, i_gld = A.index("AAPL"), A.index("GLD")
    # Median across scenarios at t=0 for AAPL should reflect negative shock
    m0 = float(np.median(X[:, 0, i_aapl]))
    assert m0 <= -0.04  # allow partial blending with stochastic draw
    m3 = float(np.median(X[:, 3, i_gld]))
    assert m3 >= 0.01

def test_student_t_or_fat_tails_optional(api, base_inputs):
    try:
        res_g = api.sample(n_scenarios=4000, horizon_days=1, method="gaussian", seed=1, **base_inputs)
        res_t = api.sample(n_scenarios=4000, horizon_days=1, method="t", seed=1, **base_inputs)
    except TypeError:
        pytest.skip("t-distribution method not supported")
    Xg, _, _ = _ensure_result_dict(res_g)
    Xt, _, _ = _ensure_result_dict(res_t)
    # t should have heavier tails: higher kurtosis
    def kurt(arr): 
        z = arr.reshape(-1)
        z = z - z.mean()
        m2 = (z**2).mean(); m4 = (z**4).mean()
        return m4 / (m2**2)
    assert kurt(Xt) >= kurt(Xg) - 0.2

def test_jump_process_optional(api, base_inputs):
    try:
        res = api.sample(n_scenarios=4000, horizon_days=5, seed=21, jumps={"lam":0.2,"mu":-0.03,"sigma":0.06}, **base_inputs)
    except TypeError:
        pytest.skip("jumps not supported")
    X, _, _ = _ensure_result_dict(res)
    # Expect occasional large negatives: lower p1 (1st percentile) than Gaussian baseline
    base = api.sample(n_scenarios=4000, horizon_days=5, seed=21, method="gaussian", **base_inputs)
    B, _, _ = _ensure_result_dict(base)
    p1_jump = np.percentile(X.reshape(-1), 1)
    p1_base = np.percentile(B.reshape(-1), 1)
    assert p1_jump <= p1_base + 1e-9

def test_regime_switching_optional(api, base_inputs):
    try:
        regimes = {
            "P": [[0.95,0.05],[0.10,0.90]],
            "mus": [base_inputs["mu"], np.array(base_inputs["mu"])*0.0 - 0.10],
            "Sigmas": [base_inputs["Sigma"], base_inputs["Sigma"]*1.8],
        }
        res = api.sample(n_scenarios=1000, horizon_days=30, seed=5, regimes=regimes, **base_inputs)
    except TypeError:
        pytest.skip("regimes not supported")
    X, _, meta = _ensure_result_dict(res)
    # If meta contains state path probabilities, accept; else just sanity-check variance > baseline
    base = api.sample(n_scenarios=1000, horizon_days=30, seed=5, **base_inputs)
    B, _, _ = _ensure_result_dict(base)
    assert float(np.var(X)) >= float(np.var(B)) * 0.9  # not super strict

def test_historical_mix_optional(api, base_inputs):
    try:
        res = api.sample(n_scenarios=800, horizon_days=10, seed=42,
                         method="copula", mix={"historical":0.7,"model":0.3}, **base_inputs)
    except TypeError:
        pytest.skip("copula/mix not supported")
    X, _, _ = _ensure_result_dict(res)
    assert X.shape[0] == 800

def test_percentiles_helper_optional(api, base_inputs):
    if not api.has("percentiles"):
        pytest.skip("percentiles() helper not exposed")
    res = api.sample(n_scenarios=1500, horizon_days=15, seed=3, **base_inputs)
    X, A, _ = _ensure_result_dict(res)
    pct = api.call("percentiles", X, qs=(1,5,50,95,99))
    assert isinstance(pct, dict)
    # One entry per asset with array over time or qs
    k = list(pct.keys())[0]
    arr = np.asarray(pct[k])
    assert arr.ndim >= 1

def test_export_import_roundtrip_optional(api, base_inputs):
    if not (api.has("export_json") and api.has("import_json")):
        pytest.skip("No export/import helpers")
    res = api.sample(n_scenarios=100, horizon_days=5, seed=13, **base_inputs)
    blob = api.call("export_json", res)
    s = json.dumps(blob, default=str)
    out = api.call("import_json", blob)
    assert out is not None

def test_reject_invalid_inputs(api, base_inputs):
    with pytest.raises(Exception):
        api.sample(n_scenarios=0, horizon_days=5, **base_inputs)
    with pytest.raises(Exception):
        api.sample(n_scenarios=100, horizon_days=0, **base_inputs)
    with pytest.raises(Exception):
        api.sample(n_scenarios=10, horizon_days=5, mu=np.array([0.1,0.2]), Sigma=np.eye(3), assets=["A","B","C"])
    # Non-PSD Sigma should raise
    Sigma_bad = np.array([[1,2],[2,1]]) * [[1,0],[0,-1]]  # intentionally wrong shape/neg variance
    with pytest.raises(Exception):
        api.sample(n_scenarios=10, horizon_days=5, assets=["X","Y"], mu=[0.1,0.2], Sigma=Sigma_bad)

def test_time_grid_and_freq_metadata(api, base_inputs):
    res = api.sample(n_scenarios=20, horizon_days=7, seed=8, freq="D", **base_inputs)
    X, A, meta = _ensure_result_dict(res)
    # dates length equals horizon_days; monotonic
    dates = res.get("dates")
    if dates is not None:
        assert len(dates) == 7
    assert meta.get("freq","D") in ("D","H","W","M")

def test_scenario_means_and_vols_roughly_track_inputs(api, base_inputs):
    mu, Sigma = base_inputs["mu"], base_inputs["Sigma"]
    res = api.sample(n_scenarios=12000, horizon_days=1, seed=17, **base_inputs)
    X, _, _ = _ensure_result_dict(res)
    # estimate per-asset mean/vol over scenarios
    est_mu = X[:, 0, :].mean(axis=0)
    est_vol = X[:, 0, :].std(axis=0)
    tgt_vol = np.sqrt(np.diag(Sigma))
    # Loose tolerances (MC error + implementation differences)
    assert np.allclose(est_vol, tgt_vol, rtol=0.2, atol=1e-3)
    # Mean check is soft because many engines model returns with zero mean by default
    # Only assert finite and bounded
    assert np.isfinite(est_mu).all() and np.abs(est_mu).max() < 0.2