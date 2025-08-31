# tests/test_optimizer.py
import importlib
import math
import numpy as np
import pytest # type: ignore
from typing import Any, Dict, Optional, Tuple, List

"""
Expected public APIs (any one is fine)

Class-style:
------------
class Optimizer:
    def optimize(self, mu, Sigma, *, risk_aversion=None, target_vol=None, target_return=None,
                 bounds=None, budget=1.0, leverage=None, costs=None, prev_w=None, turnover_lim=None,
                 sector=None, sector_bounds=None, l2_reg=None, l1_reg=None, long_only=None, cards=None, **kw) -> dict:
        # returns {"w": np.ndarray, "ret": float, "vol": float, "sharpe": float, ...}

Function-style:
---------------
def optimize(...same args...) -> dict
# (common alternates: solve(), optimize_portfolio(), mean_variance(), risk_parity(), hrp())

Optional extras (auto-skipped if missing):
- risk_parity(mu, Sigma, bounds=None, **kw) -> {"w": ...}
- hrp(returns | Sigma, **kw) -> {"w": ...}
- frontier(mu, Sigma, n=50, **kw) -> [{"ret":...,"vol":...,"w":...}, ...]

This test is tolerant to:
- different return key names (w/weights, ret/return, vol/risk)
- numpy arrays or Python lists
"""

# ------------------------ Import resolver ------------------------

IMPORT_CANDIDATES = [
    "backend.quant.optimizer",
    "backend.research.optimizer",
    "backend.risk.optimizer",
    "quant.optimizer",
    "optimizer",
]

def _load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import optimizer from {IMPORT_CANDIDATES} ({last})")

class API:
    def __init__(self, mod):
        self.mod = mod
        self.obj = None
        if hasattr(mod, "Optimizer"):
            Cls = getattr(mod, "Optimizer")
            try:
                self.obj = Cls()
            except TypeError:
                self.obj = Cls
        # Find primary optimize entrypoint
        self.opt_name = None
        for nm in ("optimize", "solve", "optimize_portfolio", "mean_variance"):
            if hasattr(self.obj or mod, nm):
                self.opt_name = nm; break
        if not self.opt_name:
            pytest.skip("No optimize()/solve()/optimize_portfolio()/mean_variance() found.")
        # Optional extras
        self.rp_name = "risk_parity" if hasattr(self.obj or mod, "risk_parity") else None
        self.hrp_name = "hrp" if hasattr(self.obj or mod, "hrp") else None
        self.frontier_name = "frontier" if hasattr(self.obj or mod, "frontier") else None

    def call(self, name, *args, **kw):
        target = self.obj if (self.obj and hasattr(self.obj, name)) else self.mod
        return getattr(target, name)(*args, **kw)

    def optimize(self, **kw):
        return self.call(self.opt_name, **kw)

    def risk_parity(self, **kw):
        if not self.rp_name: pytest.skip("No risk_parity() API")
        return self.call(self.rp_name, **kw)

    def hrp(self, **kw):
        if not self.hrp_name: pytest.skip("No hrp() API")
        return self.call(self.hrp_name, **kw)

    def frontier(self, **kw):
        if not self.frontier_name: pytest.skip("No frontier() API")
        return self.call(self.frontier_name, **kw)

# ------------------------ Helpers ------------------------

def _result_w(res) -> np.ndarray:
    if isinstance(res, dict):
        for k in ("w", "weights", "x"):
            if k in res:
                return np.asarray(res[k], dtype=float)
    if isinstance(res, (list, tuple, np.ndarray)):
        return np.asarray(res, dtype=float)
    raise AssertionError("Cannot find weights in optimize() result")

def _result_ret(res) -> Optional[float]:
    if isinstance(res, dict):
        for k in ("ret","return","mean","mu"):
            if k in res: return float(res[k])
    return None

def _result_vol(res) -> Optional[float]:
    if isinstance(res, dict):
        for k in ("vol","risk","stdev","sigma"):
            if k in res: return float(res[k])
    return None

def _is_psd(M: np.ndarray) -> bool:
    eig = np.linalg.eigvalsh((M+M.T)/2)
    return np.all(eig >= -1e-8) # type: ignore

def _make_pd_cov(n=8, seed=7) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T
    # scale to reasonable vols/corrs
    d = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(d, d)
    vols = rng.uniform(0.1, 0.35, size=n)
    Sigma = Sigma * np.outer(vols, vols)
    mu = rng.normal(0.08, 0.06, size=n)
    return mu, Sigma

def _sharpe(mu, Sigma, w, rf=0.0):
    m = float(np.dot(w, mu))
    v = float(np.sqrt(max(1e-12, w @ Sigma @ w)))
    return (m - rf) / v

# ------------------------ Fixtures ------------------------

@pytest.fixture(scope="module")
def api():
    return API(_load_mod())

@pytest.fixture()
def data():
    mu, Sigma = _make_pd_cov(n=10, seed=11)
    assert _is_psd(Sigma)
    return {"mu": mu, "Sigma": Sigma}

# ------------------------ Tests ------------------------

def test_basic_mean_variance_long_only(api, data):
    mu, Sigma = data["mu"], data["Sigma"]
    res = api.optimize(mu=mu, Sigma=Sigma, risk_aversion=3.0, long_only=True, bounds=(0.0, 0.2))
    w = _result_w(res)
    assert w.shape == (len(mu),)
    # budget
    assert pytest.approx(1.0, abs=1e-6) == float(np.sum(w))
    # bounds & non-negativity
    assert np.all(w >= -1e-8) and np.all(w <= 0.2000001)
    # finite Sharpe
    S = _sharpe(mu, Sigma, w)
    assert math.isfinite(S)

def test_unconstrained_solution_matches_closed_form(api, data):
    """
    If optimizer honors equality budget and no bounds/regularizers, the solution
    should match the classic MVO closed-form up to scaling by lambda and γ.
    We check by solving the KKT system explicitly and comparing direction.
    """
    mu, Sigma = data["mu"], data["Sigma"]
    lam = 5.0  # risk_aversion
    # ask optimizer with no bounds/long_only
    res = api.optimize(mu=mu, Sigma=Sigma, risk_aversion=lam, long_only=False, bounds=None)
    w = _result_w(res)
    # Closed-form (budget=1): solve [ 2λΣ  1 ][w] = [μ] up to affine; we project to sum(w)=1.
    # We'll compute the maximum Sharpe (tangency) and then scale to budget.
    inv = np.linalg.inv(Sigma)
    one = np.ones_like(mu)
    A = one @ inv @ one
    B = one @ inv @ mu
    C = mu  @ inv @ mu
    w_tan = (inv @ (mu - (B/A)*one))
    w_cf = w_tan / np.sum(w_tan)  # budget 1
    # Compare direction (cosine similarity close to 1)
    cos = float(np.dot(w, w_cf) / (np.linalg.norm(w) * np.linalg.norm(w_cf)))
    assert cos >= 0.98

def test_target_vol_or_return_if_supported(api, data):
    mu, Sigma = data["mu"], data["Sigma"]
    try:
        res = api.optimize(mu=mu, Sigma=Sigma, target_vol=0.12, long_only=True, bounds=(0.0, 0.3))
    except TypeError:
        pytest.skip("target_vol not supported")
    w = _result_w(res)
    vol = _result_vol(res)
    assert np.isfinite(w).all()
    if vol is not None:
        assert vol <= 0.14  # allow slack
        assert vol >= 0.05

def test_transaction_costs_and_turnover_limit_if_supported(api, data):
    mu, Sigma = data["mu"], data["Sigma"]
    prev_w = np.full(len(mu), 1.0/len(mu))
    try:
        res = api.optimize(mu=mu, Sigma=Sigma, long_only=True, bounds=(0.0, 0.3),
                           prev_w=prev_w, costs=0.0005, turnover_lim=0.25)
    except TypeError:
        pytest.skip("prev_w/costs/turnover_lim not supported")
    w = _result_w(res)
    # turnover <= 25%
    t = 0.5 * np.sum(np.abs(w - prev_w))
    assert t <= 0.2500001

def test_sector_constraints_if_supported(api, data):
    mu, Sigma = data["mu"], data["Sigma"]
    n = len(mu)
    # 3 sectors
    sector = np.array([0]*3 + [1]*3 + [2]*(n-6))
    bounds = {0: (0.0, 0.40), 1: (0.10, 0.35), 2: (0.20, 0.60)}
    try:
        res = api.optimize(mu=mu, Sigma=Sigma, long_only=True, bounds=(0.0, 0.4),
                           sector=sector, sector_bounds=bounds)
    except TypeError:
        pytest.skip("sector/sector_bounds not supported")
    w = _result_w(res)
    for g, (lo, hi) in bounds.items():
        wg = float(np.sum(w[sector == g]))
        assert lo - 1e-6 <= wg <= hi + 1e-6

def test_l2_regularization_shrinks_extremes_if_supported(api, data):
    mu, Sigma = data["mu"], data["Sigma"]
    try:
        res_base = api.optimize(mu=mu, Sigma=Sigma, long_only=True, bounds=(0.0, 0.5))
        res_l2   = api.optimize(mu=mu, Sigma=Sigma, long_only=True, bounds=(0.0, 0.5), l2_reg=10.0)
    except TypeError:
        pytest.skip("l2_reg not supported")
    w0 = _result_w(res_base); w1 = _result_w(res_l2)
    # L2 should reduce weight concentration (lower L2 norm)
    assert np.linalg.norm(w1, 2) < np.linalg.norm(w0, 2) + 1e-9

def test_leverage_and_bounds(api, data):
    mu, Sigma = data["mu"], data["Sigma"]
    # allow long/short with leverage cap
    res = api.optimize(mu=mu, Sigma=Sigma, long_only=False, bounds=(-0.2, 0.4), leverage=1.5)
    w = _result_w(res)
    assert np.all(w <= 0.4000001) and np.all(w >= -0.2000001)
    assert np.sum(np.abs(w)) <= 1.5000001
    assert pytest.approx(1.0, abs=1e-6) == float(np.sum(w))

def test_better_sharpe_than_equal_weight(api, data):
    mu, Sigma = data["mu"], data["Sigma"]
    w_eq = np.full(len(mu), 1.0/len(mu))
    S_eq = _sharpe(mu, Sigma, w_eq)
    res = api.optimize(mu=mu, Sigma=Sigma, risk_aversion=3.0, long_only=True, bounds=(0.0, 0.3))
    w = _result_w(res)
    S = _sharpe(mu, Sigma, w)
    # Optimizer should not be worse than naive EW by more than tiny slack
    assert S >= S_eq - 1e-6

def test_risk_parity_optional(api, data):
    if not api.rp_name:
        pytest.skip("risk_parity() not exposed")
    mu, Sigma = data["mu"], data["Sigma"]
    res = api.risk_parity(Sigma=Sigma, bounds=(0.0, 0.4))
    w = _result_w(res)
    assert w.shape == (len(mu),)
    # Risk contributions roughly equal
    Sigma = np.asarray(Sigma)
    rc = w * (Sigma @ w)
    assert np.max(rc) - np.min(rc) <= 0.10 * np.mean(rc) + 1e-6

def test_hrp_optional(api, data):
    if not api.hrp_name:
        pytest.skip("hrp() not exposed")
    mu, Sigma = data["mu"], data["Sigma"]
    res = api.hrp(Sigma=Sigma)
    w = _result_w(res)
    assert w.shape == (len(mu),)
    assert np.all(w >= -1e-8)
    assert pytest.approx(1.0, abs=1e-6) == float(np.sum(w))

def test_efficient_frontier_monotonic_if_supported(api, data):
    if not api.frontier_name:
        pytest.skip("frontier() not exposed")
    mu, Sigma = data["mu"], data["Sigma"]
    pts = api.frontier(mu=mu, Sigma=Sigma, n=10)
    assert isinstance(pts, (list, tuple)) and len(pts) >= 3
    vols = [float(p.get("vol", p.get("risk"))) for p in pts]
    rets = [float(p.get("ret", p.get("return"))) for p in pts]
    # Frontier should be (weakly) increasing in vol
    assert all(vols[i] <= vols[i+1] + 1e-9 for i in range(len(vols)-1))
    # And returns should generally increase with vol (not strict, but trend)
    assert rets[-1] >= rets[0] - 1e-6

def test_reject_bad_inputs(api, data):
    mu, Sigma = data["mu"], data["Sigma"]
    with pytest.raises(Exception):
        api.optimize(mu=mu, Sigma=Sigma*0, long_only=True)
    with pytest.raises(Exception):
        api.optimize(mu=np.ones_like(mu), Sigma=Sigma, bounds=(0.6, 0.7), long_only=True)  # infeasible budget
    with pytest.raises(Exception):
        api.optimize(mu=mu, Sigma=Sigma, leverage=0.5)  # too tight leverage with budget=1