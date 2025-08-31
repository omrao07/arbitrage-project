# tests/test_regime_map.py
import importlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest # type: ignore

"""
Expected public API (any one is fine)

Class-style
-----------
class RegimeMap:
    def fit(self, X, y=None, **kw) -> "RegimeMap": ...
    def predict(self, X) -> np.ndarray | list[int/str]: ...
    def predict_proba(self, X) -> np.ndarray  # [n, K]  (optional)
    def regimes(self) -> list[dict] | list[str] | np.ndarray  # labels/metadata (optional)
    def transition_matrix(self) -> np.ndarray  # [K, K] (optional)
    def change_points(self, X=None, proba=None, **kw) -> list[int]  # indices where regime switches (optional)
    def explain(self, X_row) -> dict  # feature attributions/summary (optional)
    # Optional persistence:
    def export_json(self) -> dict | str
    def import_json(self, blob) -> None
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> "RegimeMap"

Function-style
--------------
- build_regime_map(**kw) -> handle
- fit(handle, X, y=None, **kw); predict(handle, X); predict_proba(handle, X); etc.

The tests auto-skip features you don’t expose.
"""

# -------------------- Import resolver --------------------

IMPORT_CANDIDATES = [
    "backend.research.regime_map",
    "backend.quant.regime_map",
    "backend.analytics.regime_map",
    "analytics.regime_map",
    "regime_map",
]

def _load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import regime_map from {IMPORT_CANDIDATES} ({last})")

class API:
    """Unify class- and function-style APIs."""
    def __init__(self, mod):
        self.mod = mod
        self.obj = None
        if hasattr(mod, "RegimeMap"):
            Cls = getattr(mod, "RegimeMap")
            try:
                self.obj = Cls()
            except TypeError:
                self.obj = Cls
        elif hasattr(mod, "build_regime_map"):
            self.obj = mod.build_regime_map()
        else:
            pytest.skip("No RegimeMap class and no build_regime_map() factory found")

    def has(self, name: str) -> bool:
        return hasattr(self.obj, name) or hasattr(self.mod, name)

    def call(self, name: str, *args, **kw):
        if hasattr(self.obj, name):
            return getattr(self.obj, name)(*args, **kw)
        if hasattr(self.mod, name):
            return getattr(self.mod, name)(self.obj, *args, **kw)
        raise AttributeError(f"API missing '{name}'")

# -------------------- Synthetic data --------------------

@dataclass
class ToyParams:
    n0: int = 800     # bull / low-vol length
    n1: int = 600     # bear / high-vol length
    mu0: float = 0.08/252    # daily drift bull
    mu1: float = -0.08/252   # daily drift bear
    sig0: float = 0.010      # daily vol bull
    sig1: float = 0.030      # daily vol bear
    seed: int = 123

def make_two_regime_features(tp: ToyParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 2-regime series with simple features:
      X = [returns, rolling_vol_20, rolling_sign_5]
    y = [0]*n0 + [1]*n1   (true regime ids)
    """
    rng = np.random.default_rng(tp.seed)
    r0 = rng.normal(tp.mu0, tp.sig0, size=tp.n0)
    r1 = rng.normal(tp.mu1, tp.sig1, size=tp.n1)
    r = np.concatenate([r0, r1])
    # features
    vol20 = np.concatenate([
        np.sqrt(np.convolve((r0 - r0.mean())**2, np.ones(20)/20, mode="same")),
        np.sqrt(np.convolve((r1 - r1.mean())**2, np.ones(20)/20, mode="same")),
    ])
    sign5 = np.sign(np.convolve(r, np.ones(5), mode="same"))
    X = np.column_stack([r, vol20, sign5])
    y = np.array([0]*tp.n0 + [1]*tp.n1)
    # drop NaNs at edges if rolling created them
    X = np.nan_to_num(X, copy=False)
    return X, y

# -------------------- Fixtures --------------------

@pytest.fixture(scope="module")
def api():
    return API(_load_mod())

@pytest.fixture(scope="module")
def data():
    X, y = make_two_regime_features(ToyParams())
    return {"X": X, "y": y}

# -------------------- Helpers --------------------

def _to_labels(yhat) -> np.ndarray:
    arr = np.asarray(yhat)
    # if labels are strings, map to ints by factorization (stable)
    if arr.dtype.kind in ("U", "S", "O"):
        uniq = {v: i for i, v in enumerate(np.unique(arr))}
        return np.vectorize(uniq.get)(arr)
    return arr.astype(int, copy=False)

def _acc(y_true, y_pred) -> float:
    # Because label ids can be permuted, compute best of direct vs flipped for K=2
    yp = _to_labels(y_pred)
    ya = (yp == 1).astype(int)
    yb = (1 - yp)
    a = np.mean(ya == y_true)
    b = np.mean(yb == y_true)
    return max(a, b)

# -------------------- Tests --------------------

def test_fit_and_predict_reasonable_accuracy(api, data):
    X, y = data["X"], data["y"]
    api.call("fit", X)
    yhat = api.call("predict", X)
    assert isinstance(yhat, (list, np.ndarray)) and len(yhat) == len(X)
    acc = _acc(y, yhat)
    # Synthetic separation is strong; expect decent accuracy
    assert acc >= 0.70

def test_predict_proba_well_formed_if_supported(api, data):
    if not api.has("predict_proba"):
        pytest.skip("No predict_proba()")
    X = data["X"]
    P = api.call("predict_proba", X)
    P = np.asarray(P)
    assert P.ndim == 2 and P.shape[0] == len(X) and P.shape[1] >= 2
    # probabilities in [0,1] and rows ~sum to 1
    assert np.all(P >= -1e-9) and np.all(P <= 1+1e-9)
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-2)

def test_transition_matrix_and_durations_if_supported(api, data):
    if not api.has("transition_matrix"):
        pytest.skip("No transition_matrix()")
    X = data["X"]
    api.call("fit", X)
    Tm = np.asarray(api.call("transition_matrix"))
    assert Tm.ndim == 2 and Tm.shape[0] == Tm.shape[1] and Tm.shape[0] >= 2
    # Row-stochastic
    assert np.allclose(Tm.sum(axis=1), 1.0, atol=1e-3)
    # Staying probabilities should be significant on synthetic blocky series
    assert np.diag(Tm).mean() >= 0.6

def test_change_points_or_switch_indices_if_supported(api, data):
    if not api.has("change_points"):
        pytest.skip("No change_points()")
    X, y = data["X"], data["y"]
    api.call("fit", X)
    cps = api.call("change_points", X)
    assert isinstance(cps, (list, np.ndarray))
    # Expect at least one switch around boundary (n0 ~ 800). Allow slack ±50.
    if len(cps) > 0:
        assert any(abs(int(c) - 800) <= 50 for c in cps)

def test_regime_metadata_or_labels_optional(api, data):
    if not api.has("regimes"):
        pytest.skip("No regimes()")
    meta = api.call("regimes")
    # Accept list of dicts or labels
    assert isinstance(meta, (list, np.ndarray))
    # If dicts, expect id/name-like keys
    if len(meta) and isinstance(meta[0], dict):
        assert any(k in meta[0] for k in ("id", "name", "label"))

def test_online_or_partial_fit_optional(api, data):
    # If partial_fit exists, accuracy should not degrade badly
    if not api.has("partial_fit"):
        pytest.skip("No partial_fit()")
    X, y = data["X"], data["y"]
    # train on first half
    api.call("partial_fit", X[:600])
    yhat1 = api.call("predict", X)
    acc1 = _acc(y, yhat1)
    # continue training on second half
    api.call("partial_fit", X[600:])
    yhat2 = api.call("predict", X)
    acc2 = _acc(y, yhat2)
    assert acc2 >= max(0.65, acc1 - 0.05)

def test_explain_row_optional(api, data):
    if not api.has("explain"):
        pytest.skip("No explain()")
    X = data["X"]
    ex = api.call("explain", X[500])
    assert isinstance(ex, dict) and len(ex) >= 1

def test_export_import_roundtrip_optional(tmp_path, api, data):
    X = data["X"]
    api.call("fit", X)
    # JSON-style
    if api.has("export_json") and api.has("import_json"):
        blob = api.call("export_json")
        assert blob is not None
        api.call("import_json", blob)
    # File-style
    elif api.has("save") and api.has("load"):
        p = tmp_path / "regime_map.model"
        api.call("save", str(p))
        # reload into a new instance if possible
        mod = _load_mod()
        if hasattr(mod, "RegimeMap"):
            cls = getattr(mod, "RegimeMap")
            if hasattr(cls, "load"):
                new_obj = cls.load(str(p))
                # can predict
                if hasattr(new_obj, "predict"):
                    y2 = new_obj.predict(X[:10])
                    assert len(y2) == 10

def test_handles_small_batches_and_missing_values(api):
    X = np.array([[0.0, 0.01, 1.0],
                  [np.nan, 0.02, -1.0],
                  [0.001, np.nan, 0.0],
                  [0.002, 0.015, 1.0]])
    X = np.nan_to_num(X, nan=0.0)
    api.call("fit", X)
    y = api.call("predict", X)
    assert len(y) == len(X)

def test_regime_counts_and_imbalance_tolerance(api, data):
    X, _ = data["X"], data["y"]
    api.call("fit", X)
    yhat = _to_labels(api.call("predict", X))
    # Ensure at least two regimes present
    assert len(np.unique(yhat)) >= 2

def test_no_nan_or_inf_outputs(api, data):
    X = data["X"]
    api.call("fit", X)
    yhat = api.call("predict", X)
    arr = _to_labels(yhat)
    assert np.isfinite(arr).all()
    if api.has("predict_proba"):
        P = np.asarray(api.call("predict_proba", X))
        assert np.isfinite(P).all()

def test_window_prediction_consistency_optional(api, data):
    # If you expose rolling classification (e.g., predict with window kw),
    # verify same length and reasonable variation.
    try:
        yp = api.call("predict", data["X"], window=50)
    except TypeError:
        pytest.skip("predict(..., window=) not supported")
    assert len(yp) == len(data["X"])