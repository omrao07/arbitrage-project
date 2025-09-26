# ai_brain/regime_oracle.py
"""
RegimeOracle: real-time market regime classification

Features
--------
• Unsupervised regime discovery with Gaussian Mixture (diagonal covariances).
• Sticky HMM-style filtering using an estimated transition matrix (A) to
  smooth per-step probabilities and avoid regime flapping.
• Auto-labeling of regimes: {"calm", "trend", "panic"} based on centroid
  volatility & return signs (you still get numeric regime ids).
• Allocator-ready signals: risk_multiplier (0..1.5+), hedge_bias hints.
• End-to-end pipeline: build_features() from price series, fit/predict,
  describe(), save()/load(), and from_time_series() convenience.

Dependencies
------------
numpy, pandas, scikit-learn
(Install scikit-learn if missing: `pip install scikit-learn`)

Typical usage
-------------
from ai_brain.regime_oracle import RegimeOracle, build_features

# 1) Build features from a price series (close prices) and optional extra cols.
X, meta = build_features(prices_df)  # returns standardized feature matrix (pd.DataFrame)

# 2) Fit oracle
oracle = RegimeOracle(n_regimes=3, sticky=0.95).fit(X)

# 3) Predict latest regime + probs
proba = oracle.predict_proba(X.iloc[-1:])
regime, label = oracle.predict_label(X.iloc[-1:])
sig = oracle.regime_signal(proba.iloc[-1].to_dict())  # {'risk_multiplier': ..., 'hedge_bias': ...}

# 4) Persist
oracle.save(".models/regime_oracle.joblib")
# ...later...
oracle = RegimeOracle.load(".models/regime_oracle.joblib")
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture # type: ignore
    from sklearn.preprocessing import StandardScaler # type: ignore
    from sklearn.pipeline import Pipeline # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "regime_oracle requires scikit-learn. Install with: pip install scikit-learn"
    ) from e

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # we handle save/load gracefully


# ---------- Feature engineering ----------

_DEF_RET_WIN = 5
_DEF_VOL_WIN = 20
_DEF_MOM_WIN = 60


def _rolling_kurt(x: pd.Series, w: int) -> pd.Series:
    # Fisher kurtosis with small-sample correction (approx)
    def f(s):
        if len(s) < 4:
            return np.nan
        m = s.mean()
        v = s.var(ddof=1)
        if v <= 0:
            return 0.0
        z = ((s - m) ** 4).mean() / (v**2)
        return float(z - 3.0)
    return x.rolling(w, min_periods=max(4, w // 2)).apply(f, raw=False)


def _rolling_skew(x: pd.Series, w: int) -> pd.Series:
    def f(s):
        if len(s) < 3:
            return np.nan
        m = s.mean()
        v = s.std(ddof=1)
        if v <= 0:
            return 0.0
        z = ((s - m) ** 3).mean() / (v**3)
        return float(z)
    return x.rolling(w, min_periods=max(3, w // 2)).apply(f, raw=False)


def build_features(
    prices: pd.DataFrame,
    *,
    col: str = "close",
    ret_win: int = _DEF_RET_WIN,
    vol_win: int = _DEF_VOL_WIN,
    mom_win: int = _DEF_MOM_WIN,
    extra: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Build regime features from a price series (per-asset or index).
    Required columns in `prices`: at least a 'close' (default) or pass `col`.

    Returns:
        X (pd.DataFrame): standardized numerical features ready for the oracle.
        meta (dict): metadata such as window params.
    """
    if col not in prices.columns:
        raise ValueError(f"prices must contain column '{col}'")
    s = prices[col].astype(float).dropna()

    ret1 = s.pct_change().rename("ret_1")
    ret_r = s.pct_change(ret_win).rename(f"ret_{ret_win}")
    vola = ret1.rolling(vol_win, min_periods=max(5, vol_win // 2)).std(ddof=1).rename(f"vol_{vol_win}")
    mom = s.pct_change(mom_win).rename(f"mom_{mom_win}")
    dd = (s / s.cummax() - 1.0).rename("drawdown")
    skew = _rolling_skew(ret1, vol_win).rename("skew")
    kurt = _rolling_kurt(ret1, vol_win).rename("kurt")

    feats = pd.concat([ret1, ret_r, vola, mom, dd, skew, kurt], axis=1)

    if extra is not None:
        # include extra columns (e.g., VIX, MOVE, credit spreads) if provided
        feats = pd.concat([feats, extra.reindex(feats.index)], axis=1)

    X = feats.dropna().copy()
    meta = {"ret_win": ret_win, "vol_win": vol_win, "mom_win": mom_win}
    return X, meta


# ---------- Core oracle ----------

@dataclass
class OracleState:
    n_regimes: int
    sticky: float
    feature_names: List[str]
    # learned artifacts
    scaler: Optional[StandardScaler] = None
    gmm_means: Optional[np.ndarray] = None
    gmm_covars: Optional[np.ndarray] = None
    gmm_weights: Optional[np.ndarray] = None
    trans_matrix: Optional[np.ndarray] = None  # regime transition matrix (A)
    labels_map: Optional[Dict[int, str]] = None  # id -> {"calm","trend","panic"}

    def to_json(self) -> str:
        obj = asdict(self)
        # numpy arrays -> lists
        for k in ("gmm_means", "gmm_covars", "gmm_weights", "trans_matrix"):
            if obj[k] is not None:
                obj[k] = np.asarray(obj[k]).tolist()
        return json.dumps(obj)

    @staticmethod
    def from_json(s: str) -> "OracleState":
        obj = json.loads(s)
        for k in ("gmm_means", "gmm_covars", "gmm_weights", "trans_matrix"):
            if obj.get(k) is not None:
                obj[k] = np.asarray(obj[k], dtype=float)
        st = OracleState(
            n_regimes=obj["n_regimes"],
            sticky=obj["sticky"],
            feature_names=list(obj["feature_names"]),
            scaler=None,  # restored separately
            gmm_means=obj.get("gmm_means"),
            gmm_covars=obj.get("gmm_covars"),
            gmm_weights=obj.get("gmm_weights"),
            trans_matrix=obj.get("trans_matrix"),
            labels_map=obj.get("labels_map"),
        )
        return st


class RegimeOracle:
    """
    Unsupervised regime classifier with sticky smoothing.

    Parameters
    ----------
    n_regimes : int, default=3
        Number of regimes to learn.
    sticky : float, default=0.95
        Persistence prior (0..1). Higher values reduce chattering.
    random_state : int, optional
    """

    def __init__(self, n_regimes: int = 3, sticky: float = 0.95, random_state: Optional[int] = 42):
        assert 2 <= n_regimes <= 8, "n_regimes should be between 2 and 8"
        assert 0.0 <= sticky <= 0.999, "sticky must be in [0, 0.999]"
        self.n_regimes = n_regimes
        self.sticky = sticky
        self.random_state = random_state

        self._pipe: Optional[Pipeline] = None
        self._gmm: Optional[GaussianMixture] = None
        self._scaler: Optional[StandardScaler] = None
        self._A: Optional[np.ndarray] = None
        self._labels_map: Dict[int, str] = {}

        self._features: List[str] = []

    # ------------ fitting & artifacts ------------
    def fit(self, X: pd.DataFrame) -> "RegimeOracle":
        self._features = list(X.columns)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.values)

        gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="diag",
            random_state=self.random_state,
            n_init=5,
            reg_covar=1e-6,
        ).fit(Xs)

        self._gmm = gmm
        self._A = self._estimate_transitions(Xs, gmm)
        self._labels_map = self._derive_labels(X, gmm)

        return self

    # Estimate transition matrix A via responsibilities argmax path
    def _estimate_transitions(self, Xs: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
        z = gmm.predict(Xs)
        K = gmm.n_components # type: ignore
        A = np.ones((K, K), dtype=float) * (1 - self.sticky) / (K - 1)
        # count transitions
        for i in range(1, len(z)):
            A[z[i - 1], z[i]] += 1.0
        # row-normalize with persistence bias
        A = A / A.sum(axis=1, keepdims=True)
        # blend with sticky prior
        A = self.sticky * np.eye(K) + (1 - self.sticky) * A
        # normalize again
        A = A / A.sum(axis=1, keepdims=True)
        return A

    # Auto label regimes by centroid characteristics
    def _derive_labels(self, X: pd.DataFrame, gmm: GaussianMixture) -> Dict[int, str]:
        # compute centroid mean/var in original feature space approximation
        # use inverse scaling on gmm means (rough heuristic)
        means = gmm.means_  # in standardized space
        if self._scaler is not None:
            means = means * self._scaler.scale_ + self._scaler.mean_ # type: ignore

        # locate column indices
        def col_idx(name: str) -> Optional[int]:
            try:
                return self._features.index(name)
            except Exception:
                return None

        i_vol = col_idx(f"vol_{_DEF_VOL_WIN}") or col_idx("vol")
        i_ret = col_idx("ret_1")

        # rank by volatility first
        vols = np.abs(means[:, i_vol]) if i_vol is not None else np.linalg.norm(means, axis=1) # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
        order = np.argsort(vols)  # low vol -> high vol

        labels: Dict[int, str] = {}
        if i_ret is None:
            # fallback: low vol = calm, mid = trend, high = panic
            if len(order) == 2:
                labels[order[0]] = "calm"
                labels[order[1]] = "panic"
            else:
                labels[order[0]] = "calm"
                labels[order[-1]] = "panic"
                for k in order[1:-1]:
                    labels[k] = "trend"
            return labels

        # with returns available, decide sign for high-vol cluster
        for rank, k in enumerate(order):
            ret = means[k, i_ret] # type: ignore
            if rank == 0:
                labels[k] = "calm"
            elif rank == len(order) - 1:
                labels[k] = "panic" if ret < 0 else "trend" # type: ignore
            else:
                labels[k] = "trend" if ret >= 0 else "panic" # type: ignore
        return labels

    # ------------ inference ------------
    def _ensure_ready(self):
        if self._gmm is None or self._scaler is None or self._A is None:
            raise RuntimeError("RegimeOracle is not fitted. Call .fit(X) first.")

    def predict_proba(self, X: pd.DataFrame, *, smooth: bool = True, p0: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Return per-regime probabilities for each row in X.
        If smooth=True, applies sticky HMM-style filtering via transition A.
        """
        self._ensure_ready()
        Xs = self._scaler.transform(X.values)  # type: ignore
        logprob = self._gmm._estimate_log_prob(Xs) + np.log(self._gmm.weights_ + 1e-12)  # type: ignore
        ll = logprob  # [T, K]
        K = ll.shape[1]

        if not smooth:
            P = np.exp(ll - ll.max(axis=1, keepdims=True))
            P = P / P.sum(axis=1, keepdims=True)
            return pd.DataFrame(P, index=X.index, columns=[f"reg_{i}" for i in range(K)])

        A = self._A  # type: ignore
        # forward filter
        P = np.zeros_like(ll)
        if p0 is None:
            p_prev = np.ones(K) / K
        else:
            p_prev = p0 / p0.sum()

        for t in range(ll.shape[0]):
            # predict
            p_pred = A.T @ p_prev # type: ignore
            # update
            unnorm = np.exp(ll[t] - ll[t].max()) * p_pred
            p_t = unnorm / unnorm.sum()
            P[t] = p_t
            p_prev = p_t

        return pd.DataFrame(P, index=X.index, columns=[f"reg_{i}" for i in range(K)])

    def predict(self, X: pd.DataFrame, *, smooth: bool = True) -> pd.Series:
        P = self.predict_proba(X, smooth=smooth)
        z = P.values.argmax(axis=1)
        return pd.Series(z, index=X.index, name="regime_id")

    def predict_label(self, X: pd.DataFrame, *, smooth: bool = True) -> Tuple[int, str]:
        """Convenience for the most-recent row in X."""
        rid = int(self.predict(X.tail(1), smooth=smooth).iloc[-1])
        label = self._labels_map.get(rid, f"reg_{rid}")
        return rid, label

    # ------------ allocator hints ------------
    def regime_signal(self, proba: Dict[str, float]) -> Dict[str, float]:
        """
        Convert regime posterior into allocator hints.
        Returns:
            risk_multiplier: scale factor for gross/cap (e.g., 0.4 in panic).
            hedge_bias:  +ve suggests adding index puts / long vol,
                         -ve suggests carry/short vol, ~0 neutral.
        """
        # map posterior to label probs
        label_probs: Dict[str, float] = {"calm": 0.0, "trend": 0.0, "panic": 0.0}
        # invert labels_map: id -> name
        inv = self._labels_map or {}
        for k, v in proba.items():
            # k like "reg_0"
            try:
                rid = int(k.split("_")[1])
            except Exception:
                continue
            lab = inv.get(rid, "trend")
            label_probs[lab] += float(v)

        calm = label_probs["calm"]
        trend = label_probs["trend"]
        panic = label_probs["panic"]

        # Heuristic policy
        risk_mult = 0.4 * panic + 0.9 * trend + 1.2 * calm  # cap later
        risk_mult = float(np.clip(risk_mult, 0.2, 1.5))
        hedge_bias = float(np.clip(1.5 * panic - 0.5 * calm, -1.0, 1.0))
        return {"risk_multiplier": risk_mult, "hedge_bias": hedge_bias}

    # ------------ reporting ------------
    def describe(self) -> pd.DataFrame:
        """Return centroid stats and label mapping for interpretability."""
        self._ensure_ready()
        means_std = self._gmm.means_ # type: ignore
        cov = self._gmm.covariances_ # type: ignore
        if self._scaler is not None:
            means = means_std * self._scaler.scale_ + self._scaler.mean_ # type: ignore
            stds = np.sqrt(cov) * self._scaler.scale_
        else:
            means, stds = means_std, np.sqrt(cov)

        rows = []
        for i in range(self.n_regimes):
            rows.append(
                {
                    "regime_id": i,
                    "label": self._labels_map.get(i, f"reg_{i}"),
                    **{f"mu_{f}": float(means[i, j]) for j, f in enumerate(self._features)}, # type: ignore
                    **{f"sd_{f}": float(stds[i, j]) for j, f in enumerate(self._features)},
                }
            )
        return pd.DataFrame(rows)

    # ------------ persistence ------------
    def save(self, path: str) -> None:
        """Persist oracle (scaler, gmm params, A, labels_map)."""
        self._ensure_ready()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # dump sklearn objects via joblib (if available)
        if joblib is None:
            warnings.warn("joblib not available; saving JSON-only artifacts.")
        else:
            joblib.dump(self._scaler, p.with_suffix(".scaler.joblib"))  # type: ignore
            joblib.dump(self._gmm, p.with_suffix(".gmm.joblib"))        # type: ignore

        state = OracleState(
            n_regimes=self.n_regimes,
            sticky=self.sticky,
            feature_names=self._features,
            scaler=None,  # saved separately
            gmm_means=self._gmm.means_,               # type: ignore
            gmm_covars=self._gmm.covariances_,        # type: ignore
            gmm_weights=self._gmm.weights_,           # type: ignore
            trans_matrix=self._A,
            labels_map=self._labels_map,
        )
        Path(str(p) + ".json").write_text(state.to_json())

    @staticmethod
    def load(path: str) -> "RegimeOracle":
        p = Path(path)
        state = OracleState.from_json(Path(str(p) + ".json").read_text())

        oracle = RegimeOracle(n_regimes=state.n_regimes, sticky=state.sticky)
        oracle._features = state.feature_names
        oracle._labels_map = state.labels_map or {}

        if joblib is not None and p.with_suffix(".gmm.joblib").exists():
            oracle._gmm = joblib.load(p.with_suffix(".gmm.joblib"))
            oracle._scaler = joblib.load(p.with_suffix(".scaler.joblib"))
        else:
            # Reconstruct minimal GMM from params (predict_proba still works)
            gmm = GaussianMixture(n_components=state.n_regimes, covariance_type="diag")
            gmm.weights_ = np.asarray(state.gmm_weights)
            gmm.means_ = np.asarray(state.gmm_means)
            gmm.covariances_ = np.asarray(state.gmm_covars)
            gmm.precisions_cholesky_ = 1.0 / np.sqrt(gmm.covariances_)
            oracle._gmm = gmm
            oracle._scaler = StandardScaler()  # identity-like
            oracle._scaler.mean_ = np.zeros(len(state.feature_names))
            oracle._scaler.scale_ = np.ones(len(state.feature_names))

        oracle._A = np.asarray(state.trans_matrix)
        return oracle

    # ------------ convenience ------------
    @staticmethod
    def from_time_series(
        prices: pd.DataFrame,
        *,
        feature_col: str = "close",
        extra: Optional[pd.DataFrame] = None,
        n_regimes: int = 3,
        sticky: float = 0.95,
        random_state: Optional[int] = 42,
    ) -> Tuple["RegimeOracle", pd.DataFrame]:
        """
        Build features from time series, fit the oracle, return (oracle, proba_df)
        """
        X, _ = build_features(prices, col=feature_col, extra=extra)
        oracle = RegimeOracle(n_regimes=n_regimes, sticky=sticky, random_state=random_state).fit(X)
        P = oracle.predict_proba(X)
        return oracle, P