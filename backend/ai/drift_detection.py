# backend/analytics/drift_detection.py
from __future__ import annotations

"""
Drift Detection
---------------
Lightweight detectors for:
  • Data drift (features): PSI, KS (if scipy), mean/std shift, missing-rate delta
  • Prediction drift (scores / classes): PSI, KL (if scipy)
  • Concept drift (optional labels): accuracy drop, label shift JS-div (if scipy)
  • Embedding drift (optional): mean cosine shift if numpy

No hard deps. If available, uses:
    numpy (faster bins, cosine)
    scipy.stats (KS test, entropy)
Env (optional bus):
    REDIS_HOST/REDIS_PORT, DRIFT_OUT_STREAM (default "monitor.drift")

CLI:
  python -m backend.analytics.drift_detection --ref ref.json --new new.json --out report.json
where JSON looks like:
{
  "features": {"feature_name": [1.0, 2.0, ...], "cat_feature": ["A","B", ...]},
  "pred_scores": [0.12, 0.33, ...],          # optional
  "pred_classes": ["buy","sell",...],        # optional
  "labels": ["up","down",...],               # optional, for concept checks
  "embeddings": [[...], [...], ...]          # optional, equal dims
}
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
import json
import math
import os
import time

# ---- Optional libs ---------------------------------------------------
try:
    import numpy as _np
except Exception:
    _np = None

try:
    from scipy import stats as _stats
except Exception:
    _stats = None

# ---- Optional Redis bus ----------------------------------------------
try:
    import redis as _redis
except Exception:
    _redis = None

try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
DRIFT_OUT_STREAM = os.getenv("DRIFT_OUT_STREAM", "monitor.drift")

# ---- Helpers ----------------------------------------------------------

def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _to_array(xs: List[Any]):
    if _np is None:
        # filter numerics only
        return [float(x) for x in xs if _is_num(x)]
    return _np.asarray(xs, dtype=float)

def _unique(xs: List[Any]) -> List[Any]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _quantiles(data: List[float], q: List[float]) -> List[float]:
    if not data:
        return [0.0 for _ in q]
    if _np is not None:
        return _np.quantile(_np.asarray(data, dtype=float), _np.asarray(q)).tolist()
    # pure python approximate quantiles
    s = sorted(data)
    n = len(s)
    res = []
    for qi in q:
        k = qi * (n - 1)
        f = int(math.floor(k)); c = min(n - 1, f + 1)
        if c == f: res.append(s[f]); continue
        d = k - f
        res.append(s[f] * (1 - d) + s[c] * d)
    return res

def _hist_counts_numeric(data: List[float], edges: List[float]) -> List[int]:
    # edges include both ends; len(edges)=B+1
    if _np is not None:
        h, _ = _np.histogram(_np.asarray(data, dtype=float), bins=_np.asarray(edges))
        return h.astype(int).tolist()
    # python
    counts = [0] * (len(edges) - 1)
    for v in data:
        # find bin
        for i in range(len(edges) - 1):
            if (v >= edges[i]) and (v <= edges[i+1] if i == len(edges)-2 else v < edges[i+1]):
                counts[i] += 1
                break
    return counts

def _psi(p: List[float], q: List[float], eps: float = 1e-8) -> float:
    # Population Stability Index
    s = 0.0
    for pi, qi in zip(p, q):
        pi = max(eps, pi); qi = max(eps, qi)
        s += (pi - qi) * math.log(pi / qi)
    return float(s)

def _kl(p: List[float], q: List[float], eps: float = 1e-8) -> float:
    if _stats is not None:
        return float(_stats.entropy(p, q))
    # manual KL(P||Q)
    s = 0.0
    for pi, qi in zip(p, q):
        pi = max(eps, pi); qi = max(eps, qi)
        s += pi * math.log(pi / qi)
    return float(s)

def _js(p: List[float], q: List[float], eps: float = 1e-8) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m, eps) + 0.5 * _kl(q, m, eps)

def _ks(ref: List[float], new: List[float]) -> Optional[float]:
    if _stats is None:  # p-value unavailable without scipy
        return None
    try:
        _, p = _stats.ks_2samp(ref, new, alternative="two-sided", mode="auto") # type: ignore
        return float(p)
    except Exception:
        return None

def _cosine(a, b) -> Optional[float]:
    if _np is None:
        return None
    a = _np.asarray(a, dtype=float); b = _np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    num = (a * b).sum()
    den = (a**2).sum()**0.5 * (b**2).sum()**0.5
    if den == 0: return None
    return float(num / den)

# ---- Data models -----------------------------------------------------

@dataclass
class FeatureDrift:
    name: str
    psi: Optional[float] = None
    ks_pvalue: Optional[float] = None
    mean_ref: Optional[float] = None
    mean_new: Optional[float] = None
    std_ref: Optional[float] = None
    std_new: Optional[float] = None
    miss_ref: Optional[float] = None
    miss_new: Optional[float] = None
    alert: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredDrift:
    psi_scores: Optional[float] = None
    kl_scores: Optional[float] = None
    psi_classes: Optional[float] = None
    js_classes: Optional[float] = None
    acc_ref: Optional[float] = None
    acc_new: Optional[float] = None
    acc_drop: Optional[float] = None
    alert: bool = False

@dataclass
class EmbeddingDrift:
    mean_cosine: Optional[float] = None  # cosine between mean(ref) and mean(new)
    alert: bool = False

@dataclass
class DriftReport:
    ts_ms: int
    features: List[FeatureDrift]
    prediction: Optional[PredDrift]
    embedding: Optional[EmbeddingDrift]
    overall_alert: bool
    summary: Dict[str, Any]

# ---- Core monitor ----------------------------------------------------

class DriftMonitor:
    """
    fit_reference(): store your baseline windows (features/preds/labels/embeddings).
    compare(): compute drift metrics against the baseline.
    Thresholds are configurable per metric; sensible defaults are provided.
    """

    def __init__(
        self,
        *,
        feature_bins: int = 10,
        psi_alert: float = 0.25,          # ~ moderate drift: 0.1-0.25; strong > 0.25
        ks_alert_p: float = 0.01,         # p < 0.01 → reject same distribution
        score_psi_alert: float = 0.2,
        class_psi_alert: float = 0.2,
        acc_drop_alert: float = 0.05,     # 5pp accuracy drop
        embed_cosine_alert: float = 0.92, # below this signals shift (1 = identical)
        out_stream: str = DRIFT_OUT_STREAM
    ):
        self.feature_bins = max(4, feature_bins)
        self.psi_alert = psi_alert
        self.ks_alert_p = ks_alert_p
        self.score_psi_alert = score_psi_alert
        self.class_psi_alert = class_psi_alert
        self.acc_drop_alert = acc_drop_alert
        self.embed_cosine_alert = embed_cosine_alert
        self.out_stream = out_stream

        self.ref: Dict[str, Any] = {}  # stored baselines

        # optional Redis client
        self._r = None
        if _redis is not None:
            try:
                self._r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            except Exception:
                self._r = None

    # --------- public API ---------

    def fit_reference(
        self,
        *,
        features: Dict[str, List[Any]],
        pred_scores: Optional[List[float]] = None,
        pred_classes: Optional[List[Any]] = None,
        labels: Optional[List[Any]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        self.ref = {
            "features": features or {},
            "pred_scores": pred_scores or [],
            "pred_classes": pred_classes or [],
            "labels": labels or [],
            "embeddings": embeddings or [],
        }

    def compare(
        self,
        *,
        features: Dict[str, List[Any]],
        pred_scores: Optional[List[float]] = None,
        pred_classes: Optional[List[Any]] = None,
        labels: Optional[List[Any]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> DriftReport:
        ts = int(time.time() * 1000)
        feats = self._feature_drift(self.ref.get("features", {}), features or {})
        pred = self._prediction_drift(
            self.ref.get("pred_scores", []),
            self.ref.get("pred_classes", []),
            self.ref.get("labels", []),
            pred_scores or [], pred_classes or [], labels or []
        )
        emb = self._embedding_drift(self.ref.get("embeddings", []), embeddings or [])

        overall = any(fd.alert for fd in feats) or (pred.alert if pred else False) or (emb.alert if emb else False)

        report = DriftReport(
            ts_ms=ts,
            features=feats,
            prediction=pred,
            embedding=emb,
            overall_alert=overall,
            summary=self._summarize(feats, pred, emb)
        )

        # emit to bus (optional)
        payload = asdict(report)
        publish_stream(self.out_stream, payload)
        return report

    # --------- metrics implementations ---------

    def _feature_drift(self, ref: Dict[str, List[Any]], new: Dict[str, List[Any]]) -> List[FeatureDrift]:
        out: List[FeatureDrift] = []
        keys = sorted(set(ref.keys()) | set(new.keys()))
        for k in keys:
            r = ref.get(k, [])
            n = new.get(k, [])
            # missing rates
            miss_r = 1.0 - (len([x for x in r if x is not None]) / max(1, len(r)))
            miss_n = 1.0 - (len([x for x in n if x is not None]) / max(1, len(n)))

            # numeric vs categorical
            r_num = [x for x in r if _is_num(x)]
            n_num = [x for x in n if _is_num(x)]

            fd = FeatureDrift(name=k, miss_ref=miss_r, miss_new=miss_n)

            if len(r_num) >= 5 and len(n_num) >= 5:
                # numeric PSI via quantile bins of reference
                edges = self._ref_edges(r_num, self.feature_bins)
                p = self._probs(r_num, edges)
                q = self._probs(n_num, edges)
                fd.psi = _psi(p, q)
                # KS p-value if available
                fd.ks_pvalue = _ks(r_num, n_num)
                # moments
                fd.mean_ref = float(sum(r_num) / len(r_num))
                fd.mean_new = float(sum(n_num) / len(n_num))
                sr = (sum((x - fd.mean_ref) ** 2 for x in r_num) / max(1, len(r_num) - 1)) ** 0.5
                sn = (sum((x - fd.mean_new) ** 2 for x in n_num) / max(1, len(n_num) - 1)) ** 0.5
                fd.std_ref, fd.std_new = float(sr), float(sn)
                # alert rule
                fd.alert = (fd.psi is not None and fd.psi >= self.psi_alert) or \
                           (fd.ks_pvalue is not None and fd.ks_pvalue <= self.ks_alert_p) or \
                           (abs((fd.mean_new or 0) - (fd.mean_ref or 0)) > 3.0 * (fd.std_ref or 1e-9))
            else:
                # categorical PSI on frequency of top categories
                cats = _unique([x for x in r if x is not None] + [x for x in n if x is not None])
                if cats:
                    p = self._cat_probs(r, cats)
                    q = self._cat_probs(n, cats)
                    fd.psi = _psi(p, q)
                    fd.alert = (fd.psi is not None and fd.psi >= self.class_psi_alert)
                else:
                    fd.alert = False

            out.append(fd)
        return out

    def _prediction_drift(
        self,
        ref_scores: List[float],
        ref_classes: List[Any],
        ref_labels: List[Any],
        new_scores: List[float],
        new_classes: List[Any],
        new_labels: List[Any],
    ) -> Optional[PredDrift]:
        if not ref_scores and not ref_classes:
            return None

        pd = PredDrift()
        # scores
        if ref_scores and new_scores:
            edges = self._ref_edges(ref_scores, 10)
            p = self._probs(ref_scores, edges)
            q = self._probs(new_scores, edges)
            pd.psi_scores = _psi(p, q)
            pd.kl_scores = _kl(p, q)
        # classes
        if ref_classes and new_classes:
            cats = _unique(ref_classes + new_classes)
            p = self._cat_probs(ref_classes, cats)
            q = self._cat_probs(new_classes, cats)
            pd.psi_classes = _psi(p, q)
            pd.js_classes = _js(p, q)
        # accuracy drop (if labels available)
        if ref_labels and ref_classes and len(ref_labels) == len(ref_classes):
            acc_r = sum(1 for a, b in zip(ref_classes, ref_labels) if a == b) / max(1, len(ref_labels))
            pd.acc_ref = float(acc_r)
        if new_labels and new_classes and len(new_labels) == len(new_classes):
            acc_n = sum(1 for a, b in zip(new_classes, new_labels) if a == b) / max(1, len(new_labels))
            pd.acc_new = float(acc_n)
        if pd.acc_ref is not None and pd.acc_new is not None:
            pd.acc_drop = float(pd.acc_ref - pd.acc_new)

        # alert
        pd.alert = (
            (pd.psi_scores is not None and pd.psi_scores >= self.score_psi_alert) or
            (pd.psi_classes is not None and pd.psi_classes >= self.class_psi_alert) or
            (pd.acc_drop is not None and pd.acc_drop >= self.acc_drop_alert)
        )
        return pd

    def _embedding_drift(self, ref_emb: List[List[float]], new_emb: List[List[float]]) -> Optional[EmbeddingDrift]:
        if not ref_emb or not new_emb:
            return None
        if _np is None:
            return EmbeddingDrift(mean_cosine=None, alert=False)
        ref = _np.asarray(ref_emb, dtype=float)
        new = _np.asarray(new_emb, dtype=float)
        mu_r = ref.mean(axis=0)
        mu_n = new.mean(axis=0)
        cos = _cosine(mu_r, mu_n)
        alert = (cos is not None) and (cos < self.embed_cosine_alert)
        return EmbeddingDrift(mean_cosine=cos, alert=bool(alert))

    # --------- utilities ---------

    def _ref_edges(self, ref: List[float], bins: int) -> List[float]:
        qs = [i / bins for i in range(bins + 1)]
        edges = _quantiles(ref, qs)
        # ensure strictly increasing
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + 1e-9
        return edges

    def _probs(self, data: List[float], edges: List[float]) -> List[float]:
        n = max(1, len(data))
        counts = _hist_counts_numeric(data, edges)
        return [c / n for c in counts]

    def _cat_probs(self, xs: List[Any], cats: List[Any]) -> List[float]:
        n = max(1, len(xs))
        freq = {c: 0 for c in cats}
        for x in xs:
            if x in freq:
                freq[x] += 1
        return [freq[c] / n for c in cats]

    def _summarize(self, feats: List[FeatureDrift], pred: Optional[PredDrift], emb: Optional[EmbeddingDrift]) -> Dict[str, Any]:
        top_feats = sorted(
            [f for f in feats if f.psi is not None],
            key=lambda f: f.psi, reverse=True # type: ignore
        )[:5] # type: ignore
        return {
            "top_feature_psi": [{ "name": f.name, "psi": f.psi } for f in top_feats],
            "pred": asdict(pred) if pred else None,
            "embedding": asdict(emb) if emb else None,
            "alerts": {
                "features": sum(1 for f in feats if f.alert),
                "prediction": bool(pred.alert) if pred else False,
                "embedding": bool(emb.alert) if emb else False,
            }
        }

# ---- CLI ------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Drift detection (data, prediction, concept, embeddings)")
    p.add_argument("--ref", required=True, help="Reference JSON path")
    p.add_argument("--new", required=True, help="New batch JSON path")
    p.add_argument("--out", required=False, help="Write report JSON here")
    args = p.parse_args()

    ref = _load_json(args.ref)
    new = _load_json(args.new)

    mon = DriftMonitor()
    mon.fit_reference(
        features=ref.get("features", {}),
        pred_scores=ref.get("pred_scores"),
        pred_classes=ref.get("pred_classes"),
        labels=ref.get("labels"),
        embeddings=ref.get("embeddings"),
    )
    rep = mon.compare(
        features=new.get("features", {}),
        pred_scores=new.get("pred_scores"),
        pred_classes=new.get("pred_classes"),
        labels=new.get("labels"),
        embeddings=new.get("embeddings"),
    )
    payload = asdict(rep)
    if args.out:
        _save_json(args.out, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":  # pragma: no cover
    _main()