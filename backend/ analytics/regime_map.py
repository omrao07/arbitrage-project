# backend/analytics/regime_map.py
from __future__ import annotations

"""
Regime Map
----------
Classifies market regime from a small set of features:
  - Realized volatility (rv)
  - Trend strength (tr) from moving average slope / zscore
  - Average pairwise correlation (corr)
  - Liquidity proxy (liq): e.g., turnover or 1/spread
  - Credit spread proxy (cred): e.g., HY - IG, or funding stress metric

Outputs one of:
  'CALM_TREND', 'RANGE_BOUND', 'RISK_ON', 'LIQUIDITY_CRUNCH', 'CRISIS', 'RECOVERY'

Hysteresis & dwell time prevent rapid oscillations.

Streams (env):
  features.market : {"ts_ms","rv","tr","corr","liq","cred"} (any subset allowed)
  prices.bars     : {"ts_ms","symbol","close"}  (we can derive rv/tr if desired)
  regime.state    : {"ts_ms","regime","score","features":{...},"notes": "..."}
"""

import os, json, time, math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

# -------- deps (graceful) ----------------------------------------------------
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("regime_map requires numpy") from e

try:
    import pandas as pd  # optional (only for convenience in backfill)
except Exception:
    pd = None  # type: ignore

# -------- optional redis -----------------------------------------------------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

# -------- env / streams ------------------------------------------------------
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FEATURES_STREAM  = os.getenv("FEATURES_STREAM", "features.market")
PRICES_BARS      = os.getenv("PRICES_BARS_STREAM", "prices.bars")  # optional raw feed
REGIME_STREAM    = os.getenv("REGIME_OUT_STREAM", "regime.state")
MAXLEN           = int(os.getenv("REGIME_MAXLEN", "5000"))

# -------- utility ------------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

def _nan_to(val: Optional[float], alt: float) -> float:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)): return alt
        return float(val)
    except Exception:
        return alt

# -------- configuration ------------------------------------------------------
@dataclass
class Thresholds:
    # Volatility (daily or annualized; keep consistent with your pipeline)
    rv_low: float = 0.10
    rv_high: float = 0.30
    # Trend (|zscore| or scaled slope)
    tr_strong: float = 1.0
    tr_weak: float = 0.25
    # Average pairwise correlation
    corr_low: float = 0.20
    corr_high: float = 0.60
    # Liquidity (higher is more liquid)
    liq_low: float = 0.2
    liq_high: float = 0.7
    # Credit spread stress (higher = worse)
    cred_ok: float = 0.3
    cred_stress: float = 0.7

@dataclass
class Hysteresis:
    # add buffers so exiting a regime needs stronger move
    rv_buf: float = 0.02
    tr_buf: float = 0.10
    corr_buf: float = 0.05
    liq_buf: float = 0.05
    cred_buf: float = 0.05
    min_dwell_bars: int = 30    # do not switch until at least N bars passed

@dataclass
class RegimeConfig:
    thresholds: Thresholds = Thresholds()
    hysteresis: Hysteresis = Hysteresis()
    # derive features from prices if feed not present
    derive_from_prices: bool = False
    bar_ms: int = int(os.getenv("REGIME_BAR_MS", "60000"))  # 1m bars
    # trend calc window & vol window (in bars)
    tr_win: int = 60
    rv_win: int = 60

# -------- state --------------------------------------------------------------
@dataclass
class RegimeState:
    ts_ms: int = 0
    regime: str = "RECOVERY"
    dwell: int = 0
    last_features: Dict[str, float] = None # type: ignore

# -------- regime mapper core -------------------------------------------------
class RegimeMapper:
    def __init__(self, cfg: Optional[RegimeConfig] = None):
        self.cfg = cfg or RegimeConfig()
        self.st = RegimeState(ts_ms=0, regime="RECOVERY", dwell=0, last_features={})

        # buffers for price-derived features (if needed)
        self._px: List[float] = []
        self._rets: List[float] = []

    # ---- public entry: classify --------------------------------------------
    def classify(self, ts_ms: int, *, rv: Optional[float], tr: Optional[float],
                 corr: Optional[float], liq: Optional[float], cred: Optional[float]) -> Tuple[str, float, Dict[str, float], str]:
        """
        Returns (regime, score, features, notes)
        """
        T = self.cfg.thresholds
        H = self.cfg.hysteresis

        # use last features for hysteresis buffers
        prev = self.st.last_features or {}
        prev_reg = self.st.regime

        # normalize inputs with defaults
        f_rv   = _nan_to(rv,   prev.get("rv",   T.rv_low))
        f_tr   = _nan_to(tr,   prev.get("tr",   0.0))
        f_corr = _nan_to(corr, prev.get("corr", T.corr_low))
        f_liq  = _nan_to(liq,  prev.get("liq",  0.5))
        f_cred = _nan_to(cred, prev.get("cred", T.cred_ok))

        # Apply hysteresis (soft): shift thresholds depending on current regime
        rv_hi = T.rv_high + (H.rv_buf if prev_reg in ("CRISIS","LIQUIDITY_CRUNCH") else 0.0)
        rv_lo = T.rv_low  - (H.rv_buf if prev_reg in ("CALM_TREND","RISK_ON") else 0.0)
        tr_str = T.tr_strong - (H.tr_buf if prev_reg in ("CALM_TREND","RISK_ON") else 0.0)
        tr_wk  = T.tr_weak   + (H.tr_buf if prev_reg in ("RANGE_BOUND","RECOVERY") else 0.0)
        corr_hi = T.corr_high + (H.corr_buf if prev_reg in ("CRISIS","LIQUIDITY_CRUNCH") else 0.0)
        corr_lo = T.corr_low  - (H.corr_buf if prev_reg in ("CALM_TREND","RISK_ON") else 0.0)
        liq_lo  = T.liq_low   + (H.liq_buf if prev_reg in ("CRISIS","LIQUIDITY_CRUNCH") else 0.0)
        liq_hi  = T.liq_high  - (H.liq_buf if prev_reg in ("CALM_TREND","RISK_ON") else 0.0)
        cred_ok = T.cred_ok   - (H.cred_buf if prev_reg in ("CALM_TREND","RISK_ON") else 0.0)
        cred_str= T.cred_stress + (H.cred_buf if prev_reg in ("CRISIS","LIQUIDITY_CRUNCH") else 0.0)

        notes = []

        # Rule blocks (ordered by severity)
        if (f_rv >= rv_hi and f_corr >= corr_hi) or (f_cred >= cred_str):
            regime = "CRISIS"
            notes.append("High vol + high correlation clustering or credit stress.")
        elif (f_liq <= liq_lo and f_corr >= corr_hi) or (f_cred > T.cred_ok and f_rv > T.rv_low):
            regime = "LIQUIDITY_CRUNCH"
            notes.append("Liquidity thin with elevated correlation; credit tightness present.")
        elif (f_rv <= rv_lo and abs(f_tr) >= tr_str and f_corr <= corr_hi and f_liq >= liq_hi):
            regime = "CALM_TREND"
            notes.append("Low vol, strong trend, decent liquidity.")
        elif (f_rv <= T.rv_high and abs(f_tr) >= tr_wk and f_corr <= corr_hi and f_liq >= T.liq_low):
            regime = "RISK_ON"
            notes.append("Moderate vol, trending, correlations contained.")
        elif (f_rv <= T.rv_high and abs(f_tr) < tr_wk) and (f_corr <= corr_hi):
            regime = "RANGE_BOUND"
            notes.append("Contained vol, weak trend → chop.")
        else:
            # default transition state
            regime = "RECOVERY"
            notes.append("Mixed signals; healing/transition phase.")

        # dwell guard
        if regime != prev_reg and self.st.dwell < H.min_dwell_bars:
            # stick with previous unless severe deterioration (CRISIS override)
            if regime == "CRISIS":
                notes.append("Overriding dwell due to CRISIS severity.")
            else:
                regime = prev_reg
                notes.append(f"Kept {prev_reg} due to min dwell {H.min_dwell_bars} bars.")

        # score: simple composite in [-1, +1]
        # negative = stress, positive = benign trend
        s_stress = 0.0
        s_benefit = 0.0
        # stress components
        s_stress += _sigmoid((f_rv - T.rv_low) / max(1e-6, T.rv_high - T.rv_low))
        s_stress += _sigmoid((f_corr - T.corr_low) / max(1e-6, T.corr_high - T.corr_low))
        s_stress += _sigmoid((f_cred - T.cred_ok) / max(1e-6, T.cred_stress - T.cred_ok))
        s_stress += _sigmoid((T.liq_low - f_liq) / max(1e-6, T.liq_low))  # illiquidity
        # benefit components
        s_benefit += _sigmoid((abs(f_tr) - T.tr_weak) / max(1e-6, T.tr_strong - T.tr_weak))
        s_benefit += _sigmoid((f_liq - T.liq_low) / max(1e-6, T.liq_high - T.liq_low))
        score = float(np.tanh(0.5 * (s_benefit - s_stress)))

        # update state
        if regime == prev_reg:
            self.st.dwell += 1
        else:
            self.st.dwell = 1
        self.st.regime = regime
        self.st.ts_ms = ts_ms
        feats = {"rv": f_rv, "tr": f_tr, "corr": f_corr, "liq": f_liq, "cred": f_cred}
        self.st.last_features = feats

        return regime, score, feats, "; ".join(notes)

    # ---- optional: derive features from prices ------------------------------
    def push_price(self, px: float):
        if px is None or px <= 0: return
        if self._px:
            self._rets.append(math.log(px) - math.log(self._px[-1]))
            if len(self._rets) > max(self.cfg.rv_win * 5, self.cfg.tr_win * 5):
                self._rets = self._rets[-max(self.cfg.rv_win * 5, self.cfg.tr_win * 5):]
        self._px.append(px)
        if len(self._px) > max(self.cfg.rv_win * 5, self.cfg.tr_win * 5) + 1:
            self._px = self._px[-(max(self.cfg.rv_win * 5, self.cfg.tr_win * 5) + 1):]

    def derive_features(self) -> Dict[str, float]:
        r = np.asarray(self._rets, dtype=float)
        out = {"rv": float("nan"), "tr": float("nan"), "corr": float("nan"), "liq": float("nan"), "cred": float("nan")}
        if r.size >= self.cfg.rv_win:
            out["rv"] = float(np.std(r[-self.cfg.rv_win:], ddof=0) * math.sqrt(252 * (self.cfg.bar_ms / 86_400_000)))
        if r.size >= self.cfg.tr_win + 1:
            seg = r[-self.cfg.tr_win:]
            # trend ~ zscore of cumulative return slope
            cum = np.cumsum(seg)
            x = np.arange(cum.size)
            # slope via simple covariance / var
            vx = float(np.var(x)) + 1e-12
            beta = float(np.cov(x, cum)[0,1] / vx) if np.ndim(np.cov(x, cum)) == 2 else 0.0
            z = float(beta / (np.std(cum) + 1e-12))
            out["tr"] = z
        # corr/liq/cred can be plugged from other subsystems; keep NaN if unknown
        return out

# -------- helper -------------------------------------------------------------
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except Exception:
        return 0.5

# -------- worker (Redis) -----------------------------------------------------
class RegimeWorker:
    """
    Live worker:
      - If FEATURES_STREAM present, consume features directly.
      - Else, if derive_from_prices=True, consume PRICES_BARS for a proxy (rv,tr).
    Emits regime snapshots to REGIME_STREAM.
    """
    def __init__(self, cfg: Optional[RegimeConfig] = None):
        self.cfg = cfg or RegimeConfig()
        self.rm = RegimeMapper(self.cfg)
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.last_feat_id = "$"
        self.last_px_id = "$"

    async def connect(self):
        if not USE_REDIS: return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def run(self):
        await self.connect()
        if not self.r:
            print("[regime] no redis; worker idle"); return

        while True:
            try:
                streams = {}
                streams[FEATURES_STREAM] = self.last_feat_id
                if self.cfg.derive_from_prices:
                    streams[PRICES_BARS] = self.last_px_id

                resp = await self.r.xread(streams, count=500, block=2000)  # type: ignore
                if not resp:
                    continue

                out_ready = False
                feats: Dict[str, float] = {}
                ts = now_ms()

                for stream, entries in resp:
                    for _id, fields in entries:
                        j = {}
                        try:
                            j = json.loads(fields.get("json","{}"))
                        except Exception:
                            continue
                        if stream == FEATURES_STREAM:
                            self.last_feat_id = _id
                            ts = int(j.get("ts_ms") or ts)
                            feats.update({
                                "rv": j.get("rv"),
                                "tr": j.get("tr"),
                                "corr": j.get("corr"),
                                "liq": j.get("liq"),
                                "cred": j.get("cred"),
                            })
                            out_ready = True
                        elif stream == PRICES_BARS and self.cfg.derive_from_prices:
                            self.last_px_id = _id
                            px = j.get("close")
                            if px is not None:
                                self.rm.push_price(float(px))
                                ts = int(j.get("ts_ms") or ts)
                                # try building minimal features on the fly
                                feats.update(self.rm.derive_features())
                                out_ready = True

                if out_ready:
                    regime, score, f_used, notes = self.rm.classify(
                        ts_ms=ts,
                        rv=feats.get("rv"),
                        tr=feats.get("tr"),
                        corr=feats.get("corr"),
                        liq=feats.get("liq"),
                        cred=feats.get("cred"),
                    )
                    await self._publish({
                        "ts_ms": ts,
                        "regime": regime,
                        "score": round(score, 6),
                        "features": {k: (None if (v is None or (isinstance(v,float) and math.isnan(v))) else float(v)) for k,v in f_used.items()},
                        "notes": notes,
                        "dwell": self.rm.st.dwell
                    })

            except Exception as e:
                await self._publish({"ts_ms": now_ms(), "error": str(e)})

    async def _publish(self, obj: Dict):
        if self.r:
            try:
                await self.r.xadd(REGIME_STREAM, {"json": json.dumps(obj)}, maxlen=MAXLEN, approximate=True)  # type: ignore
            except Exception:
                pass
        else:
            print("[regime]", obj)

# -------- backfill helper ----------------------------------------------------
def backfill(features: Dict[str, List[float]], cfg: Optional[RegimeConfig] = None) -> List[Dict]:
    """
    features: dict with lists for keys among ['rv','tr','corr','liq','cred'] aligned in time.
    Returns list of snapshots suitable for plotting a timeline.
    """
    cfg = cfg or RegimeConfig()
    rm = RegimeMapper(cfg)
    T = max(len(v) for v in features.values() if v is not None) if features else 0
    out = []
    ts = now_ms() - T * cfg.bar_ms
    for i in range(T):
        rv   = _seq(features.get("rv"), i)
        tr   = _seq(features.get("tr"), i)
        corr = _seq(features.get("corr"), i)
        liq  = _seq(features.get("liq"), i)
        cred = _seq(features.get("cred"), i)
        reg, sc, f, notes = rm.classify(ts, rv=rv, tr=tr, corr=corr, liq=liq, cred=cred)
        out.append({"ts_ms": ts, "regime": reg, "score": sc, "features": f, "notes": notes, "dwell": rm.st.dwell})
        ts += cfg.bar_ms
    return out

def _seq(a: Optional[List[float]], i: int) -> Optional[float]:
    if a is None: return None
    if i < len(a): return a[i]
    return None

# -------- CLI / demo ---------------------------------------------------------
def _demo():
    rng = np.random.default_rng(7)
    T = 800
    # simulate regimes by stitching different parameter blocks
    rv = np.concatenate([
        0.10 + 0.02 * rng.standard_normal(200),
        0.18 + 0.03 * rng.standard_normal(200),
        0.35 + 0.05 * rng.standard_normal(200),
        0.12 + 0.02 * rng.standard_normal(200),
    ])
    tr = np.concatenate([
        1.2  + 0.3 * rng.standard_normal(200),  # strong trend
        0.2  + 0.4 * rng.standard_normal(200),  # weak
        0.1  + 0.3 * rng.standard_normal(200),  # breaks
        0.8  + 0.3 * rng.standard_normal(200),  # recovery trend
    ])
    corr = np.concatenate([
        0.25 + 0.08 * rng.standard_normal(200),
        0.40 + 0.10 * rng.standard_normal(200),
        0.75 + 0.08 * rng.standard_normal(200),
        0.35 + 0.10 * rng.standard_normal(200),
    ])
    liq = np.concatenate([
        0.8  + 0.05 * rng.standard_normal(200),
        0.6  + 0.06 * rng.standard_normal(200),
        0.25 + 0.07 * rng.standard_normal(200),
        0.75 + 0.05 * rng.standard_normal(200),
    ])
    cred = np.concatenate([
        0.25 + 0.05 * rng.standard_normal(200),
        0.35 + 0.05 * rng.standard_normal(200),
        0.80 + 0.05 * rng.standard_normal(200),
        0.30 + 0.05 * rng.standard_normal(200),
    ])
    rows = backfill({"rv":rv.tolist(),"tr":tr.tolist(),"corr":corr.tolist(),"liq":liq.tolist(),"cred":cred.tolist()})
    print("demo rows:", len(rows), "last:", rows[-1]["regime"], rows[-1]["score"])

if __name__ == "__main__":
    import argparse, asyncio
    ap = argparse.ArgumentParser("regime_map")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--derive-from-prices", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
    elif args.worker:
        cfg = RegimeConfig(derive_from_prices=args.derive_from_prices)
        asyncio.run(RegimeWorker(cfg).run())
    else:
        _demo()