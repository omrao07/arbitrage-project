# backend/analytics/mood_index.py
"""
Market Mood Index (MMI)

Blends fast+slow signals into a single, smoothed index in [-1, +1]:
- News sentiment (headline/article)  → from your news parsers
- Social/exhaust sentiment            → from corp_exhaust SignalComposer
- Market internals                    → breadth, vol regime, credit spreads, FX
- Flow/positioning proxies            → risk-on vs safe-haven
- PnL stress / drawdown               → feedback from your PnL X-Ray

Design goals
------------
- Pure Python; pandas optional for pretty tables
- Works with partial inputs (missing features default to neutral)
- Kalman-like 1D smoother (alpha-beta filter) for stability
- Regime bands with hysteresis to avoid flapping
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional pandas (for to_dataframe)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# =============================================================================
# Helpers
# =============================================================================

def _safe(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _zscore(x: float, hist: List[float], floor: float = 1e-9) -> float:
    if not hist:
        return 0.0
    mu = statistics.fmean(hist)
    try:
        sd = statistics.pstdev(hist)
    except Exception:
        sd = 0.0
    sd = max(sd, floor)
    return (x - mu) / sd

def _tanh_norm(x: float, scale: float = 1.0) -> float:
    """Squash to [-1,1] with tunable scale."""
    return math.tanh(x / max(1e-9, scale))


# =============================================================================
# Input schema
# =============================================================================

@dataclass
class MoodInputs:
    """
    Provide the latest snapshot of inputs (any can be omitted).
    All scores should be roughly in [-1, +1] already if possible; raw values will be standardized.
    """
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    # --- Sentiment (already scaled to [-1, +1] ideally) ---
    news_sentiment: Optional[float] = None        # from your news_yahoo/news_moneycontrol sentiment
    social_sentiment: Optional[float] = None      # from corp_exhaust SignalComposer score
    earnings_tone: Optional[float] = None         # optional: ER call tone

    # --- Market internals (raw → normalized inside) ---
    adv_decline_ratio: Optional[float] = None     # (>1 bullish breadth)
    realized_vol: Optional[float] = None          # e.g., 20d stdev or VIX-like
    credit_spread_bp: Optional[float] = None      # e.g., IG spread bps
    usd_dxy_ret: Optional[float] = None           # recent DXY % return
    gold_ret: Optional[float] = None              # recent XAU % return
    oil_ret: Optional[float] = None               # recent Brent/WTI % return
    em_fx_ret: Optional[float] = None             # basket % return (risk-on proxy)
    crypto_ret: Optional[float] = None            # BTC/ETH basket % return

    # --- Flow / positioning proxies ---
    etf_inflows_std: Optional[float] = None       # standardized equity ETF flows
    futures_positioning_std: Optional[float] = None

    # --- Feedback from your system ---
    pnl_rolling_dd: Optional[float] = None        # rolling drawdown fraction [0..1]
    pnl_sign: Optional[float] = None              # sign(last pnl delta) in [-1,1]

    # --- Optional custom features map (already scaled) ---
    extra: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Output schema
# =============================================================================

@dataclass
class MoodReading:
    ts_ms: int
    raw_components: Dict[str, float]
    composite_raw: float
    composite_smoothed: float
    regime: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# =============================================================================
# Core engine
# =============================================================================

class MoodIndex:
    """
    Streaming mood index with a Kalman-ish 1D smoother and regime bands.
    """

    def __init__(
        self,
        *,
        weights: Optional[Dict[str, float]] = None,
        half_life_s: float = 600.0,          # smoothing half-life (~10 min)
        hysteresis: float = 0.1,             # regime switch buffer
        history_len: int = 5_000,
    ):
        # Component weights (sum doesn't have to be 1.0)
        self.weights = dict({
            "news": 0.35,
            "social": 0.15,
            "breadth": 0.15,
            "vol": 0.15,
            "credit": 0.10,
            "flows": 0.05,
            "risk_on": 0.10,
            "pnl_feedback": 0.05,
        })
        if weights:
            self.weights.update(weights)

        # Smoother params
        self.alpha = 1.0 - math.exp(-math.log(2.0) / max(1e-9, half_life_s))  # EMA alpha from half-life
        self.hysteresis = float(hysteresis)
        self.history_len = int(history_len)

        # State
        self._hist_raw: List[float] = []
        self._hist_smooth: List[float] = []
        self._hist_ts: List[int] = []
        self._last_s: float = 0.0
        self._last_regime: str = "neutral"

        # Rolling stats for standardization of raw internals
        self._roll: Dict[str, List[float]] = {}

    # ------------------------------- Public API -------------------------------

    def update(self, inp: MoodInputs) -> MoodReading:
        """
        Ingest one snapshot and emit a new MoodReading.
        """
        c = self._compute_components(inp)
        raw = sum(c[k] * self.weights.get(k, 0.0) for k in c)

        # smooth
        s = (1 - self.alpha) * self._last_s + self.alpha * raw if self._hist_smooth else raw
        s = _clip(s, -1.0, 1.0)

        # regime with hysteresis
        regime = self._decide_regime(s)

        # persist history
        self._push_hist(inp.ts_ms, raw, s)

        reading = MoodReading(
            ts_ms=inp.ts_ms,
            raw_components=c,
            composite_raw=_clip(raw, -1.5, 1.5),
            composite_smoothed=s,
            regime=regime,
            meta={"alpha": self.alpha, "weights": self.weights},
        )
        return reading

    def history(self, *, last_n: Optional[int] = None) -> List[MoodReading]:
        n = len(self._hist_ts)
        rng = range(max(0, n - (last_n or n)), n)
        out: List[MoodReading] = []
        for i in rng:
            out.append(MoodReading(
                ts_ms=self._hist_ts[i],
                raw_components={},  # omit to keep light; reconstruct if needed
                composite_raw=self._hist_raw[i],
                composite_smoothed=self._hist_smooth[i],
                regime=self._regime_from_value(self._hist_smooth[i]),
            ))
        return out

    def to_dataframe(self):
        if pd is None:
            raise RuntimeError("pandas not installed")
        return pd.DataFrame({
            "ts_ms": self._hist_ts,
            "raw": self._hist_raw,
            "smoothed": self._hist_smooth,
            "regime": [self._regime_from_value(x) for x in self._hist_smooth],
        })

    # ------------------------------- Internals --------------------------------

    def _compute_components(self, inp: MoodInputs) -> Dict[str, float]:
        comps: Dict[str, float] = {}

        # ---- Sentiment (already scaled or z-normalized if raw) ----
        news = _clip(_safe(inp.news_sentiment, 0.0), -1.0, 1.0)
        social = _clip(_safe(inp.social_sentiment, 0.0), -1.0, 1.0)
        earn = _clip(_safe(inp.earnings_tone, 0.0), -1.0, 1.0)
        sent = self._blend([news, social, 0.5 * earn])
        comps["news"] = sent
        comps["social"] = social  # keep separately for diagnostics

        # ---- Breadth (adv/decl) → map >1 to positive ----
        breadth = self._standardize("breadth_raw", _safe(inp.adv_decline_ratio))
        # adv/decl above 1 -> positive; below 1 -> negative
        comps["breadth"] = _tanh_norm(breadth, scale=1.0)

        # ---- Volatility (higher vol = risk-off) ----
        volz = - self._standardize("realized_vol", _safe(inp.realized_vol))   # negative sign
        comps["vol"] = _tanh_norm(volz, scale=1.5)

        # ---- Credit spreads (wider = risk-off) ----
        credz = - self._standardize("credit_spread", _safe(inp.credit_spread_bp))
        comps["credit"] = _tanh_norm(credz, scale=1.5)

        # ---- Risk-on basket (EM FX + crypto + oil beta) & Safe-haven (USD, gold) ----
        risk_on = self._blend([
            + self._standardize("em_fx_ret", _safe(inp.em_fx_ret)),
            + self._standardize("crypto_ret", _safe(inp.crypto_ret)),
            + 0.5 * self._standardize("oil_ret", _safe(inp.oil_ret)),
            - self._standardize("dxy_ret", _safe(inp.usd_dxy_ret)),   # stronger USD → risk-off
            - 0.5 * self._standardize("gold_ret", _safe(inp.gold_ret))
        ])
        comps["risk_on"] = _tanh_norm(risk_on, scale=1.5)

        # ---- Flows / positioning ----
        flow = self._blend([
            _safe(inp.etf_inflows_std, 0.0),          # already standardized
            0.5 * _safe(inp.futures_positioning_std, 0.0),
        ])
        comps["flows"] = _clip(flow, -1.0, 1.0)

        # ---- PnL feedback (drawdown dampens) ----
        dd = _safe(inp.pnl_rolling_dd, 0.0)          # [0..1]
        pnl_sign = _safe(inp.pnl_sign, 0.0)          # [-1..1]
        pnl_fb = _clip(pnl_sign - 1.25 * dd, -1.0, 1.0)
        comps["pnl_feedback"] = pnl_fb

        # ---- Extras (already scaled) ----
        for k, v in (inp.extra or {}).items():
            comps[f"extra:{k}"] = _clip(_safe(v), -1.0, 1.0)

        return comps

    def _blend(self, arr: Iterable[float]) -> float:
        xs = [x for x in arr if x is not None]
        if not xs:
            return 0.0
        # robust mean
        m = statistics.fmean(xs)
        return _clip(m, -1.0, 1.0)

    def _standardize(self, key: str, x: float) -> float:
        """
        Maintain a rolling mean/std per feature for z-normalization.
        """
        hist = self._roll.setdefault(key, [])
        if x == 0.0 and not hist:
            return 0.0
        hist.append(x)
        if len(hist) > 3000:
            del hist[: len(hist) - 3000]
        return _zscore(x, hist[:-1] if len(hist) > 1 else hist)

    def _decide_regime(self, s: float) -> str:
        """
        Regime bands with hysteresis:
          Bear:   s <= -0.35 - H
          Neutral: otherwise within [-0.15-H, +0.15+H]
          Bull:   s >= +0.35 + H
        """
        H = self.hysteresis
        last = self._last_regime
        if last == "bull":
            if s < +0.15 - H:
                last = "neutral"
            if s < -0.35 - H:
                last = "bear"
        elif last == "bear":
            if s > -0.15 + H:
                last = "neutral"
            if s > +0.35 + H:
                last = "bull"
        else:  # neutral
            if s >= +0.35 + H:
                last = "bull"
            elif s <= -0.35 - H:
                last = "bear"
        self._last_regime = last
        self._last_s = s
        return last

    def _regime_from_value(self, s: float) -> str:
        if s >= +0.35:
            return "bull"
        if s <= -0.35:
            return "bear"
        return "neutral"

    def _push_hist(self, ts_ms: int, raw: float, smooth: float) -> None:
        self._hist_ts.append(ts_ms)
        self._hist_raw.append(_clip(raw, -1.5, 1.5))
        self._hist_smooth.append(smooth)
        if len(self._hist_ts) > self.history_len:
            cut = len(self._hist_ts) - self.history_len
            del self._hist_ts[:cut]
            del self._hist_raw[:cut]
            del self._hist_smooth[:cut]


# =============================================================================
# Tiny demo
# =============================================================================

if __name__ == "__main__":
    mmi = MoodIndex(half_life_s=120.0, hysteresis=0.08)

    # Simulate 60 steps with noisy inputs
    import random
    ts = int(time.time() * 1000)
    for i in range(60):
        # toy: cycle from risk-off to risk-on
        phase = math.sin(i / 9.0)
        inp = MoodInputs(
            ts_ms=ts + i * 60_000,
            news_sentiment=0.6 * phase + 0.1 * random.uniform(-1, 1),
            social_sentiment=0.5 * phase + 0.2 * random.uniform(-1, 1),
            adv_decline_ratio=1.0 + 0.5 * phase + 0.1 * random.uniform(-1, 1),
            realized_vol=20.0 - 8.0 * phase + 0.5 * random.uniform(-1, 1),       # higher in risk-off
            credit_spread_bp=120.0 - 40.0 * phase + 2 * random.uniform(-1, 1),   # tighter in risk-on
            usd_dxy_ret=-0.01 * phase + 0.002 * random.uniform(-1, 1),
            gold_ret=-0.02 * phase + 0.003 * random.uniform(-1, 1),
            oil_ret=0.015 * phase + 0.005 * random.uniform(-1, 1),
            em_fx_ret=0.01 * phase + 0.003 * random.uniform(-1, 1),
            crypto_ret=0.03 * phase + 0.01 * random.uniform(-1, 1),
            etf_inflows_std=0.7 * phase,
            futures_positioning_std=0.3 * phase,
            pnl_rolling_dd=max(0.0, 0.2 - 0.2 * phase),
            pnl_sign=0.5 * phase,
        )
        r = mmi.update(inp)
        if i % 10 == 0:
            print(f"t={i:02d} | raw={r.composite_raw:+.3f} smooth={r.composite_smoothed:+.3f} regime={r.regime}")

    if pd:
        print(mmi.to_dataframe().tail())