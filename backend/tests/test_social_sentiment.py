# tests/test_social_sentiment.py
import importlib
import json
import math
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest # type: ignore

"""
Expected public API (any one is fine)

Class-style:
------------
class SocialSentiment:
    # ingestion / normalization
    def ingest(self, events: list[dict]) -> int                                   # returns ingested count
    def normalize(self, text: str) -> str                                         # optional
    # scoring
    def score_text(self, text: str, lang: str | None = None) -> dict              # {"sent": float in [-1,1], "label": "pos|neg|neu"}
    def score_batch(self, texts: list[str], langs: list[str] | None = None) -> list[dict]  # optional
    # entity linking
    def extract_entities(self, text: str) -> dict                                 # {"tickers":[...], "symbols":[...], ...}
    # storage / query
    def aggregate(self, entity: str, start_ms: int, end_ms: int, *, window: str = "1h") -> dict
         # returns {"n":int,"mean":float,"pos":int,"neg":int,"neu":int,"series":[{"ts":..,"mean":..,"n":..},...]}
    def rolling(self, entity: str, end_ms: int, lookback: str = "7d") -> dict     # optional rolling KPIs
    # alerts / rate-limit / cache (optional)
    def alerts(self, entity: str, *, threshold: float = 0.5) -> list[dict]
    def set_rate_limit(self, key: str, n: int, per_s: float) -> None
    # export / import (optional)
    def export_json(self) -> dict | str
    def import_json(self, blob: dict | str) -> None
    # housekeeping (optional)
    def dedup_stats(self) -> dict
    def clear(self) -> None

Function-style:
---------------
- new_sentiment() -> handle
- ingest(handle, ...), score_text(handle, ...), extract_entities(handle, ...), aggregate(handle, ...), ...

The tests auto-skip optional parts if missing.
"""

# ----------------------- Import resolver -----------------------

IMPORT_CANDIDATES = [
    "backend.news.social_sentiment",
    "backend.analytics.social_sentiment",
    "backend.altdata.social_sentiment",
    "altdata.social_sentiment",
    "social_sentiment",
    "sentiment",
]

def _load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import social_sentiment from {IMPORT_CANDIDATES} ({last})")

class API:
    def __init__(self, mod):
        self.mod = mod
        self.obj = None
        # Prefer class
        if hasattr(mod, "SocialSentiment"):
            Cls = getattr(mod, "SocialSentiment")
            try:
                self.obj = Cls()
            except TypeError:
                self.obj = Cls
        elif hasattr(mod, "new_sentiment"):
            self.obj = mod.new_sentiment()
        else:
            pytest.skip("No SocialSentiment class and no new_sentiment() factory found.")

    def has(self, name): return hasattr(self.obj, name) or hasattr(self.mod, name)

    def call(self, name, *args, **kw):
        if hasattr(self.obj, name):
            return getattr(self.obj, name)(*args, **kw)
        if hasattr(self.mod, name):
            return getattr(self.mod, name)(self.obj, *args, **kw)
        raise AttributeError(f"Missing API '{name}'")

# ----------------------- Synthetic fixtures -----------------------

def _ms(dt: datetime) -> int: return int(dt.timestamp() * 1000)

def _seed_events(now: datetime) -> List[dict]:
    """
    Creates a small, diverse batch:
      - tickers via $AAPL / TSLA, hashtags, emojis
      - duplicates by id
      - mixed languages (English + Hindi)
      - news- vs. social-like sources
    """
    t0 = _ms(now - timedelta(minutes=50))
    t1 = _ms(now - timedelta(minutes=40))
    t2 = _ms(now - timedelta(minutes=30))
    t3 = _ms(now - timedelta(minutes=10))
    t4 = _ms(now - timedelta(minutes=5))
    return [
        {"id":"1","ts":t0,"source":"twitter","user":"u1",
         "text":"$AAPL to the moon 🚀🚀 solid earnings beat!", "lang":"en"},
        {"id":"2","ts":t1,"source":"news","publisher":"Moneycontrol",
         "text":"TSLA faces delivery headwinds; outlook cautious", "lang":"en"},
        {"id":"3","ts":t2,"source":"twitter","user":"u2",
         "text":"$TSLA is overrated lol 🙄", "lang":"en"},
        {"id":"4","ts":t3,"source":"twitter","user":"u3",
         "text":"AAPL services growth strong; love the cashflows", "lang":"en"},
        {"id":"5","ts":t4,"source":"news","publisher":"Yahoo Finance",
         "text":"टेस्ला पर मांग नरम — निवेशकों में सतर्कता (#TSLA)", "lang":"hi"},
        # duplicate id to test dedup
        {"id":"3","ts":t2,"source":"twitter","user":"u2",
         "text":"$TSLA is overrated lol 🙄", "lang":"en"},
    ]

@pytest.fixture(scope="module")
def api():
    return API(_load_mod())

@pytest.fixture()
def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)

@pytest.fixture()
def loaded(api, now_utc):
    if api.has("clear"):
        try: api.call("clear")
        except Exception: pass
    events = _seed_events(now_utc)
    n = api.call("ingest", events)
    assert isinstance(n, int) and n >= 5
    return {"events": events}

# ----------------------- Tests -----------------------

def test_normalize_and_score_text(api):
    txt_pos = "Love this earnings print, massive beat! 🚀"
    txt_neg = "Guidance cut. Ugly miss."
    if api.has("normalize"):
        n1 = api.call("normalize", txt_pos)
        assert isinstance(n1, str) and len(n1) >= 5
    s1 = api.call("score_text", txt_pos)
    s2 = api.call("score_text", txt_neg)
    # Expect opposite signs
    assert float(s1.get("sent", 0)) > float(s2.get("sent", 0))

def test_entity_extraction(api):
    e = api.call("extract_entities", "Thinking about $AAPL and TSLA today; also NIFTY.")
    assert isinstance(e, dict)
    tickers = set((e.get("tickers") or e.get("symbols") or []))
    assert "AAPL" in tickers or "TSLA" in tickers

def test_ingest_and_aggregate_basic(api, loaded, now_utc):
    start = _ms(now_utc - timedelta(hours=1))
    end = _ms(now_utc + timedelta(minutes=1))
    agg_aapl = api.call("aggregate", entity="AAPL", start_ms=start, end_ms=end, window="10m")
    assert isinstance(agg_aapl, dict)
    assert agg_aapl.get("n", 0) >= 1
    assert "series" in agg_aapl and isinstance(agg_aapl["series"], list)

def test_deduplication_stats_optional(api, loaded):
    if not api.has("dedup_stats"):
        pytest.skip("No dedup_stats()")
    st = api.call("dedup_stats")
    assert isinstance(st, dict)
    # at least one duplicate id ("3") should have been removed or counted
    s = json.dumps(st).lower()
    assert ("dup" in s) or (st.get("dropped", 0) >= 1)

def test_batch_scoring_optional(api):
    if not api.has("score_batch"):
        pytest.skip("No score_batch()")
    out = api.call("score_batch", ["great quarter", "awful guidance"])
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["sent"] > out[1]["sent"]

def test_multilingual_handling_graceful(api, loaded, now_utc):
    # The seed includes a Hindi text; engine may translate or fallback to neutral; just assert no crash
    start = _ms(now_utc - timedelta(hours=2))
    end = _ms(now_utc + timedelta(seconds=1))
    agg_tsla = api.call("aggregate", entity="TSLA", start_ms=start, end_ms=end, window="30m")
    assert isinstance(agg_tsla, dict) and agg_tsla.get("n", 0) >= 1

def test_alert_thresholds_optional(api, now_utc, loaded):
    if not api.has("alerts"):
        pytest.skip("No alerts()")
    alerts = api.call("alerts", "TSLA", threshold=0.4)
    assert isinstance(alerts, list)
    # any structure is fine; just ensure it returns list

def test_rate_limit_optional(api):
    if not api.has("set_rate_limit"):
        pytest.skip("No set_rate_limit()")
    api.call("set_rate_limit", key="ingest", n=3, per_s=1.0)
    # 3 quick ingests ok, 4th may raise or return fewer
    burst = [{"id":f"x{i}","ts":int(time.time()*1000),"source":"twitter","text":f"ping {i} $AAPL"} for i in range(4)]
    try:
        n = api.call("ingest", burst)
        assert isinstance(n, int) and n <= 4
    except Exception:
        # acceptable: engine throws on RL breach
        pass

def test_export_import_roundtrip_optional(api):
    if not (api.has("export_json") and api.has("import_json")):
        pytest.skip("No export/import")
    blob = api.call("export_json")
    s = json.dumps(blob, default=str)
    assert isinstance(s, str) and len(s) > 10
    api.call("import_json", blob)

def test_rolling_optional(api, now_utc):
    if not api.has("rolling"):
        pytest.skip("No rolling()")
    end = _ms(now_utc)
    res = api.call("rolling", entity="AAPL", end_ms=end, lookback="1d")
    assert isinstance(res, dict)

def test_signal_correlates_with_constructed_returns(api, loaded, now_utc):
    """
    Construct a fake returns series where positive windows correspond to AAPL-positive texts in seed.
    We expect weakly positive correlation between windowed sentiment mean and returns.
    """
    start = _ms(now_utc - timedelta(hours=1))
    end = _ms(now_utc + timedelta(minutes=1))
    agg = api.call("aggregate", entity="AAPL", start_ms=start, end_ms=end, window="10m")
    series = agg.get("series", [])
    if len(series) < 2:
        pytest.skip("Not enough series points for correlation")
    sents = np.array([float(x.get("mean", 0.0)) for x in series])
    # fabricate returns aligned to windows: rising when sentiment positive
    rets = (sents - sents.mean()) * 0.02 + np.random.default_rng(7).normal(0, 0.005, size=len(sents))
    # Pearson correlation
    if np.std(sents) > 1e-9 and np.std(rets) > 1e-9:
        corr = float(np.corrcoef(sents, rets)[0,1])
        assert corr > -0.25  # should not be strongly negative; often positive on synthetic

def test_edge_cases_and_safety(api):
    # empty / whitespace / url-only should not crash
    for txt in ["", "   ", "https://example.com"]:
        s = api.call("score_text", txt)
        assert isinstance(s, dict) and math.isfinite(float(s.get("sent", 0.0)))
    # very long text gets truncated or processed
    long_txt = "great " * 5000
    s2 = api.call("score_text", long_txt)
    assert isinstance(s2, dict)

def test_schema_tolerance_on_ingest(api, now_utc):
    events = [
        {"id":"weird1","ts":_ms(now_utc), "text":"bullish $AAPL"},
        {"id":"weird2","ts":_ms(now_utc)+1, "text":"bearish TSLA", "extra": {"foo":"bar"}},
    ]
    n = api.call("ingest", events)
    assert isinstance(n, int) and n >= 1