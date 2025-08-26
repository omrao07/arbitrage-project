# backend/io/signals_adapter.py
"""
SignalsAdapter: unify, transform, and publish signals.

Why this exists
---------------
- You have many feeds (altdata, social, health, climate, synthetic).
- Each emits dict-like payloads at different cadences.
- Downstream agents need a single {key: float} snapshot with stable names.

What it does
------------
1) Pull: gather from pluggable sources (dicts, callables, file loaders).
2) Transform (optional):
   - drop/keep filters
   - rename keys (mapping or prefixing)
   - scaling (×a + b), abs(), clip(lo, hi)
   - z-score standardization using OnlineStats (rolling)
3) Mix: combine sources via backend.common.mixer.SignalMixer.
4) Publish: send the final dict to a sink (e.g., signal_bus.publish).

Zero external deps by default; YAML loading is enabled if pyyaml is present.

Typical usage
-------------
from backend.io.signals_adapter import (
    SignalsAdapter, SourceSpec, TransformSpec, MixerHook
)
from backend.common.mixer import MixerConfig
from backend.common.predictor import OnlineStats

adapter = SignalsAdapter(
    sources=[
        SourceSpec(name="altdata", kind="dict", payload={"btc_tx": 1.2, "AAPL.web": 0.6}),
        SourceSpec(name="sentiment", kind="callable", payload=lambda: {"BTC.sent": 0.35, "AAPL.sent": 0.12}),
        SourceSpec(name="file_yaml", kind="yaml", path="configs/sentiment.yaml", prefix=None),
    ],
    transforms={
        "altdata": TransformSpec(prefix="alt.", clip=(-5, 5)),
        "sentiment": TransformSpec(prefix="nlp.", scale=(1.0, 0.0)),
    },
    mixer=MixerHook(config=MixerConfig(
        mode="linear",
        weights={"altdata": 0.5, "sentiment": 0.5},
        norm="z", winsor=(0.05, 0.95), clip=(-3, 3)
    )),
    publisher=lambda d: signal_bus_publish(d)   # your func
)
snapshot = adapter.snapshot(now_ts=time.time(), with_raw=True)
"""

from __future__ import annotations

import json
import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Optional YAML support
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# Optional mixer
try:
    from backend.common.mixer import SignalMixer, MixerConfig # type: ignore
    _HAVE_MIXER = True
except Exception:
    _HAVE_MIXER = False
    SignalMixer = object  # type: ignore
    class MixerConfig:  # type: ignore
        def __init__(self, **kw): pass

# Optional stats for z-score
try:
    from backend.common.predictor import OnlineStats # type: ignore
except Exception:
    @dataclass
    class OnlineStats:  # minimal fallback
        n: int = 0
        mean: float = 0.0
        _m2: float = 0.0
        def update(self, x: float) -> None:
            self.n += 1
            d = x - self.mean
            self.mean += d / self.n
            self._m2 += d * (x - self.mean)
        @property
        def std(self) -> float:
            if self.n <= 1: return 0.0
            var = self._m2 / (self.n - 1)
            return math.sqrt(max(0.0, var))


Number = Union[int, float]
SignalMap = Dict[str, Number]


# ----------------------------- Specs -----------------------------------

@dataclass
class SourceSpec:
    """
    Declare a source.
    kind:
      - "dict": payload is a dict or a callable returning dict
      - "callable": payload() -> dict
      - "json": path points to a JSON file with a dict
      - "yaml": path points to a YAML file with a dict
    """
    name: str
    kind: str
    payload: Any = None
    path: Optional[str] = None
    prefix: Optional[str] = None
    debounce_sec: float = 0.0          # cache duration; 0 disables
    required: bool = False             # if True, errors bubble up


@dataclass
class TransformSpec:
    """Per-source transforms applied AFTER loading raw map."""
    keep: Optional[List[str]] = None        # whitelist substrings or exact keys
    drop: Optional[List[str]] = None        # blacklist substrings or exact keys
    rename: Optional[Dict[str, str]] = None # exact key rename map
    prefix: Optional[str] = None            # add "prefix." to each key
    scale: Tuple[float, float] = (1.0, 0.0) # (a, b): v' = a*v + b
    absolute: bool = False
    clip: Optional[Tuple[float, float]] = None
    zscore: bool = False                    # standardize per-key with OnlineStats
    z_eps: float = 1e-6


@dataclass
class MixerHook:
    """Wrap SignalMixer (optional)."""
    config: MixerConfig
    _mx: Optional[SignalMixer] = field(default=None, init=False) # type: ignore
    def get(self) -> SignalMixer: # type: ignore
        if not _HAVE_MIXER:
            raise RuntimeError("backend.common.mixer not available")
        if self._mx is None:
            self._mx = SignalMixer(self.config) # type: ignore
        return self._mx


# ----------------------------- Adapter ---------------------------------

@dataclass
class SignalsAdapter:
    sources: List[SourceSpec]
    transforms: Dict[str, TransformSpec] = field(default_factory=dict)
    mixer: Optional[MixerHook] = None
    publisher: Optional[Callable[[SignalMap], None]] = None

    # runtime
    _cache: Dict[str, Tuple[float, SignalMap]] = field(default_factory=dict, init=False)
    _zstats: Dict[str, OnlineStats] = field(default_factory=dict, init=False)  # per-key zscore stats

    # ---- public API ----

    def snapshot(self, *, now_ts: Optional[float] = None, with_raw: bool = False) -> Dict[str, Any]:
        """
        Pull -> transform -> (optionally mix) -> publish.
        Returns {"ts":..., "signals": final_map, "raw": per_source_raw?}
        """
        now = float(now_ts if now_ts is not None else time.time())
        per_src: Dict[str, SignalMap] = {}
        timestamps: Dict[str, float] = {}

        # Load + transform per source
        for spec in self.sources:
            try:
                raw = self._load_source(spec, now)
                xform = self.transforms.get(spec.name)
                cooked = self._apply_transform(raw, xform, now)
                per_src[spec.name] = cooked
                timestamps[spec.name] = now
            except Exception as e:
                if spec.required:
                    raise
                # else, skip source
                per_src[spec.name] = {}
                timestamps[spec.name] = now

        # Mix or merge
        if self.mixer is not None:
            mx = self.mixer.get()
            merged = mx.mix(per_src, timestamps=timestamps, now_ts=now)
        else:
            # simple union (later sources overwrite)
            merged = {}
            for name in [s.name for s in self.sources]:
                merged.update(per_src.get(name, {}))

        # Publish
        if self.publisher:
            try:
                self.publisher(merged)
            except Exception:
                pass

        out = {"ts": now, "signals": merged}
        if with_raw:
            out["raw"] = per_src
        return out

    # ---- loaders ----

    def _load_source(self, spec: SourceSpec, now: float) -> SignalMap:
        # cache
        if spec.debounce_sec > 0:
            t_last, cached = self._cache.get(spec.name, (0.0, {}))
            if now - t_last < spec.debounce_sec:
                return dict(cached)

        if spec.kind == "dict":
            data = spec.payload() if callable(spec.payload) else (spec.payload or {})
            out = self._coerce_map(data, prefix=spec.prefix)
        elif spec.kind == "callable":
            data = spec.payload()  # must return dict
            out = self._coerce_map(data, prefix=spec.prefix)
        elif spec.kind == "json":
            with open(spec.path or "", "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            out = self._coerce_map(data, prefix=spec.prefix)
        elif spec.kind == "yaml":
            if not _HAVE_YAML:
                raise RuntimeError("pyyaml not installed; cannot load YAML")
            with open(spec.path or "", "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            out = self._coerce_map(data, prefix=spec.prefix)
        else:
            raise ValueError(f"unknown source kind '{spec.kind}'")

        if spec.debounce_sec > 0:
            self._cache[spec.name] = (now, dict(out))
        return out

    @staticmethod
    def _coerce_map(obj: Any, *, prefix: Optional[str] = None) -> SignalMap:
        """
        Flatten nested {group:{k:v}} into {"group.k": v} and coerce to float.
        Non-numeric values are dropped.
        """
        flat: SignalMap = {}

        def walk(base: Optional[str], x: Any) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    kk = f"{base}.{k}" if base else str(k)
                    walk(kk, v)
            else:
                try:
                    flat[str(base)] = float(x)  # type: ignore[arg-type]
                except Exception:
                    # drop non-numeric leaves
                    pass

        if isinstance(obj, dict):
            walk(None, obj)
        else:
            # try to interpret as already-flat mapping sequence
            try:
                for k, v in dict(obj).items():  # type: ignore
                    flat[str(k)] = float(v)
            except Exception:
                pass

        if prefix:
            flat = {f"{prefix}{k}": v for k, v in flat.items()}
        return flat

    # ---- transforms ----

    def _apply_transform(self, m: SignalMap, spec: Optional[TransformSpec], now: float) -> SignalMap:
        if not spec:
            return dict(m)

        out: SignalMap = dict(m)

        # keep/drop
        if spec.keep is not None:
            keep_set = set(spec.keep)
            out = {k: v for k, v in out.items() if (k in keep_set or any(p in k for p in keep_set))}
        if spec.drop is not None:
            drop_set = set(spec.drop)
            out = {k: v for k, v in out.items() if not (k in drop_set or any(p in k for p in drop_set))}

        # rename
        if spec.rename:
            out = {spec.rename.get(k, k): v for k, v in out.items()}

        # prefix
        if spec.prefix:
            out = {f"{spec.prefix}{k}": v for k, v in out.items()}

        # scale
        a, b = spec.scale
        if a != 1.0 or b != 0.0:
            out = {k: (a * v + b) for k, v in out.items()}

        # absolute
        if spec.absolute:
            out = {k: abs(v) for k, v in out.items()}

        # clip
        if spec.clip is not None:
            lo, hi = spec.clip
            out = {k: max(lo, min(hi, v)) for k, v in out.items()}

        # z-score (online, per-key)
        if spec.zscore:
            zed: SignalMap = {}
            for k, v in out.items():
                st = self._zstats.get(k)
                if st is None:
                    st = OnlineStats(); self._zstats[k] = st
                st.update(float(v))
                sd = st.std or spec.z_eps
                zed[k] = (float(v) - st.mean) / (sd if sd > 0 else spec.z_eps)
            out = zed

        return out


# ----------------------------- Tiny demo --------------------------------

if __name__ == "__main__":
    def fake_bus_publish(d: Dict[str, float]) -> None:
        print("[bus] published", len(d), "signals. sample:", list(d.items())[:5])

    sadp = SignalsAdapter(
        sources=[
            SourceSpec(name="alt", kind="dict", payload={"AAPL.web": 0.6, "BTC.tx": 1.2}, debounce_sec=0.0),
            SourceSpec(name="nlp", kind="dict", payload={"BTC.sent": 0.35, "AAPL.sent": 0.12}),
            SourceSpec(name="syn", kind="dict", payload=lambda: {"EURUSD.carry": 0.015, "risk_z": 0.4}),
        ],
        transforms={
            "alt": TransformSpec(prefix="alt.", clip=(-5, 5)),
            "nlp": TransformSpec(prefix="nlp.", zscore=True),
            "syn": TransformSpec(),
        },
        mixer=MixerHook(config=MixerConfig(
            mode="linear",
            weights={"alt": 0.5, "nlp": 0.3, "syn": 0.2},
            norm="z", winsor=(0.05, 0.95), clip=(-3, 3)
        )),
        publisher=fake_bus_publish
    )
    snap = sadp.snapshot(with_raw=True)
    print(json.dumps({"ts": snap["ts"], "n_signals": len(snap["signals"])}, indent=2))