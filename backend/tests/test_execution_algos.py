# tests/test_execution_algos.py
import importlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest # type: ignore

"""
What this covers
----------------
- TWAP: near-uniform slicing over the horizon; respects clip size
- POV: participation <= target by time; reacts to volume
- VWAP: executed avg price close to tape VWAP over horizon
- Implementation Shortfall (IS): produces a schedule; does not exceed basic risk/clip bounds

API flexibility
---------------
Each module can be one of:
  A) class Algo with .plan(parent, market, **cfg) -> list[child_orders]
  B) function plan(parent, market=None, **cfg) -> list[child_orders]
  C) function schedule(...) -> list[child_orders]

We try common module paths; edit IMPORT_PATHS if needed.

Parent order shape (flexible):
  { symbol, side: 'buy'|'sell', qty, start_ts, end_ts, venue?, clip_size? }

Child order shape (flexible):
  { ts, qty, side, symbol, price_limit? }

Tape shape:
  list of trades: { ts, px, sz } (ms, price, size)
"""

# -------------------------- Import helpers --------------------------

ALGO_IMPORT_PATHS = {
    "twap": [
        "backend.exec.twap",
        "backend.execution.twap",
        "backend.oms.twap",
        "execution.twap",
        "twap",
    ],
    "vwap": [
        "backend.exec.vwap",
        "backend.execution.vwap",
        "backend.oms.vwap",
        "execution.vwap",
        "vwap",
    ],
    "pov": [
        "backend.exec.pov",
        "backend.execution.pov",
        "backend.oms.pov",
        "execution.pov",
        "pov",
    ],
    "is": [
        "backend.exec.implementation_shortfall",
        "backend.execution.implementation_shortfall",
        "backend.oms.implementation_shortfall",
        "execution.implementation_shortfall",
        "implementation_shortfall",
        "impl_shortfall",
        "is_algo",
    ],
}

def _load_module(kind: str):
    last = None
    for modpath in ALGO_IMPORT_PATHS[kind]:
        try:
            return importlib.import_module(modpath)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import {kind.upper()} module from candidates {ALGO_IMPORT_PATHS[kind]} ({last})")

def _resolve_plan(mod):
    """
    Return a callable (parent, market, **cfg) -> list[children]
    """
    # Class with .plan
    if hasattr(mod, "Algo"):
        A = getattr(mod, "Algo")
        def _call(parent, market=None, **cfg):
            inst = A(**cfg) if _ctor_kw_ok(A) else A()
            if hasattr(inst, "plan"):
                return inst.plan(parent, market, **cfg)
            raise AttributeError("Algo class has no .plan(...)")
        return _call
    if hasattr(mod, "TWAP"):  # sometimes class named after algo
        A = getattr(mod, "TWAP")
        def _call(parent, market=None, **cfg):
            inst = A(**cfg) if _ctor_kw_ok(A) else A()
            return inst.plan(parent, market, **cfg)
        return _call
    # Functions
    for name in ("plan", "schedule", "build"):
        if hasattr(mod, name) and callable(getattr(mod, name)):
            fn = getattr(mod, name)
            def _call(parent, market=None, **cfg):
                try:
                    return fn(parent=parent, market=market, **cfg)
                except TypeError:
                    # some accept (parent, **cfg) or (parent, market_ctx)
                    try:
                        return fn(parent, market, **cfg)
                    except TypeError:
                        return fn(parent, **cfg)
            return _call
    pytest.skip("No Algo class or plan/schedule/build function exported")

def _ctor_kw_ok(cls) -> bool:
    try:
        import inspect
        sig = inspect.signature(cls)
        return any(p.kind in (p.VAR_KEYWORD, p.KEYWORD_ONLY) for p in sig.parameters.values())
    except Exception:
        return True


# -------------------------- Synthetic data --------------------------

@dataclass
class Parent:
    symbol: str = "AAPL"
    side: str = "buy"   # 'buy' | 'sell'
    qty: float = 100_000
    start_ts: int = 1_700_000_000_000
    end_ts: int = 1_700_000_360_000  # +6 min
    venue: Optional[str] = None
    clip_size: Optional[float] = None

def make_tape(parent: Parent, base_px=185.0, drift_bps=5.0, noise_bps=2.0, trades_per_min=60) -> List[Dict[str, float]]:
    """
    Deterministic pseudo-tape: slow drift + small oscillation; constant micro-lots.
    """
    tape: List[Dict[str, float]] = []
    minutes = max(1, int((parent.end_ts - parent.start_ts) / 60_000))
    step_ms = int(60_000 / trades_per_min)
    px = base_px
    t = parent.start_ts
    for i in range(minutes * trades_per_min):
        # deterministic "noise"
        delta = (drift_bps / 1e4) * base_px / (minutes * trades_per_min)
        osc = (((i % 10) - 5) / 5.0) * (noise_bps / 1e4) * base_px * 0.1
        px = max(0.01, px + delta + osc)
        tape.append({"ts": t, "px": px, "sz": 100 + (i % 5) * 25})
        t += step_ms
    return tape

def vwap_px(tape, t0, t1) -> float:
    notional = 0.0
    vol = 0.0
    for tr in tape:
        if t0 <= tr["ts"] <= t1:
            notional += tr["px"] * tr["sz"]
            vol += tr["sz"]
    return notional / vol if vol > 0 else float("nan")

def exec_price(schedule: List[Dict[str, Any]], tape: List[Dict[str, float]]) -> float:
    """
    Match each child to nearest-in-time trade on the tape for pricing; size-weighted.
    """
    if not schedule:
        return float("nan")
    tot_notional = 0.0
    tot_qty = 0.0
    for ch in schedule:
        ts = ch.get("ts") or ch.get("time") or ch.get("t")
        qty = float(ch.get("qty") or ch.get("size") or 0.0)
        if qty <= 0:
            continue
        # nearest trade
        nearest = min(tape, key=lambda tr: abs(tr["ts"] - ts)) # type: ignore
        px = nearest["px"]
        tot_notional += px * qty
        tot_qty += qty
    return tot_notional / tot_qty if tot_qty > 0 else float("nan")

def total_qty(schedule: List[Dict[str, Any]]) -> float:
    return sum(float(ch.get("qty") or ch.get("size") or 0.0) for ch in schedule)

def within_horizon(schedule: List[Dict[str, Any]], p: Parent) -> bool:
    for ch in schedule:
        ts = ch.get("ts") or ch.get("time") or ch.get("t")
        if ts is None or ts < p.start_ts or ts > p.end_ts:
            return False
    return True


# ------------------------------ Fixtures ------------------------------

@pytest.fixture()
def parent() -> Parent:
    return Parent()

@pytest.fixture()
def market_ctx(parent: Parent) -> Dict[str, Any]:
    return {
        "symbol": parent.symbol,
        "bid": 185.24,
        "ask": 185.26,
        "mid": 185.25,
        "tz": "UTC",
        "session": "REG",
    }

@pytest.fixture()
def tape(parent: Parent) -> List[Dict[str, float]]:
    return make_tape(parent, base_px=185.0, drift_bps=4.0, noise_bps=1.5, trades_per_min=60)


# ------------------------------- Tests --------------------------------

def test_twap_uniform_and_respects_clip(parent, market_ctx):
    mod = _load_module("twap")
    plan = _resolve_plan(mod)

    p = parent
    p.clip_size = 8_000  # enforce clips

    schedule = plan(parent=vars(p), market=market_ctx, clip_size=p.clip_size) # type: ignore
    assert isinstance(schedule, list) and len(schedule) > 0
    assert math.isclose(total_qty(schedule), p.qty, rel=0, abs=1e-6) # type: ignore
    assert within_horizon(schedule, p)

    # near-uniform slices: stddev / mean should be small
    sizes = [float(ch.get("qty") or ch.get("size") or 0.0) for ch in schedule]
    mean = sum(sizes) / len(sizes)
    var = sum((x - mean) ** 2 for x in sizes) / len(sizes)
    std = math.sqrt(var)
    assert std <= 0.25 * mean  # allow some rounding

    # respect clip
    assert max(sizes) <= p.clip_size + 1e-6


def test_pov_participation_cap(parent, market_ctx, tape):
    mod = _load_module("pov")
    plan = _resolve_plan(mod)

    target_pov = 0.1  # 10%
    p = Parent(qty=120_000, start_ts=parent.start_ts, end_ts=parent.end_ts, symbol=parent.symbol)

    # Build cumulative tape volume by time
    t0, t1 = p.start_ts, p.end_ts
    vol_by_time = {}
    cum = 0.0
    for tr in tape:
        if t0 <= tr["ts"] <= t1:
            cum += tr["sz"]
            vol_by_time[tr["ts"]] = cum
    total_tape_vol = cum
    assert total_tape_vol > 0

    schedule = plan(parent=vars(p), market=market_ctx, target_pov=target_pov) # type: ignore
    assert isinstance(schedule, list) and len(schedule) > 0
    assert within_horizon(schedule, p)

    # For every child at time t, cumulative child volume <= target * tape cumulative vol at t
    child_cum = 0.0
    for ch in sorted(schedule, key=lambda x: x.get("ts") or x.get("time") or 0):
        child_cum += float(ch.get("qty") or ch.get("size") or 0.0)
        ts = ch.get("ts") or ch.get("time")
        # find nearest tape ts not after 'ts'
        t_keys = [tt for tt in vol_by_time.keys() if tt <= ts]
        if not t_keys:
            continue
        near_ts = max(t_keys)
        pov_used = child_cum / max(1.0, vol_by_time[near_ts])
        assert pov_used <= target_pov + 0.02  # allow small slack


def test_vwap_tracks_tape_vwap(parent, market_ctx, tape):
    mod = _load_module("vwap")
    plan = _resolve_plan(mod)

    p = parent
    # If your VWAP uses an expected curve, we still validate ex-post vs actual tape
    schedule = plan(parent=vars(p), market=market_ctx) # type: ignore
    assert isinstance(schedule, list) and len(schedule) > 0
    assert within_horizon(schedule, p)
    assert math.isclose(total_qty(schedule), p.qty, rel=0, abs=1e-6) # type: ignore

    px_exec = exec_price(schedule, tape)
    px_vwap = vwap_px(tape, p.start_ts, p.end_ts)

    # Ensure executed avg price is reasonably close to tape VWAP
    # Drift/noise may introduce error; 10-20 bps tolerance is typical for synthetic tape
    bps_err = abs(px_exec - px_vwap) / px_vwap * 1e4
    assert bps_err <= 20.0, f"VWAP exec {px_exec:.4f} deviates {bps_err:.1f}bps from tape VWAP {px_vwap:.4f}"


def test_implementation_shortfall_outputs_schedule(parent, market_ctx):
    mod = _load_module("is")
    plan = _resolve_plan(mod)

    p = Parent(qty=60_000, start_ts=parent.start_ts, end_ts=parent.end_ts, symbol=parent.symbol)
    bench_px = 185.10  # arrival price benchmark
    schedule = plan(parent=vars(p), market=market_ctx, benchmark_px=bench_px, max_clip=12_000) # type: ignore
    assert isinstance(schedule, list) and len(schedule) > 0
    assert within_horizon(schedule, p)
    assert total_qty(schedule) == pytest.approx(p.qty)

    # Check clips respected if supported
    sizes = [float(ch.get("qty") or ch.get("size") or 0.0) for ch in schedule]
    assert max(sizes) <= 12_000 + 1e-6