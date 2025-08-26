# tests/test_latency.py
"""
Latency adapter tests (duck-typed)

Covers:
- Default/venue/path overrides from config/latency.yaml
- Jitter is applied within bounds
- Burst mode spikes (if supported)
- Loss probability drop path (if supported)
- No real sleeping: time.sleep() is monkeypatched

Expected adapter APIs (any one is fine):
  A) simulate_latency(adapter, venue, phase="order_send") -> sleeps OR returns ms
  B) LatencyModel(config_dict_or_path).simulate(adapter, venue, phase="order_send") -> sleeps/returns
"""

import os
import json
import time
import types
import pytest # type: ignore

yaml = pytest.importorskip("yaml", reason="pyyaml not installed")
la   = pytest.importorskip("backend.adapters.latency_adapter", reason="latency_adapter module not found")

CFG_PATH = "config/latency.yaml"

def _load_cfg():
    if not os.path.exists(CFG_PATH):
        pytest.skip(f"{CFG_PATH} not found")
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

# --- Small probe over adapter API --------------------------------------------

def _make_sim(cfg):
    # If adapter exposes a LatencyModel class, prefer it.
    if hasattr(la, "LatencyModel"):
        model = la.LatencyModel(cfg)
        def _sim(adapter, venue, phase):
            return model.simulate(adapter=adapter, venue=venue, phase=phase)
        return _sim
    # Else if file-level simulate_latency exists, use it directly
    if hasattr(la, "simulate_latency"):
        return lambda adapter, venue, phase: la.simulate_latency(adapter, venue, phase)
    # Else, look for a module-level object with simulate()
    lm = getattr(la, "LAT", None) or getattr(la, "latency", None)
    if lm and hasattr(lm, "simulate"):
        return lambda adapter, venue, phase: lm.simulate(adapter=adapter, venue=venue, phase=phase)
    pytest.skip("No simulate API found (LatencyModel.simulate or simulate_latency).")

# --- Monkeypatch helpers ------------------------------------------------------

class SleepRecorder:
    def __init__(self):
        self.calls = []
    def __call__(self, secs: float):
        # record, don't actually sleep
        self.calls.append(secs)

class RNGFixed:
    """Deterministic RNG shim to control jitter/burst/loss behavior."""
    def __init__(self, uniform_seq=None, random_seq=None):
        self.uniform_seq = list(uniform_seq or [0.0])  # jitter offset
        self.random_seq = list(random_seq or [1.0])    # prob checks (1.0 = never triggers)
    def uniform(self, a, b):
        if self.uniform_seq:
            x = self.uniform_seq.pop(0)
        else:
            x = 0.0
        # map x in [-1,1] to [a,b] if we passed normalized; else assume x is absolute
        # Here we assume x is absolute offset; just clamp
        return max(a, min(b, x))
    def random(self):
        if self.random_seq:
            return self.random_seq.pop(0)
        return 1.0

@pytest.fixture
def cfg():
    return _load_cfg()

@pytest.fixture
def sim(cfg):
    return _make_sim(cfg)

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    rec = SleepRecorder()
    monkeypatch.setattr(time, "sleep", rec, raising=True)
    return rec

# Try to patch random.* used inside adapter, if it imports Python's stdlib random
@pytest.fixture
def rng_control(monkeypatch):
    try:
        import random as stdrand
    except Exception:
        pytest.skip("random module unavailable?")
    rng = RNGFixed()
    monkeypatch.setattr(stdrand, "uniform", rng.uniform, raising=True)
    monkeypatch.setattr(stdrand, "random", rng.random, raising=True)
    return rng

# --- Tests --------------------------------------------------------------------

def _ms_from_sleep(recorder):
    """Sum recorded sleeps (in seconds) and convert to ms."""
    total_ms = sum(recorder.calls) * 1000.0
    return total_ms

def test_path_override_for_zerodha_nse(cfg, sim, no_sleep, rng_control):
    """
    Validate that path override for Zerodha→NSE 'order_send' is honored.
    In sample config: ~28 ms + jitter.
    We force jitter=0 via rng_control.uniform sequence.
    """
    # Force jitter 0
    rng_control.uniform_seq = [0.0]
    base = cfg["defaults"]["order_send"]
    # Validate config contains the path override we expect
    expect = None
    for path in cfg.get("paths", []):
        if path.get("adapter") == "zerodha" and path.get("venue") == "NSE":
            expect = path.get("order_send", None)
            break
    if expect is None:
        pytest.skip("No Zerodha/NSE path override in config; skipping")

    sim("zerodha", "NSE", "order_send")
    slept_ms = _ms_from_sleep(no_sleep)
    assert abs(slept_ms - float(expect)) < 1.0, f"expected ~{expect}ms, got {slept_ms:.2f}ms" # type: ignore

def test_default_vs_venue_override(cfg, sim, no_sleep, rng_control):
    """
    Compare default 'venue_roundtrip' vs NASDAQ override.
    """
    rng_control.uniform_seq = [0.0, 0.0]
    default_vrt = float(cfg["defaults"]["venue_roundtrip"])
    nasdaq_vrt = float(cfg["venues"]["NASDAQ"]["venue_roundtrip"])

    # Generic (unknown) venue → expect default
    sim("ibkr", "UNKNOWN", "venue_roundtrip")
    ms_default = _ms_from_sleep(no_sleep)
    # Reset recorder
    no_sleep.calls.clear()

    # NASDAQ → expect override
    sim("ibkr", "NASDAQ", "venue_roundtrip")
    ms_nasdaq = _ms_from_sleep(no_sleep)

    assert abs(ms_default - default_vrt) < 1.0
    assert abs(ms_nasdaq - nasdaq_vrt) < 1.0
    assert ms_nasdaq != ms_default

def test_jitter_bounds(cfg, sim, no_sleep, rng_control):
    """
    Jitter should stay within ±jitter_ms from config defaults unless path/venue overrides it.
    We'll run twice with -jitter and +jitter.
    """
    j = float(cfg["defaults"]["jitter_ms"])
    base = float(cfg["defaults"]["order_send"])
    # First call: -jitter
    rng_control.uniform_seq = [-j]
    sim("paper", "NYSE", "order_send")
    ms1 = _ms_from_sleep(no_sleep); no_sleep.calls.clear()

    # Second call: +jitter
    rng_control.uniform_seq = [ +j ]
    sim("paper", "NYSE", "order_send")
    ms2 = _ms_from_sleep(no_sleep); no_sleep.calls.clear()

    assert base - j - 0.5 <= ms1 <= base + 0.5
    assert base - 0.5 <= ms2 <= base + j + 0.5
    assert ms2 - ms1 >= j - 1.0  # roughly the spread

def test_burst_mode_spike_if_supported(cfg, sim, no_sleep, rng_control):
    """
    If simulation.burst_mode.enabled, and adapter uses random.random() < prob,
    we force a spike and assert the extra spike_ms is added.
    """
    sim_cfg = cfg.get("simulation", {}).get("burst_mode", {})
    if not sim_cfg or not sim_cfg.get("enabled", False):
        pytest.skip("Burst mode disabled in config; skipping")

    spike_ms = float(sim_cfg.get("spike_ms", 0))
    if spike_ms <= 0:
        pytest.skip("No spike_ms configured; skipping")

    # Force jitter 0, and force random.random() below prob to trigger spike
    rng_control.uniform_seq = [0.0]
    rng_control.random_seq = [0.0]  # always trigger

    # Use a known path value to compute expected
    base = float(cfg["defaults"]["order_send"])
    sim("ibkr", "NYSE", "order_send")
    ms = _ms_from_sleep(no_sleep)

    assert ms >= base + spike_ms - 0.5, f"expected >= base+spike ({base}+{spike_ms}), got {ms:.2f}ms"

def test_loss_probability_drop_path_if_supported(cfg, sim, no_sleep, rng_control):
    """
    If adapter supports 'loss_prob' dropping messages, we simulate a 'send' phase
    and assert it returns quickly (no sleep) or marks a drop.
    As APIs vary, we only check that when random() < loss_prob, the sleep does NOT equal normal path.
    """
    loss_prob = float(cfg["defaults"].get("loss_prob", 0.0))
    if loss_prob <= 0:
        pytest.skip("loss_prob not configured; skipping")

    # Set random() to definitely trigger the drop
    rng_control.random_seq = [0.0]
    rng_control.uniform_seq = [0.0]  # zero jitter

    sim("paper", "NASDAQ", "order_send")
    ms = _ms_from_sleep(no_sleep)

    # If adapter skips sleeping on drop, ms ~ 0; otherwise it may mark differently.
    # We only require it's not the normal base latency.
    base = float(cfg["defaults"]["order_send"])
    assert abs(ms - base) > 1.0, "loss-triggered path should not behave like normal latency"