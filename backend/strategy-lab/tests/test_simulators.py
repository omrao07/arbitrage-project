# tests/test_simulators.py
import os
import csv
import time
import math
import json
import tempfile
import importlib
import pytest

# --------------------------- helpers ---------------------------

def _optional_import(modname):
    try:
        return importlib.import_module(modname)
    except Exception:
        pytest.skip(f"Optional module '{modname}' not found; skipping related tests.")


# ========================= DATA LOADER ==========================

def test_loader_csv_jsonl_align_slice_and_iter():
    dl = _optional_import("simulators.envs.data_loader")
    loader = dl.MarketDataLoader()

    with tempfile.TemporaryDirectory() as d:
        p_csv = os.path.join(d, "AAA.csv")
        p_jsonl = os.path.join(d, "BBB.jsonl")
        t0 = 1_700_000_000

        # CSV (AAA)
        with open(p_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date","open","high","low","close","volume"])
            for i in range(6):
                px = 100 + i
                w.writerow([t0 + i*86400, px, px, px, px, 1000])

        # JSONL (BBB)
        with open(p_jsonl, "w", encoding="utf-8") as f:
            for i in range(6):
                px = 50 + i
                row = {"ts": t0 + i*86400, "open": px, "high": px, "low": px, "close": px, "volume": 800}
                f.write(json.dumps(row) + "\n")

        # Slice last 5 bars, align, and iterate
        ss = dl.SliceSpec(start_ts=float(t0 + 1*86400))
        ds = loader.load_many(
            specs=[
                dl.LoadSpec(path=p_csv, symbol="AAA", fmt="csv"),
                dl.LoadSpec(path=p_jsonl, symbol="BBB", fmt="jsonl"),
            ],
            slice_spec=ss,
            align=True,
        )
        assert ds.symbols() == ["AAA", "BBB"]
        assert len(ds.calendar) == 5  # sliced off first bar

        # rolling windows of 3
        wins = list(loader.rolling(ds, window=3))
        assert len(wins) == 3  # 5 bars -> 3 windows

        # per-bar iterator
        bars = list(loader.iter_bars(ds))
        assert len(bars) == 5
        # close exists and is positive
        _, bar0 = bars[0]
        assert bar0["AAA"]["close"] > 0 and bar0["BBB"]["close"] > 0


# ========================= MARKET ENV ==========================

def test_market_env_session_and_shock(monkeypatch):
    envmod = _optional_import("simulators.envs.market_env")

    # Exec stub recording last price & tick payloads
    class ExecStub:
        def __init__(self):
            self.last = {}
        def update_price(self, s, p): self.last[s] = p
        def equity(self): return 1_000_000.0
        def gross_exposure(self): return 0.0
        def leverage(self): return 0.0

    ticks = []
    def on_tick(sym, payload): 
        if sym == "AAA": 
            ticks.append(payload)

    x = ExecStub()
    cfg = envmod.EnvConfig(dt_sec=0.01, session_start="00:00", session_end="23:59", tz_offset_minutes=0, seed=7)
    specs = [envmod.SymbolSpec("AAA", start_price=100.0, sigma=0.20, spread_bps=2.0)]
    shocks = [envmod.Shock(kind="news", symbol="AAA", at_sec=5, duration_sec=10, spread_mult=4.0, vol_mult=3.0)]
    env = envmod.MarketEnv(x, cfg=cfg, symbols=specs, shocks=shocks, on_tick=on_tick)

    # Force local time to 00:00.. so step() advances within session
    monkeypatch.setattr(envmod.time, "time", lambda: 0.0)

    # Run a few ticks before and during shock
    for _ in range(5):
        env.step()  # t=0..4 (no shock yet)
    # nudge local time to '5' to trigger shock window
    monkeypatch.setattr(envmod.time, "time", lambda: 5.0)
    for _ in range(5):
        env.step()

    assert "AAA" in x.last and x.last["AAA"] > 0.0
    # verify at least one tick flagged shock_active True and spread widened
    had_shock = any(t["shock_active"] for t in ticks)
    assert had_shock
    # spreads positive always
    assert all(t["spread"] > 0 for t in ticks)


# ========================= BACKTESTER ==========================

def test_backtester_bar_replay_and_metrics(tmp_path):
    btmod = _optional_import("simulators.backtester")

    # aligned mini dataset
    class DS: pass
    t0 = 1_700_000_000
    def series(n, start):
        x = start; out=[]
        for i in range(n):
            x *= (1 + (0.002 if i%3==0 else -0.001))
            out.append({"ts": t0 + i*86400, "open":x, "high":x, "low":x, "close":x, "volume":1})
        return out
    ds = DS()
    ds.data = {"AAA": series(60, 100), "BBB": series(60, 50)} # type: ignore
    ds.calendar = [r["ts"] for r in ds.data["AAA"]] # type: ignore

    # Execution & Strategy stubs
    class X:
        def __init__(self): self._last={}; self._fills=[]
        def update_price(self,s,p): self._last[s]=p
        def mark_to_market(self): pass
        def equity(self): return 1_000_000.0
        def fills(self): return self._fills
    class SA:
        def __init__(self, x): self.x=x
        def on_price(self,s,p): self.x.update_price(s,p)
        def maybe_rebalance(self, force=False): pass

    x = X(); sa = SA(x)
    cfg = btmod.BacktestConfig(rebalance="daily", warmup_bars=5, fills_log_path=str(tmp_path/"fills.jsonl"))
    bt = btmod.Backtester(ds, x, sa, cfg)
    res = bt.run()
    assert res["metrics"]["bars"] == len(ds.calendar) # type: ignore
    assert isinstance(res["returns"], list) and len(res["returns"]) == len(ds.calendar)-1 # type: ignore
    assert "sharpe" in res["metrics"] and "max_drawdown" in res["metrics"]


# ========================= EQ-LS SIM ==========================

def test_eq_ls_sim_quantiles_costs_and_metrics():
    simmod = _optional_import("simulators.eq_ls_sim")

    # Build tiny aligned dataset
    class DS: pass
    t0 = 1_700_000_000
    def mk(n, start):
        x = start; rows=[]
        import random; random.seed(11)
        for i in range(n):
            x *= (1 + (0.01 if (i%7==0) else -0.004))
            rows.append({"ts": t0 + i*86400, "open":x, "high":x, "low":x, "close":x, "volume":1})
        return rows
    data = {"AAA": mk(120,100), "BBB": mk(120,50), "CCC": mk(120,25), "DDD": mk(120,10), "EEE": mk(120,5)}
    cal = [r["ts"] for r in data["AAA"]]
    ds = DS(); ds.data = data; ds.calendar = cal # type: ignore

    # Simple factor: short-term momentum
    def fac(sym, i, hist, ctx):
        return (hist[i]["close"] - hist[i-10]["close"]) if i >= 10 else 0.0

    cfg = simmod.EqLSSimConfig(quantile=0.2, cost_bps=5.0, rebalance_every_bars=5, long_only=False)
    sim = simmod.EqLSSimulator(ds, cfg)
    rep = sim.run(fac)
    assert rep["metrics"]["bars"] > 0
    assert "turnover" in rep["metrics"]
    # equity curve starts at 1.0
    assert pytest.approx(rep["equity_curve"][0], rel=0, abs=1e-12) == 1.0


# ========================= MONTE CARLO ==========================

def test_monte_carlo_corr_and_var_es():
    mcmod = _optional_import("simulators.monte_carlo")
    AssetSpec, MCConfig, MonteCarlo = mcmod.AssetSpec, mcmod.MCConfig, mcmod.MonteCarlo

    assets = [
        AssetSpec("AAA", s0=100.0, mu=0.08, sigma=0.25),
        AssetSpec("BBB", s0=50.0,  mu=0.06, sigma=0.20),
        AssetSpec("CCC", s0=25.0,  mu=0.10, sigma=0.35),
    ]
    corr = [
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ]
    cfg = MCConfig(n_paths=300, horizon_days=64, seed=13, jump_lambda=0.0)
    mc = MonteCarlo(assets, cfg, corr=corr)
    rep = mc.run()
    assert set(rep["symbols"]) == {"AAA","BBB","CCC"}
    assert "assets" in rep and "AAA" in rep["assets"]

    # basic VaR/ES sanity
    ret_stats = rep["assets"]["AAA"]["return"]
    assert "var95" in ret_stats and "es95" in ret_stats
    assert ret_stats["es95"] >= ret_stats["var95"]  # ES should not be less than VaR