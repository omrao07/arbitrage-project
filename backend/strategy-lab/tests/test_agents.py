# tests/test_agents.py
import math
import time
import types
import importlib
import pytest

# --------------------------- helpers ---------------------------

def _optional_import(modname):
    try:
        return importlib.import_module(modname)
    except Exception:
        pytest.skip(f"Optional module '{modname}' not found; skipping related tests.")


# ========================= AB TESTS / ROUTERS ==========================

def test_abtest_hash_routing_and_attribution():
    ab = _optional_import("selector.ab_tests")

    # Minimal ExecutionAgent stub with fills + price/equity accounting
    class ExecStub:
        def __init__(self):
            self._last = {}
            self._fills = []
            self.cash = 1_000_000.0
            self._pos = {}
        def update_price(self, sym, px): self._last[sym] = px
        def last_price(self, sym): return self._last.get(sym, 0.0)
        def position(self, sym): return self._pos.get(sym, 0.0)
        def equity(self):
            mtm = sum(self._pos.get(s, 0.0) * self._last.get(s, 0.0) for s in self._pos)
            return self.cash + mtm
        def gross_exposure(self): return sum(abs(self._pos.get(s,0.0)*self._last.get(s,0.0)) for s in self._pos)
        def leverage(self): 
            eq = self.equity()
            return (self.gross_exposure() / eq) if eq else 0.0
        class _OID:
            def __init__(self, v): self.val=v
        def submit_order(self, symbol, side, qty, type=None, limit_price=None, stop_price=None):
            px = self._last.get(symbol, limit_price or stop_price or 0.0)
            # immediate fill at current price
            oid = ExecStub._OID(f"OID-{symbol}-{time.time_ns()}")
            fee = 0.0
            self._fills.append(types.SimpleNamespace(order_id=oid, symbol=symbol, side=side, qty=qty, price=px, fee=fee))
            # update position & cash (BUY positive qty, SELL negative qty if side has .SELL)
            sign = +1 if str(getattr(side, "BUY", "BUY")) in (getattr(side, "name", "BUY"), "BUY") else -1
            qty_signed = sign * qty
            self._pos[symbol] = self._pos.get(symbol, 0.0) + qty_signed
            self.cash -= qty_signed * px  # BUY reduces cash; SELL increases (qty_signed negative)
            return oid
        def fills(self): return self._fills
        def open_orders(self): return []

    # Enum shim (to match ab_tests expectations loosely)
    class Side: BUY="BUY"; SELL="SELL"
    class OrderType: MARKET="MARKET"; LIMIT="LIMIT"

    x = ExecStub()
    router = ab.HashSplitRouter(["A","B"], share_map={"A":0.5, "B":0.5})
    runner = ab.ABTestRunner(x, router)

    # Build two tiny strategy agents that try to trade the same symbols
    class MiniSA:
        def __init__(self, exec_proxy, side):
            self.x = exec_proxy; self.side = side
        def on_price(self, sym, px): self.x.update_price(sym, px)
        def maybe_rebalance(self, force=False):
            # place a tiny order on every call; router will allow only owner's symbols
            for sym in ("SBIN","AAPL","MSFT","TSLA"):
                self.x.submit_order(sym, Side.BUY if self.side=="BUY" else Side.SELL, qty=1, type=OrderType.MARKET)

    sa_A = MiniSA(runner.make_proxy("A"), "BUY")
    sa_B = MiniSA(runner.make_proxy("B"), "SELL")
    runner.register("A", sa_A)
    runner.register("B", sa_B)

    # Feed prices and trigger a rebalance
    for sym, px in [("SBIN",800.0), ("AAPL",200.0), ("MSFT",350.0), ("TSLA",250.0)]:
        runner.on_price(sym, px)
    runner.maybe_rebalance_all(force=True)
    runner.tick()

    snap = runner.snapshot()
    assert "arms" in snap and set(snap["arms"].keys()) == {"A","B"}
    # Each arm should show some cashflow (non-zero) after fills
    a_cf = snap["arms"]["A"]["realized_cashflow"]
    b_cf = snap["arms"]["B"]["realized_cashflow"]
    assert (a_cf != 0.0) or (b_cf != 0.0)


def test_policy_router_rule_ordering():
    pr = _optional_import("selector.policy_router")
    ab = _optional_import("selector.ab_tests")

    default = ab.HashSplitRouter(["A","B"])
    router = pr.PolicyRouter(
        arms=["A","B"],
        rules=[
            pr.AllowDenyRule(allow={"TSLA":"B"}),
            pr.GroupRule(mapping={"PSU": (["SBIN","PNB"], "A")}),
        ],
        default=default,
    )
    assert router.owner("TSLA") == "B"
    assert router.owner("SBIN") == "A"
    # For an unknown symbol, falls back to hash (deterministic bucket)
    owner = router.owner("AAPL")
    assert owner in {"A","B"}


# ========================= EVALUATOR ==========================

def test_evaluator_sampling_and_summary(tmp_path):
    evm = _optional_import("selector.evaluator")

    # Tiny stub runner producing a predictable equity walk
    class Runner:
        def __init__(self):
            self.t = time.time()
            self.eq = 1_000_000.0
            self.cfA = 0.0; self.cfB = 0.0
        def snapshot(self):
            self.t += 60
            self.eq *= 1.0005
            self.cfA += 10.0; self.cfB += -5.0
            return {"t": self.t, "equity": self.eq, "gross_exposure": 0.0, "leverage": 0.0,
                    "arms": {"A":{"realized_cashflow": self.cfA, "trades":1, "win_rate":0.6},
                             "B":{"realized_cashflow": self.cfB, "trades":1, "win_rate":0.4}}}

    out = tmp_path / "ab_eval.jsonl"
    ev = evm.Evaluator(Runner(), monitoring=None, cfg=evm.EvalConfig(sample_every_sec=0.1, output_path=str(out)))
    for _ in range(10):
        row = ev.tick()
        assert row is not None
    summ = ev.summary()
    assert "portfolio" in summ and "arms" in summ
    assert set(summ["arms"].keys()) == {"A","B"}


# ========================= DATA LOADER & SIMS ==========================

def test_data_loader_align_and_iter():
    dl = _optional_import("simulators.envs.data_loader")
    loader = dl.MarketDataLoader()

    # build two tiny in-memory series via temp files
    import tempfile, os, csv
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "AAA.csv")
        p2 = os.path.join(d, "BBB.csv")
        t0 = 1_700_000_000
        with open(p1, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["date","open","high","low","close","volume"])
            for i in range(5):
                px = 100 + i
                w.writerow([t0 + i*86400, px, px, px, px, 1000])
        with open(p2, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["date","open","high","low","close","volume"])
            for i in range(5):
                px = 50 + i
                w.writerow([t0 + i*86400, px, px, px, px, 1000])

        ds = loader.load_dir(os.path.join(d, "*.csv"), fmt="csv", align=True)
        assert ds.symbols() == ["AAA","BBB"]
        assert len(ds.calendar) == 5

        # rolling windows
        wins = list(loader.rolling(ds, window=3))
        assert len(wins) == 3  # 5 bars -> 3 windows of size 3


def test_eq_ls_sim_runs():
    simmod = _optional_import("simulators.eq_ls_sim")

    # small synthetic dataset (aligned)
    class DS: pass
    t0 = 1_700_000_000
    def mk(n, start):
        x = start; rows=[]
        import random; random.seed(7)
        for i in range(n):
            x *= (1 + (0.01 if (i%10==0) else -0.005))  # mild pattern
            rows.append({"ts": t0 + i*86400, "open": x, "high": x, "low": x, "close": x, "volume": 1})
        return rows
    data = {"AAA": mk(60,100), "BBB": mk(60,50), "CCC": mk(60,25), "DDD": mk(60,10)}
    cal = [r["ts"] for r in data["AAA"]]
    ds = DS(); ds.data = data; ds.calendar = cal # type: ignore

    def fac(sym, i, hist, ctx):  # simple momentum
        return (hist[i]["close"] - hist[i-5]["close"]) if i>=5 else 0.0

    cfg = simmod.EqLSSimConfig(quantile=0.25, cost_bps=1.0)
    sim = simmod.EqLSSimulator(ds, cfg)
    rep = sim.run(fac)
    assert "metrics" in rep and rep["metrics"]["bars"] > 0


# ========================= MARKET ENV (synthetic) ==========================

def test_market_env_ticks_and_exec():
    envmod = _optional_import("simulators.envs.market_env")

    # Exec stub with update_price() observed
    class ExecStub:
        def __init__(self): self.last={}
        def update_price(self, s, p): self.last[s]=p
        def equity(self): return 1_000_000.0
        def gross_exposure(self): return 0.0
        def leverage(self): return 0.0

    x = ExecStub()
    specs = [envmod.SymbolSpec("AAA", start_price=100.0, sigma=0.20)]
    env = envmod.MarketEnv(x, cfg=envmod.EnvConfig(dt_sec=0.01, session_start=None, session_end=None, seed=1), symbols=specs)
    for _ in range(5):
        env.step()
    assert "AAA" in x.last and x.last["AAA"] > 0.0


# ========================= BACKTESTER (bar replay) ==========================

def test_backtester_runs_smoke(tmp_path):
    btmod = _optional_import("simulators.backtester")

    # Build tiny aligned dataset
    class DS: pass
    t0 = 1_700_000_000
    def series(n, start):
        x=start; out=[]
        for i in range(n):
            x *= (1 + (0.001 if i%2==0 else -0.0005))
            out.append({"ts": t0+i*86400, "open":x, "high":x, "low":x, "close":x, "volume":1})
        return out
    ds = DS()
    ds.data = {"AAA": series(40,100), "BBB": series(40,50)} # type: ignore
    ds.calendar = [r["ts"] for r in ds.data["AAA"]] # type: ignore

    # Minimal ExecutionAgent + StrategyAgent stubs
    class X:
        def __init__(self): self._last={}; self._fills=[]
        def update_price(self,s,p): self._last[s]=p
        def mark_to_market(self): pass
        def equity(self): return 1_000_000.0 + sum(self._last.values())*0.0
        def fills(self): return self._fills

    class StrategyAgentStub:
        def __init__(self, x): self.x=x
        def on_price(self,s,p): self.x.update_price(s,p)
        def maybe_rebalance(self, force=False): pass

    x = X()
    sa = StrategyAgentStub(x)
    cfg = btmod.BacktestConfig(rebalance="daily", warmup_bars=5, fills_log_path=str(tmp_path/"fills.jsonl"))
    bt = btmod.Backtester(ds, x, sa, cfg)
    res = bt.run()
    assert res["metrics"]["bars"] > 0
    assert isinstance(res["returns"], list)


# ========================= STRATEGIES (signals only) ==========================

@pytest.mark.parametrize("modname,clsname,kwargs",
    [
        ("strategies.mean_reversion", "MeanReversionStrategy", {"lookback":5}),
        ("strategies.momentum", "MomentumStrategy", {"lookback":20,"skip":2}),
        ("strategies.pairs_trading", "PairsTradingStrategy", {"pairs":[("AAA","BBB")], "lookback":20}),
    ]
)
def test_strategies_emit_scores(modname, clsname, kwargs):
    mod = _optional_import(modname)
    Strat = getattr(mod, clsname)
    strat = Strat(**kwargs)

    # feed a tiny tape
    pxA = [100,101,99,100,102,101,103,104,103,105]
    pxB = [50,50.5,50.2,50.4,50.8,51.0,50.7,51.2,51.1,51.5]
    for i in range(len(pxA)):
        if hasattr(strat, "on_price"):
            strat.on_price("AAA", pxA[i])
            if "pairs_trading" in modname:
                strat.on_price("BBB", pxB[i])
    scores = strat.generate_signals(time.time())
    assert isinstance(scores, dict)


# ========================= MONTE CARLO ==========================

def test_monte_carlo_basic():
    mcmod = _optional_import("simulators.monte_carlo")
    AssetSpec = mcmod.AssetSpec
    MCConfig = mcmod.MCConfig
    MonteCarlo = mcmod.MonteCarlo
    assets = [AssetSpec("AAA", s0=100.0, mu=0.08, sigma=0.2),
              AssetSpec("BBB", s0=50.0, mu=0.06, sigma=0.25)]
    cfg = MCConfig(n_paths=200, horizon_days=64, seed=7, jump_lambda=0.0)
    mc = MonteCarlo(assets, cfg, corr=[[1.0,0.3],[0.3,1.0]])
    report = mc.run()
    assert set(report["symbols"]) == {"AAA","BBB"}
    assert "assets" in report and "AAA" in report["assets"]