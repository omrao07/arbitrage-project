# backend/scenarios/scenario_runner.py
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.backtest.backtester import ( # type: ignore
    Backtester, DataFeed, Candle, TradeTick, Quote,
    CommissionModel, SlippageModel, LatencyModel, ExecConfig
)
from backend.common.schemas import PortfolioSnapshot, LedgerEvent # type: ignore


# ---------------- Scenario Definitions ----------------

@dataclass
class ScenarioResult:
    name: str
    params: Dict[str, Any]
    final_snapshot: PortfolioSnapshot
    equity_curve: List[Tuple[int, float]]
    logs: List[LedgerEvent]

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "params": self.params,
            "final_nav": self.final_snapshot.nav,
            "equity_curve": self.equity_curve,
        }, indent=2, default=str)


@dataclass
class Scenario:
    name: str
    description: str
    data: DataFeed
    strategies: List[Tuple[str, Callable]]
    exec_cfg: ExecConfig
    params: Dict[str, Any] = None # type: ignore

    def run(self, initial_cash: float = 1_000_000.0) -> ScenarioResult:
        bt = Backtester(initial_cash=initial_cash, exec_cfg=self.exec_cfg)
        for strat_name, strat_cb in self.strategies:
            bt.attach_strategy(strat_name, on_bar=strat_cb)
        bt.load_data(self.data)
        res = bt.run()
        return ScenarioResult(
            name=self.name,
            params=self.params or {},
            final_snapshot=res.final_snapshot,
            equity_curve=res.equity_curve,
            logs=res.logs
        )


# ---------------- Scenario Runner ----------------

class ScenarioRunner:
    def __init__(self):
        self._scenarios: Dict[str, Scenario] = {}

    def register(self, scenario: Scenario) -> None:
        self._scenarios[scenario.name] = scenario

    def run(self, name: str, *, initial_cash: float = 1_000_000.0) -> ScenarioResult:
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not registered")
        return self._scenarios[name].run(initial_cash=initial_cash)

    def run_all(self, *, initial_cash: float = 1_000_000.0) -> Dict[str, ScenarioResult]:
        results: Dict[str, ScenarioResult] = {}
        for name, sc in self._scenarios.items():
            results[name] = sc.run(initial_cash=initial_cash)
        return results

    def export_results(self, results: Dict[str, ScenarioResult], path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        out = {name: asdict(res.final_snapshot) for name, res in results.items()}
        path.write_text(json.dumps(out, indent=2, default=str))


# ---------------- Example Usage ----------------

if __name__ == "__main__":  # demo
    import time, random

    # Fake candle generator for testing
    now = int(time.time() * 1000)
    px = 100.0
    candles: List[Candle] = []
    for i in range(60):  # 1h of 1m bars
        px *= (1 + random.uniform(-0.002, 0.002))
        candles.append(Candle(
            symbol="DEMO", ts_ms=now+i*60_000,
            open=px*0.999, high=px*1.001, low=px*0.998, close=px,
            volume=1000, interval="1m"
        ))

    feed = DataFeed(candles=candles)

    # trivial strategy: buy on dip, sell on rise
    class ToyStrategy:
        def __init__(self): self.avg=None
        def on_bar(self, bar, api):
            self.avg = bar.close if self.avg is None else 0.98*self.avg+0.02*bar.close
            diff_bps=(bar.close-self.avg)/self.avg*1e4
            if diff_bps<-5: api.order(bar.symbol,"buy",10)
            elif diff_bps>5: api.order(bar.symbol,"sell",10)

    toy = ToyStrategy()

    exec_cfg = ExecConfig(
        commission=CommissionModel(bps=0.5),
        slippage=SlippageModel(k_spread=0.3, k_participation=0.1),
        latency=LatencyModel(ms=10, jitter_ms=5)
    )

    sc = Scenario(
        name="mean_reversion_test",
        description="Buy dips / sell rallies on synthetic DEMO data",
        data=feed,
        strategies=[("toy", toy.on_bar)],
        exec_cfg=exec_cfg,
        params={"bps": 5}
    )

    runner = ScenarioRunner()
    runner.register(sc)
    result = runner.run("mean_reversion_test")
    print(result.to_json())