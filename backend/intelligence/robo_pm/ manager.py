# backend/runtime/manager.py
"""
Run-time Manager for the arbitrage swarm.

Responsibilities
----------------
- Wire data providers (prices, signals) and balances/constraints into MarketContext
- Instantiate agents (explicit list or via registry) and a Coordinator
- Run a step: propose -> negotiate -> (optional) execute
- Persist explainable reports (stdout and optional files)
- Provide a clean loop with backoff, stop(), and errors surfaced clearly

No external deps. Designed to be extended with your own providers/router.

Quick start
-----------
from backend.runtime.manager import Manager, StaticPrices, StaticSignals
from agents.crypto import CryptoAgent
from coordinator import Coordinator
from agents.base import MarketContext, Constraints

mgr = Manager(
    price_provider=StaticPrices({"BTCUSDT": 65000, "ETHUSDT": 3200, "AAPL": 210}),
    signal_provider=StaticSignals({"social_sent_btc": 0.35, "mom_z_AAPL": 1.0}),
    agents=[CryptoAgent()],
    constraints=Constraints(max_notional_usd=250_000),
    log_dir="runs"
)
mgr.run_once(do_execute=False)
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal as _os_signal
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from agents.base import MarketContext, Constraints # type: ignore
from coordinator import Coordinator # type: ignore
from backend.common.explainer import explain_proposal, explain_decision, md_report # type: ignore


# ----------------------------- provider protocols -----------------------------

class PriceProvider(Protocol):
    def fetch_prices(self) -> Dict[str, float]: ...

class SignalProvider(Protocol):
    def fetch_signals(self) -> Dict[str, float]: ...

class BalanceProvider(Protocol):
    def fetch_balances(self) -> Dict[str, float]: ...

class FXProvider(Protocol):
    def fetch_fx(self) -> Dict[str, float]: ...


# ----------------------------- tiny default providers -------------------------

@dataclass
class StaticPrices:
    data: Dict[str, float]
    def fetch_prices(self) -> Dict[str, float]:
        return dict(self.data)

@dataclass
class StaticSignals:
    data: Dict[str, float]
    def fetch_signals(self) -> Dict[str, float]:
        return dict(self.data)

@dataclass
class StaticBalances:
    data: Dict[str, float]
    def fetch_balances(self) -> Dict[str, float]:
        return dict(self.data)

@dataclass
class StaticFX:
    data: Dict[str, float]
    def fetch_fx(self) -> Dict[str, float]:
        return dict(self.data)


# ----------------------------- manager config ---------------------------------

@dataclass
class ManagerConfig:
    step_interval_sec: float = 5.0          # loop sleep between steps
    save_reports: bool = True               # write md/json under log_dir
    log_dir: str = "runs"
    report_prefix: str = "run"
    max_backoff_sec: float = 30.0           # failure backoff cap
    print_explainers: bool = True           # print to stdout


# ----------------------------- manager ----------------------------------------

@dataclass
class Manager:
    price_provider: PriceProvider
    signal_provider: SignalProvider
    agents: List[Any]                     # AgentBase instances
    
    balances_provider: Optional[BalanceProvider] = None
    fx_provider: Optional[FXProvider] = None
    constraints: Constraints = field(default_factory=Constraints)
    coordinator: Optional[Coordinator] = None
    cfg: ManagerConfig = field(default_factory=ManagerConfig)

    _should_stop: bool = field(default=False, init=False)

    # ---- lifecycle ----

    def __post_init__(self) -> None:
        if self.coordinator is None:
            # Default coordinator using provided agents
            self.coordinator = Coordinator(agents=self.agents, enable_router=False)
        self._install_signal_handlers()
        if self.cfg.save_reports:
            os.makedirs(self.cfg.log_dir, exist_ok=True)

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):
            self._should_stop = True
            print(f"[manager] received signal {signum}; stopping after current step...", file=sys.stderr)
        for sig in (getattr(_os_signal, "SIGINT", None), getattr(_os_signal, "SIGTERM", None)):
            if sig is not None:
                try:
                    _os_signal.signal(sig, _handler)
                except Exception:
                    pass

    # ---- core step ----

    def build_context(self) -> MarketContext:
        prices = self.price_provider.fetch_prices() if self.price_provider else {}
        signals = self.signal_provider.fetch_signals() if self.signal_provider else {}
        balances = self.balances_provider.fetch_balances() if self.balances_provider else {}
        fx = self.fx_provider.fetch_fx() if self.fx_provider else {}

        return MarketContext.now(
            prices=prices,
            fx_usd_per_base=fx,
            balances=balances,
            signals=signals,
            constraints=self.constraints
        )

    def run_once(self, *, do_execute: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        ctx = self.build_context()

        # Collect per-agent proposals to include in report
        agent_blocks: List[str] = []
        outcomes_json: Dict[str, Any] = {}

        # Coordinator does propose+risk+negotiate internally;
        # but for explainability we can also run per-agent explainers
        # by asking coordinator._gather (exposed via a small shim)
        # Since _gather is internal, we replicate logic here safely.
        # (We still rely on coordinator.step for the final slate.)

        # Build slate
        decision = self.coordinator.step(ctx, do_execute=do_execute) # type: ignore

        # Best-effort: pull internal outcomes for explainers if present in diagnostics
        diags = decision.diagnostics or {}
        outs = diags.get("outcomes", {})
        for name, info in outs.items():
            try:
                # Rehydrate a minimal proposal-like dict for explainer
                p = {
                    "orders": info.get("orders", []),
                    "thesis": info.get("thesis", ""),
                    "score": info.get("score", 0.0),
                    "confidence": info.get("confidence", 0.0),
                    "horizon_sec": 3600,
                    "tags": [],
                }
                # risk can't be reconstructed perfectly; show gross/net if present
                r = {
                    "ok": info.get("ok", False),
                    "gross_notional_usd": 0.0,
                    "exposure_usd": 0.0,
                    "notes": "",
                }
                txt = explain_proposal(name, p, r, markdown=False)
                agent_blocks.append(txt)
                outcomes_json[name] = info
            except Exception:
                # ignore individual explainer errors
                pass

        # Coordinator explainer
        slate_txt = explain_decision(decision, markdown=False)

        # Build Markdown report
        md = md_report(
            title="Arb Swarm Run",
            context_summary={"ts": int(ctx.ts), "constraints": self._to_jsonable(self.constraints),
                             "prices": self._shorten(ctx.prices), "signals": self._shorten(ctx.signals, 18)},
            agent_blocks=agent_blocks,
            decision_block=slate_txt,
            notes=f"do_execute={do_execute}"
        )

        if self.cfg.print_explainers:
            print("\n" + md + "\n")

        # Save artifacts
        run_id = f"{self.cfg.report_prefix}_{int(t0)}"
        paths = {}
        if self.cfg.save_reports:
            md_path = os.path.join(self.cfg.log_dir, f"{run_id}.md")
            json_path = os.path.join(self.cfg.log_dir, f"{run_id}.json")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": run_id,
                    "ts": int(ctx.ts),
                    "prices": ctx.prices,
                    "signals": ctx.signals,
                    "balances": ctx.balances,
                    "decision": self._to_jsonable(decision),
                    "outcomes": outcomes_json
                }, f, indent=2, sort_keys=True)
            paths = {"md": md_path, "json": json_path}

        return {"ok": bool(decision.ok), "decision": decision, "paths": paths}

    # ---- loop runner ----

    def run_loop(self, *, do_execute: bool = False) -> None:
        backoff = 0.0
        step = 0
        while not self._should_stop:
            try:
                step += 1
                res = self.run_once(do_execute=do_execute)
                backoff = 0.0  # reset backoff on success
            except Exception as e:
                print("[manager] step error:", e, file=sys.stderr)
                traceback.print_exc()
                backoff = min(self.cfg.max_backoff_sec, (backoff * 1.5) + 1.0)
            # sleep with backoff
            sleep_s = max(self.cfg.step_interval_sec, backoff)
            for _ in range(int(sleep_s * 10)):
                if self._should_stop:
                    break
                time.sleep(0.1)

    def stop(self) -> None:
        self._should_stop = True

    # ---- helpers ----

    def _shorten(self, d: Dict[str, Any], max_items: int = 12) -> Dict[str, Any]:
        if len(d) <= max_items:
            return d
        keys = sorted(d.keys())[:max_items]
        out = {k: d[k] for k in keys}
        out["__truncated__"] = f"{len(d) - max_items} more..."
        return out

    def _to_jsonable(self, obj: Any) -> Any:
        try:
            import dataclasses
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj) # type: ignore
        except Exception:
            pass
        try:
            # naive: pull __dict__ if present
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            return str(obj)


# ----------------------------- CLI entry --------------------------------------

if __name__ == "__main__":
    # Minimal demo run (dry-run)
    from agents.crypto import CryptoAgent # type: ignore
    from agents.equities import EquitiesAgent # type: ignore
    from agents.fx import FXAgent # type: ignore

    prices = {"BTCUSDT": 65000, "ETHUSDT": 3200, "AAPL": 210.0, "EURUSD": 1.09}
    signals = {
        "btc_basis_annual": 0.08, "btc_funding_8h": 0.0001, "social_sent_btc": 0.35, "vol_z_btc": 0.2,
        "mom_z_AAPL": 1.0, "earn_surprise_AAPL": 0.06, "sent_AAPL": 0.3,
        "carry_EURUSD": 0.012, "mom_z_EURUSD": 0.8, "ppp_gap_EURUSD": -0.10,
    }

    mgr = Manager(
        price_provider=StaticPrices(prices),
        signal_provider=StaticSignals(signals),
        agents=[CryptoAgent(), EquitiesAgent(), FXAgent()],
        constraints=Constraints(max_notional_usd=250_000),
        cfg=ManagerConfig(step_interval_sec=10.0, log_dir="runs", print_explainers=True)
    )

    mgr.run_once(do_execute=False)