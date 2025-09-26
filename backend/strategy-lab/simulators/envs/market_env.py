# simulators/envs/market_env.py
"""
MarketEnv (stdlib-only)
-----------------------
A lightweight market environment for strategy-lab sims/backtests.

What it does
- Generates per-symbol mid-price paths with configurable dynamics (GBM / mean-reverting).
- Derives bid/ask from mid and spread; supports temporary spread/liquidity shocks (“news”).
- Emits ticks at a fixed dt and calls ExecutionAgent.update_price(...) so your
  InMemoryImmediateBroker re-evaluates LIMIT/STOP orders and fills when marketable.
- Optional trading hours (pause ticks outside the session).
- Deterministic seeding; easy to unit-test.

What it does NOT do
- Full order-book queues or volume matching (kept simple to work with your existing broker).
- VWAP/TWAP slicing (do that with a custom Broker or StrategyAgent cadence).

Quick start
-----------
from agents.execution_agent import ExecutionAgent
from simulators.envs.market_env import MarketEnv, EnvConfig, SymbolSpec, Shock

x = ExecutionAgent(starting_cash=1_000_000)
env = MarketEnv(
    exec_agent=x,
    cfg=EnvConfig(dt_sec=1.0, session_start="09:15", session_end="15:30", tz_offset_minutes=330, seed=42),
    symbols=[
        SymbolSpec("AAPL", start_price=200.0, sigma=0.20, mu=0.05, spread_bps=2.0),
        SymbolSpec("MSFT", start_price=350.0, sigma=0.18, mu=0.06, spread_bps=1.5),
    ],
    shocks=[
        Shock(kind="news", symbol="AAPL", at_sec=5*60, duration_sec=120, spread_mult=3.0, vol_mult=2.5),
    ],
)
env.run(seconds=600)  # emit 10 minutes of ticks
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ------------------------------ Config models ---------------------------------

@dataclass
class EnvConfig:
    dt_sec: float = 1.0                       # seconds between ticks
    session_start: Optional[str] = None       # "HH:MM" local market time, None => always on
    session_end: Optional[str] = None         # "HH:MM" local market time
    tz_offset_minutes: int = 0                # local offset vs UTC in minutes
    seed: Optional[int] = None                # RNG seed
    latency_ms: Tuple[int, int] = (5, 20)     # jitter applied to tick callbacks (simulated network)
    price_mode: str = "gbm_mr"                # "gbm" | "mr" | "gbm_mr"
    mr_kappa: float = 0.05                    # mean-reversion strength (used in "mr" or "gbm_mr")
    mr_theta_mult: float = 1.0                # long-run level multiplier * start_price for theta


@dataclass
class SymbolSpec:
    symbol: str
    start_price: float
    sigma: float = 0.20          # annualized vol (for GBM part)
    mu: float = 0.05             # annualized drift
    spread_bps: float = 2.0      # baseline bid-ask half-spread in bps of mid (per side)
    tick_size: float = 0.01      # price grid
    min_price: float = 0.01      # floor


@dataclass
class Shock:
    """
    Simple “news” shock:
    - at_sec from env start (0 = immediately)
    - duration_sec the effect lasts
    - spread_mult multiplies baseline spread during the shock
    - vol_mult multiplies sigma during the shock
    """
    kind: str = "news"
    symbol: Optional[str] = None          # None => applies to all symbols
    at_sec: int = 0
    duration_sec: int = 60
    spread_mult: float = 2.0
    vol_mult: float = 2.0


# ------------------------------ Market state ----------------------------------

@dataclass
class SymState:
    spec: SymbolSpec
    mid: float
    bid: float
    ask: float
    t_local_sec: int = 0
    in_shock_until: int = 0


# ------------------------------ MarketEnv -------------------------------------

class MarketEnv:
    """
    Drives synthetic prices and calls exec_agent.update_price(symbol, last_price)
    each tick (using mid, or bid/ask midpoint).
    """

    def __init__(
        self,
        exec_agent,
        cfg: Optional[EnvConfig] = None,
        symbols: Optional[List[SymbolSpec]] = None,
        shocks: Optional[List[Shock]] = None,
        on_tick: Optional[Callable[[str, Dict], None]] = None,  # optional observer hook
    ):
        self.x = exec_agent
        self.cfg = cfg or EnvConfig()
        self.rng = random.Random(self.cfg.seed)
        self.on_tick = on_tick
        self._t0 = time.time()
        self._ticks = 0

        self.states: Dict[str, SymState] = {}
        for sp in (symbols or []):
            half_spread = sp.spread_bps / 1e4 * sp.start_price
            half_spread = max(sp.tick_size, half_spread)
            bid = self._round_px(sp.start_price - half_spread, sp.tick_size, down=True)
            ask = self._round_px(sp.start_price + half_spread, sp.tick_size, down=False)
            self.states[sp.symbol] = SymState(spec=sp, mid=sp.start_price, bid=bid, ask=ask)

        # index shocks by symbol
        self.shocks: List[Shock] = shocks or []

    # ------------------------- public control ---------------------------------

    def run(self, seconds: int, realtime: bool = False) -> None:
        """
        Advance the environment for `seconds` seconds, emitting ticks each dt_sec.
        Set realtime=True to sleep between ticks; else runs as fast as possible.
        """
        steps = max(0, int(seconds / max(1e-6, self.cfg.dt_sec)))
        for _ in range(steps):
            self.step()
            if realtime:
                time.sleep(self.cfg.dt_sec)

    def step(self) -> None:
        """
        One tick for all symbols (if in session).
        """
        self._ticks += 1
        t_local = self._local_seconds_since_midnight()

        in_session = self._in_session(t_local)
        for sym, st in self.states.items():
            st.t_local_sec = t_local
            if not in_session:
                # still update observers with stale quotes if needed
                self._emit_tick(sym, st, session_open=False)
                continue

            # apply shocks if any start now
            self._update_shock_state(sym, t_local)

            # evolve price
            mid_next = self._evolve_mid(st)
            mid_next = max(st.spec.min_price, mid_next)

            # compute spread (baseline * shock multiplier if active)
            half_spread = self._half_spread(st)
            bid = self._round_px(mid_next - half_spread, st.spec.tick_size, down=True)
            ask = self._round_px(mid_next + half_spread, st.spec.tick_size, down=False)
            # ensure bid < ask
            if bid >= ask:
                ask = bid + st.spec.tick_size

            # commit
            st.mid, st.bid, st.ask = mid_next, bid, ask

            # push last trade price to execution (use mid; your broker already handles slippage)
            self.x.update_price(sym, st.mid)
            self._emit_tick(sym, st, session_open=True)

    # ------------------------- price dynamics ----------------------------------

    def _evolve_mid(self, st: SymState) -> float:
        sp = st.spec
        dt_years = self.cfg.dt_sec / (252.0 * 6.5 * 3600.0)  # ~252d * 6.5h trading
        # adjust sigma for active shock
        sigma_eff = sp.sigma * self._shock_vol_mult(st)
        z = self.rng.gauss(0.0, 1.0)

        if self.cfg.price_mode == "gbm":
            # GBM: S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)dt + sigma * sqrt(dt) * z)
            incr = (sp.mu - 0.5 * sigma_eff * sigma_eff) * dt_years + sigma_eff * math.sqrt(dt_years) * z
            return st.mid * math.exp(incr)

        elif self.cfg.price_mode == "mr":
            # Ornstein-Uhlenbeck on price level around theta
            theta = sp.start_price * self.cfg.mr_theta_mult
            mr = self.cfg.mr_kappa * (theta - st.mid) * dt_years
            noise = sigma_eff * math.sqrt(dt_years) * z * st.mid * 0.1  # scale noise to price magnitude
            return max(sp.min_price, st.mid + mr + noise)

        else:  # "gbm_mr" hybrid: gbm trend + small pull toward theta
            gbm_incr = (sp.mu - 0.5 * sigma_eff * sigma_eff) * dt_years + sigma_eff * math.sqrt(dt_years) * z
            gbm_next = st.mid * math.exp(gbm_incr)
            theta = sp.start_price * self.cfg.mr_theta_mult
            mr_adj = 1.0 + self.cfg.mr_kappa * (theta / max(1e-9, gbm_next) - 1.0) * dt_years
            return max(sp.min_price, gbm_next * mr_adj)

    # ------------------------- shocks & spreads --------------------------------

    def _update_shock_state(self, symbol: str, t_local: int) -> None:
        st = self.states[symbol]
        # start shocks whose at_sec == current local seconds since midnight
        for sh in self.shocks:
            if sh.symbol not in (None, symbol):
                continue
            if t_local == sh.at_sec:
                st.in_shock_until = max(st.in_shock_until, t_local + max(1, sh.duration_sec))

    def _half_spread(self, st: SymState) -> float:
        sp = st.spec
        mult = self._shock_spread_mult(st)
        half = (sp.spread_bps / 1e4) * st.mid * mult
        return max(sp.tick_size, half)

    def _shock_spread_mult(self, st: SymState) -> float:
        if st.in_shock_until > st.t_local_sec:
            # find the most intense applicable shock at this moment
            mult = 1.0
            for sh in self.shocks:
                if sh.symbol not in (None, st.spec.symbol):
                    continue
                if st.t_local_sec >= sh.at_sec and st.t_local_sec < sh.at_sec + sh.duration_sec:
                    mult = max(mult, float(sh.spread_mult or 1.0))
            return mult
        return 1.0

    def _shock_vol_mult(self, st: SymState) -> float:
        if st.in_shock_until > st.t_local_sec:
            mult = 1.0
            for sh in self.shocks:
                if sh.symbol not in (None, st.spec.symbol):
                    continue
                if st.t_local_sec >= sh.at_sec and st.t_local_sec < sh.at_sec + sh.duration_sec:
                    mult = max(mult, float(sh.vol_mult or 1.0))
            return mult
        return 1.0

    # ------------------------- time/session helpers ----------------------------

    def _local_seconds_since_midnight(self) -> int:
        """
        Compute local time-of-day seconds from UTC now() plus tz offset,
        then mod by 86400 to stay within a day.
        """
        t = time.time() + self.cfg.tz_offset_minutes * 60
        sod = int(t % 86400)
        return sod

    def _in_session(self, local_sod: int) -> bool:
        # If no session is configured, always on
        if not self.cfg.session_start or not self.cfg.session_end:
            return True
        s = _hhmm_to_seconds(self.cfg.session_start)
        e = _hhmm_to_seconds(self.cfg.session_end)
        if s <= e:
            return s <= local_sod <= e
        # wrap-around (overnight)
        return (local_sod >= s) or (local_sod <= e)

    # ------------------------- utilities / hooks --------------------------------

    @staticmethod
    def _round_px(px: float, tick: float, down: bool = True) -> float:
        if tick <= 0:
            return px
        n = int(px / tick)
        base = n * tick
        if down:
            return max(tick, round(base, 8))
        else:
            return max(tick, round(base + (0 if abs(px - base) < 1e-12 else tick), 8))

    def _emit_tick(self, symbol: str, st: SymState, session_open: bool) -> None:
        if not self.on_tick:
            return
        # Optional latency jitter to emulate wire delay
        lo, hi = self.cfg.latency_ms
        delay = self.rng.uniform(float(lo), float(hi)) / 1000.0
        # We don't actually sleep here to avoid slowing sims; include the delay in payload.
        payload = {
            "t": time.time(),
            "local_t": st.t_local_sec,
            "session_open": session_open,
            "mid": st.mid,
            "bid": st.bid,
            "ask": st.ask,
            "spread": max(0.0, st.ask - st.bid),
            "shock_active": st.in_shock_until > st.t_local_sec,
            "latency_ms": delay * 1000.0,
        }
        try:
            self.on_tick(symbol, payload)
        except Exception:
            pass


# ------------------------------ helpers ---------------------------------------

def _hhmm_to_seconds(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 3600 + int(mm) * 60


# ------------------------------ smoke test ------------------------------------

if __name__ == "__main__":
    # Minimal demo that prints a few ticks and triggers fills via ExecutionAgent broker.
    from agents.execution_agent import ExecutionAgent, Side, OrderType

    x = ExecutionAgent(starting_cash=100_000.0)
    specs = [
        SymbolSpec("AAPL", start_price=200.0, sigma=0.25, spread_bps=2.0),
        SymbolSpec("MSFT", start_price=350.0, sigma=0.20, spread_bps=1.5),
    ]
    shocks = [Shock(kind="news", symbol="AAPL", at_sec=60, duration_sec=60, spread_mult=3.0, vol_mult=2.0)]
    env = MarketEnv(
        exec_agent=x,
        cfg=EnvConfig(dt_sec=0.1, session_start=None, session_end=None, seed=7),
        symbols=specs,
        shocks=shocks,
        on_tick=lambda sym, p: (print(f"{sym} mid={p['mid']:.2f} bid={p['bid']:.2f} ask={p['ask']:.2f}") if (sym=="AAPL" and int(p["local_t"])%5==0) else None)
    )

    # Seed last prices so LIMIT orders can be marketable
    for s in ["AAPL", "MSFT"]:
        x.update_price(s, next(iter(specs)).start_price if s == "AAPL" else 350.0)

    # Place a couple of orders; LIMITs will fill when mid crosses (via env.step -> exec.update_price)
    x.submit_order("AAPL", Side.BUY, qty=50, type=OrderType.LIMIT, limit_price=198.0)
    x.submit_order("AAPL", Side.SELL, qty=50, type=OrderType.STOP, stop_price=205.0)

    env.run(seconds=10, realtime=False)
    print("Equity:", round(x.equity(), 2), "Cash:", round(x.cash, 2), "Pos AAPL:", x.position("AAPL"))