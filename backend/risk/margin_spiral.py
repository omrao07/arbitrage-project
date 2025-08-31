# backend/risk/margin_spiral.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import math
import time

# ----------------------------- data models -----------------------------

@dataclass
class Asset:
    symbol: str
    price: float                    # current mid
    adv_qty: float                  # average daily volume (shares/contracts)
    margin_rate: float = 0.30       # maintenance margin fraction on gross exposure (e.g., 0.30 = 30%)
    eff_spread_bps: float = 8.0     # effective half-spread when taking
    impact_k_sqrt: float = 25.0     # bps * sqrt(q/ADV) per slice (temp+perm proxy)
    vol_day: float = 0.02           # daily vol (fraction), optional (for diagnostics)

@dataclass
class Position:
    symbol: str
    qty: float                      # signed (+ long, - short)

@dataclass
class BrokerTerms:
    maint_margin_mult: float = 1.0  # multiply asset.margin_rate to get maintenance requirement
    call_lag_min: int = 5           # minutes to cure before broker starts forcing
    liquidation_pov: float = 0.15   # max participation of ADV per slice (0..1)
    buffer_cash_frac: float = 0.05  # extra cushion on top of shortfall (5%)
    fee_bps_extra: float = 1.5      # addl fees/penalties bps when under call

@dataclass
class StepState:
    t_index: int
    ts_ms: int
    prices: Dict[str, float]
    equity: float
    req_margin: float
    shortfall: float
    forced_notional: float
    forced_qty: Dict[str, float]
    impact_bps_next: Dict[str, float]
    realized_cost_cash: float       # spread/fees realized this step
    alerts: List[str] = field(default_factory=list)

# ----------------------------- engine -----------------------------

class MarginSpiralEngine:
    """
    Simulate margin call dynamics with endogenous liquidation and price impact.

    Key ideas:
      - Requirement = Σ |qty| * price * (asset.margin_rate * maint_margin_mult)
      - Equity = cash + Σ qty * price  (selling mostly rebalances between position and cash;
        equity drops by trading costs/penalties)
      - If Equity < Requirement -> margin shortfall -> compute notional to liquidate.
      - Liquidation constrained by ADV × POV per slice; generates spread/fee costs now,
        and price impact (sqrt-law) that depresses next step’s prices.
    """

    def __init__(self,
                 assets: Dict[str, Asset],
                 positions: Dict[str, Position],
                 cash: float,
                 terms: Optional[BrokerTerms] = None):
        self.assets: Dict[str, Asset] = {k: Asset(**asdict(v)) if isinstance(v, Asset) else Asset(**v)  # type: ignore
                                         for k, v in assets.items()}
        self.pos: Dict[str, Position] = {k: Position(**asdict(v)) if isinstance(v, Position) else Position(**v)  # type: ignore
                                         for k, v in positions.items()}
        self.cash: float = float(cash)
        self.terms = terms or BrokerTerms()
        self._impact_carryover_bps: Dict[str, float] = {k: 0.0 for k in self.assets}
        self._history: List[StepState] = []

    # ---------- bookkeeping ----------
    def _equity(self) -> float:
        return self.cash + sum(self.pos[s].qty * self.assets[s].price for s in self.pos)

    def _requirement(self) -> float:
        t = self.terms
        req = 0.0
        for s, p in self.pos.items():
            a = self.assets[s]
            req += abs(p.qty) * a.price * (a.margin_rate * t.maint_margin_mult)
        return req

    # ---------- liquidation sizing ----------
    def _needed_liquidation_notional(self, shortfall: float) -> float:
        """
        Selling |ΔN| notional reduces requirement by (margin_rate * |ΔN|).
        Equity barely changes except for trading costs, so to close shortfall S:
            |ΔN| >= S / margin_rate (+ buffer)
        Use weighted average margin rate of *sellable* legs (longs to sell, shorts to buy).
        """
        # compute weighted margin rate over absolute exposure
        gross = 0.0; mr_weighted = 0.0
        for s, p in self.pos.items():
            a = self.assets[s]
            expo = abs(p.qty) * a.price
            if expo <= 0:
                continue
            gross += expo
            mr_weighted += expo * a.margin_rate
        mr = (mr_weighted / gross) if gross > 0 else 0.30
        buf = (1.0 + self.terms.buffer_cash_frac)
        if mr <= 1e-9:
            return shortfall * buf
        return (shortfall * buf) / mr

    def _slice_capacity_notional(self, minutes_per_step: int) -> Dict[str, float]:
        """
        Capacity per asset per step: POV × ADV × (minutes/1440) × price.
        """
        cap: Dict[str, float] = {}
        scale = (minutes_per_step / (60.0 * 24.0))
        for s, a in self.assets.items():
            cap[s] = max(0.0, self.terms.liquidation_pov) * max(0.0, a.adv_qty) * scale * a.price
        return cap

    def _allocate_sales(self, need_notional: float, caps_notional: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate forced notional to sell across symbols proportionally to gross exposure and capacity.
        Only *reduce absolute exposure*: sell longs (positive qty), buy to cover shorts (negative qty).
        Returns: target_notional_to_trade per symbol (positive number means reduce |position|).
        """
        if need_notional <= 1e-9:
            return {s: 0.0 for s in caps_notional}

        # weights = min(cap, gross) with preference to more liquid (higher cap) & bigger exposures
        weights: Dict[str, float] = {}
        tot_w = 0.0
        for s, p in self.pos.items():
            a = self.assets[s]
            expo = abs(p.qty) * a.price
            cap = caps_notional.get(s, 0.0)
            # only allocate if position exists and capacity is non-zero
            w = min(cap, expo)
            if expo > 0 and cap > 0:
                weights[s] = w
                tot_w += w

        if tot_w <= 0:
            # no capacity; return zeros (spiral continues)
            return {s: 0.0 for s in caps_notional}

        alloc: Dict[str, float] = {}
        for s, w in weights.items():
            x = need_notional * (w / tot_w)
            alloc[s] = min(x, caps_notional[s])
        return alloc

    def _apply_trades(self, alloc_notional: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        Execute forced trades at current mid with costs:
          - Realized spread/fees immediately reduce equity.
          - Update positions (reduce absolute exposure).
          - Compute impact bps to be applied to *next step* mid price.
        Returns (impact_bps_by_symbol, realized_cost_cash).
        """
        impact_next: Dict[str, float] = {}
        realized_cost = 0.0
        for s, notional in alloc_notional.items():
            if notional <= 0.0:
                impact_next[s] = 0.0
                continue
            a = self.assets[s]
            px = a.price
            qty_to_reduce = min(abs(self.pos[s].qty), notional / max(1e-12, px))

            if qty_to_reduce <= 0:
                impact_next[s] = 0.0
                continue

            # Direction: reduce absolute exposure (sell longs / buy shorts)
            if self.pos[s].qty > 0:
                trade_qty = -qty_to_reduce   # sell
            else:
                trade_qty = +qty_to_reduce   # buy to cover

            # Realized cost now: effective spread + penalty bps on notional
            loss_bps = a.eff_spread_bps + self.terms.fee_bps_extra
            realized_cost += (loss_bps / 1e4) * (abs(trade_qty) * px)

            # Update position
            self.pos[s].qty += trade_qty

            # Impact for next step (sqrt-law)
            adv = max(1e-9, a.adv_qty)
            impact_bps = a.impact_k_sqrt * math.sqrt((abs(trade_qty)) / adv)
            impact_next[s] = impact_bps
        return impact_next, realized_cost

    def _apply_price_updates(self,
                             exogenous_pct: Dict[str, float],
                             impact_bps_from_prev: Dict[str, float]) -> None:
        """
        Update prices for this step: apply exogenous shock + *previous* step's impact carryover.
        """
        for s, a in self.assets.items():
            exo = exogenous_pct.get(s, exogenous_pct.get("*", 0.0))
            imp = - (impact_bps_from_prev.get(s, 0.0) / 1e4)
            a.price = max(1e-9, a.price * (1.0 + exo + imp))

    # ---------------------- public simulation API ----------------------

    def simulate(self,
                 steps: int,
                 minutes_per_step: int = 5,
                 exogenous_path: Optional[List[Dict[str, float]]] = None,
                 start_ts_ms: Optional[int] = None) -> List[StepState]:
        """
        Run a multi-step spiral simulation.
        - steps: number of time steps
        - minutes_per_step: trading slice duration
        - exogenous_path: list of dicts (per step) of % shocks; use {"*": -0.01} for broad shocks
        """
        self._history.clear()
        ts = start_ts_ms or int(time.time() * 1000)

        for t in range(steps):
            # 1) prices move by prior impact + exogenous this step
            exo = (exogenous_path[t] if (exogenous_path and t < len(exogenous_path)) else {"*": 0.0})
            self._apply_price_updates(exo, self._impact_carryover_bps)

            # 2) recompute equity & requirement
            eq = self._equity()
            req = self._requirement()
            shortfall = max(0.0, req - eq)

            alerts: List[str] = []
            forced_notional = 0.0
            forced_qty: Dict[str, float] = {s: 0.0 for s in self.assets}
            realized_cost = 0.0
            impact_next: Dict[str, float] = {s: 0.0 for s in self.assets}

            # 3) if under maintenance margin, initiate cure
            if shortfall > 0:
                alerts.append(f"MARGIN CALL: shortfall ${shortfall:,.0f}")
                need_notional = self._needed_liquidation_notional(shortfall)
                # capacity this slice (minutes)
                caps = self._slice_capacity_notional(minutes_per_step)
                alloc = self._allocate_sales(need_notional, caps)

                forced_notional = sum(alloc.values())
                # realize trading costs, update positions, compute impact into next step
                impact_next, realized_cost = self._apply_trades(alloc)
                self.cash += (forced_notional / 1.0) - realized_cost  # cash from selling minus costs

                # Note: equity change now = +proceeds - trading costs; the reduction of position value offsets proceeds;
                # so effective equity delta ≈ - realized_cost. Requirement is recomputed next step due to lower |qty|.

                if forced_notional <= 1e-6:
                    alerts.append("NO CAPACITY: unable to cure (POV/ADV constrained)")

            # 4) carry impact to *next* step
            self._impact_carryover_bps = impact_next

            # 5) finalize step snapshot
            prices_now = {s: self.assets[s].price for s in self.assets}
            # reconstruct forced qty signed for reporting
            for s in self.pos:
                # we can't know exact per-step delta without storing prev qty; for UI, approximate:
                forced_qty[s] = 0.0  # (optional: track prev pos to compute delta precisely)

            st = StepState(
                t_index=t,
                ts_ms=ts + t * minutes_per_step * 60_000,
                prices=prices_now,
                equity=self._equity(),
                req_margin=self._requirement(),
                shortfall=max(0.0, self._requirement() - self._equity()),
                forced_notional=forced_notional,
                forced_qty=forced_qty,
                impact_bps_next=impact_next,
                realized_cost_cash=realized_cost,
                alerts=alerts
            )
            self._history.append(st)

        return self._history

    # ---------------------- convenience ----------------------

    def snapshot(self) -> Dict[str, float]:
        eq = self._equity()
        req = self._requirement()
        return {
            "equity": eq,
            "required_margin": req,
            "shortfall": max(0.0, req - eq),
            "cash": self.cash,
            "gross_exposure": sum(abs(self.pos[s].qty) * self.assets[s].price for s in self.pos),
            "net_exposure": sum(self.pos[s].qty * self.assets[s].price for s in self.pos),
        }

    def history(self) -> List[StepState]:
        return self._history

# ----------------------------- demo -----------------------------
if __name__ == "__main__":
    # Toy example: one equity & one crypto, both hit by a 6% exogenous drop at t=0,
    # then no further exogenous moves. Observe spiral dynamics with limited capacity.
    assets = {
        "EQ": Asset(symbol="EQ", price=100.0, adv_qty=5_000_000, margin_rate=0.30, eff_spread_bps=6.0, impact_k_sqrt=20.0),
        "CR": Asset(symbol="CR", price=50_000.0, adv_qty=2_000, margin_rate=0.40, eff_spread_bps=10.0, impact_k_sqrt=35.0),
    }
    positions = {
        "EQ": Position(symbol="EQ", qty=800_000),     # $80m long
        "CR": Position(symbol="CR", qty=10),          # $500k long
    }
    terms = BrokerTerms(maint_margin_mult=1.0, call_lag_min=5, liquidation_pov=0.12, buffer_cash_frac=0.05)
    eng = MarginSpiralEngine(assets, positions, cash=8_000_000.0, terms=terms)

    # Exogenous shock path: 1st step -6% on all, then flat
    path = [{"*": -0.06}] + [{"*": 0.0} for _ in range(24)]
    hist = eng.simulate(steps=len(path), minutes_per_step=10, exogenous_path=path)

    # Pretty print key lines
    for st in hist[:6]:
        print(f"t={st.t_index:02d} equity=${st.equity:,.0f} req=${st.req_margin:,.0f} "
              f"shortfall=${st.shortfall:,.0f} forced=${st.forced_notional:,.0f} alerts={st.alerts}")