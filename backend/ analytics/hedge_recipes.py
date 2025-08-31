# backend/strategies/hedge_recipes.py
from __future__ import annotations

"""
Hedge Recipes
-------------
Dependency-light toolkit to define/compose hedging strategies and emit executable
order intents for your OMS/risk pipeline.

Key ideas
- A HedgeRecipe takes (position snapshot, market snapshot, option chain accessor)
  and produces a list of order intents (dicts) + rationale notes.
- Batteries included: protective_put, collar, put_spread, call_overwrite,
  futures_overlay, delta_hedge, tail_put, calendar_put_spread, vol_target overlay.
- No broker/venue coupling here—just clean intents for your router/risk to handle.
- No third-party deps; plug your own option-chain accessor.

Conventions
- Quantities are "natural units" (shares, option contracts, futures qty); your OMS
  will convert to notional if needed.
- Sides: 'buy' / 'sell'
- Option symbols are abstracted as {underlier, right, strike, expiry}; your
  downstream normalizer should format the venue-specific symbol/ticker.

Example
-------
recipes = HedgeKitchen(accessor=my_chain_fn)
intents, notes = recipes.collar(
    symbol="AAPL", pos_shares=10_000, spot=200.10,
    put_delta=-0.20, call_delta=0.20, expiry_days=45, ratio=1.0
)
# -> feed intents into risk/OMS; show `notes` in Explain UI.
"""

import math
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# ------------------------------- Types ---------------------------------------

@dataclass
class OptionQuote:
    symbol: str             # venue-ready if you have it; else build later
    underlier: str
    right: str              # "C" or "P"
    strike: float
    expiry: str             # ISO date "YYYY-MM-DD"
    mid: float
    delta: float            # + for calls, - for puts
    iv: Optional[float] = None

@dataclass
class ChainSnapshot:
    underlier: str
    spot: float
    ts_ms: int
    options: List[OptionQuote]  # flattened grid across expiries/strikes

@dataclass
class OrderIntent:
    symbol: str            # equity/future/option ticker; or a dict for option spec
    side: str              # 'buy' | 'sell'
    qty: float
    typ: str = "market"
    limit_price: Optional[float] = None
    venue: Optional[str] = None
    asset_class: Optional[str] = None  # 'equity' | 'option' | 'future'
    meta: Dict[str, Any] = None # type: ignore

def _oi(**kw) -> OrderIntent:
    kw.setdefault("meta", {})
    return OrderIntent(**kw)

# Option chain accessor signature:
#   get_chain(symbol:str, max_expiry_days:int|None) -> ChainSnapshot
ChainAccessor = Callable[[str, Optional[int]], ChainSnapshot]

# --------------------------- Utility functions -------------------------------

def _pick_by_delta(options: List[OptionQuote], right: str, target_delta: float, expiry_days: Optional[int]) -> OptionQuote:
    """
    Pick option whose delta is closest to target (call positive, put negative).
    If expiry_days provided, prefer the nearest expiry >= requested days; else nearest overall.
    """
    if not options:
        raise ValueError("empty option chain")
    # Partition by right
    pool = [o for o in options if o.right.upper() == right.upper()]
    if not pool:
        raise ValueError(f"no {right} options in chain")
    # If expiry constraint, pick expiries around target horizon
    if expiry_days is not None:
        # naive: group by expiry, choose expiry with min |days - expiry_days|
        # Expect caller's accessor to have mapped expiry->days in meta if needed; use string diff as proxy.
        def _days_from_iso(iso: str) -> int:
            # very rough; downstream can be precise. Expect format YYYY-MM-DD.
            y, m, d = [int(x) for x in iso.split("-")]
            return y * 365 + m * 30 + d  # ordinal-ish
        target_ord = _days_from_iso(_fake_day_offset(expiry_days))
        by_exp: Dict[str, List[OptionQuote]] = {}
        for o in pool:
            by_exp.setdefault(o.expiry, []).append(o)
        # pick expiry closest by ordinal heuristic
        def _ord(iso: str) -> int:
            y, m, d = [int(x) for x in iso.split("-")]
            return y * 365 + m * 30 + d
        expiry = min(by_exp.keys(), key=lambda e: abs(_ord(e) - target_ord))
        pool = by_exp[expiry]
    # Closest delta
    best = min(pool, key=lambda o: abs(float(o.delta) - float(target_delta)))
    return best

def _fake_day_offset(days: int) -> str:
    """Tiny helper to fake 'today + days' ISO for rough ordering without datetime deps."""
    # Not used to trade; only to pick an expiry bucket roughly near target.
    # Use YYYY-00-(days) placeholder to keep monotonic ordering against the ordinal heuristic.
    return f"2025-00-{max(1, min(365, days))}"

def _contracts_for_shares(shares: float, contract_multiplier: int = 100) -> float:
    """Approx contracts to hedge 'shares' underlier with 1:contract_multiplier contract size."""
    return shares / float(contract_multiplier)

def _round_contracts(x: float) -> float:
    # Round to nearest whole contract; keep as float for intent but integer-ish.
    return float(int(round(x)))

# ------------------------------ Kitchen --------------------------------------

class HedgeKitchen:
    """
    Construct hedges from recipes. Provide an `accessor` that returns a ChainSnapshot.
    """
    def __init__(self, accessor: ChainAccessor, *, contract_multiplier: int = 100):
        self.get_chain = accessor
        self.mult = int(contract_multiplier)

    # ----------- Core recipes ------------------------------------------------

    def protective_put(
        self,
        *,
        symbol: str,
        pos_shares: float,
        spot: float,
        put_delta: float = -0.25,
        expiry_days: int = 30,
        ratio: float = 1.0,
        max_expiry_days: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        Buy ~|delta| put(s) against long stock. qty = ratio * shares / mult.
        """
        chain = self.get_chain(symbol, max_expiry_days or expiry_days)
        opt = _pick_by_delta(chain.options, "P", put_delta, expiry_days)
        qty = _round_contracts(ratio * _contracts_for_shares(pos_shares, self.mult))
        intents = [
            _oi(symbol=_fmt_opt(opt), side="buy", qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"protective_put", "target_delta":put_delta})
        ]
        notes = [
            f"Protective put: {symbol}, {qty}x {opt.expiry} P{opt.strike} (~Δ {opt.delta:+.2f}) vs {pos_shares} sh @ {spot:.2f}."
        ]
        return intents, notes

    def call_overwrite(
        self,
        *,
        symbol: str,
        pos_shares: float,
        spot: float,
        call_delta: float = 0.20,
        expiry_days: int = 30,
        ratio: float = 1.0,
        max_expiry_days: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        Sell covered calls against long stock. qty = ratio * shares / mult.
        """
        chain = self.get_chain(symbol, max_expiry_days or expiry_days)
        opt = _pick_by_delta(chain.options, "C", call_delta, expiry_days)
        qty = _round_contracts(ratio * _contracts_for_shares(pos_shares, self.mult))
        intents = [
            _oi(symbol=_fmt_opt(opt), side="sell", qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"call_overwrite", "target_delta":call_delta})
        ]
        notes = [
            f"Call overwrite: {symbol}, short {qty}x {opt.expiry} C{opt.strike} (~Δ {opt.delta:+.2f}) covered by {pos_shares} sh."
        ]
        return intents, notes

    def collar(
        self,
        *,
        symbol: str,
        pos_shares: float,
        spot: float,
        put_delta: float = -0.25,
        call_delta: float = 0.20,
        expiry_days: int = 45,
        ratio: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        Buy put + sell call (same expiry), sized to shares*ratio.
        """
        chain = self.get_chain(symbol, expiry_days)
        p = _pick_by_delta(chain.options, "P", put_delta, expiry_days)
        c = _pick_by_delta(chain.options, "C", call_delta, expiry_days)
        qty = _round_contracts(ratio * _contracts_for_shares(pos_shares, self.mult))
        intents = [
            _oi(symbol=_fmt_opt(p), side="buy",  qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"collar", "leg":"put", "target_delta":put_delta}),
            _oi(symbol=_fmt_opt(c), side="sell", qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"collar", "leg":"call", "target_delta":call_delta}),
        ]
        notes = [
            f"Zero/low-cost collar target: long {qty}x {p.expiry} P{p.strike} + short {qty}x C{c.strike} vs {pos_shares} sh.",
            "Adjust deltas/expiries to balance premium if needed."
        ]
        return intents, notes

    def put_spread(
        self,
        *,
        symbol: str,
        pos_shares: float,
        spot: float,
        long_put_delta: float = -0.25,
        short_put_delta: float = -0.10,
        expiry_days: int = 45,
        ratio: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        Debit put spread to cap premium while protecting tail partially.
        """
        chain = self.get_chain(symbol, expiry_days)
        p_long = _pick_by_delta(chain.options, "P", long_put_delta, expiry_days)
        p_short = _pick_by_delta(chain.options, "P", short_put_delta, expiry_days)
        qty = _round_contracts(ratio * _contracts_for_shares(pos_shares, self.mult))
        intents = [
            _oi(symbol=_fmt_opt(p_long),  side="buy",  qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"put_spread", "leg":"long"}),
            _oi(symbol=_fmt_opt(p_short), side="sell", qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"put_spread", "leg":"short"}),
        ]
        notes = [f"Put spread: +{p_long.expiry} P{p_long.strike} / -P{p_short.strike}, qty {qty}, vs {pos_shares} sh."]
        return intents, notes

    def delta_hedge(
        self,
        *,
        symbol: str,
        pos_shares: float,
        target_delta: float = 0.0,
        current_delta: Optional[float] = None,
        spot: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        Equity delta hedge to target (e.g., neutralize).
        If current_delta not provided, assume current_delta = pos_shares (1.0 per share).
        """
        cur = float(current_delta if current_delta is not None else pos_shares)
        need = target_delta - cur
        if abs(need) < 1e-6:
            return [], [f"Delta hedge: already at target Δ={target_delta:.2f}."]
        side = "buy" if need > 0 else "sell"
        qty = abs(need)  # shares
        intents = [
            _oi(symbol=symbol, side=side, qty=qty, asset_class="equity",
                meta={**(meta or {}), "recipe":"delta_hedge", "target_delta":target_delta, "current_delta":cur})
        ]
        notes = [f"Delta hedge: {side} {qty:.0f} {symbol} to move Δ {cur:.0f} → {target_delta:.0f}."]
        return intents, notes

    def futures_overlay(
        self,
        *,
        future_symbol: str,
        hedge_notional: float,
        contract_value: float,
        direction: str = "short",   # 'short' to reduce beta; 'long' to add
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        Overlay index futures to adjust beta/notional quickly.
        contract_value: $ exposure per 1 future (e.g., ES ~ $50 * index).
        """
        if contract_value <= 0:
            raise ValueError("contract_value must be > 0")
        qty = _round_contracts(abs(hedge_notional) / float(contract_value))
        side = "sell" if direction == "short" else "buy"
        intents = [
            _oi(symbol=future_symbol, side=side, qty=qty, asset_class="future",
                meta={**(meta or {}), "recipe":"futures_overlay", "contract_value":contract_value})
        ]
        notes = [f"Futures overlay: {side} {qty} {future_symbol} for ~${hedge_notional:,.0f} exposure adj."]
        return intents, notes

    def tail_put(
        self,
        *,
        symbol: str,
        pos_shares: float,
        moneyness: float = 0.8,       # long deep OTM put (e.g., 80% of spot)
        expiry_days: int = 90,
        ratio: float = 0.5,           # partial tail cover
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        Deep OTM put for crash protection. Uses strike ≈ moneyness * spot by nearest match.
        """
        chain = self.get_chain(symbol, expiry_days)
        # closest put by strike
        target_strike = moneyness * chain.spot
        pool = [o for o in chain.options if o.right == "P"]
        opt = min(pool, key=lambda o: abs(o.strike - target_strike))
        qty = _round_contracts(ratio * _contracts_for_shares(pos_shares, self.mult))
        intents = [
            _oi(symbol=_fmt_opt(opt), side="buy", qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"tail_put", "moneyness":moneyness})
        ]
        notes = [f"Tail put: buy {qty}x {opt.expiry} P{opt.strike} (~{moneyness:.0%} moneyness) vs {pos_shares} sh."]
        return intents, notes

    def calendar_put_spread(
        self,
        *,
        symbol: str,
        pos_shares: float,
        near_expiry_days: int = 30,
        far_expiry_days: int = 90,
        put_delta: float = -0.25,
        ratio: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        Long far-dated put, short near-dated put (same strike ~ Δ target).
        """
        ch_near = self.get_chain(symbol, near_expiry_days)
        ch_far  = self.get_chain(symbol, far_expiry_days)
        p_near = _pick_by_delta(ch_near.options, "P", put_delta, near_expiry_days)
        # try to find same strike in far expiry; else closest delta
        same_strike_far = [o for o in ch_far.options if o.right == "P" and abs(o.strike - p_near.strike) < 1e-6]
        p_far = same_strike_far[0] if same_strike_far else _pick_by_delta(ch_far.options, "P", put_delta, far_expiry_days)
        qty = _round_contracts(ratio * _contracts_for_shares(pos_shares, self.mult))
        intents = [
            _oi(symbol=_fmt_opt(p_far),  side="buy",  qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"calendar_put_spread", "leg":"far_long"}),
            _oi(symbol=_fmt_opt(p_near), side="sell", qty=qty, asset_class="option",
                meta={**(meta or {}), "recipe":"calendar_put_spread", "leg":"near_short"}),
        ]
        notes = [f"Calendar put spread: +{p_far.expiry} P{p_far.strike} / -{p_near.expiry} P{p_near.strike}, qty {qty}."]
        return intents, notes

    def vol_target_overlay(
        self,
        *,
        book_vol: float,
        target_vol: float,
        beta_to_index: float,
        index_future: str,
        index_contract_value: float,
        portfolio_value: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[OrderIntent], List[str]]:
        """
        If realized/forecasted book volatility exceeds target, short index futures to reduce risk
        (or long to increase if too low). Simple proportional controller.

        hedge_notional ≈ ( (book_vol - target_vol) / max(target_vol,1e-6) ) * (portfolio_value / max(|beta|,1e-6))
        """
        if target_vol <= 0 or portfolio_value <= 0:
            return [], ["vol_target_overlay: invalid target_vol or portfolio_value."]
        k = (book_vol - target_vol) / max(target_vol, 1e-6)
        if abs(k) < 1e-3:
            return [], ["vol_target_overlay: already at target."]
        # sign: if book_vol > target -> short futures; else long futures
        direction = "short" if k > 0 else "long"
        hedge_notional = abs(k) * (portfolio_value / max(abs(beta_to_index), 1e-6))
        return self.futures_overlay(
            future_symbol=index_future,
            hedge_notional=hedge_notional,
            contract_value=index_contract_value,
            direction=direction,
            meta={**(meta or {}), "recipe":"vol_target_overlay", "k":k}
        )

# -------------------------- Formatting helpers -------------------------------

def _fmt_opt(o: OptionQuote) -> str:
    """
    Produce a portable option descriptor. If you already have venue ticker in `o.symbol`,
    we pass that through; else we return a JSON-ish spec the OMS normalizer can resolve.
    """
    if o.symbol:
        return o.symbol
    return f"OPT::{o.underlier}:{o.expiry}:{o.right}:{_trim(o.strike)}"

def _trim(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s

# -------------------------- Example dummy accessor ---------------------------

def dummy_accessor(symbol: str, max_expiry_days: Optional[int]) -> ChainSnapshot:
    """
    Dumb synthetic chain for testing wire-up without a real options API.
    Creates two expiries with strikes around spot, simple delta mapping.
    """
    spot = 100.0
    expiries = ["2025-01-31", "2025-03-31"]
    opts: List[OptionQuote] = []
    for expiry in expiries:
        for k in range(-5, 6):
            strike = spot * (1 + 0.05 * k)
            # crude delta proxy
            m = (spot - strike) / max(spot, 1e-9)
            call_delta = 0.5 + 0.4 * m  # capped
            put_delta  = - (0.5 + 0.4 * (-m))
            call_delta = max(0.05, min(0.95, call_delta))
            put_delta  = -max(0.05, min(0.95, abs(put_delta)))
            mid = max(0.25, abs(spot - strike) * 0.15)
            opts.append(OptionQuote(symbol="", underlier=symbol, right="C", strike=round(strike,2), expiry=expiry, mid=mid, delta=round(call_delta,2)))
            opts.append(OptionQuote(symbol="", underlier=symbol, right="P", strike=round(strike,2), expiry=expiry, mid=mid, delta=round(put_delta,2)))
    return ChainSnapshot(underlier=symbol, spot=spot, ts_ms=0, options=opts)

# ------------------------------ Quick demo -----------------------------------

if __name__ == "__main__":
    # Run a quick self-demo with the dummy chain
    k = HedgeKitchen(accessor=dummy_accessor)
    intents, notes = k.collar(symbol="TEST", pos_shares=10_000, spot=100.0, put_delta=-0.25, call_delta=0.20, expiry_days=45)
    for n in notes: print("NOTE:", n)
    for i in intents: print("INTENT:", asdict(i))