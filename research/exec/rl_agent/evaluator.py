# research/exec/rl_agent/evaluator.py
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParentOrder:
    symbol: str
    side: str              # "BUY" or "SELL"
    qty: float             # total parent size (shares)
    start_idx: int         # index in market df to start executing
    end_idx: int           # inclusive index to stop (episode horizon)
    participation_cap: float = 0.10   # max % of bar volume
    price_band_bps: float = 50.0      # abs(price - ref) <= band (bps)
    child_qty_max_pct_remaining: float = 0.02  # ≤2% of remaining per child
    board_lot: int = 1                 # min lot multiple
    ref_px_col: str = "mid"            # reference price column for band checks


@dataclass(frozen=True)
class ChildDecision:
    qty: float           # signed (+buy, -sell)
    limit_px: Optional[float] = None   # None => marketable at mid/touch
    reason: str = "policy"


@dataclass(frozen=True)
class EpisodeResult:
    symbol: str
    side: str
    filled_qty: float
    avg_fill_px: float
    slippage_bps: float
    notional: float
    n_children: int
    violations: Dict[str, int]
    participation_max: float
    pnl_vs_vwap: float


class Guard:
    """
    Risk/compliance guard. Provide callables to enforce hard constraints.
    Return (allowed_qty, adjusted_px, violations_dict_increment)
    """
    def __init__(
        self,
        *,
        participation_cap: float,
        price_band_bps: float,
        child_qty_max_pct_remaining: float,
        board_lot: int,
        ref_px_col: str = "mid",
    ) -> None:
        self.participation_cap = participation_cap
        self.price_band_bps = price_band_bps
        self.child_pct_remaining = child_qty_max_pct_remaining
        self.board_lot = max(1, int(board_lot))
        self.ref_px_col = ref_px_col

    def apply(
        self,
        decision: ChildDecision,
        *,
        remaining: float,
        bar_volume: float,
        ref_px: float,
    ) -> Tuple[float, Optional[float], Dict[str, int]]:
        vio: Dict[str, int] = {}

        # Directional sign from decision.qty
        desired = float(decision.qty)

        # Child-size cap vs remaining
        cap_qty = self.child_pct_remaining * abs(remaining)
        if cap_qty <= 0:
            cap_qty = abs(remaining)
        if abs(desired) > cap_qty:
            desired = math.copysign(cap_qty, desired)
            vio["child_size"] = vio.get("child_size", 0) + 1

        # Participation guard vs bar volume
        if bar_volume and abs(desired) > self.participation_cap * bar_volume:
            desired = math.copysign(self.participation_cap * bar_volume, desired)
            vio["participation"] = vio.get("participation", 0) + 1

        # Board lot multiple
        if self.board_lot > 1:
            lot = self.board_lot
            desired = math.copysign((abs(desired) // lot) * lot, desired)

        # Price band guard (if limit provided)
        px = decision.limit_px
        if px is not None and ref_px > 0:
            band = self.price_band_bps * 1e-4 * ref_px
            if px > ref_px + band:
                px = ref_px + band
                vio["price_band"] = vio.get("price_band", 0) + 1
            elif px < ref_px - band:
                px = ref_px - band
                vio["price_band"] = vio.get("price_band", 0) + 1

        return desired, px, vio


def _sign(side: str) -> int:
    s = side.upper()
    if s == "BUY":
        return +1
    if s == "SELL":
        return -1
    raise ValueError(f"unknown side {side}")


def _fill_price(
    side_sign: int,
    limit_px: Optional[float],
    best_bid: float,
    best_ask: float,
    mid: float,
) -> Optional[float]:
    """
    Simple execution model:
    - If no limit: cross at touch (BUY -> ask, SELL -> bid)
    - If limit: if marketable, fill at touch; else None (no fill)
    """
    if limit_px is None:
        return best_ask if side_sign > 0 else best_bid
    # BUY: limit >= ask is marketable
    if side_sign > 0 and limit_px >= best_ask:
        return min(limit_px, best_ask)
    # SELL: limit <= bid is marketable
    if side_sign < 0 and limit_px <= best_bid:
        return max(limit_px, best_bid)
    return None  # not marketable within this bar


def evaluate_policy(
    market: pd.DataFrame,
    order: ParentOrder,
    policy_act: Callable[[Dict[str, float]], ChildDecision],
    *,
    seed: Optional[int] = None,
) -> Tuple[EpisodeResult, pd.DataFrame]:
    """
    Evaluate a policy on a historical episode.

    Parameters
    ----------
    market : DataFrame
        Must contain columns: ['bid','ask','mid','v','ts'] at least.
        Index is integer time step; use order.start_idx..order.end_idx inclusive.
    order : ParentOrder
    policy_act : function(state) -> ChildDecision
        The policy receives a dict state with:
        {
           't': step idx, 'remaining': float, 'bid': float, 'ask': float,
           'mid': float, 'bar_v': float, 'elapsed': float (0..1),
           'participation_today': float
        }

    Returns
    -------
    EpisodeResult, DataFrame(trade_log)
    """
    if seed is not None:
        np.random.seed(seed)

    # Slice episode
    start, end = int(order.start_idx), int(order.end_idx)
    if start < 0 or end >= len(market) or end < start:
        raise ValueError("invalid episode bounds")
    ep = market.iloc[start : end + 1].copy()

    req_cols = {"bid", "ask", "mid", "v"}
    missing = req_cols - set(c.lower() for c in ep.columns.map(str))
    # Try case-insensitive mapping
    colmap = {c.lower(): c for c in ep.columns}
    def col(name: str) -> str:
        return colmap.get(name, name)

    for c in ("bid", "ask", "mid", "v"):
        if col(c) not in ep.columns:
            raise ValueError(f"market missing column '{c}'")

    # State trackers
    side_sign = _sign(order.side)
    remaining = side_sign * order.qty  # signed remaining
    filled_qty = 0.0
    notional = 0.0
    n_children = 0
    participation_max = 0.0
    violations: Dict[str, int] = {}

    guard = Guard(
        participation_cap=order.participation_cap,
        price_band_bps=order.price_band_bps,
        child_qty_max_pct_remaining=order.child_qty_max_pct_remaining,
        board_lot=order.board_lot,
        ref_px_col=order.ref_px_col,
    )

    log_rows: List[Dict[str, float]] = []
    total_volume = 0.0

    for step, (_, row) in enumerate(ep.iterrows(), start=0):
        if abs(remaining) <= 0:
            break

        bid = float(row[col("bid")])
        ask = float(row[col("ask")])
        mid = float(row[col("mid")])
        bar_v = float(row[col("v")] or 0.0)
        total_volume += bar_v

        elapsed = step / max(1, (len(ep) - 1))
        today_participation = (abs(filled_qty) / total_volume) if total_volume > 0 else 0.0

        state = {
            "t": step + start,
            "remaining": remaining,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "bar_v": bar_v,
            "elapsed": elapsed,
            "participation_today": today_participation,
        }

        # Policy proposes a child order
        decision = policy_act(state)
        if not isinstance(decision, ChildDecision):
            raise TypeError("policy_act must return ChildDecision")

        # Apply guardrails
        allowed_qty, px_adj, vio = guard.apply(
            decision,
            remaining=remaining,
            bar_volume=bar_v,
            ref_px=row[col(order.ref_px_col)],
        )
        for k, v in vio.items():
            violations[k] = violations.get(k, 0) + v

        # Resolve fill price
        fill_px = _fill_price(side_sign, px_adj if decision.limit_px is not None else None, bid, ask, mid)

        filled = 0.0
        if fill_px is not None and abs(allowed_qty) > 0:
            # Fill limited by remaining
            filled = math.copysign(min(abs(allowed_qty), abs(remaining)), allowed_qty)
            # Update accumulators
            remaining -= filled
            filled_qty += filled
            notional += abs(filled) * float(fill_px)
            n_children += 1

        participation_max = max(participation_max, today_participation)

        log_rows.append(
            {
                "t": step + start,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "bar_v": bar_v,
                "decision_qty": decision.qty,
                "decision_px": (decision.limit_px if decision.limit_px is not None else np.nan),
                "allowed_qty": allowed_qty,
                "fill_px": (fill_px if fill_px is not None else np.nan),
                "filled": filled,
                "remaining": remaining,
                "violations_child_size": vio.get("child_size", 0),
                "violations_participation": vio.get("participation", 0),
                "violations_price_band": vio.get("price_band", 0),
            }
        )

    trade_log = pd.DataFrame(log_rows)

    # Metrics
    executed = abs(filled_qty)
    avg_fill_px = (notional / executed) if executed > 0 else np.nan
    ref_open = float(ep.iloc[0][col(order.ref_px_col)])
    slippage_bps = (avg_fill_px - ref_open) / ref_open * (10000.0) * (side_sign) if executed > 0 else np.nan

    # VWAP benchmark over executed horizon (volume-weighted mid as proxy)
    v = ep[col("v")].values.astype(float)
    px = ep[col("mid")].values.astype(float)
    vwap = (px * v).sum() / v.sum() if v.sum() > 0 else ref_open
    pnl_vs_vwap = (vwap - avg_fill_px) * (executed * side_sign) if executed > 0 else 0.0

    result = EpisodeResult(
        symbol=order.symbol,
        side=order.side,
        filled_qty=executed * side_sign,
        avg_fill_px=avg_fill_px if not math.isnan(avg_fill_px) else 0.0,
        slippage_bps=float(slippage_bps) if not (math.isnan(slippage_bps)) else 0.0,
        notional=notional,
        n_children=n_children,
        violations={
            "child_size": int(trade_log["violations_child_size"].sum()) if not trade_log.empty else 0,
            "participation": int(trade_log["violations_participation"].sum()) if not trade_log.empty else 0,
            "price_band": int(trade_log["violations_price_band"].sum()) if not trade_log.empty else 0,
        },
        participation_max=float(participation_max),
        pnl_vs_vwap=float(pnl_vs_vwap),
    )

    return result, trade_log


# ----------------------------- Helpers -----------------------------------

def summarize_many(
    market: pd.DataFrame,
    orders: List[ParentOrder],
    policy_act: Callable[[Dict[str, float]], ChildDecision],
) -> pd.DataFrame:
    """
    Run multiple episodes and return a summary DataFrame with one row per order.
    """
    rows: List[Dict[str, float]] = []
    for o in orders:
        res, _ = evaluate_policy(market, o, policy_act)
        row = asdict(res)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------- Example policy hooks --------------------------

def twap_policy(alpha: float = 1.0) -> Callable[[Dict[str, float]], ChildDecision]:
    """
    Simple TWAP-like policy: aims to finish linearly over time.
    alpha scales aggressiveness (1.0 = exact linear).
    """
    def act(state: Dict[str, float]) -> ChildDecision:
        remaining = state["remaining"]
        elapsed = max(1e-6, state["elapsed"])  # avoid 0
        steps_left_ratio = 1.0 - elapsed
        target_this_bar = alpha * remaining * (elapsed - (elapsed - 1.0 / max(1, 1 / elapsed)))
        # Fall back: proportional to remaining and inverse to time left
        qty = np.clip(target_this_bar, -abs(remaining), abs(remaining))
        return ChildDecision(qty=qty, limit_px=None, reason="twap")
    return act


def passive_inside_band(band_bps: float = 10.0) -> Callable[[Dict[str, float]], ChildDecision]:
    """
    Posts passively near mid within a band; marketable once price crosses.
    """
    def act(state: Dict[str, float]) -> ChildDecision:
        mid = float(state["mid"])
        side = 1 if state["remaining"] > 0 else -1
        px = mid * (1 - band_bps * 1e-4) if side > 0 else mid * (1 + band_bps * 1e-4)
        return ChildDecision(qty=0.02 * abs(state["remaining"]) * side, limit_px=px, reason="passive")
    return act