# backend/sim/darkpool_sim.py
"""
Dark-Pool Execution Simulator
-----------------------------
Simulates slicing a parent order across multiple dark-pool venues with:
  • Midpoint matching vs NBBO, minimum size, price-improvement rules
  • Poisson contra-liquidity arrivals with lognormal size
  • Venue "toxicity" → adverse selection (post-trade markouts)
  • Spread & mid-price microstructure dynamics (random walk w/ volatility)
  • Child-order scheduler & router (parallel venues with weights)
  • Per-fill audit envelopes (SHA-256) + optional Merkle ledger append
  • Summary metrics: fill rate, VWAP, slippage, effective spread capture,
    markouts (+1s/+5s/+60s), venue hit ratios, time-to-fill, fees.

Dependencies
  • numpy

Usage
-----
from backend.sim.darkpool_sim import (
    DarkPoolConfig, RouterConfig, ParentOrder, MarketSimConfig, DarkSim
)

venues = [
    DarkPoolConfig(venue_id="DP1", min_qty=500, match_rate=0.65, fee_bps=0.2, toxicity_beta=0.6),
    DarkPoolConfig(venue_id="DP2", min_qty=1000, match_rate=0.45, fee_bps=0.0, toxicity_beta=0.3),
    DarkPoolConfig(venue_id="DP3", min_qty=200, match_rate=0.80, fee_bps=0.1, toxicity_beta=0.9),
]

router = RouterConfig(
    venue_weights={"DP1": 0.5, "DP2": 0.3, "DP3": 0.2},
    child_target=1500, child_jitter=0.25, max_parallel=2, min_slice_ms=250
)

market = MarketSimConfig(
    s0=100.00, spread_bps=8.0, vol_bps_sqrtsec=25.0, dt_ms=50,
    contra_lambda_per_s=2.0, contra_logn_mu=7.0, contra_logn_sigma=0.8,
    seed=42
)

parent = ParentOrder(symbol="AAPL", side="BUY", qty=25000, start_ms=0, end_ms=60_000)

sim = DarkSim(venues, router, market)
result = sim.run(parent)
print(result["summary"])
# Per-fill envelopes in result["fills"]; venue stats in result["venues"]
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------- Bus hook (optional) -----------------------

try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        # Safe stub
        head = {k: payload[k] for k in ("ts", "symbol", "venue", "qty", "px")} if isinstance(payload, dict) else {}
        print(f"[stub publish_stream] {stream} <- {json.dumps(head)[:200]}")

def _ledger_append(payload: Dict[str, Any], ledger_path: Optional[str]) -> None:
    if not ledger_path:
        return
    try:
        from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        MerkleLedger(ledger_path).append({"type": "darkpool_fill", "payload": payload})
    except Exception:
        pass


# ----------------------- Config models -----------------------

@dataclass
class DarkPoolConfig:
    venue_id: str
    min_qty: int = 200                             # minimum executable size
    match_rate: float = 0.6                        # baseline crossing probability on contra arrival (0..1)
    fee_bps: float = 0.0                           # taker fee in bps on notional
    price_improve_bps: float = 1.0                 # minimum price improvement vs NBBO (bps of spread)
    min_rest_ms: int = 50                          # minimum time resting before eligible
    toxicity_beta: float = 0.5                     # markout drift sensitivity (>0 => more adverse)
    midpoint_only: bool = True                     # price = midpoint (default)
    priority: str = "pro_rata"                     # "fifo" | "pro_rata"

@dataclass
class RouterConfig:
    venue_weights: Dict[str, float]
    child_target: int = 1000                       # target child qty
    child_jitter: float = 0.2                      # ± percentage randomization
    max_parallel: int = 2                          # max venues per slice
    min_slice_ms: int = 200                        # minimum gap between new children
    stream_fills: str = "STREAM_SIM_FILLS"
    ledger_path: Optional[str] = None

@dataclass
class MarketSimConfig:
    s0: float = 100.0
    spread_bps: float = 10.0                       # average quoted spread (bps of price)
    vol_bps_sqrtsec: float = 20.0                  # mid-vol in bps / sqrt(second)
    dt_ms: int = 50                                # time step
    contra_lambda_per_s: float = 1.5               # Poisson rate of contra arrivals per second per venue
    contra_logn_mu: float = 7.0                    # lognormal size (mu) ~ exp(mu) mean scale
    contra_logn_sigma: float = 0.75                # lognormal dispersion
    spread_mean_revert: float = 0.2                # AR(1) for spread (toward spread_bps)
    jump_prob_per_s: float = 0.02                  # occasional jump in mid
    jump_bps_mean: float = 40.0                    # avg jump size (bps)
    seed: Optional[int] = 7

@dataclass
class ParentOrder:
    symbol: str
    side: str                  # "BUY" or "SELL"
    qty: int
    start_ms: int
    end_ms: int
    limit_px: Optional[float] = None               # None => no hard limit (midpoint cross)


# ----------------------- Core simulator -----------------------

class DarkSim:
    def __init__(self, venues: List[DarkPoolConfig], router: RouterConfig, market: MarketSimConfig) -> None:
        self.venues = {v.venue_id: v for v in venues}
        self.router = router
        self.mkt = market
        self.rng = np.random.default_rng(market.seed)

        # Normalize router weights
        wsum = sum(max(0.0, w) for w in router.venue_weights.values())
        if wsum <= 0:
            raise ValueError("Router venue_weights must have positive mass")
        self.weights = {k: float(w) / wsum for k, w in router.venue_weights.items() if k in self.venues}

        if len(self.weights) == 0:
            raise ValueError("Router venue_weights reference unknown venues")

    # ------------- public API -------------

    def run(self, order: ParentOrder) -> Dict[str, Any]:
        """Run the simulation for a single parent order window."""
        side_sign = +1 if order.side.upper() == "BUY" else -1
        T_ms = int(order.end_ms - order.start_ms)
        steps = max(1, int(np.ceil(T_ms / self.mkt.dt_ms)))
        times = np.arange(steps, dtype=int) * self.mkt.dt_ms

        # Simulate market mid & spread
        mid, spread = self._simulate_market_path(steps)

        # State
        remaining = int(order.qty)
        last_child_ts = -10**9
        live_children: Dict[str, Dict[str, Any]] = {}  # venue_id -> {qty, ts_in}
        fills: List[Dict[str, Any]] = []
        fees_total = 0.0

        for k, t in enumerate(times):
            now_ms = int(order.start_ms + t)
            px_mid = float(mid[k])
            px_bid = px_mid - 0.5 * px_mid * (spread[k] / 1e4)
            px_ask = px_mid + 0.5 * px_mid * (spread[k] / 1e4)

            # Child scheduling
            if remaining > 0 and (now_ms - last_child_ts) >= self.router.min_slice_ms:
                # Size with jitter
                base = self.router.child_target
                jitter = 1.0 + self.rng.uniform(-self.router.child_jitter, self.router.child_jitter)
                child_qty = int(max(1, min(remaining, round(base * jitter))))
                # Select venues (top-k by weights sampling without replacement)
                venues = self._sample_venues(self.router.max_parallel)
                split = self._pro_rata_split(child_qty, [self.weights[v] for v in venues])
                for v_id, q in zip(venues, split):
                    if q <= 0:
                        continue
                    live_children[v_id] = {"qty": int(q), "ts_in": now_ms}
                remaining -= int(sum(split))
                last_child_ts = now_ms

            # Venue matching
            for v_id, v in self.venues.items():
                child = live_children.get(v_id)
                if not child:
                    continue
                if (now_ms - child["ts_in"]) < v.min_rest_ms:
                    continue

                # Contra arrival?
                lam_dt = self.mkt.contra_lambda_per_s * (self.mkt.dt_ms / 1000.0)
                # Binomial thinning approx for one step
                if self.rng.random() < min(1.0, lam_dt):
                    contra_qty = int(np.maximum(1.0, self.rng.lognormal(self.mkt.contra_logn_mu, self.mkt.contra_logn_sigma)))
                    # Order-size gate & match rate
                    eligible = (contra_qty >= v.min_qty) and (self.rng.random() < v.match_rate)

                    if eligible:
                        tradable = min(child["qty"], contra_qty)
                        # Price: midpoint with required improvement vs NBBO
                        px = px_mid
                        if v.midpoint_only:
                            # Price improvement check: midpoint improves by 50% of spread; ensure >= threshold
                            eff_impr_bps = 50.0  # midpoint improvement vs spread (bps of spread)
                            if eff_impr_bps < v.price_improve_bps:
                                continue  # skip if not enough price improvement (unlikely with midpoint)
                        # Limit check
                        if order.limit_px is not None:
                            if side_sign > 0 and px > order.limit_px:
                                continue
                            if side_sign < 0 and px < order.limit_px:
                                continue

                        # Execute
                        fill_qty = int(tradable)
                        child["qty"] -= fill_qty
                        if child["qty"] <= 0:
                            live_children.pop(v_id, None)

                        # Fees
                        notional = float(fill_qty) * px
                        fee = notional * (v.fee_bps / 1e4)
                        fees_total += fee

                        env = self._fill_envelope(
                            ts_ms=now_ms,
                            symbol=order.symbol,
                            venue=v_id,
                            side=order.side.upper(),
                            qty=fill_qty,
                            px=px,
                            px_mid=px_mid,
                            px_bid=px_bid,
                            px_ask=px_ask,
                            fee=fee,
                            step_index=k,
                        )
                        publish_stream(self.router.stream_fills, env)
                        _ledger_append(env, self.router.ledger_path)
                        fills.append(env)

            # Early exit if done
            if remaining <= 0 and len(live_children) == 0:
                break

        # Metrics
        metrics, per_venue = self._compute_metrics(order, fills, mid, spread, times)
        metrics["fees_total"] = float(fees_total)
        return {"summary": metrics, "venues": per_venue, "fills": fills, "path": {"mid": mid.tolist(), "spread_bps": spread.tolist(), "times_ms": (order.start_ms + times).tolist()}}

    # ------------- market path -------------

    def _simulate_market_path(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        mid = np.empty(steps, dtype=float)
        spr = np.empty(steps, dtype=float)
        mid[0] = float(self.mkt.s0)
        spr[0] = float(self.mkt.spread_bps)
        vol = self.mkt.vol_bps_sqrtsec / 1e4  # convert bps to fraction per sqrt(sec)
        dt_s = self.mkt.dt_ms / 1000.0
        sdt = vol * math.sqrt(dt_s)
        for k in range(1, steps):
            # Mid random walk + occasional jump
            d = sdt * self.rng.standard_normal()
            if self.rng.random() < (self.mkt.jump_prob_per_s * dt_s):
                jump = (self.mkt.jump_bps_mean / 1e4) * self.rng.choice([-1.0, 1.0])
                d += jump
            mid[k] = max(0.01, mid[k - 1] * (1.0 + d))

            # Spread mean-reverting AR(1) around baseline
            spr[k] = max(1.0, self.mkt.spread_bps + self.mkt.spread_mean_revert * (spr[k - 1] - self.mkt.spread_bps) + self.rng.normal(0.0, 0.5))
        return mid, spr

    # ------------- routing helpers -------------

    def _sample_venues(self, k: int) -> List[str]:
        keys = list(self.weights.keys())
        w = np.array([self.weights[v] for v in keys], dtype=float)
        p = w / w.sum()
        # without replacement proportional draw
        idx = list(range(len(keys)))
        chosen: List[str] = []
        p_work = p.copy()
        for _ in range(min(k, len(keys))):
            i = int(self.rng.choice(idx, p=p_work / p_work.sum()))
            chosen.append(keys[i])
            # remove i
            mask = np.ones_like(p_work, dtype=bool)
            mask[i] = False
            p_work = p_work[mask]
            idx = [j for j in idx if j != i]
            keys = [keys[j] for j in range(len(keys)) if j != i]
        return chosen

    @staticmethod
    def _pro_rata_split(total: int, weights: List[float]) -> List[int]:
        w = np.maximum(0.0, np.array(weights, dtype=float))
        if w.sum() <= 0:
            share = np.full(len(weights), total // max(1, len(weights)), dtype=int)
            share[0] += total - share.sum()
            return share.tolist()
        raw = total * (w / w.sum())
        floor = np.floor(raw).astype(int)
        rem = int(total - floor.sum())
        # distribute remainder by largest fractional part
        frac_idx = np.argsort(-(raw - floor))
        floor[frac_idx[:rem]] += 1
        return floor.tolist()

    # ------------- envelopes & metrics -------------

    @staticmethod
    def _hash_env(env: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(env, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()).hexdigest()

    def _fill_envelope(
        self,
        *,
        ts_ms: int,
        symbol: str,
        venue: str,
        side: str,
        qty: int,
        px: float,
        px_mid: float,
        px_bid: float,
        px_ask: float,
        fee: float,
        step_index: int,
    ) -> Dict[str, Any]:
        env = {
            "ts": int(ts_ms),
            "adapter": "darkpool_sim",
            "symbol": symbol.upper(),
            "venue": venue,
            "side": side,
            "qty": int(qty),
            "px": float(px),
            "px_mid": float(px_mid),
            "px_bid": float(px_bid),
            "px_ask": float(px_ask),
            "fee": float(fee),
            "step": int(step_index),
            "version": 1,
        }
        env["hash"] = self._hash_env(env)
        return env

    def _compute_metrics(
        self,
        order: ParentOrder,
        fills: List[Dict[str, Any]],
        mid_path: np.ndarray,
        spread_path: np.ndarray,
        times: np.ndarray,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if len(fills) == 0:
            return (
                {
                    "filled_qty": 0,
                    "fill_rate": 0.0,
                    "vwap_px": None,
                    "arrival_mid": float(mid_path[0]),
                    "slippage_bps": None,
                    "eff_spread_capture_bps": None,
                    "markout_1s_bps": None,
                    "markout_5s_bps": None,
                    "markout_60s_bps": None,
                    "time_to_fill_ms": None,
                },
                {},
            )

        side_sign = +1 if order.side.upper() == "BUY" else -1
        arrival_mid = float(mid_path[0])

        qty = np.array([f["qty"] for f in fills], dtype=float)
        pxs = np.array([f["px"] for f in fills], dtype=float)
        t_idx = np.array([f["step"] for f in fills], dtype=int)

        notional = float((qty * pxs).sum())
        filled = int(qty.sum())
        vwap = notional / max(1.0, filled)

        # Slippage vs arrival mid (signed by side)
        slippage_bps = side_sign * (vwap - arrival_mid) / arrival_mid * 1e4

        # Effective spread capture (fill vs mid at fill time)
        mid_at_fill = mid_path[t_idx]
        esc_bps = side_sign * (mid_at_fill - pxs) / mid_at_fill * 1e4  # positive means price improvement
        eff_spread_capture_bps = float(np.average(esc_bps, weights=qty))

        # Markouts (expected adverse selection): use realized path +/- adversarial drift proxy
        mark1s = self._markout_bps(mid_path, times, fills, horizon_s=1.0, side_sign=side_sign)
        mark5s = self._markout_bps(mid_path, times, fills, horizon_s=5.0, side_sign=side_sign)
        mark60s = self._markout_bps(mid_path, times, fills, horizon_s=60.0, side_sign=side_sign)

        # Time to finish (last fill timestamp)
        t_last = int(times[int(t_idx[-1])])
        t_first = int(times[int(t_idx[0])])
        ttf = int((t_last - times[0])) if filled >= order.qty else None

        # Venue stats & toxicity-adjusted markout
        per_venue: Dict[str, Dict[str, Any]] = {}
        for v_id in self.venues.keys():
            mask = np.array([f["venue"] == v_id for f in fills], dtype=bool)
            qv = qty[mask]
            pv = pxs[mask]
            iv = t_idx[mask]
            if qv.sum() <= 0:
                continue
            notional_v = float((qv * pv).sum())
            fill_q = int(qv.sum())
            vwap_v = notional_v / max(1.0, fill_q)
            mid_v = mid_path[iv]
            esc_v = side_sign * (mid_v - pv) / mid_v * 1e4
            per_venue[v_id] = {
                "filled_qty": fill_q,
                "vwap_px": float(vwap_v),
                "eff_spread_capture_bps": float(np.average(esc_v, weights=qv)),
                "hit_ratio": float(fill_q) / float(order.qty),
            }

        summary = {
            "filled_qty": filled,
            "fill_rate": float(filled) / float(order.qty),
            "vwap_px": float(vwap),
            "arrival_mid": float(arrival_mid),
            "slippage_bps": float(slippage_bps),
            "eff_spread_capture_bps": float(eff_spread_capture_bps),
            "markout_1s_bps": float(mark1s) if mark1s is not None else None,
            "markout_5s_bps": float(mark5s) if mark5s is not None else None,
            "markout_60s_bps": float(mark60s) if mark60s is not None else None,
            "time_to_fill_ms": ttf,
        }
        return summary, per_venue

    def _markout_bps(
        self,
        mid_path: np.ndarray,
        times: np.ndarray,
        fills: List[Dict[str, Any]],
        *,
        horizon_s: float,
        side_sign: int,
    ) -> Optional[float]:
        if len(fills) == 0:
            return None
        dt_s = self.mkt.dt_ms / 1000.0
        h_steps = int(round(horizon_s / dt_s))
        qty = np.array([f["qty"] for f in fills], dtype=float)
        pxs = np.array([f["px"] for f in fills], dtype=float)
        t_idx = np.array([f["step"] for f in fills], dtype=int)

        t_end = t_idx + h_steps
        t_end = np.minimum(t_end, len(mid_path) - 1)

        # Venue-specific adverse drift from toxicity_beta
        beta = np.array([self.venues[f["venue"]].toxicity_beta for f in fills], dtype=float)
        # Map beta into extra drift per horizon (bps): higher beta → more negative markout for BUY
        tox_bps = beta * 0.5 * math.sqrt(max(horizon_s, 1e-9))  # simple concave scaling
        # Convert to price drift in absolute terms at fill mid
        mid_fill = mid_path[t_idx]
        tox_px = (tox_bps / 1e4) * mid_fill * (-side_sign)  # BUY → negative, SELL → positive

        px_future = mid_path[t_end] + tox_px
        mark = side_sign * (px_future - pxs) / pxs * 1e4  # bps
        return float(np.average(mark, weights=qty))


# ----------------------- __main__ quick demo -----------------------

if __name__ == "__main__":
    venues = [
        DarkPoolConfig(venue_id="DP1", min_qty=500, match_rate=0.65, fee_bps=0.2, toxicity_beta=0.6),
        DarkPoolConfig(venue_id="DP2", min_qty=1000, match_rate=0.45, fee_bps=0.0, toxicity_beta=0.3),
        DarkPoolConfig(venue_id="DP3", min_qty=200, match_rate=0.80, fee_bps=0.1, toxicity_beta=0.9),
    ]
    router = RouterConfig(
        venue_weights={"DP1": 0.5, "DP2": 0.3, "DP3": 0.2},
        child_target=1500, child_jitter=0.25, max_parallel=2, min_slice_ms=250
    )
    market = MarketSimConfig(
        s0=100.00, spread_bps=8.0, vol_bps_sqrtsec=25.0, dt_ms=50,
        contra_lambda_per_s=2.0, contra_logn_mu=7.0, contra_logn_sigma=0.8, seed=42
    )
    parent = ParentOrder(symbol="AAPL", side="BUY", qty=25000, start_ms=0, end_ms=60_000)
    sim = DarkSim(venues, router, market)
    res = sim.run(parent)
    print(json.dumps(res["summary"], indent=2))