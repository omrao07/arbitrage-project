# backend/execution/cost_model_v2.py
from __future__ import annotations

import math, json, os, time, statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

# -------- optional numpy (graceful) ------------------------------------------
HAVE_NP = True
try:
    import numpy as np  # type: ignore
except Exception:
    HAVE_NP = False
    np = None  # type: ignore

# -------- helpers -------------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def _safe(x: Optional[float], d=0.0) -> float:
    try:
        return float(x) # type: ignore
    except Exception:
        return float(d)

# -------- inputs / outputs ----------------------------------------------------
@dataclass
class CostInputs:
    # order & market context
    side: str                   # "buy" | "sell"
    qty: float                  # absolute shares/contracts
    mid_px: float               # current mid
    bid_px: Optional[float] = None
    ask_px: Optional[float] = None
    spread_bps: Optional[float] = None     # if bid/ask not supplied
    adv_shares: Optional[float] = None     # average daily volume (shares)
    child_ratio_of_parent: float = 1.0     # 0..1 (for children)
    # microstructure & signals
    ob_imbalance: Optional[float] = None   # [-1, +1]
    est_latency_ms: Optional[int] = 30
    realized_vol_bps_1m: Optional[float] = 8.0   # 1-min volatility bps
    tick_size: Optional[float] = None
    # venue / fees
    venue: Optional[str] = None
    taker_fee_bps: Optional[float] = None
    maker_rebate_bps: Optional[float] = None
    maker_fill_prob: Optional[float] = 0.0  # probability a resting order gets filled
    # borrow / funding
    borrow_rate_bps_per_day: Optional[float] = 0.0  # shorting fee
    holding_days: Optional[float] = 0.0
    # fx
    px_ccy: str = "USD"
    pnl_ccy: str = "USD"
    fx_rate: Optional[float] = 1.0          # px_ccy → pnl_ccy

@dataclass
class CostBreakdown:
    # currency is pnl_ccy
    spread: float
    fees: float
    impact: float
    latency: float
    adverse: float
    borrow: float
    fx_adj: float
    total: float

    def bps(self, notional: float) -> Dict[str, float]:
        if notional <= 0:
            return {k: 0.0 for k in ["spread","fees","impact","latency","adverse","borrow","fx_adj","total"]}
        to_bps = lambda v: 1e4 * float(v) / float(notional)
        return {
            "spread": to_bps(self.spread),
            "fees": to_bps(self.fees),
            "impact": to_bps(self.impact),
            "latency": to_bps(self.latency),
            "adverse": to_bps(self.adverse),
            "borrow": to_bps(self.borrow),
            "fx_adj": to_bps(self.fx_adj),
            "total": to_bps(self.total),
        }

# -------- config & defaults ---------------------------------------------------
@dataclass
class CostModelV2Config:
    # Market impact (square-root; Kyle/AC-style)
    # cost_px ≈ k * sigma * mid * sqrt(q / ADV)
    k_impact: float = 1.0
    # Latency slip: sigma_1m * mid * sqrt(latency / 60s) * c_lat
    k_latency: float = 0.5
    # Adverse selection: scale imbalance to expected drift over short horizon
    k_adverse: float = 0.25
    # Spread default (if bid/ask absent)
    default_spread_bps: float = 5.0
    # Fee schedule fallbacks
    default_taker_fee_bps: float = 0.5
    default_maker_rebate_bps: float = 0.2
    # Maker vs taker blend for marketable vs passive
    maker_weight: float = 0.0   # 0=taker only; 1=maker only. You can override per call.
    # FX neutralization
    assume_fx_neutral: bool = True

class CostModelV2:
    """
    Second-gen execution cost model with microstructure hooks.
    Costs are positive for both buys/sells (portfolio 'cost' convention).
    Return currency is inputs.pnl_ccy (using inputs.fx_rate).
    """
    def __init__(self, cfg: Optional[CostModelV2Config] = None):
        self.cfg = cfg or CostModelV2Config()
        # learned params from calibration
        self.learned_k: Optional[float] = None
        self.learned_latency_k: Optional[float] = None

    # ---- primitives ---------------------------------------------------------
    def _spread_cost(self, ci: CostInputs) -> float:
        if ci.bid_px and ci.ask_px:
            spr = max(0.0, ci.ask_px - ci.bid_px)
        else:
            spr = (self.cfg.default_spread_bps if ci.spread_bps is None else ci.spread_bps) * ci.mid_px / 1e4
        # expected cost ≈ half-spread for marketable, discounted by maker_weight*maker_fill_prob
        taker_half = 0.5 * spr
        maker_gain = (ci.maker_rebate_bps or self.cfg.default_maker_rebate_bps) * ci.mid_px / 1e4
        w_maker = clamp(ci.maker_fill_prob or 0.0, 0.0, 1.0) * clamp(self.cfg.maker_weight, 0.0, 1.0)
        # If we truly rest, we *earn* rebate and avoid half-spread; approximate blended expectation:
        exp_per_share = (1.0 - w_maker) * taker_half - w_maker * maker_gain
        return abs(ci.qty) * max(0.0, exp_per_share)

    def _fee_cost(self, ci: CostInputs) -> float:
        # fees positive cost; rebates reduce cost
        taker = (ci.taker_fee_bps if ci.taker_fee_bps is not None else self.cfg.default_taker_fee_bps)
        maker = (ci.maker_rebate_bps if ci.maker_rebate_bps is not None else self.cfg.default_maker_rebate_bps)
        w_maker = clamp(ci.maker_fill_prob or 0.0, 0.0, 1.0) * clamp(self.cfg.maker_weight, 0.0, 1.0)
        per_share_bps = (1.0 - w_maker) * taker - w_maker * maker
        return abs(ci.qty) * ci.mid_px * per_share_bps / 1e4

    def _impact_cost(self, ci: CostInputs) -> float:
        # Square-root impact: k * sigma * mid * sqrt(q/ADV)
        adv = max(1.0, _safe(ci.adv_shares, 0.0))
        q = abs(ci.qty)
        if adv <= 1.0 or q <= 0.0:
            return 0.0
        sigma = (_safe(ci.realized_vol_bps_1m, 8.0) / 1e4)  # fraction
        k = self.learned_k if self.learned_k is not None else self.cfg.k_impact
        per_share_px = k * sigma * ci.mid_px * math.sqrt(q / adv)
        return q * per_share_px

    def _latency_cost(self, ci: CostInputs) -> float:
        # Expected slip from latency during volatile periods:
        # per_share ≈ k_lat * sigma_1m * mid * sqrt(latency / 60s)
        latency_s = max(0.0, (_safe(ci.est_latency_ms, 30) / 1000.0))
        if latency_s <= 0.0:
            return 0.0
        sigma = (_safe(ci.realized_vol_bps_1m, 8.0) / 1e4)
        k_lat = self.learned_latency_k if self.learned_latency_k is not None else self.cfg.k_latency
        per_share_px = k_lat * sigma * ci.mid_px * math.sqrt(latency_s / 60.0)
        return abs(ci.qty) * per_share_px

    def _adverse_selection(self, ci: CostInputs) -> float:
        # Map L1 imbalance to expected short-horizon drift (very coarse):
        # drift_bps ≈ k_adv * imbalance * 5 (tunable scale)
        imb = clamp(_safe(ci.ob_imbalance, 0.0), -1.0, 1.0)
        k_adv = self.cfg.k_adverse
        drift_bps = k_adv * imb * 5.0
        # cost depends on side: for BUY, positive imbalance (ask>bid) hurts; for SELL, negative hurts.
        sign = +1.0 if ci.side.lower() in ("buy","b","long") else -1.0
        expected_bps_cost = max(0.0, sign * drift_bps)  # only count adverse component
        return abs(ci.qty) * ci.mid_px * expected_bps_cost / 1e4

    def _borrow_cost(self, ci: CostInputs) -> float:
        r = _safe(ci.borrow_rate_bps_per_day, 0.0)
        d = max(0.0, _safe(ci.holding_days, 0.0))
        if r <= 0.0 or d <= 0.0:
            return 0.0
        return abs(ci.qty) * ci.mid_px * (r * d) / 1e4

    def _fx_adjust(self, amount_px_ccy: float, ci: CostInputs) -> float:
        # Convert to pnl_ccy; if assume_fx_neutral, no additional penalty.
        fx = _safe(ci.fx_rate, 1.0)
        if fx <= 0: fx = 1.0
        converted = amount_px_ccy * fx
        return converted

    # ---- API ----------------------------------------------------------------
    def estimate_child(self, ci: CostInputs) -> CostBreakdown:
        """
        Estimate *child* order execution cost in pnl_ccy for the provided market state.
        """
        notional_px_ccy = abs(ci.qty) * ci.mid_px
        spread = self._spread_cost(ci)
        fees = self._fee_cost(ci)
        impact = self._impact_cost(ci)
        latency = self._latency_cost(ci)
        adverse = self._adverse_selection(ci)
        borrow = self._borrow_cost(ci)

        px_ccy_sum = spread + fees + impact + latency + adverse + borrow
        pnl_ccy_sum = self._fx_adjust(px_ccy_sum, ci)

        return CostBreakdown(
            spread=self._fx_adjust(spread, ci),
            fees=self._fx_adjust(fees, ci),
            impact=self._fx_adjust(impact, ci),
            latency=self._fx_adjust(latency, ci),
            adverse=self._fx_adjust(adverse, ci),
            borrow=self._fx_adjust(borrow, ci),
            fx_adj=0.0,                   # already applied per-component
            total=pnl_ccy_sum,
        )

    def estimate_parent(self, ci: CostInputs, *, slices: int = 10) -> CostBreakdown:
        """
        Estimate *parent* order cost by simulating slices (simple TWAP-like),
        compounding impact concavity and decreasing remaining ADV footprint.
        """
        q_total = abs(ci.qty)
        if q_total <= 0:
            return CostBreakdown(0,0,0,0,0,0,0,0)

        spread = fees = impact = latency = adverse = borrow = 0.0
        remaining = q_total
        # evenly split for now; can plug your actual schedule
        per = q_total / float(max(1, slices))
        for i in range(slices):
            if remaining <= 0: break
            dq = min(per, remaining)
            remaining -= dq
            w = dq / q_total
            # dampen impact as remaining/ADV shrinks within the loop:
            sub_ci = CostInputs(**{**asdict(ci), "qty": dq, "child_ratio_of_parent": w})
            sub = self.estimate_child(sub_ci)
            spread  += sub.spread
            fees    += sub.fees
            impact  += sub.impact * (0.8 + 0.2 * (remaining / q_total))  # mild decay during progression
            latency += sub.latency
            adverse += sub.adverse
            borrow  += sub.borrow

        total = spread + fees + impact + latency + adverse + borrow
        return CostBreakdown(spread, fees, impact, latency, adverse, borrow, 0.0, total)

    # ---- venue scoring -------------------------------------------------------
    def venue_score(self, ci: CostInputs, venues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank venues by expected cost for this child order.
        `venues` is a list of dicts with overrides like:
          {"venue":"XNAS","taker_fee_bps":0.3,"maker_rebate_bps":0.2,"maker_fill_prob":0.25,"spread_bps":4.0}
        Returns list sorted by ascending total cost.
        """
        results = []
        for v in venues:
            v_ci = CostInputs(**{**asdict(ci), **v})
            br = self.estimate_child(v_ci)
            results.append({
                "venue": v.get("venue",""),
                "total_cost": br.total,
                "cost_bps": br.bps(abs(v_ci.qty)*v_ci.mid_px)["total"],
                "breakdown": asdict(br),
            })
        results.sort(key=lambda x: (x["total_cost"], x["cost_bps"]))
        return results

    # ---- calibration (optional) ---------------------------------------------
    def calibrate_from_fills(self, fills: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calibrate k_impact and k_latency from realized slippage on historical fills.
        Expects each fill dict to include:
         { "side":"buy|sell", "qty":..., "mid_before":..., "exec_px":..., "adv":..., "latency_ms":..., "rv_bps_1m":... }
        """
        # Build simple linear fits:
        # slip_per_share ≈ k * sigma * mid * sqrt(q/ADV) + k_lat * sigma * mid * sqrt(latency/60)
        x_imp, x_lat, y = [], [], []
        for f in fills:
            try:
                side = f["side"]
                sgn = 1.0 if str(side).lower().startswith("b") else -1.0
                qty = abs(float(f["qty"]))
                mid = float(f["mid_before"])
                exec_px = float(f["exec_px"])
                adv = max(1.0, float(f.get("adv", 0.0)))
                latency_ms = max(0.0, float(f.get("latency_ms", 0.0)))
                rv = float(f.get("rv_bps_1m", 8.0)) / 1e4
                slip = sgn * (exec_px - mid)  # positive = cost for buys
                # features
                xi = rv * mid * math.sqrt(qty / adv)
                xl = rv * mid * math.sqrt((latency_ms/1000.0) / 60.0)
                x_imp.append(xi); x_lat.append(xl); y.append(slip)
            except Exception:
                continue

        if not y or sum(abs(v) for v in y) == 0:
            return {"k_impact": self.cfg.k_impact, "k_latency": self.cfg.k_latency}

        # simple 2D linear regression (no intercept) by normal equations
        if HAVE_NP:
            X = np.vstack([np.array(x_imp), np.array(x_lat)]).T  # type: ignore # (n,2)
            Y = np.array(y) # type: ignore
            # Solve min ||X k - Y||^2 → k = (X^T X)^-1 X^T Y
            try:
                k_hat = np.linalg.lstsq(X, Y, rcond=None)[0] # type: ignore
                self.learned_k = float(k_hat[0])
                self.learned_latency_k = float(k_hat[1])
            except Exception:
                pass
        else:
            # manual normal equations
            s11 = sum(v*v for v in x_imp)
            s22 = sum(v*v for v in x_lat)
            s12 = sum(a*b for a,b in zip(x_imp, x_lat))
            sy1 = sum(a*b for a,b in zip(x_imp, y))
            sy2 = sum(a*b for a,b in zip(x_lat, y))
            det = s11*s22 - s12*s12
            if abs(det) > 1e-12:
                self.learned_k = (s22*sy1 - s12*sy2) / det
                self.learned_latency_k = (s11*sy2 - s12*sy1) / det

        return {
            "k_impact": float(self.learned_k if self.learned_k is not None else self.cfg.k_impact),
            "k_latency": float(self.learned_latency_k if self.learned_latency_k is not None else self.cfg.k_latency)
        }

# -------- CLI ----------------------------------------------------------------
def _cli():
    import argparse, sys
    ap = argparse.ArgumentParser("cost_model_v2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="Estimate cost for a child/parent order")
    e.add_argument("--mode", choices=["child","parent"], default="child")
    e.add_argument("--qty", type=float, required=True)
    e.add_argument("--side", required=True, choices=["buy","sell"])
    e.add_argument("--mid", type=float, required=True)
    e.add_argument("--bid", type=float, default=None)
    e.add_argument("--ask", type=float, default=None)
    e.add_argument("--spread-bps", type=float, default=None)
    e.add_argument("--adv", type=float, default=None)
    e.add_argument("--taker-fee-bps", type=float, default=None)
    e.add_argument("--maker-rebate-bps", type=float, default=None)
    e.add_argument("--maker-prob", type=float, default=0.0)
    e.add_argument("--imb", type=float, default=None)
    e.add_argument("--latency-ms", type=int, default=30)
    e.add_argument("--rv-bps-1m", type=float, default=8.0)
    e.add_argument("--borrow-bps-day", type=float, default=0.0)
    e.add_argument("--days", type=float, default=0.0)
    e.add_argument("--fx", type=float, default=1.0)
    e.add_argument("--parent-slices", type=int, default=10)

    v = sub.add_parser("venues", help="Rank venues by expected child cost")
    v.add_argument("--qty", type=float, required=True)
    v.add_argument("--side", required=True, choices=["buy","sell"])
    v.add_argument("--mid", type=float, required=True)
    v.add_argument("--adv", type=float, default=None)
    v.add_argument("--rv-bps-1m", type=float, default=8.0)
    v.add_argument("--venues-json", required=True, help="Path to JSON list of venue overrides")

    c = sub.add_parser("calibrate", help="Learn k_impact / k_latency from fills JSONL")
    c.add_argument("--fills-jsonl", required=True)

    args = ap.parse_args()
    cm = CostModelV2()

    if args.cmd == "estimate":
        ci = CostInputs(
            side=args.side, qty=args.qty, mid_px=args.mid,
            bid_px=args.bid, ask_px=args.ask, spread_bps=args.spread_bps,
            adv_shares=args.adv, taker_fee_bps=args.taker_fee_bps, maker_rebate_bps=args.maker_rebate_bps,
            maker_fill_prob=args.maker_prob, ob_imbalance=args.imb, est_latency_ms=args.latency_ms,
            realized_vol_bps_1m=args.rv_bps_1m, borrow_rate_bps_per_day=args.borrow_bps_day, holding_days=args.days,
            fx_rate=args.fx
        )
        br = cm.estimate_child(ci) if args.mode=="child" else cm.estimate_parent(ci, slices=args.parent_slices)
        notional = abs(ci.qty)*ci.mid_px
        out = {
            "inputs": asdict(ci),
            "breakdown_ccy": asdict(br),
            "breakdown_bps": br.bps(notional),
            "notional": notional,
        }
        print(json.dumps(out, indent=2))

    elif args.cmd == "venues":
        with open(args.venues_json, "r", encoding="utf-8") as f:
            venues = json.load(f)
        base = CostInputs(
            side=args.side, qty=args.qty, mid_px=args.mid, adv_shares=args.adv,
            realized_vol_bps_1m=args.rv_bps_1m
        )
        ranked = cm.venue_score(base, venues)
        print(json.dumps(ranked, indent=2))

    elif args.cmd == "calibrate":
        fills = []
        with open(args.fills_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    fills.append(json.loads(line))
                except Exception:
                    continue
        params = cm.calibrate_from_fills(fills)
        print(json.dumps({"learned": params}, indent=2))

if __name__ == "__main__":
    _cli()