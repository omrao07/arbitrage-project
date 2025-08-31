# backend/portfolio/allocator.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Literal

Side = Literal["buy", "sell"]

# --------- Optional imports (graceful fallbacks) ----------
try:
    from backend.risk.drawdown_speed import DrawdownAlert  # type: ignore # suggested_risk_mult
except Exception:
    @dataclass
    class DrawdownAlert:
        level: str = "ok"
        suggested_risk_mult: float = 1.0

# (All other modules are optional; pass their outputs numerically via inputs)

# ==========================================================
# Data structures
# ==========================================================

@dataclass
class SymbolInfo:
    price: float                      # latest mark/mid
    vol_day: float = 0.02             # daily stdev of returns (e.g., 0.02 = 2%)
    adv: float = 1_000_000            # average daily volume (shares/contracts)
    spread_bps: float = 5.0           # quoted spread in bps of mid
    region: Optional[str] = None
    sector: Optional[str] = None

@dataclass
class PortfolioState:
    cash: float
    nav: float                        # current NAV (cash + MTM)
    positions: Dict[str, float]       # symbol -> qty (can be negative)
    last_price: Dict[str, float] = field(default_factory=dict)

@dataclass
class Constraints:
    max_gross: float = 0.50           # max gross exposure as fraction of NAV
    max_leverage: float = 1.5         # NAV leverage cap (gross/NAV)
    max_symbol_w: float = 0.07        # per-symbol abs weight cap
    max_region_w: Dict[str, float] = field(default_factory=dict)   # region -> cap
    max_sector_w: Dict[str, float] = field(default_factory=dict)   # sector -> cap
    turnover_cap: float = 0.35        # max |delta notional| / NAV per rebalance
    min_trade_notional: float = 500.0 # skip dust trades
    borrow_ok: bool = True            # allow negative weights

@dataclass
class Costs:
    fee_bps: float = 0.3              # commissions/taxes in bps
    impact_eta: float = 25.0          # temporary impact bps per 100% participation
    risk_buffer_bps: float = 5.0      # extra bps cushion to avoid churning

@dataclass
class RiskKnobs:
    risk_mult: float = 1.0            # global multiplier (e.g., from drawdown speed)
    var_cap_frac: Optional[float] = None  # cap on 1d VaR as fraction of NAV (e.g., 0.03)
    es_cap_frac: Optional[float] = None   # cap on ES as fraction of NAV
    lv_ar_cap_bps: Optional[float] = None # LVaR cap per symbol in bps of notional

@dataclass
class TradeIntent:
    symbol: str
    side: Side
    qty: float
    target_qty: float
    reason: str
    est_cost_bps: float
    notes: Dict[str, float] = field(default_factory=dict)

@dataclass
class AllocationResult:
    target_weights: Dict[str, float]       # symbol -> target weight [-, +]
    target_qty: Dict[str, float]           # symbol -> target qty
    trades: List[TradeIntent]
    applied_multiplier: float              # final scale after all caps
    diagnostics: Dict[str, float]          # helpful summary for logs/UI

# ==========================================================
# Allocator
# ==========================================================

class Allocator:
    """
    Score→weight allocator with risk & cost constraints.

    Inputs:
      - signals: symbol -> score in [-1, +1]
      - symbols: symbol -> SymbolInfo
      - state:   PortfolioState
      - constraints, costs, risk knobs
      - optional: per-symbol LVaR bps function and portfolio risk estimators

    Method:
      1) Raw target weights: w_i ∝ (score_i / vol_i) with spread/ADV penalties
      2) Scale by global risk multiplier & normalize to gross <= max_gross
      3) Enforce per-symbol/region/sector caps
      4) Turnover and cost-aware shrink
      5) Convert to target qty and emit trades (delta vs. current)
    """

    def __init__(self):
        pass

    # ---- public API -------------------------------------------------

    def allocate(
        self,
        *,
        signals: Dict[str, float],
        symbols: Dict[str, SymbolInfo],
        state: PortfolioState,
        constraints: Constraints = Constraints(),
        costs: Costs = Costs(),
        risk: RiskKnobs = RiskKnobs(),
        dd_alert: Optional[DrawdownAlert] = None,
        # Optional hooks (provide simple callables returning floats)
        portfolio_var_fn: Optional[callable] = None,      # -> var_frac for a proposal # type: ignore
        portfolio_es_fn: Optional[callable] = None,       # -> es_frac for a proposal # type: ignore
        symbol_lvar_bps_fn: Optional[callable] = None,    # sym, notional -> bps # type: ignore
    ) -> AllocationResult:

        # 0) precompute helpers
        nav = max(1e-9, state.nav)
        # drawdown-based risk cut
        dd_mult = (dd_alert.suggested_risk_mult if dd_alert else 1.0)
        g_mult = min(max(risk.risk_mult, 0.0), 1.0) * min(1.0, max(0.0, dd_mult))

        # 1) raw score → weight (vol-inverse scaled; penalize illiquidity/spread)
        raw_w: Dict[str, float] = {}
        liq_pen: Dict[str, float] = {}
        for sym, s in signals.items():
            inf = symbols.get(sym)
            if not inf or inf.price <= 0: 
                continue
            v = max(1e-6, inf.vol_day)
            # liquidity penalty ~ (1 + λ_spread + λ_participation)
            spread_pen = 1.0 + (inf.spread_bps / 10000.0) * 3.0
            # participation proxy at 1% NAV position over price/ADV
            nominal_1pct_nav = 0.01 * nav
            est_qty = nominal_1pct_nav / max(1e-6, inf.price)
            part = min(1.0, est_qty / max(1.0, inf.adv))
            part_pen = 1.0 + 2.0 * part
            pen = spread_pen * part_pen
            liq_pen[sym] = pen
            # base weight before global scaling
            base = (s / v) / pen
            raw_w[sym] = base

        if not raw_w:
            return AllocationResult({}, {}, [], g_mult, {"reason": 0.0})

        # 2) normalize to gross and apply global multiplier
        gross = sum(abs(w) for w in raw_w.values())
        if gross <= 1e-12:
            scaled = {k: 0.0 for k in raw_w}
        else:
            scaled = {k: (w / gross) * constraints.max_gross * g_mult for k, w in raw_w.items()}

        # 3) per-symbol caps, borrowing and sign
        for k in list(scaled.keys()):
            w = scaled[k]
            if not constraints.borrow_ok:
                # clamp to [0, cap]
                w = max(0.0, w)
            cap = constraints.max_symbol_w
            if abs(w) > cap:
                w = cap if w > 0 else -cap
            scaled[k] = w

        # 4) region/sector caps
        scaled = self._enforce_buckets(scaled, symbols, constraints.max_region_w)
        scaled = self._enforce_buckets(scaled, symbols, constraints.max_sector_w, attr="sector")

        # 5) portfolio-level VaR/ES guardrails (optional)
        # Build a temporary proposal vector for checks
        prop_notional = {k: scaled[k] * nav for k in scaled}
        if portfolio_var_fn and risk.var_cap_frac:
            var_frac = portfolio_var_fn(prop_notional)  # user-provided function
            if var_frac > risk.var_cap_frac:
                shrink = max(0.01, risk.var_cap_frac / max(1e-9, var_frac))
                scaled = {k: v * shrink for k, v in scaled.items()}
                g_mult *= shrink
        if portfolio_es_fn and risk.es_cap_frac:
            es_frac = portfolio_es_fn(prop_notional)
            if es_frac > risk.es_cap_frac:
                shrink = max(0.01, risk.es_cap_frac / max(1e-9, es_frac))
                scaled = {k: v * shrink for k, v in scaled.items()}
                g_mult *= shrink

        # 6) LVaR per-symbol guardrail (if hook provided)
        if symbol_lvar_bps_fn and risk.lv_ar_cap_bps is not None:
            for k in list(scaled.keys()):
                w = scaled[k]
                notional = abs(w) * nav
                bps = symbol_lvar_bps_fn(k, notional)  # expected LVaR bps
                if bps > risk.lv_ar_cap_bps:
                    # shrink this symbol proportionally
                    shrink = max(0.05, risk.lv_ar_cap_bps / max(1e-6, bps))
                    scaled[k] = w * shrink

        # 7) turnover & cost-aware shrink
        # Estimate turnover and skip dust trades
        trades: List[TradeIntent] = []
        target_qty: Dict[str, float] = {}
        target_w = scaled.copy()

        est_turnover = 0.0
        for sym, w in target_w.items():
            inf = symbols[sym]
            tgt_notional = w * nav
            tgt_qty = tgt_notional / max(1e-6, inf.price)
            cur_qty = state.positions.get(sym, 0.0)
            d_qty = tgt_qty - cur_qty
            d_not = abs(d_qty * inf.price)
            est_turnover += d_not
            target_qty[sym] = tgt_qty

        if est_turnover / nav > constraints.turnover_cap:
            shr = max(0.05, constraints.turnover_cap / max(1e-9, est_turnover / nav))
            target_w = {k: v * shr for k, v in target_w.items()}
            target_qty = {k: q * shr for k, q in target_qty.items()}
            g_mult *= shr

        # Build trade intents with rough cost model
        for sym, tgt_q in target_qty.items():
            inf = symbols[sym]
            cur_q = state.positions.get(sym, 0.0)
            d_q = tgt_q - cur_q
            d_not = abs(d_q * inf.price)
            if d_not < constraints.min_trade_notional:
                continue
            side: Side = "buy" if d_q > 0 else "sell"
            # simple cost estimate: half-spread + fee + participation impact
            # participation proxy vs ADV
            part = min(1.0, abs(d_q) / max(1.0, inf.adv))
            est_bps = (0.5 * inf.spread_bps) + costs.fee_bps + costs.impact_eta * part
            # add small cushion
            est_bps += costs.risk_buffer_bps
            trades.append(TradeIntent(
                symbol=sym, side=side, qty=abs(d_q), target_qty=tgt_q,
                reason="rebalance", est_cost_bps=est_bps,
                notes={"part": part, "spread_bps": inf.spread_bps}
            ))

        diags = {
            "gross_target": sum(abs(target_w[k]) for k in target_w),
            "applied_multiplier": g_mult,
            "symbols": float(len(target_w)),
        }
        return AllocationResult(
            target_weights=target_w,
            target_qty=target_qty,
            trades=trades,
            applied_multiplier=g_mult,
            diagnostics=diags,
        )

    # ---- helpers ----------------------------------------------------

    def _enforce_buckets(
        self,
        weights: Dict[str, float],
        symbols: Dict[str, SymbolInfo],
        caps: Dict[str, float],
        *,
        attr: str = "region"
    ) -> Dict[str, float]:
        if not caps:
            return weights
        # compute bucket exposures
        bucket_sum: Dict[str, float] = {}
        for sym, w in weights.items():
            info = symbols.get(sym)
            key = getattr(info, attr) if info else None
            if not key: 
                continue
            bucket_sum[key] = bucket_sum.get(key, 0.0) + abs(w)
        # apply shrink where needed
        out = dict(weights)
        for key, gross in bucket_sum.items():
            cap = caps.get(key)
            if cap is None:
                continue
            if gross > cap and gross > 1e-12:
                shrink = cap / gross
                for sym, w in list(out.items()):
                    info = symbols.get(sym)
                    if info and getattr(info, attr) == key:
                        out[sym] = w * shrink
        return out