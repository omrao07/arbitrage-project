# agents/buffett.py
"""
BuffettAgent
------------
Quality + value stock selector with a margin-of-safety discipline.

Expected signals in MarketContext.signals (all optional; rules skip if missing)
Per symbol SYM (e.g., AAPL):
- f"fcf_yield_{SYM}"        : Free cash flow yield (e.g., 0.06 = 6%)
- f"roic_{SYM}"             : Return on invested capital (fraction)
- f"net_debt_ebitda_{SYM}"  : Leverage (negative allowed if net cash)
- f"moat_{SYM}"             : Moat/quality score in [0,1] (0=narrow, 1=wide)
- f"buyback_yield_{SYM}"    : Net buyback yield (fraction; negative if issuing)
- f"earn_var_{SYM}"         : Earnings volatility proxy (std/mean, lower is better)
- f"growth_5y_{SYM}"        : 5y revenue/FCF CAGR proxy (fraction)
- f"margin_trend_{SYM}"     : 2–3y margin trend (fraction Δ), positive is good
Optional macro:
- "rate_10y"                : 10y yield (discount-rate proxy)

Behavior
--------
- Compute a composite "quality × value" score with penalties for leverage/volatility.
- Apply a safety buffer vs. the discount-rate (higher rates -> demand higher FCF yield).
- Rank names; BUY top positives, SELL rich/low-quality names if score << 0.
- Horizon: multi-quarter by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import ( # type: ignore
    AgentBase, MarketContext, Proposal, OrderPlan, Constraints,
    soft_score_from_edge, clamp
)


# ------------------- configuration -------------------

@dataclass
class BuffettItem:
    symbol: str
    venue: Optional[str] = "NYSE"
    base_qty: float = 8.0
    sector: str = "GEN"


@dataclass
class BuffettConfig:
    watchlist: List[BuffettItem] = field(default_factory=lambda: [
        BuffettItem("AAPL", sector="MEGA_GROWTH", base_qty=10.0),
        BuffettItem("MSFT", sector="MEGA_GROWTH", base_qty=8.0),
        BuffettItem("BRK.B", sector="FINANCIALS", base_qty=6.0),
        BuffettItem("KO", sector="DEFENSIVE", base_qty=10.0),
        BuffettItem("JNJ", sector="DEFENSIVE", base_qty=8.0),
        BuffettItem("XOM", sector="VALUE", base_qty=8.0),
    ])

    # Minimums / thresholds
    min_moat: float = 0.5             # prefer at least “narrow+”
    max_net_debt_ebitda: float = 3.0  # soft cap; >3x penalized
    max_earn_var: float = 0.4         # earnings volatility cap

    # Discount-rate guardrail (demand FCF yield > rate + spread)
    fcf_spread_min: float = 0.02      # min spread over 10y to buy
    fcf_spread_strong: float = 0.04   # stronger conviction if > this

    # Weights for composite
    w_fcf: float = 0.45
    w_roic: float = 0.25
    w_moat: float = 0.15
    w_buyback: float = 0.10
    w_growth: float = 0.05

    # Penalties
    pen_leverage: float = 0.15        # per 1x above cap
    pen_vol: float = 0.20             # per 0.1 of earn_var above cap
    pen_neg_margin_trend: float = 0.10

    # Edges → score scaling (bps)
    edge_bps_value: float = 120.0     # from FCF spread & valuey bits
    edge_bps_quality: float = 80.0    # from ROIC/moat/buybacks
    horizon_sec: float = 90 * 24 * 3600  # ~1 quarter

    # Selection
    min_abs_score_to_trade: float = 0.18
    max_legs: int = 8


# ------------------- agent -------------------

class BuffettAgent(AgentBase):
    name = "buffett"

    def __init__(self, cfg: Optional[BuffettConfig] = None):
        self.cfg = cfg or BuffettConfig()

    # -------- propose --------

    def propose(self, context: MarketContext) -> Proposal:
        s = context.signals or {}
        prices = context.prices or {}

        rate = _f(s.get("rate_10y"), 0.03)  # default 3% if not provided
        parts: List[str] = []
        tags: List[str] = []
        legs: List[OrderPlan] = []

        scored: Dict[str, float] = {}
        confs: Dict[str, float] = {}

        for item in self.cfg.watchlist:
            sym = item.symbol
            if sym not in prices:
                continue

            fcf = _f(s.get(f"fcf_yield_{sym}"), None)
            roic = _f(s.get(f"roic_{sym}"), None)
            moat = _f(s.get(f"moat_{sym}"), None)
            lev = _f(s.get(f"net_debt_ebitda_{sym}"), None)
            bb  = _f(s.get(f"buyback_yield_{sym}"), 0.0)
            ev  = _f(s.get(f"earn_var_{sym}"), None)
            g5  = _f(s.get(f"growth_5y_{sym}"), 0.0)
            mtrend = _f(s.get(f"margin_trend_{sym}"), 0.0)

            # Require core metrics
            if fcf is None or roic is None or moat is None:
                continue

            # Margin of safety: FCF yield must exceed (rate + spread)
            spread = fcf - (rate + self.cfg.fcf_spread_min) # type: ignore
            value_edge_bps = max(0.0, spread) * self.cfg.edge_bps_value * 10  # ×10 to convert % to units
            # Additional value juice: buybacks & growth (small)
            value_edge_bps += max(0.0, bb) * 100 * 0.3 # type: ignore
            value_edge_bps += max(0.0, g5) * 100 * 0.2 # type: ignore

            # Quality edge
            q = 0.0
            q += self.cfg.w_roic * max(0.0, roic) * 100
            q += self.cfg.w_moat * max(0.0, moat) * 100
            q += self.cfg.w_buyback * max(0.0, bb) * 100 # type: ignore
            q_edge_bps = q * (self.cfg.edge_bps_quality / 100.0)

            # Penalties
            pen = 0.0
            if lev is not None and lev > self.cfg.max_net_debt_ebitda:
                pen += (lev - self.cfg.max_net_debt_ebitda) * (self.cfg.pen_leverage * 100)
            if ev is not None and ev > self.cfg.max_earn_var:
                pen += ((ev - self.cfg.max_earn_var) / 0.1) * (self.cfg.pen_vol * 100)
            if mtrend < 0: # type: ignore
                pen += abs(mtrend) * (self.cfg.pen_neg_margin_trend * 100) # type: ignore

            edge_total_bps = max(0.0, value_edge_bps + q_edge_bps - pen)

            # Convert to normalized score [-1,1]; negative if spread deeply negative or quality poor
            score = soft_score_from_edge(edge_total_bps, self.cfg.horizon_sec, cap=1.0)
            # Flip to negative if safety fails by a lot or moat thin with weak ROIC
            if spread < -0.01 and (moat < self.cfg.min_moat or (roic is not None and roic < 0.06)):
                score = -min(1.0, abs(score) + 0.25)

            # Confidence increases with data completeness + moat strength
            conf = 0.5 + 0.3 * (1 if ev is not None and lev is not None else 0) + 0.2 * clamp(moat, 0.0, 1.0)
            conf = clamp(conf, 0.4, 0.95)

            # Guardrails
            if moat < self.cfg.min_moat:
                # degrade enthusiasm if moat too low
                score *= 0.6

            scored[sym] = clamp(score, -1.0, 1.0)
            confs[sym] = conf

            parts.append(f"{sym}: fcf={_pct(fcf)} vs 10y={_pct(rate)} (spr={_pct(spread)}), "
                         f"ROIC={_pct(roic)}, moat={moat:.2f}, lev={_fmt(lev)}, bb={_pct(bb)} → sc={score:.2f}")
            tags += [item.sector.lower(), "value", "quality"]

        if not scored:
            return Proposal(orders=[], thesis="No Buffett candidates (insufficient signals).",
                            score=0.0, horizon_sec=self.cfg.horizon_sec, confidence=0.4, tags=["idle"])

        # Rank & select
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        # choose positive BUYs and strong negative SELLs (optional short)
        legs: List[OrderPlan] = []
        chosen = 0
        for sym, sc in ranked:
            if abs(sc) < self.cfg.min_abs_score_to_trade:
                continue
            item = next(i for i in self.cfg.watchlist if i.symbol == sym)
            side = "BUY" if sc > 0 else "SELL"
            qty = abs(sc) * item.base_qty
            legs.append(OrderPlan(symbol=sym, side=side, qty=qty, type="MARKET",
                                  venue=item.venue, meta={"score": sc, "sector": item.sector}))
            chosen += 1
            if chosen >= self.cfg.max_legs:
                break

        if not legs:
            return Proposal(orders=[], thesis="Signals available but no names cleared margin-of-safety.",
                            score=0.0, horizon_sec=self.cfg.horizon_sec, confidence=0.5, tags=["value","quality"])

        avg_score = sum(scored[s] for s in [o.symbol for o in legs]) / max(1, len(legs))
        avg_conf = sum(confs[s] for s in [o.symbol for o in legs]) / max(1, len(legs))
        thesis = " | ".join(parts[-12:])

        return Proposal(
            orders=legs,
            thesis=thesis,
            score=clamp(avg_score, -1.0, 1.0),
            horizon_sec=self.cfg.horizon_sec,
            confidence=avg_conf,
            tags=list(dict.fromkeys(tags + ["buffett"])),
            diagnostics={"rate_10y": rate, "picked": [o.symbol for o in legs]},
        )

    # -------- risk --------

    def risk(self, proposal: Proposal, context: MarketContext):
        return self.base_risk(proposal, context)

    # -------- explain --------

    def explain(self, proposal: Proposal, risk=None) -> str:
        if not proposal.orders:
            return f"[{self.name}] {proposal.thesis}"
        legs_txt = ", ".join([f"{o.side} {o.qty:g} {o.symbol}" + (f"@{o.venue}" if o.venue else "") for o in proposal.orders])
        risk_txt = ""
        if risk:
            ok = "PASS" if risk.ok else "FAIL"
            risk_txt = f" | risk={ok} gross=${risk.gross_notional_usd:,.0f} net=${risk.exposure_usd:,.0f}"
        return f"[{self.name}] {legs_txt} | score={proposal.score:.2f}, conf={proposal.confidence:.2f} | {proposal.thesis}{risk_txt}"


# ------------------- tiny utils -------------------

def _f(x, default=None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.2f}x"
    except Exception:
        return str(x)

def _pct(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return str(x)