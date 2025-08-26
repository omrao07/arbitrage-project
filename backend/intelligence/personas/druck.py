# agents/druck.py
"""
DruckAgent
----------
"Ride the horse that's running." — A Druckenmiller‑style macro agent.

Philosophy
----------
- Concentrate in the *strongest* trends (momentum + breakouts), not diworsified baskets.
- Align with the *policy/liquidity tide* (QE/tightening, USD trend, risk regime).
- Respect crowding/positioning and regime flips; size up only when winds agree.

Expected MarketContext.signals (all optional; rules skip if missing)
Per symbol SYM (e.g., NDX, TLT, DXY, GLD, COPP, CL):
- f"mom_z_{SYM}"         : multi-horizon momentum z-score
- f"breakout_z_{SYM}"    : breakout/price-location z (e.g., 55d/200d)
- f"cftc_pos_z_{SYM}"    : positioning z (positive = crowded long)
- f"ivol_z_{SYM}"        : implied/realized vol regime (z); high -> de-risk

Global macro/regime:
- "liquidity_z"          : global liquidity (positive = easier conditions)
- "policy_impulse"       : CB/fiscal impulse (+ easing, - tightening)
- "risk_z"               : risk regime (e.g., VIX z; + = risk-off)
- "usd_trend_z"          : USD broad trend (DXY or BBDXY) z
- Optional: "infl_nowcast": inflation nowcast z (helps gold/commodities if >0)

Symbols universe (override via config):
- "NDX"  (NASDAQ 100 proxy)
- "TLT"  (long UST duration)
- "DXY"  (USD index proxy)
- "GLD"  (gold)
- "COPP" (copper proxy; use HG or CPER if preferred)
- "CL"   (WTI crude; use front month code)
- "EEM"  (EM equities ETF)

Output
------
- A concentrated multi-leg Proposal with per-leg meta score.
- Average score/confidence reflects alignment & data completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import ( # type: ignore
    AgentBase, MarketContext, Proposal, OrderPlan, Constraints,
    soft_score_from_edge, clamp
)


# ------------------- configuration -------------------

@dataclass
class AssetItem:
    symbol: str
    bucket: str                 # "risk" | "duration" | "usd" | "gold" | "commodity" | "em"
    base_qty: float = 1.0       # natural units (shares, contracts, etc.)
    venue: Optional[str] = None


@dataclass
class DruckConfig:
    # Default cross-asset set
    universe: List[AssetItem] = field(default_factory=lambda: [
        AssetItem("NDX",  bucket="risk",      base_qty=1.0,  venue="INDEX"),
        AssetItem("TLT",  bucket="duration",  base_qty=10.0, venue="NYSE"),
        AssetItem("DXY",  bucket="usd",       base_qty=500.0, venue="FX"),
        AssetItem("GLD",  bucket="gold",      base_qty=5.0,  venue="NYSE"),
        AssetItem("COPP", bucket="commodity", base_qty=20.0, venue="COMEX"),  # use "HG" or "CPER" in your feed
        AssetItem("CL",   bucket="commodity", base_qty=1.0,  venue="NYMEX"),
        AssetItem("EEM",  bucket="em",        base_qty=10.0, venue="NYSE"),
    ])

    # Weights for per-asset trend score
    w_mom: float = 0.55
    w_breakout: float = 0.35
    w_crowding_pen: float = 0.20        # penalty per |cftc_pos_z|

    # Macro alignment multipliers by bucket
    liq_boost_risk: float = 0.40        # liquidity/policy helps risk assets
    liq_boost_em: float = 0.45
    liq_pen_duration: float = -0.30     # liquidity hurts duration (bonds) in tightening
    liq_gold: float = 0.20
    liq_commod: float = 0.35
    liq_usd: float = -0.35              # easier liq tends to weaken USD (risk-on)

    risk_pen_hi: float = 0.35           # risk-off penalizes risk buckets
    risk_boost_usd: float = 0.25
    risk_boost_duration: float = 0.20
    vol_pen: float = 0.15               # per +1 z of ivol_z

    # USD trend coupling (if USD trend up, fade EM/commodities; favor USD)
    usd_couple_usd: float = 0.35
    usd_couple_em: float = -0.25
    usd_couple_commod: float = -0.20
    usd_couple_gold: float = -0.05

    # Inflation tilt (optional)
    infl_boost_gold: float = 0.25
    infl_boost_commod: float = 0.20
    infl_pen_duration: float = -0.20

    # Edge scaling and horizon
    edge_bps: float = 120.0
    horizon_sec: float = 7 * 24 * 3600

    # Selection / concentration
    min_abs_score_to_trade: float = 0.20
    max_legs: int = 4                    # concentrate!
    top1_size_boost: float = 1.5
    top2_size_boost: float = 1.2


# ------------------- agent -------------------

class DruckAgent(AgentBase):
    name = "druck"

    def __init__(self, cfg: Optional[DruckConfig] = None):
        self.cfg = cfg or DruckConfig()

    # -------- propose --------

    def propose(self, context: MarketContext) -> Proposal:
        s = context.signals or {}
        prices = context.prices or {}

        # Global regime vars
        liq = _f(s.get("liquidity_z"), 0.0)
        pol = _f(s.get("policy_impulse"), 0.0)
        riskz = _f(s.get("risk_z"), 0.0)
        usd_tr = _f(s.get("usd_trend_z"), 0.0)
        infl = _f(s.get("infl_nowcast"), 0.0)

        per_sym_score: Dict[str, float] = {}
        per_sym_conf: Dict[str, float] = {}
        notes: List[str] = []
        tags: List[str] = ["macro", "trend", "druck"]

        # Compute per-asset scores
        for item in self.cfg.universe:
            sym = item.symbol
            if sym not in prices:
                continue

            mom = _f(s.get(f"mom_z_{sym}"), None) # type: ignore
            brk = _f(s.get(f"breakout_z_{sym}"), None) # type: ignore
            pos = _f(s.get(f"cftc_pos_z_{sym}"), 0.0)
            ivz = _f(s.get(f"ivol_z_{sym}"), 0.0)

            if mom is None and brk is None:
                continue

            # Base trend score
            base = 0.0
            w = 0.0
            if mom is not None:
                base += self.cfg.w_mom * mom; w += self.cfg.w_mom
            if brk is not None:
                base += self.cfg.w_breakout * brk; w += self.cfg.w_breakout
            base = base / max(1e-9, w)

            # Crowding penalty (discourage joining late crowded longs)
            base -= self.cfg.w_crowding_pen * abs(pos)

            # Macro alignment multiplier
            macro = 0.0
            # liquidity + policy
            liq_pol = liq + pol
            if item.bucket in ("risk",):
                macro += self.cfg.liq_boost_risk * liq_pol
                macro -= self.cfg.risk_pen_hi * max(0.0, riskz)
            elif item.bucket in ("em",):
                macro += self.cfg.liq_boost_em * liq_pol
                macro -= (self.cfg.risk_pen_hi + 0.05) * max(0.0, riskz)
                macro += self.cfg.usd_couple_em * usd_tr   # USD up hurts EM
            elif item.bucket in ("commodity",):
                macro += self.cfg.liq_commod * liq_pol
                macro += self.cfg.usd_couple_commod * usd_tr
            elif item.bucket in ("gold",):
                macro += self.cfg.liq_gold * liq_pol
                macro += self.cfg.usd_couple_gold * usd_tr
            elif item.bucket in ("duration",):
                macro += self.cfg.liq_pen_duration * liq_pol
                macro += self.cfg.risk_boost_duration * max(0.0, riskz)
            elif item.bucket in ("usd",):
                macro += self.cfg.liq_usd * liq_pol
                macro += self.cfg.risk_boost_usd * max(0.0, riskz)
                macro += self.cfg.usd_couple_usd * usd_tr

            # Inflation tilt
            if infl != 0.0:
                if item.bucket in ("gold",):
                    macro += self.cfg.infl_boost_gold * infl
                elif item.bucket in ("commodity",):
                    macro += self.cfg.infl_boost_commod * infl
                elif item.bucket in ("duration",):
                    macro += self.cfg.infl_pen_duration * infl

            # Vol regime penalty
            macro -= self.cfg.vol_pen * max(0.0, ivz)

            # Final directional score
            sc_raw = base + macro
            sc = clamp(sc_raw, -1.5, 1.5)  # pre-edge clamp

            # Convert to normalized decision score via soft edge
            edge_bps = abs(sc) * self.cfg.edge_bps * 1.0
            score = soft_score_from_edge(edge_bps, self.cfg.horizon_sec, cap=1.0)
            score = score if sc >= 0 else -score

            # Confidence: alignment + data completeness
            have = (mom is not None) + (brk is not None)
            align = 0.5 + 0.2 * _sgn_agree(base, macro) + 0.1 * (1 if abs(pos) < 1.0 else 0)
            conf = clamp(0.35 + 0.15 * have + 0.3 * align, 0.35, 0.9)

            per_sym_score[sym] = score
            per_sym_conf[sym] = conf

            notes.append(f"{sym}: base={base:+.2f}, macro={macro:+.2f}, pos={pos:+.2f}, ivz={ivz:+.2f} -> sc={score:+.2f}")

        if not per_sym_score:
            return Proposal(orders=[], thesis="No Druck edges (missing trends/feeds).",
                            score=0.0, horizon_sec=self.cfg.horizon_sec, confidence=0.4, tags=["idle"])

        # Rank by |score|; concentrate into top ideas
        ranked = sorted(per_sym_score.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ranked = [kv for kv in ranked if abs(kv[1]) >= self.cfg.min_abs_score_to_trade][: self.cfg.max_legs]

        legs: List[OrderPlan] = []
        for i, (sym, sc) in enumerate(ranked):
            item = next(it for it in self.cfg.universe if it.symbol == sym)
            side = "BUY" if sc >= 0 else "SELL"
            qty = abs(sc) * item.base_qty
            # concentration boost for top picks
            if i == 0:
                qty *= self.cfg.top1_size_boost
            elif i == 1:
                qty *= self.cfg.top2_size_boost
            legs.append(OrderPlan(symbol=sym, side=side, qty=qty, type="MARKET", venue=item.venue,
                                  meta={"score": sc, "bucket": item.bucket}))
            notes.append(f"{sym} {side} size≈{qty:g} (score={sc:+.2f})")

        avg_score = sum(sc for _, sc in ranked) / max(1, len(ranked))
        avg_conf = sum(per_sym_conf[sym] for sym, _ in ranked) / max(1, len(ranked))
        thesis = " | ".join(notes[-14:])

        return Proposal(
            orders=legs,
            thesis=thesis,
            score=clamp(avg_score, -1.0, 1.0),
            horizon_sec=self.cfg.horizon_sec,
            confidence=avg_conf,
            tags=list(dict.fromkeys(tags)),
            diagnostics={
                "liquidity_z": liq, "policy_impulse": pol, "risk_z": riskz,
                "usd_trend_z": usd_tr, "infl_nowcast": infl,
                "scores": per_sym_score
            },
        )

    # -------- risk --------

    def risk(self, proposal: Proposal, context: MarketContext):
        # Reuse base risk (constraint checks + notionals)
        return self.base_risk(proposal, context)

    # -------- explain --------

    def explain(self, proposal: Proposal, risk=None) -> str:
        if not proposal.orders:
            return f"[{self.name}] {proposal.thesis}"
        legs_txt = ", ".join([f"{o.side} {o.qty:g} {o.symbol}" + (f'@{o.venue}' if o.venue else "") for o in proposal.orders])
        rtxt = ""
        if risk:
            ok = "PASS" if risk.ok else "FAIL"
            rtxt = f" | risk={ok} gross=${risk.gross_notional_usd:,.0f} net=${risk.exposure_usd:,.0f}"
        return f"[{self.name}] {legs_txt} | score={proposal.score:.2f}, conf={proposal.confidence:.2f} | {proposal.thesis}{rtxt}"


# ------------------- tiny utils -------------------

def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _sgn_agree(a: float, b: float) -> int:
    """Return 1 if a,b same sign (reinforcing), 0 if either ~0, -1 if opposite."""
    if abs(a) < 1e-9 or abs(b) < 1e-9:
        return 0
    return 1 if (a > 0 and b > 0) or (a < 0 and b < 0) else -1