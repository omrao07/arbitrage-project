# agents/fx.py
"""
FXAgent
-------
Signal-driven FX agent that blends:
- Carry (rate differentials) → long high-yield vs low-yield
- Momentum (trend-follow with z-score)
- PPP/Fair-value gap (mean reversion)
- Central-bank surprise impulse (hawkish/dovish shock)
- Global risk regime (risk-on/off tilt via VIX or similar)

Expected MarketContext.signals (optional; rules skip if missing)
For each pair KEY in watchlist:
  - f"carry_{KEY}"           : annualized carry / rate differential (e.g., 0.025 = 2.5%/yr for base over quote)
  - f"mom_z_{KEY}"           : momentum z-score of returns (lookback vs mean/σ)
  - f"ppp_gap_{KEY}"         : % gap (spot/PPP - 1). Positive => overvalued base (tends to SELL base)
  - f"cb_surprise_{BASE}"    : central bank surprise (hawkish>0, dovish<0) for the BASE currency
Global:
  - "risk_z"                 : risk regime z-score (e.g., VIX z). Positive = risk-off, negative = risk-on.

Conventions
-----------
Pairs are quoted BASE/QUOTE (e.g., EURUSD). Side="BUY" → long BASE / short QUOTE.
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
class FxItem:
    pair: str                  # e.g., "EURUSD"
    venue: Optional[str] = None
    base_qty: float = 100_000  # 100k notional per leg (units of BASE); tune for your adapters
    # risk bucket for tilt: "carry", "risk", "commodity" (AUD, NZD, CAD), "safe" (USD, JPY, CHF)
    bucket: str = "neutral"


@dataclass
class FXConfig:
    watchlist: List[FxItem] = field(default_factory=lambda: [
        FxItem("EURUSD", venue="FXSIM", base_qty=50_000, bucket="neutral"),
        FxItem("USDJPY", venue="FXSIM", base_qty=50_000, bucket="safe"),
        FxItem("GBPUSD", venue="FXSIM", base_qty=40_000, bucket="neutral"),
        FxItem("AUDUSD", venue="FXSIM", base_qty=40_000, bucket="commodity"),
    ])

    # Carry thresholds (annualized)
    carry_buy_th: float = +0.01       # BUY base if carry >= +1%
    carry_sell_th: float = -0.01      # SELL base if carry <= -1%

    # Momentum thresholds (z)
    mom_buy_z: float = +0.6
    mom_sell_z: float = -0.6

    # PPP gap thresholds (mean-revert)
    ppp_sell_over_th: float = +0.08   # if base is >8% over PPP → SELL base
    ppp_buy_under_th: float = -0.08   # if base is < -8% under PPP → BUY base

    # Central bank surprise (per BASE)
    cb_buy_th: float = +0.25          # hawkish surprise → BUY base
    cb_sell_th: float = -0.25         # dovish surprise → SELL base

    # Edge scaling to bps
    edge_bps_carry: float = 120.0     # per 1.0 of (carry - th), annualized
    edge_bps_mom: float = 70.0
    edge_bps_ppp: float = 90.0
    edge_bps_cb: float = 100.0

    # Horizons
    horizon_carry: float = 10 * 24 * 3600
    horizon_mom: float = 3 * 24 * 3600
    horizon_ppp: float = 20 * 24 * 3600
    horizon_cb: float = 5 * 24 * 3600

    # Risk-on/off tilt (risk_z > 0 == risk-off): downweight commodity/risky bases on BUY, upweight safe
    risk_tilt_on: bool = True
    riskz_neutral: float = 0.0
    riskz_high: float = +1.5
    risky_penalty_max: float = 0.6
    safe_boost_max: float = 1.25

    # Caps & selection
    per_pair_score_cap: float = 1.2
    min_abs_score_to_trade: float = 0.18
    max_legs: int = 8


# ------------------- agent -------------------

class FXAgent(AgentBase):
    name = "fx"

    def __init__(self, cfg: Optional[FXConfig] = None):
        self.cfg = cfg or FXConfig()

    # -------- propose --------

    def propose(self, context: MarketContext) -> Proposal:
        s = context.signals or {}
        prices = context.prices or {}

        per_pair_score: Dict[str, float] = {}
        per_pair_conf: Dict[str, float] = {}
        notes: List[str] = []
        tags: List[str] = []

        # Global risk tilt
        riskz = _f(s.get("risk_z"), None)

        def _risk_mult(bucket: str, long_base: bool) -> float:
            if not self.cfg.risk_tilt_on or riskz is None:
                return 1.0
            # Map riskz to [0,1] between neutral and high risk-off
            if riskz <= self.cfg.riskz_neutral:
                x = 0.0
            elif riskz >= self.cfg.riskz_high:
                x = 1.0
            else:
                x = (riskz - self.cfg.riskz_neutral) / max(1e-9, self.cfg.riskz_high - self.cfg.riskz_neutral)
            if bucket in ("commodity", "carry"):
                # penalize BUY of risky bases when risk-off; SELL gets mild boost
                return (1.0 - x * (1.0 - self.cfg.risky_penalty_max)) if long_base else (1.0 + 0.15 * x)
            if bucket in ("safe",):
                # safe bases (USD/JPY/CHF) benefit under risk-off when long
                return (1.0 + (self.cfg.safe_boost_max - 1.0) * x) if long_base else (1.0 - 0.1 * x)
            return 1.0

        # Scoring per pair
        for item in self.cfg.watchlist:
            pair = item.pair
            if pair not in prices:   # require a price for size sanity; otherwise skip
                continue

            sc = 0.0
            w = 0.0
            conf = 0.0
            local_notes: List[str] = []

            carry = _f(s.get(f"carry_{pair}"), None)
            momz  = _f(s.get(f"mom_z_{pair}"), None)
            ppp   = _f(s.get(f"ppp_gap_{pair}"), None)

            # Extract BASE for CB surprise key (e.g., "EURUSD" → "EUR")
            base_ccy = pair[:3]
            cb = _f(s.get(f"cb_surprise_{base_ccy}"), None)

            # 1) Carry
            if carry is not None:
                if carry >= self.cfg.carry_buy_th:
                    edge = (carry - self.cfg.carry_buy_th) * self.cfg.edge_bps_carry
                    sc1 = soft_score_from_edge(edge, self.cfg.horizon_carry)
                    sc += sc1; w += 1.0; conf += 0.6
                    local_notes.append(f"{pair}: carry={carry:.3f}→BUY edge≈{edge:.0f}bps sc={sc1:.2f}")
                elif carry <= self.cfg.carry_sell_th:
                    edge = (abs(carry - self.cfg.carry_sell_th)) * self.cfg.edge_bps_carry
                    sc1 = -soft_score_from_edge(edge, self.cfg.horizon_carry)
                    sc += sc1; w += 1.0; conf += 0.6
                    local_notes.append(f"{pair}: carry={carry:.3f}→SELL edge≈{edge:.0f}bps sc={sc1:.2f}")

            # 2) Momentum
            if momz is not None:
                if momz >= self.cfg.mom_buy_z:
                    edge = (momz - self.cfg.mom_buy_z) * self.cfg.edge_bps_mom
                    sc2 = soft_score_from_edge(edge, self.cfg.horizon_mom)
                    sc += sc2; w += 1.0; conf += 0.5
                    local_notes.append(f"{pair}: mom z={momz:.2f}↑ sc={sc2:.2f}")
                elif momz <= self.cfg.mom_sell_z:
                    edge = (abs(momz - self.cfg.mom_sell_z)) * self.cfg.edge_bps_mom
                    sc2 = -soft_score_from_edge(edge, self.cfg.horizon_mom)
                    sc += sc2; w += 1.0; conf += 0.5
                    local_notes.append(f"{pair}: mom z={momz:.2f}↓ sc={sc2:.2f}")

            # 3) PPP / Fair-value gap (mean reversion)
            if ppp is not None:
                if ppp >= self.cfg.ppp_sell_over_th:
                    edge = (ppp - self.cfg.ppp_sell_over_th) * self.cfg.edge_bps_ppp
                    sc3 = -soft_score_from_edge(edge, self.cfg.horizon_ppp)  # overvalued base -> SELL
                    sc += sc3; w += 0.8; conf += 0.45
                    local_notes.append(f"{pair}: PPP gap={ppp:.2f} over → SELL sc={sc3:.2f}")
                elif ppp <= self.cfg.ppp_buy_under_th:
                    edge = (abs(ppp - self.cfg.ppp_buy_under_th)) * self.cfg.edge_bps_ppp
                    sc3 = soft_score_from_edge(edge, self.cfg.horizon_ppp)
                    sc += sc3; w += 0.8; conf += 0.45
                    local_notes.append(f"{pair}: PPP gap={ppp:.2f} under → BUY sc={sc3:.2f}")

            # 4) CB surprise (impulse)
            if cb is not None:
                if cb >= self.cfg.cb_buy_th:
                    edge = (cb - self.cfg.cb_buy_th) * self.cfg.edge_bps_cb
                    sc4 = soft_score_from_edge(edge, self.cfg.horizon_cb)
                    sc += sc4; w += 0.8; conf += 0.55
                    local_notes.append(f"{pair}: {base_ccy} CB hawkish {cb:.2f} → BUY sc={sc4:.2f}")
                elif cb <= self.cfg.cb_sell_th:
                    edge = (abs(cb - self.cfg.cb_sell_th)) * self.cfg.edge_bps_cb
                    sc4 = -soft_score_from_edge(edge, self.cfg.horizon_cb)
                    sc += sc4; w += 0.8; conf += 0.55
                    local_notes.append(f"{pair}: {base_ccy} CB dovish {cb:.2f} → SELL sc={sc4:.2f}")

            if w == 0.0:
                continue

            # Aggregate and cap
            sc = clamp(sc / max(1.0, w), -self.cfg.per_pair_score_cap, self.cfg.per_pair_score_cap)

            # Risk tilt multiplier based on risk regime & bucket
            long_base = sc >= 0.0
            sc_final = sc * _risk_mult(item.bucket, long_base=long_base)

            per_pair_score[pair] = clamp(sc_final, -1.0, 1.0)
            per_pair_conf[pair] = clamp(conf / max(1.0, w), 0.0, 1.0)
            notes += local_notes
            tags += [item.bucket]

        if not per_pair_score:
            return Proposal(orders=[], thesis="No FX edges triggered.", score=0.0,
                            horizon_sec=self.cfg.horizon_mom, confidence=0.3, tags=["idle"])

        # Rank & select
        ranked = sorted(per_pair_score.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ranked = [kv for kv in ranked if abs(kv[1]) >= self.cfg.min_abs_score_to_trade][: self.cfg.max_legs]

        legs: List[OrderPlan] = []
        for pair, sc in ranked:
            item = next(i for i in self.cfg.watchlist if i.pair == pair)
            qty = abs(sc) * item.base_qty
            side = "BUY" if sc >= 0 else "SELL"
            legs.append(OrderPlan(symbol=pair, side=side, qty=qty, type="MARKET", venue=item.venue,
                                  meta={"score": sc, "bucket": item.bucket}))
            notes.append(f"{pair} {side} size≈{qty:,.0f} (score={sc:.2f})")

        # Aggregate score/confidence on selected legs
        avg_score = sum(sc for _, sc in ranked) / max(1.0, len(ranked))
        avg_conf = sum(per_pair_conf[p] for p, _ in ranked) / max(1.0, len(ranked))

        thesis = " | ".join(notes[-14:])  # keep concise

        return Proposal(
            orders=legs,
            thesis=thesis,
            score=clamp(avg_score, -1.0, 1.0),
            horizon_sec=max(self.cfg.horizon_carry, self.cfg.horizon_ppp, self.cfg.horizon_mom, self.cfg.horizon_cb),
            confidence=avg_conf,
            tags=list(dict.fromkeys(tags + ["fx"])),
            diagnostics={"per_pair_scores": per_pair_score, "risk_z": riskz},
        )

    # -------- risk --------

    def risk(self, proposal: Proposal, context: MarketContext):
        return self.base_risk(proposal, context)

    # -------- explain --------

    def explain(self, proposal: Proposal, risk=None) -> str:
        if not proposal.orders:
            return f"[{self.name}] {proposal.thesis}"
        legs_txt = ", ".join([f"{o.side} {o.qty:,.0f} {o.symbol}" + (f"@{o.venue}" if o.venue else "") for o in proposal.orders])
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