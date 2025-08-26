# agents/equities.py
"""
EquitiesAgent
-------------
Signal-blended equities agent (momentum + earnings + valuation + sentiment)
with a macro rates tilt (10y).

Expected signals in MarketContext.signals (all optional; rules skip if missing):
- f"mom_z_{SYM}"            : momentum z-score (lookback vs mean/σ)
- f"earn_surprise_{SYM}"    : last EPS surprise (% of consensus, e.g., +0.08 = +8%)
- f"val_z_{SYM}"            : valuation z-score (e.g., PE/EVS vs sector history)
- f"sent_{SYM}"             : normalized sentiment [-1, 1]
- "rate_10y"                : US 10y yield (fraction, e.g., 0.045 = 4.5%)

You can feed these from your feature pipeline (alt/social/ fundamentals loaders).

Symbols & sizing come from EquitiesConfig.watchlist.
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
class EqItem:
    symbol: str
    venue: Optional[str] = None
    base_qty: float = 10.0         # shares per unit trade
    sector: str = "GEN"            # optional tag (growth/value tilt use)


@dataclass
class EquitiesConfig:
    # Watchlist of tickers (override to your list)
    watchlist: List[EqItem] = field(default_factory=lambda: [
        EqItem("AAPL", venue="NYSE", base_qty=10.0, sector="MEGA_GROWTH"),
        EqItem("MSFT", venue="NYSE", base_qty=8.0,  sector="MEGA_GROWTH"),
        EqItem("NVDA", venue="NYSE", base_qty=4.0,  sector="MEGA_GROWTH"),
        EqItem("TSLA", venue="NYSE", base_qty=6.0,  sector="GROWTH"),
        EqItem("AMZN", venue="NYSE", base_qty=6.0,  sector="GROWTH"),
        EqItem("JPM",  venue="NYSE", base_qty=8.0,  sector="VALUE"),
        EqItem("XOM",  venue="NYSE", base_qty=8.0,  sector="VALUE"),
    ])

    # Thresholds
    mom_buy_z: float = +0.7
    mom_sell_z: float = -0.7
    earn_buy_surprise: float = +0.05        # +5%
    earn_sell_surprise: float = -0.05       # -5%
    val_buy_z: float = -0.8                 # cheap -> buy
    val_sell_z: float = +0.8                # rich -> sell
    sent_buy_th: float = +0.25
    sent_sell_th: float = -0.3

    # Per-signal scaling to "edge bps"
    edge_bps_mom: float = 60.0
    edge_bps_earn: float = 120.0
    edge_bps_val: float = 70.0
    edge_bps_sent: float = 40.0

    # Horizons (seconds)
    horizon_mom: float = 3 * 24 * 3600
    horizon_earn: float = 5 * 24 * 3600
    horizon_val: float = 15 * 24 * 3600
    horizon_sent: float = 12 * 3600

    # Macro tilt: higher 10y -> downweight growth longs, upweight value
    rates_tilt_on: bool = True
    rate10y_neutral: float = 0.035            # 3.5%
    rate10y_high: float = 0.05                # 5.0%
    growth_penalty_max: float = 0.6           # size multiplier at very high rates
    value_boost_max: float = 1.25

    # Caps
    per_name_score_cap: float = 1.2           # cap before compression
    final_score_cap: float = 1.0               # [-1, 1] after compression
    min_abs_score_to_trade: float = 0.15
    max_legs: int = 10                         # top-N by |score|


# ------------------- agent -------------------

class EquitiesAgent(AgentBase):
    name = "equities"

    def __init__(self, cfg: Optional[EquitiesConfig] = None):
        self.cfg = cfg or EquitiesConfig()

    # -------- propose --------

    def propose(self, context: MarketContext) -> Proposal:
        s = context.signals or {}
        prices = context.prices or {}

        legs: List[OrderPlan] = []
        parts: List[str] = []
        tags: List[str] = []
        per_name_scores: Dict[str, float] = {}
        per_name_conf: Dict[str, float] = {}

        # Macro: rates tilt sizing/score
        rate = _f(s.get("rate_10y"), None)
        def _macro_mult(sector: str, long_side: bool) -> float:
            if not self.cfg.rates_tilt_on or rate is None:
                return 1.0
            # Map 10y into [0,1] between neutral and high
            if rate <= self.cfg.rate10y_neutral:
                x = 0.0
            elif rate >= self.cfg.rate10y_high:
                x = 1.0
            else:
                x = (rate - self.cfg.rate10y_neutral) / max(1e-9, self.cfg.rate10y_high - self.cfg.rate10y_neutral)
            if sector in ("MEGA_GROWTH", "GROWTH"):
                # penalize growth *longs* when rates are high; shorts get mild boost
                return (1.0 - x * (1.0 - self.cfg.growth_penalty_max)) if long_side else (1.0 + 0.15 * x)
            if sector in ("VALUE", "FINANCIALS", "ENERGY"):
                # value/financials benefit from higher rates (size up longs)
                return (1.0 + (self.cfg.value_boost_max - 1.0) * x) if long_side else (1.0 - 0.1 * x)
            return 1.0

        # Per-name scoring
        for item in self.cfg.watchlist:
            sym = item.symbol
            if sym not in prices:
                continue

            # Pull signals (None if missing)
            mom_z = _f(s.get(f"mom_z_{sym}"), None)
            earn = _f(s.get(f"earn_surprise_{sym}"), None)
            valz = _f(s.get(f"val_z_{sym}"), None)
            sent = _f(s.get(f"sent_{sym}"), None)

            score = 0.0
            w = 0.0
            conf = 0.0
            notes: List[str] = []

            # Momentum (trend-follow)
            if mom_z is not None:
                if mom_z >= self.cfg.mom_buy_z:
                    edge = (mom_z - self.cfg.mom_buy_z) * self.cfg.edge_bps_mom
                    sc = soft_score_from_edge(edge, self.cfg.horizon_mom)
                    score += sc; w += 1.0; conf += 0.55
                    notes.append(f"{sym}: mom z={mom_z:.2f}↑ edge≈{edge:.0f}bps sc={sc:.2f}")
                elif mom_z <= self.cfg.mom_sell_z:
                    edge = (abs(mom_z - self.cfg.mom_sell_z)) * self.cfg.edge_bps_mom
                    sc = -soft_score_from_edge(edge, self.cfg.horizon_mom)
                    score += sc; w += 1.0; conf += 0.55
                    notes.append(f"{sym}: mom z={mom_z:.2f}↓ edge≈{edge:.0f}bps sc={sc:.2f}")

            # Earnings surprise (post-earn drift)
            if earn is not None:
                if earn >= self.cfg.earn_buy_surprise:
                    edge = (earn - self.cfg.earn_buy_surprise) * self.cfg.edge_bps_earn * 10  # convert % to units
                    sc = soft_score_from_edge(edge, self.cfg.horizon_earn)
                    score += sc; w += 1.0; conf += 0.65
                    notes.append(f"{sym}: EPS surprise={earn:.2%}↑ edge≈{edge:.0f}bps sc={sc:.2f}")
                elif earn <= self.cfg.earn_sell_surprise:
                    edge = (abs(earn - self.cfg.earn_sell_surprise)) * self.cfg.edge_bps_earn * 10
                    sc = -soft_score_from_edge(edge, self.cfg.horizon_earn)
                    score += sc; w += 1.0; conf += 0.65
                    notes.append(f"{sym}: EPS surprise={earn:.2%}↓ edge≈{edge:.0f}bps sc={sc:.2f}")

            # Valuation mean-reversion
            if valz is not None:
                if valz <= self.cfg.val_buy_z:
                    edge = (abs(valz - self.cfg.val_buy_z)) * self.cfg.edge_bps_val
                    sc = soft_score_from_edge(edge, self.cfg.horizon_val)
                    score += sc; w += 0.7; conf += 0.5
                    notes.append(f"{sym}: cheap val z={valz:.2f} edge≈{edge:.0f}bps sc={sc:.2f}")
                elif valz >= self.cfg.val_sell_z:
                    edge = (abs(valz - self.cfg.val_sell_z)) * self.cfg.edge_bps_val
                    sc = -soft_score_from_edge(edge, self.cfg.horizon_val)
                    score += sc; w += 0.7; conf += 0.5
                    notes.append(f"{sym}: rich val z={valz:.2f} edge≈{edge:.0f}bps sc={sc:.2f}")

            # Sentiment (short-horizon)
            if sent is not None:
                if sent >= self.cfg.sent_buy_th:
                    edge = (sent - self.cfg.sent_buy_th) * self.cfg.edge_bps_sent
                    sc = soft_score_from_edge(edge, self.cfg.horizon_sent)
                    score += sc; w += 0.5; conf += 0.45
                    notes.append(f"{sym}: sentiment={sent:.2f}↑ edge≈{edge:.0f}bps sc={sc:.2f}")
                elif sent <= self.cfg.sent_sell_th:
                    edge = (abs(sent - self.cfg.sent_sell_th)) * self.cfg.edge_bps_sent
                    sc = -soft_score_from_edge(edge, self.cfg.horizon_sent)
                    score += sc; w += 0.5; conf += 0.45
                    notes.append(f"{sym}: sentiment={sent:.2f}↓ edge≈{edge:.0f}bps sc={sc:.2f}")

            if w == 0.0:
                continue

            # Aggregate & cap
            score = clamp(score / max(1.0, w), -self.cfg.per_name_score_cap, self.cfg.per_name_score_cap)

            # Convert to trade intent
            side_long = score >= 0.0
            # Macro tilt multiplier
            size_mult = _macro_mult(item.sector, long_side=side_long)
            final = clamp(score * size_mult, -self.cfg.final_score_cap, self.cfg.final_score_cap)

            per_name_scores[sym] = final
            per_name_conf[sym] = clamp(conf / max(1.0, w), 0.0, 1.0)
            parts += notes
            tags += [item.sector.lower()]

        if not per_name_scores:
            return Proposal(orders=[], thesis="No equity edges triggered.", score=0.0,
                            horizon_sec=self.cfg.horizon_sent, confidence=0.3, tags=["idle"])

        # Rank by |score|; take top N; enforce min score
        ranked = sorted(per_name_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ranked = [kv for kv in ranked if abs(kv[1]) >= self.cfg.min_abs_score_to_trade][: self.cfg.max_legs]

        for sym, sc in ranked:
            item = next(i for i in self.cfg.watchlist if i.symbol == sym)
            qty_units = max(0.0, abs(sc)) * item.base_qty
            side = "BUY" if sc >= 0 else "SELL"
            legs.append(OrderPlan(symbol=sym, side=side, qty=qty_units, type="MARKET", venue=item.venue,
                                  meta={"score": sc, "sector": item.sector}))
            parts.append(f"{sym} {side} size≈{qty_units:g} (score={sc:.2f})")

        # Aggregate score/confidence (simple mean of selected)
        if ranked:
            avg_score = sum(sc for _, sc in ranked) / max(1.0, len(ranked))
            avg_conf = sum(per_name_conf[s] for s, _ in ranked) / max(1.0, len(ranked))
        else:
            avg_score, avg_conf = 0.0, 0.4

        thesis = " | ".join(parts[-12:])  # keep message compact

        return Proposal(
            orders=legs,
            thesis=thesis,
            score=clamp(avg_score, -1.0, 1.0),
            horizon_sec=max(self.cfg.horizon_earn, self.cfg.horizon_mom, self.cfg.horizon_val, self.cfg.horizon_sent),
            confidence=avg_conf,
            tags=list(dict.fromkeys(tags + ["equities"])),
            diagnostics={"per_name_scores": per_name_scores, "rate_10y": rate},
        )

    # -------- risk --------

    def risk(self, proposal: Proposal, context: MarketContext):
        # Reuse base constraint checks + notional summary
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