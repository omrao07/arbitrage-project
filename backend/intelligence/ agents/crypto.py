# agents/crypto.py
"""
CryptoAgent
-----------
Signal-driven crypto agent for BTC/ETH.

Signals expected in `MarketContext.signals` (all optional; skipped if missing):
- "btc_basis_annual"        : annualized futures/perp basis for BTC (e.g., 0.08 = 8%/yr)
- "eth_basis_annual"        : same for ETH
- "btc_funding_8h"          : next 8h funding rate (fraction, e.g., 0.00015 = 1.5 bps)
- "eth_funding_8h"          : same for ETH
- "social_sent_btc"         : normalized sentiment score [-1, 1]
- "social_sent_eth"         : normalized sentiment score [-1, 1]
- "vol_z_btc"               : z-score of realized/IV vs lookback (regime tag)
- "vol_z_eth"               : z-score for ETH
- "coinbase_premium_btc_bps": spot premium (Coinbase vs Binance) in bps (for venue choice / micro-meanrevert)
- "coinbase_premium_eth_bps": same for ETH

Symbols (override via config if your codes differ):
- BTCUSDT, ETHUSDT (spot or perpetual proxy)

Strategy motifs (each may add a leg):
1) **Carry/Basis**: Go long spot when basis is meaningfully positive & funding small.
2) **Sentiment**: Lean with social momentum on short horizons.
3) **Vol regime**: If vol_z very low, favor mean-reversion sizing; if very high, reduce size (risk).
4) **Venue micro-edge (optional)**: Nudge venue tag towards cheaper venue when premium is rich.

Output:
- A single Proposal possibly with multiple legs (BTC and/or ETH).

Tune via `CryptoConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .base import ( # type: ignore
    AgentBase, MarketContext, Proposal, OrderPlan, Constraints,
    soft_score_from_edge, clamp
)


# ------------------- configuration -------------------

@dataclass
class CryptoConfig:
    # Symbols
    sym_btc: str = "BTCUSDT"
    sym_eth: str = "ETHUSDT"

    # Base sizes
    qty_btc: float = 0.05
    qty_eth: float = 1.0

    # Carry/Basis thresholds
    min_basis_to_buy_btc: float = 0.04   # 4% annualized -> bullish carry (long spot)
    min_basis_to_buy_eth: float = 0.05
    max_funding_8h_abs_btc: float = 0.0003  # cap on near-term funding to avoid rich payers
    max_funding_8h_abs_eth: float = 0.0004

    # Sentiment thresholds
    sent_buy_th: float = 0.25
    sent_sell_th: float = -0.35

    # Edge scaling (bps per unit of signal “excess”)
    edge_bps_per_basis: float = 160.0        # per 1.0 of (basis - threshold), annualized
    edge_bps_per_sent: float = 60.0          # per 1.0 of (sent - threshold)

    # Horizon (seconds)
    horizon_basis_sec: float = 3 * 24 * 3600
    horizon_sent_sec: float = 2 * 3600

    # Volatility regime modifiers
    vol_z_reduce_thresh: float = 1.5         # if |vol_z| > this, reduce size
    vol_z_boost_thresh: float = -0.8         # if vol_z < this (quiet), small boost for mean-revert/micro alpha
    size_reduce_factor_hi_vol: float = 0.5
    size_boost_factor_low_vol: float = 1.25
    score_reduce_factor_hi_vol: float = 0.7

    # Venue tilt based on spot premium (bps)
    venue_binance: str = "BINANCE"
    venue_coinbase: str = "COINBASE"         # if you have/plan an adapter
    premium_tilt_bps: float = 5.0            # if premium > +5 bps on Coinbase vs Binance, prefer Binance for buys, etc.


# ------------------- agent -------------------

class CryptoAgent(AgentBase):
    name = "crypto"

    def __init__(self, cfg: Optional[CryptoConfig] = None):
        self.cfg = cfg or CryptoConfig()

    # -------- propose --------

    def propose(self, context: MarketContext) -> Proposal:
        s = context.signals or {}
        prices = context.prices or {}

        legs: List[OrderPlan] = []
        parts: List[str] = []
        tags: List[str] = []
        score_accum = 0.0
        w_accum = 0.0
        conf_accum = 0.0

        # ---- derive per-asset motifs
        for asset in ("BTC", "ETH"):
            sym = self.cfg.sym_btc if asset == "BTC" else self.cfg.sym_eth
            if sym not in prices:
                continue

            # Inputs
            basis = _f(s.get(f"{asset.lower()}_basis_annual"), None)
            funding = _f(s.get(f"{asset.lower()}_funding_8h"), None)
            sent = _f(s.get(f"social_sent_{asset.lower()}"), None)
            volz = _f(s.get(f"vol_z_{asset.lower()}"), 0.0)
            premium_bps = _f(s.get(f"coinbase_premium_{asset.lower()}_bps"), 0.0)

            # Sizing modifiers from vol regime
            size_mult, score_mult = self._vol_modifiers(volz) # type: ignore

            # Base qty
            qty = (self.cfg.qty_btc if asset == "BTC" else self.cfg.qty_eth) * size_mult

            # Venue tilt (optional)
            venue = self._pick_venue(premium_bps, side_hint="BUY")  # type: ignore # default tilt for long motifs

            # --- 1) Basis/Carry motif (long spot if annualized basis is rich & funding tame)
            if basis is not None:
                min_basis = (self.cfg.min_basis_to_buy_btc if asset == "BTC" else self.cfg.min_basis_to_buy_eth)
                max_fund = (self.cfg.max_funding_8h_abs_btc if asset == "BTC" else self.cfg.max_funding_8h_abs_eth)
                if basis >= min_basis and (funding is None or abs(funding) <= max_fund):
                    excess = basis - min_basis
                    edge_bps = excess * self.cfg.edge_bps_per_basis
                    score = soft_score_from_edge(edge_bps, self.cfg.horizon_basis_sec, cap=1.0) * score_mult
                    if qty > 0:
                        legs.append(OrderPlan(symbol=sym, side="BUY", qty=qty, type="MARKET", venue=venue,
                                              meta={"motif": "carry", "basis": basis, "funding_8h": funding}))
                        parts.append(f"{asset}: basis={basis:.3f} (>{min_basis:.3f}), funding={_fmt(funding)} → edge≈{edge_bps:.0f}bps, score={score:.2f}")
                        tags += ["carry", "basis", asset.lower()]
                        score_accum += score; w_accum += 1.0; conf_accum += 0.65

            # --- 2) Sentiment motif (short-horizon)
            if sent is not None:
                if sent >= self.cfg.sent_buy_th:
                    excess = sent - self.cfg.sent_buy_th
                    edge_bps = excess * self.cfg.edge_bps_per_sent
                    score = soft_score_from_edge(edge_bps, self.cfg.horizon_sent_sec, cap=1.0) * score_mult
                    if qty > 0:
                        venue2 = self._pick_venue(premium_bps, side_hint="BUY") # type: ignore
                        legs.append(OrderPlan(symbol=sym, side="BUY", qty=0.5 * qty, type="MARKET", venue=venue2,
                                              meta={"motif": "sent_up", "sent": sent}))
                        parts.append(f"{asset}: sentiment={sent:.2f}↑ → edge≈{edge_bps:.0f}bps, score={score:.2f}")
                        tags += ["sentiment", asset.lower()]
                        score_accum += score * 0.8; w_accum += 0.8; conf_accum += 0.55
                elif sent <= self.cfg.sent_sell_th:
                    excess = abs(sent - self.cfg.sent_sell_th)
                    edge_bps = excess * self.cfg.edge_bps_per_sent
                    score = soft_score_from_edge(edge_bps, self.cfg.horizon_sent_sec, cap=1.0) * score_mult
                    if qty > 0:
                        venue2 = self._pick_venue(premium_bps, side_hint="SELL") # type: ignore
                        legs.append(OrderPlan(symbol=sym, side="SELL", qty=0.5 * qty, type="MARKET", venue=venue2,
                                              meta={"motif": "sent_down", "sent": sent}))
                        parts.append(f"{asset}: sentiment={sent:.2f}↓ → edge≈{edge_bps:.0f}bps, score={score:.2f}")
                        tags += ["sentiment", asset.lower()]
                        score_accum += score * 0.8; w_accum += 0.8; conf_accum += 0.55

        # If no legs, return neutral
        if not legs:
            return Proposal(orders=[], thesis="No crypto edges triggered.", score=0.0, horizon_sec=3600.0, confidence=0.3, tags=["idle"])

        avg_score = clamp(score_accum / max(1.0, w_accum), -1.0, 1.0)
        avg_conf = clamp(conf_accum / max(1.0, w_accum), 0.0, 1.0)
        thesis = " | ".join(parts)

        return Proposal(
            orders=legs,
            thesis=thesis,
            score=avg_score,
            horizon_sec=max(self.cfg.horizon_basis_sec, self.cfg.horizon_sent_sec),
            confidence=avg_conf,
            tags=list(dict.fromkeys(tags)),
            diagnostics={"vol_mod": "applied" if avg_score != score_accum else "none"},
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

    # ------------------- helpers -------------------

    def _vol_modifiers(self, volz: float) -> tuple[float, float]:
        """
        Return (size_mult, score_mult) based on volatility regime.
        """
        if abs(volz) >= self.cfg.vol_z_reduce_thresh:
            return (self.cfg.size_reduce_factor_hi_vol, self.cfg.score_reduce_factor_hi_vol)
        if volz <= self.cfg.vol_z_boost_thresh:
            return (self.cfg.size_boost_factor_low_vol, 1.0)
        return (1.0, 1.0)

    def _pick_venue(self, premium_bps: float, *, side_hint: str) -> Optional[str]:
        """
        If Coinbase trades richer than Binance by > premium_tilt_bps,
        prefer the cheaper venue to buy, and the richer venue to sell.
        """
        tilt = self.cfg.premium_tilt_bps
        if premium_bps is None:
            return None
        if side_hint.upper() == "BUY":
            # If Coinbase premium is high, prefer Binance to buy cheaper
            if premium_bps >= tilt:
                return self.cfg.venue_binance
        else:
            # If Coinbase premium is high, prefer selling on Coinbase
            if premium_bps >= tilt:
                return self.cfg.venue_coinbase
        return None


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
        return f"{float(x):.5f}"
    except Exception:
        return str(x)