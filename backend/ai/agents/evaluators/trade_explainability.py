# backend/ai/agents/analysis/trade_explainability.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Data schemas (lightweight)
# ---------------------------------------------------------------------
@dataclass
class TradeOrder:
    order_id: str
    strategy: str
    symbol: str
    side: str                      # "buy" | "sell"
    qty: float
    order_type: str = "market"     # "market" | "limit"
    limit_price: Optional[float] = None
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g., {"signal":0.62,"features":{"mom_5m":..}}

@dataclass
class TradeFill:
    order_id: str
    fill_id: str
    symbol: str
    qty: float
    price: float
    venue: Optional[str] = None
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g., {"lat_ms": 21}

@dataclass
class MarketSnapshot:
    # Prices around the decision/arrival & benchmark
    arrival_px: float                 # mid or reference when order was sent
    benchmark_px: Optional[float] = None  # e.g., open/close/VWAP if known
    prev_close_px: Optional[float] = None
    last_px: Optional[float] = None       # current/exit
    bid_px: Optional[float] = None
    ask_px: Optional[float] = None
    spread_px: Optional[float] = None     # if set, overrides bid/ask calc
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))

@dataclass
class RiskSnapshot:
    var_1d: Optional[float] = None         # as fraction, e.g., 0.025
    beta: Optional[float] = None
    exposure_after: Optional[float] = None  # $ notional post-trade
    leverage: Optional[float] = None
    limits: Dict[str, Any] = field(default_factory=dict)  # {"max_notional":..}

@dataclass
class FactorCoeffs:
    # Simple linear model for feature attributions: contrib = w * feature
    weights: Dict[str, float] = field(default_factory=dict)   # {"mom_5m":0.8,"rev_1h":-0.3,...}
    intercept: float = 0.0

@dataclass
class ExplainConfig:
    shortfall_benchmark: str = "arrival"     # "arrival" | "prev_close" | "custom"
    custom_benchmark_px: Optional[float] = None
    round_prices_to: int = 4                 # decimals
    assume_half_spread: bool = True          # use half-spread to approximate arrival slippage if bid/ask known

# ---------------------------------------------------------------------
# Core Explainer
# ---------------------------------------------------------------------
class TradeExplainer:
    """
    Provides human + machine explanations for trades:
      • Rationale from signals/features
      • Expected pnl vs realized pnl
      • Implementation shortfall & slippage bps
      • Basic factor attribution (linear, dependency-free)
      • Venue/latency notes and risk context
      • Replayable timeline

    All inputs are passed in—no external I/O. Safe to call anywhere.
    """

    def __init__(self, config: Optional[ExplainConfig] = None):
        self.cfg = config or ExplainConfig()

    # ---------------- Public API ----------------
    def explain_trade(
        self,
        order: TradeOrder,
        fills: List[TradeFill],
        market: MarketSnapshot,
        *,
        risk: Optional[RiskSnapshot] = None,
        factors: Optional[FactorCoeffs] = None,
    ) -> Dict[str, Any]:
        """
        Returns a structured dict with 'headline', 'rationale', 'metrics', 'attribution', 'risk', 'timeline'.
        """
        direction = 1.0 if order.side.lower() == "buy" else -1.0
        qty = float(order.qty)
        arrival = float(market.arrival_px)
        bench_px = self._pick_benchmark_px(market)

        # Aggregate fills
        tot_qty, vwap_px, first_ts, last_ts, venues = self._aggregate_fills(fills)

        # Metrics
        slippage_abs, slippage_bps = self._slippage(bench_px, vwap_px, direction)
        shortfall_abs, shortfall_bps = self._slippage(arrival, vwap_px, direction)
        spread = self._spread(market)

        exp_pnl, realized_pnl = self._pnl_estimates(order, vwap_px, market, direction, qty)

        # Attribution
        features = (order.meta or {}).get("features", {})
        signal = (order.meta or {}).get("signal", None)
        attribution, expected_alpha = self._factor_attribution(features, factors)

        # Risk
        risk_dict = asdict(risk) if risk else {}
        risk_dict["exposure_change"] = direction * qty * vwap_px

        # Narrative (human)
        headline = self._headline(order, vwap_px, shortfall_bps, realized_pnl)
        rationale = self._rationale(order, signal, features, attribution, expected_alpha)

        # Timeline
        timeline = self._timeline(order, fills, market)

        # Compliance notes (light)
        compliance = self._compliance(order, market, slippage_bps, shortfall_bps)

        # Result
        metrics = {
            "qty": qty,
            "arrival_px": self._rp(arrival),
            "benchmark_px": self._rp(bench_px) if bench_px is not None else None,
            "fill_vwap_px": self._rp(vwap_px),
            "slippage_vs_benchmark_abs": self._rp(slippage_abs),
            "slippage_vs_benchmark_bps": round(slippage_bps, 3),
            "implementation_shortfall_abs": self._rp(shortfall_abs),
            "implementation_shortfall_bps": round(shortfall_bps, 3),
            "spread_px": self._rp(spread) if spread is not None else None,
            "expected_pnl": self._rp(exp_pnl) if exp_pnl is not None else None,
            "realized_pnl": self._rp(realized_pnl) if realized_pnl is not None else None,
            "latency_ms": (fills[0].meta.get("lat_ms") if fills and isinstance(fills[0].meta, dict) else None),
            "venues": venues,
        }

        return {
            "headline": headline,
            "rationale": rationale,
            "metrics": metrics,
            "attribution": {
                "features": features,
                "weights": (asdict(factors) if factors else {"weights": {}, "intercept": 0.0}),
                "expected_alpha": expected_alpha,
                "contributions": attribution,
            },
            "risk": risk_dict,
            "compliance": compliance,
            "timeline": timeline,
            "raw": {
                "order": asdict(order),
                "fills": [asdict(f) for f in fills],
                "market": asdict(market),
            }
        }

    def explain_portfolio(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Minimal roll-up across multiple explain_trade() outputs.
        """
        total_realized = 0.0
        total_expected = 0.0
        n = 0
        out: List[Dict[str, Any]] = []
        for t in trades:
            m = t.get("metrics", {})
            if m.get("realized_pnl") is not None:
                total_realized += float(m["realized_pnl"])
            if m.get("expected_pnl") is not None:
                total_expected += float(m["expected_pnl"])
            n += 1
            out.append({"symbol": t.get("raw", {}).get("order", {}).get("symbol"),
                        "headline": t.get("headline"),
                        "realized_pnl": t.get("metrics", {}).get("realized_pnl")})
        return {
            "summary": {
                "n_trades": n,
                "sum_expected_pnl": round(total_expected, 2),
                "sum_realized_pnl": round(total_realized, 2)
            },
            "trades": out
        }

    # ---------------- Internals ----------------
    def _pick_benchmark_px(self, m: MarketSnapshot) -> Optional[float]:
        if self.cfg.shortfall_benchmark == "arrival":
            return m.arrival_px
        if self.cfg.shortfall_benchmark == "prev_close":
            return m.prev_close_px
        if self.cfg.shortfall_benchmark == "custom":
            return self.cfg.custom_benchmark_px
        return m.arrival_px

    def _aggregate_fills(self, fills: List[TradeFill]) -> Tuple[float, float, Optional[int], Optional[int], List[str]]:
        if not fills:
            return 0.0, float("nan"), None, None, []
        qty = sum(float(f.qty) for f in fills)
        vwap = sum(float(f.qty) * float(f.price) for f in fills) / max(qty, 1e-12)
        first_ts = min(f.ts_ms for f in fills)
        last_ts = max(f.ts_ms for f in fills)
        venues = sorted(set([f.venue for f in fills if f.venue]))
        return qty, vwap, first_ts, last_ts, venues

    def _slippage(self, bench_px: Optional[float], vwap_px: float, direction: float) -> Tuple[float, float]:
        if bench_px is None or not math.isfinite(vwap_px):
            return float("nan"), float("nan")
        diff = (vwap_px - bench_px) * direction
        bps = (diff / bench_px) * 1e4 if bench_px else float("nan")
        return diff, bps

    def _spread(self, m: MarketSnapshot) -> Optional[float]:
        if m.spread_px is not None:
            return float(m.spread_px)
        if m.bid_px is not None and m.ask_px is not None:
            return float(m.ask_px) - float(m.bid_px)
        return None

    def _pnl_estimates(self, order: TradeOrder, vwap_px: float, market: MarketSnapshot, direction: float, qty: float) -> Tuple[Optional[float], Optional[float]]:
        # Expected PnL: use (last - fill) * dir * qty if last is provided (or zero if unknown)
        exp = None
        if market.last_px is not None and math.isfinite(vwap_px):
            exp = (float(market.last_px) - vwap_px) * direction * qty
        # Realized PnL (if order.meta carries an exit_px)
        exit_px = (order.meta or {}).get("exit_px")
        realized = None
        if exit_px is not None and math.isfinite(vwap_px):
            realized = (float(exit_px) - vwap_px) * direction * qty
        return exp, realized

    def _factor_attribution(self, features: Dict[str, Any], coeffs: Optional[FactorCoeffs]) -> Tuple[Dict[str, float], Optional[float]]:
        if not coeffs or not coeffs.weights:
            return {}, None
        contribs: Dict[str, float] = {}
        total = coeffs.intercept
        for k, w in coeffs.weights.items():
            x = float(features.get(k, 0.0) or 0.0)
            c = float(w) * x
            contribs[k] = round(c, 6)
            total += c
        return contribs, round(total, 6)

    def _headline(self, order: TradeOrder, vwap_px: float, shortfall_bps: float, realized_pnl: Optional[float]) -> str:
        side = order.side.upper()
        sym = order.symbol.upper()
        sf = "n/a" if (not math.isfinite(shortfall_bps)) else f"{shortfall_bps:+.1f} bps"
        rp = "" if realized_pnl is None else f" | Realized PnL={realized_pnl:+.2f}"
        return f"{side} {int(order.qty)} {sym} @ VWAP {self._rp(vwap_px)} | Impl. shortfall={sf}{rp}"

    def _rationale(self, order: TradeOrder, signal: Optional[float], features: Dict[str, Any],
                   attribution: Dict[str, float], expected_alpha: Optional[float]) -> Dict[str, Any]:
        sig_str = None if signal is None else round(float(signal), 4)
        ranked = sorted(attribution.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        top_list = [{"feature": k, "contrib": v} for k, v in ranked]
        return {
            "strategy": order.strategy,
            "signal_score": sig_str,
            "top_feature_contributions": top_list,
            "expected_alpha": expected_alpha
        }

    def _timeline(self, order: TradeOrder, fills: List[TradeFill], market: MarketSnapshot) -> List[Dict[str, Any]]:
        tl = [{
            "t": order.ts_ms,
            "type": "order_submitted",
            "data": {"order_id": order.order_id, "side": order.side, "qty": order.qty}
        }]
        for f in sorted(fills, key=lambda x: x.ts_ms):
            tl.append({"t": f.ts_ms, "type": "fill", "data": {"fill_id": f.fill_id, "qty": f.qty, "price": self._rp(f.price), "venue": f.venue}})
        tl.append({"t": market.ts_ms, "type": "market_snapshot", "data": {"arrival_px": self._rp(market.arrival_px), "last_px": self._rp(market.last_px) if market.last_px is not None else None}})
        return tl

    def _compliance(self, order: TradeOrder, market: MarketSnapshot, slip_bps: float, shortfall_bps: float) -> Dict[str, Any]:
        notes = []
        if math.isfinite(slip_bps) and abs(slip_bps) > 50:
            notes.append("High slippage vs benchmark (>50 bps).")
        spread = self._spread(market)
        if spread is not None and spread > 0 and self.cfg.assume_half_spread:
            notes.append("Arrival mid approximated using half-spread.")
        if order.order_type == "market" and spread and spread > 0.01 * market.arrival_px:
            notes.append("Wide spread for market order.")
        return {"notes": notes} if notes else {}

    def _rp(self, x: Optional[float]) -> Optional[float]:
        if x is None or not math.isfinite(float(x)):
            return None
        return round(float(x), self.cfg.round_prices_to)

# ---------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    order = TradeOrder(
        order_id="PB-123",
        strategy="example_buy_dip",
        symbol="AAPL",
        side="buy",
        qty=100,
        order_type="limit",
        limit_price=190.0,
        meta={"signal": 0.72, "features": {"mom_5m": 0.8, "rev_1h": -0.2, "vol_z": -0.5}, "exit_px": 191.3}
    )
    fills = [
        TradeFill(order_id="PB-123", fill_id="F1", symbol="AAPL", qty=60, price=189.95, venue="SIM", ts_ms=int(time.time()*1000)-600),
        TradeFill(order_id="PB-123", fill_id="F2", symbol="AAPL", qty=40, price=190.05, venue="SIM", ts_ms=int(time.time()*1000)-300),
    ]
    market = MarketSnapshot(arrival_px=190.0, prev_close_px=188.2, last_px=191.0, bid_px=189.9, ask_px=190.1)
    risk = RiskSnapshot(var_1d=0.022, beta=1.1, exposure_after=100*190.0, leverage=1.2, limits={"max_notional": 1_000_000})
    coeffs = FactorCoeffs(weights={"mom_5m": 0.4, "rev_1h": -0.3, "vol_z": -0.1}, intercept=0.02)

    explainer = TradeExplainer()
    out = explainer.explain_trade(order, fills, market, risk=risk, factors=coeffs)
    from pprint import pprint
    pprint(out)