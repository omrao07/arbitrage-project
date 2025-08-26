# backend/oms/cost_model.py
"""
Unified Trading Cost Model
--------------------------
Purpose
- Provide ex-ante cost estimates for routing/position sizing
- Attach ex-post realized costs for TCA/PNL attribution

Supports
- Asset classes: equities/ETF, futures, options, crypto (basic)
- Venues: US (NYSE/NASDAQ), IN (NSE/BSE), generic (configurable)
- Costs: commission, exchange/clearing, taxes (incl. India STT approx),
         spread, slippage/impact (linear or sqrt), borrow/funding,
         FX conversion (home currency reporting)

Usage
-----
cm = CostModel.from_yaml("config/costs.yaml")  # or CostModel() for defaults
est = cm.estimate(
    side="buy", qty=1500, price=100.25, symbol="RELIANCE.NS",
    venue="NSE", instrument_type="equity",
    adv=10_000_000, vol=0.22, spread=0.02, notional_ccy="INR",
    broker="zerodha"
)
# est.total_bps, est.total_ccy, est.breakdown -> dict

# Ex-post on fill:
fill_adj = cm.apply_fill(price=100.30, exec_price=100.34, side="buy",
                         qty=500, symbol="AAPL", venue="NASDAQ",
                         instrument_type="equity")

Config (YAML) example (optional)
--------------------------------
brokers:
  zerodha:
    equity:
      commission_per_order: 20.0         # INR
      clearing_bps: 0.00325              # 0.00325% = 0.325 bps
      stt_bps: 0.1                       # buy STT 0.1 bps (illustrative)
      stt_bps_sell: 10.0                 # sell STT 10 bps (illustrative)
      gst_bps: 1.8                       # on brokerage (illustrative)
venues:
  NSE:
    tick_size: 0.05
    lot_size: 1
defaults:
  impact_model: sqrt
  impact_coef:
    sqrt_k: 0.75          # multiplier for sqrt(Q/ADV)*sigma
    linear_k: 0.05        # for linear model
  fx:
    base_ccy: INR
    rates: { "USDINR": 83.0, "BTCUSD": 1.0 }
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import yaml  # pip install pyyaml if using YAML config
except Exception:
    yaml = None  # type: ignore


# ----------------------------- data models -----------------------------

@dataclass
class CostBreakdown:
    # currency = notional_ccy for ex-ante; for ex-post, same as trade ccy
    commission: float = 0.0
    exchange_fees: float = 0.0
    clearing_fees: float = 0.0
    taxes: float = 0.0               # STT/SEC fee/GST etc.
    borrow_funding: float = 0.0      # borrow (equity short) or perp funding
    spread_cost: float = 0.0         # half-spread slippage estimate
    impact_cost: float = 0.0         # market impact/slippage beyond spread
    fx_conv: float = 0.0             # conversion cost to base ccy (if applied)
    other: float = 0.0

    def total(self) -> float:
        return (self.commission + self.exchange_fees + self.clearing_fees +
                self.taxes + self.borrow_funding + self.spread_cost +
                self.impact_cost + self.fx_conv + self.other)

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclass
class CostEstimate:
    symbol: str
    side: str              # 'buy'|'sell'
    venue: str
    instrument_type: str   # 'equity'|'future'|'option'|'crypto'
    qty: float
    price: float           # reference/mark
    notional: float
    notional_ccy: str
    breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    total_ccy: float = 0.0
    total_bps: float = 0.0     # relative to notional
    model_meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------- core model -----------------------------

class CostModel:
    """
    Config-driven cost model.
    If no config provided, reasonable defaults are used and can be overridden per-call.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or self._default_config()

    # ---- convenience constructors ----
    @classmethod
    def from_yaml(cls, path: str) -> "CostModel":
        if yaml is None:
            raise RuntimeError("pyyaml not installed. Use CostModel(config=...) or pip install pyyaml")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(cfg)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "CostModel":
        return cls(cfg)

    # ---- public API ----
    def estimate(
        self,
        *,
        side: str,
        qty: float,
        price: float,
        symbol: str,
        venue: str,
        instrument_type: str = "equity",
        # microstructure inputs (if unknown, pass None and we’ll infer fallbacks)
        adv: Optional[float] = None,            # average daily value (notional) in trade ccy
        vol: Optional[float] = None,            # daily stdev (sigma) ~ 20% = 0.2
        spread: Optional[float] = None,         # full spread in price units
        lot_size: Optional[int] = None,
        contract_multiplier: Optional[float] = None,  # for futures/options
        option_per_contract_fee: Optional[float] = None,
        # economics
        notional_ccy: str = "USD",
        broker: Optional[str] = None,
        home_ccy: Optional[str] = None,         # reporting currency; default from config
        fx_pair: Optional[str] = None,          # e.g., USDINR
        # model knobs
        impact_model: Optional[str] = None,     # 'sqrt'|'linear'|'none'
        override: Optional[Dict[str, Any]] = None,
    ) -> CostEstimate:
        """
        Returns a fully-populated CostEstimate object.
        """
        cfg = self._merge_runtime(overrides=override)
        base_ccy = (home_ccy or cfg["defaults"]["fx"]["base_ccy"]).upper()
        venue_cfg = (cfg.get("venues") or {}).get(venue, {})
        broker_cfg = ((cfg.get("brokers") or {}).get(broker or "", {}) or {}).get(instrument_type, {})
        lot = lot_size or venue_cfg.get("lot_size") or 1
        mult = float(contract_multiplier or venue_cfg.get("contract_multiplier") or 1.0)

        # Compute notional in trade currency
        trade_notional = float(qty) * float(price) * (mult if instrument_type in ("future", "option") else 1.0)

        # Spread: if not supplied, infer from venue tick or default bps
        if spread is None:
            tick = venue_cfg.get("tick_size", 0.01)
            # crude heuristic: 1–2 ticks typical for liquid; else fallback bps
            spread = max(tick, price * (cfg["defaults"].get("spread_bps", 1.0) / 1e4))

        # ADV/vol fallbacks
        adv = float(adv or cfg["defaults"].get("adv_fallback", 5_000_000))
        vol = float(vol or cfg["defaults"].get("vol_fallback", 0.20))

        # impact model
        imodel = (impact_model or cfg["defaults"].get("impact_model") or "sqrt").lower()

        # ---- components ----
        bd = CostBreakdown()

        # commissions & fixed
        if instrument_type == "option" and option_per_contract_fee is not None:
            bd.commission += float(option_per_contract_fee) * qty
        else:
            # per-order or bps of notional
            per_order = float(broker_cfg.get("commission_per_order", 0.0))
            per_share = float(broker_cfg.get("commission_per_share", 0.0)) * qty
            bps = float(broker_cfg.get("commission_bps", 0.0)) / 1e4 * trade_notional
            bd.commission += per_order + per_share + bps

        # exchange/clearing fees
        bd.exchange_fees += float(broker_cfg.get("exchange_bps", 0.0)) / 1e4 * trade_notional
        bd.clearing_fees += float(broker_cfg.get("clearing_bps", 0.0)) / 1e4 * trade_notional

        # taxes (very rough defaults; set to 0 or override in config for production)
        stt_buy_bps = float(broker_cfg.get("stt_bps", 0.0))
        stt_sell_bps = float(broker_cfg.get("stt_bps_sell", stt_buy_bps))
        gst_bps_on_brokerage = float(broker_cfg.get("gst_bps", 0.0))  # applied to commission in some locales
        if venue.upper() in ("NSE", "BSE", "NSE_FO", "BSE_FO"):
            # illustrative: apply STT on sells (cash equity) and on premium for options (customize via config)
            if instrument_type == "equity":
                stt_rate = stt_sell_bps if side.lower() == "sell" else stt_buy_bps
                bd.taxes += stt_rate / 1e4 * trade_notional
            elif instrument_type == "option":
                premium_notional = qty * price * mult
                bd.taxes += stt_sell_bps / 1e4 * premium_notional if side.lower() == "sell" else 0.0
        # generic GST on brokerage line
        if gst_bps_on_brokerage > 0 and bd.commission > 0:
            bd.taxes += gst_bps_on_brokerage / 1e4 * bd.commission

        # borrow/funding (if short or leveraged/perp)
        if side.lower() == "sell" and instrument_type == "equity":
            borrow_bps_daily = float(broker_cfg.get("borrow_bps_daily", 0.0))  # per day
            bd.borrow_funding += (borrow_bps_daily / 1e4) * trade_notional  # 1-day placeholder

        # spread cost (crossing half-spread on a marketable order)
        bd.spread_cost += 0.5 * float(spread) * qty * (mult if instrument_type in ("future", "option") else 1.0) # type: ignore

        # impact/slippage estimate
        if imodel == "sqrt":
            k = float(cfg["defaults"]["impact_coef"].get("sqrt_k", 1.0))
            participation = min(1.0, max(0.0, trade_notional / max(1.0, adv)))
            impact_bps = k * vol * math.sqrt(max(1e-9, participation)) * 1e4  # convert to bps
            bd.impact_cost += impact_bps / 1e4 * trade_notional
        elif imodel == "linear":
            k = float(cfg["defaults"]["impact_coef"].get("linear_k", 0.05))
            participation = min(1.0, max(0.0, trade_notional / max(1.0, adv)))
            impact_bps = k * vol * participation * 1e4
            bd.impact_cost += impact_bps / 1e4 * trade_notional
        else:
            # none
            pass

        # FX conversion to reporting currency
        report_ccy = base_ccy
        trade_ccy = (notional_ccy or report_ccy).upper()
        fx_cost_bps = float(cfg["defaults"].get("fx_cost_bps", 0.0))
        fx_conv_cost = 0.0
        if trade_ccy != report_ccy:
            pair = fx_pair or f"{trade_ccy}{report_ccy}"
            rate = self._fx_rate(cfg, pair)
            if rate is None:
                # if missing, assume 1 and no conv cost (better than exploding)
                conv_notional = trade_notional
            else:
                conv_notional = trade_notional * rate
            fx_conv_cost = fx_cost_bps / 1e4 * conv_notional
            bd.fx_conv += fx_conv_cost
            total_ccy = (bd.total() - fx_conv_cost) * rate + fx_conv_cost # type: ignore
        else:
            total_ccy = bd.total()

        # compute totals
        total_bps = (total_ccy / max(1.0, trade_notional)) * 1e4
        meta = {
            "impact_model": imodel,
            "participation": min(1.0, trade_notional / max(1.0, adv)),
            "sigma": vol,
            "spread": spread,
            "multiplier": mult,
            "lot_size": lot,
            "report_ccy": report_ccy,
            "trade_ccy": trade_ccy,
            "fx_pair": fx_pair,
        }

        return CostEstimate(
            symbol=symbol.upper(),
            side=side.lower(),
            venue=venue,
            instrument_type=instrument_type.lower(),
            qty=qty,
            price=price,
            notional=trade_notional,
            notional_ccy=trade_ccy,
            breakdown=bd,
            total_ccy=total_ccy,
            total_bps=total_bps,
            model_meta=meta,
        )

    def apply_fill(
        self,
        *,
        price: float,          # reference/mark at time of order (or arrival)
        exec_price: float,     # actual fill price
        side: str,
        qty: float,
        symbol: str,
        venue: str,
        instrument_type: str = "equity",
        notional_ccy: str = "USD",
        broker: Optional[str] = None,
    ) -> CostEstimate:
        """
        Ex-post realized: converts realized slippage (vs mark) + adds broker/fees/taxes
        using broker/venue defaults (no impact model).
        """
        base = self.estimate(
            side=side, qty=qty, price=price, symbol=symbol, venue=venue,
            instrument_type=instrument_type, notional_ccy=notional_ccy,
            broker=broker, impact_model="none"
        )
        # realized slippage on top (signed)
        signed = 1 if side.lower() == "buy" else -1
        realized = (exec_price - price) * signed * qty * (base.model_meta.get("multiplier", 1.0))
        base.breakdown.impact_cost += max(0.0, realized)  # attribute adverse move as impact
        base.total_ccy = base.breakdown.total()
        base.total_bps = (base.total_ccy / max(1.0, base.notional)) * 1e4
        base.model_meta["realized_exec_price"] = exec_price
        return base

    # ----------------------- helpers & defaults -----------------------

    def _fx_rate(self, cfg: Dict[str, Any], pair: str) -> Optional[float]:
        rates = ((cfg.get("defaults") or {}).get("fx") or {}).get("rates") or {}
        return rates.get(pair)

    def _merge_runtime(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(self.config)
        if overrides:
            # shallow merge for simplicity
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        return cfg

    def _default_config(self) -> Dict[str, Any]:
        # conservative global defaults; tune in config/costs.yaml
        return {
            "brokers": {
                "default": {
                    "equity": {"commission_bps": 0.5, "exchange_bps": 0.2, "clearing_bps": 0.05},
                    "future": {"commission_bps": 0.3, "exchange_bps": 0.1, "clearing_bps": 0.05},
                    "option": {"commission_bps": 0.3, "exchange_bps": 0.1, "clearing_bps": 0.05},
                    "crypto": {"commission_bps": 1.0},
                },
                "zerodha": {
                    "equity": {
                        "commission_per_order": 20.0,   # INR flat (illustrative)
                        "exchange_bps": 0.00325,        # 0.00325% (illustrative)
                        "clearing_bps": 0.0,
                        "stt_bps": 0.1,                  # illustrate buy
                        "stt_bps_sell": 10.0,            # illustrate sell
                        "gst_bps": 18.0,                 # GST on brokerage (illustrative, *per 100* here)
                    },
                    "option": {
                        "commission_per_order": 20.0,
                        "exchange_bps": 0.0,
                        "clearing_bps": 0.0,
                        "stt_bps": 0.0,
                        "stt_bps_sell": 5.0,
                        "gst_bps": 18.0,
                    }
                }
            },
            "venues": {
                "NSE": {"tick_size": 0.05, "lot_size": 1},
                "BSE": {"tick_size": 0.05, "lot_size": 1},
                "NASDAQ": {"tick_size": 0.01, "lot_size": 1},
                "NYSE": {"tick_size": 0.01, "lot_size": 1},
                "CME": {"tick_size": 0.25, "contract_multiplier": 50.0},  # ES example
            },
            "defaults": {
                "impact_model": "sqrt",
                "impact_coef": {"sqrt_k": 0.75, "linear_k": 0.05},
                "spread_bps": 1.0,         # used if tick not available
                "adv_fallback": 5_000_000,
                "vol_fallback": 0.20,
                "fx": {"base_ccy": "USD", "rates": {"USDINR": 83.0}},
                "fx_cost_bps": 0.5,        # cost of conversion
            }
        }


# ----------------------------- CLI probe -----------------------------
if __name__ == "__main__":
    cm = CostModel()
    # Ex-ante example
    e = cm.estimate(side="buy", qty=10000, price=100.0, symbol="AAPL",
                    venue="NASDAQ", instrument_type="equity",
                    adv=2e9, vol=0.25, spread=0.01, notional_ccy="USD", broker="default")
    print("EX-ANTE:", round(e.total_bps, 3), "bps", "@", e.notional_ccy, "=>", round(e.total_ccy, 2))
    print(e.breakdown.to_dict())
    # Ex-post example
    r = cm.apply_fill(price=100.0, exec_price=100.06, side="buy",
                      qty=2500, symbol="AAPL", venue="NASDAQ",
                      instrument_type="equity", notional_ccy="USD", broker="default")
    print("EX-POST:", round(r.total_bps, 3), "bps", "@", r.notional_ccy, "=>", round(r.total_ccy, 2))