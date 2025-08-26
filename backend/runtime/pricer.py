# backend/execution/pricer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List


# ----------------------------- Core types -----------------------------

@dataclass
class Quote:
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None

    def mid(self) -> Optional[float]:
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last if (self.last and self.last > 0) else None

    def spread_bps(self) -> Optional[float]:
        if self.bid and self.ask and self.bid > 0:
            return (self.ask - self.bid) / self.bid * 1e4
        return None


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float  # average cost in instrument currency
    currency: str = "USD"  # instrument currency (e.g., "USD", "INR")


@dataclass
class MTM:
    symbol: str
    qty: float
    avg_price: float
    mark: float
    currency: str
    market_value: float
    cost_value: float
    unrealized: float
    pnl_bps: float


# ----------------------------- Simple helpers -----------------------------

def _notional(price: float, qty: float) -> float:
    return float(price) * float(qty)

def _to_rate_pair(ccy_from: str, ccy_to: str) -> Tuple[str, int]:
    """
    Returns (pair, direction), where direction = +1 if value*rate, -1 if value/rate.
    Example: USD -> INR: returns ("USDINR", +1); INR -> USD: ("USDINR", -1)
    """
    a = ccy_from.upper(); b = ccy_to.upper()
    if a == b:
        return ("", +1)
    return (f"{a}{b}", +1) if a < b else (f"{b}{a}", -1)

def convert_ccy(value: float, from_ccy: str, to_ccy: str, fx_rates: Dict[str, float]) -> float:
    """
    Convert value from 'from_ccy' to 'to_ccy' using fx_rates dict: {"USDINR": 83.2, ...}
    """
    if from_ccy.upper() == to_ccy.upper():
        return float(value)
    pair, direction = _to_rate_pair(from_ccy, to_ccy)
    rate = fx_rates.get(pair)
    if rate is None or rate <= 0:
        raise ValueError(f"Missing FX rate for pair {pair}")
    return float(value * rate) if direction > 0 else float(value / rate)


# ----------------------------- MTM / PnL -----------------------------

def mark_position(pos: Position, quote: Quote, *, base_ccy: str = "USD",
                  fx_rates: Optional[Dict[str, float]] = None) -> Optional[MTM]:
    """
    Mark a single position to market using best available price (mid -> last).
    Converts to base_ccy if fx_rates provided and pos.currency != base_ccy.
    """
    px = quote.mid() or quote.last
    if px is None or px <= 0:
        return None

    mv = _notional(px, pos.qty)
    cv = _notional(pos.avg_price, pos.qty)
    unreal = mv - cv
    pnl_bps = (unreal / max(cv, 1e-9)) * 1e4 if cv != 0 else 0.0

    # FX conversion if needed
    if fx_rates and (pos.currency.upper() != base_ccy.upper()):
        mv = convert_ccy(mv, pos.currency, base_ccy, fx_rates)
        cv = convert_ccy(cv, pos.currency, base_ccy, fx_rates)
        unreal = mv - cv

    return MTM(
        symbol=pos.symbol, qty=pos.qty, avg_price=pos.avg_price, mark=px,
        currency=base_ccy if fx_rates else pos.currency,
        market_value=mv, cost_value=cv, unrealized=unreal, pnl_bps=pnl_bps
    )


def mark_portfolio(positions: Iterable[Position],
                   quotes: Dict[str, Quote],
                   *,
                   base_ccy: str = "USD",
                   fx_rates: Optional[Dict[str, float]] = None) -> Dict[str, any]: # type: ignore
    """
    Mark a list of positions. Returns dict with itemized MTM and totals.
    """
    items: List[MTM] = []
    tot_mv = 0.0
    tot_cv = 0.0
    for p in positions:
        q = quotes.get(p.symbol)
        if not q:
            continue
        m = mark_position(p, q, base_ccy=base_ccy, fx_rates=fx_rates)
        if m:
            items.append(m)
            tot_mv += m.market_value
            tot_cv += m.cost_value
    totals = {
        "market_value": tot_mv,
        "cost_value": tot_cv,
        "unrealized": tot_mv - tot_cv,
        "pnl_bps": ((tot_mv - tot_cv) / max(tot_cv, 1e-9)) * 1e4 if tot_cv else 0.0,
        "currency": base_ccy,
    }
    return {"items": items, "totals": totals}


# ----------------------------- Black–Scholes (vanilla) -----------------------------

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _norm_cdf(x: float) -> float:
    # Abramowitz–Stegun approximation via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _d1(S: float, K: float, r: float, q: float, vol: float, T: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))

def _d2(d1: float, vol: float, T: float) -> float:
    return d1 - vol * math.sqrt(T)

def bs_price(S: float, K: float, r: float, q: float, vol: float, T: float, call: bool = True) -> float:
    """
    Black–Scholes price for European option.
    S: spot, K: strike, r: risk-free (cont), q: dividend yield, vol: sigma (ann), T: years to expiry
    """
    if vol <= 0 or T <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = _d1(S, K, r, q, vol, T)
    d2 = _d2(d1, vol, T)
    if call:
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

def bs_greeks(S: float, K: float, r: float, q: float, vol: float, T: float, call: bool = True) -> Dict[str, float]:
    if vol <= 0 or T <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1 = _d1(S, K, r, q, vol, T)
    d2 = _d2(d1, vol, T)
    pdf = _norm_pdf(d1)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    delta = disc_q * (_norm_cdf(d1) if call else (_norm_cdf(d1) - 1.0))
    gamma = disc_q * pdf / (S * vol * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T) * 0.01  # per 1% vol
    if call:
        theta = (-S * disc_q * pdf * vol / (2 * math.sqrt(T))
                 - r * K * disc_r * _norm_cdf(d2)
                 + q * S * disc_q * _norm_cdf(d1)) / 365.0
        rho = (K * T * disc_r * _norm_cdf(d2)) * 0.01
    else:
        theta = (-S * disc_q * pdf * vol / (2 * math.sqrt(T))
                 + r * K * disc_r * _norm_cdf(-d2)
                 - q * S * disc_q * _norm_cdf(-d1)) / 365.0
        rho = (-K * T * disc_r * _norm_cdf(-d2)) * 0.01
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ----------------------------- Forwards / carry -----------------------------

def forward_price(spot: float, r: float, q: float, T: float) -> float:
    """
    Continuous-compounding forward: F = S * e^{(r - q) T}
    r: risk-free, q: dividend yield / carry, T: years
    """
    return float(spot * math.exp((r - q) * T))


# ----------------------------- Convenience: bulk price helpers -----------------------------

def quotes_from_last(last_map: Dict[str, float]) -> Dict[str, Quote]:
    """Build Quote dict if you only have last prices."""
    return {sym: Quote(symbol=sym, last=px) for sym, px in last_map.items()}

def mtm_dicts(result: Dict[str, any]) -> Dict[str, any]: # type: ignore
    """
    Convert mark_portfolio output to plain dicts for JSON (e.g., API response).
    """
    def _to(m: MTM) -> Dict[str, float | str]:
        return dict(
            symbol=m.symbol, qty=m.qty, avg_price=m.avg_price, mark=m.mark, currency=m.currency,
            market_value=m.market_value, cost_value=m.cost_value, unrealized=m.unrealized, pnl_bps=m.pnl_bps
        )
    items = [_to(it) for it in result["items"]]
    totals = dict(result["totals"])
    return {"items": items, "totals": totals}