# backend/engine/strategies/vol_smile_arb.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, DefaultDict
from collections import defaultdict, deque

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


# ----------------------------- Config -----------------------------

@dataclass
class VolSmileArbConfig:
    underlying: str = "SPY"                   # underlier to monitor
    venues: tuple[str, ...] = ("IBKR", "PAPER")

    # universe / filters
    max_tau_yr: float = 0.25                  # use options up to 3 months
    min_tau_yr: float = 5.0 / 365.0           # ignore very near exp (< 5 days)
    min_points_for_fit: int = 8               # per-expiry smile fit requirement
    fit_degree: int = 2                       # polynomial degree on log-moneyness (2 = parabola)

    # entry / exit
    enter_z: float = 2.0                      # |residual z| to open a spread
    exit_z: float = 0.8                       # z moves back inside -> don't trade / (optional unwind hook)

    # spread construction
    max_gross_premium: float = 2_500.0        # cap per trade (quote currency)
    target_vega_neutral: bool = True          # try to pair to near-zero vega
    max_vega_ratio_mismatch: float = 0.4      # |(v_long + v_short)| / (|v_long| + |v_short|)
    min_strike_dist_bps: float = 50.0         # min 0.5% strike distance between legs
    max_strike_dist_bps: float = 400.0        # max 4% distance when pairing

    # sizing
    lot_mult: float = 100.0                   # equity options contract multiplier
    slippage_bps: float = 5.0                 # execution markup for limit prices
    cooldown_ms: int = 1500                   # between trades
    default_qty: float = 1.0                  # fallback contracts

    # safety
    hard_kill: bool = False


# ------------------------------ Math ------------------------------

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _bs_d1(S: float, K: float, tau: float, vol: float, r: float = 0.0, q: float = 0.0) -> float:
    if S <= 0 or K <= 0 or tau <= 0 or vol <= 0:
        return 0.0
    return (math.log(S / K) + (r - q + 0.5 * vol * vol) * tau) / (vol * math.sqrt(tau))

def _bs_vega(S: float, K: float, tau: float, vol: float, q: float = 0.0) -> float:
    if tau <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = _bs_d1(S, K, tau, vol, 0.0, q)
    return S * math.exp(-q * tau) * math.sqrt(tau) * _norm_pdf(d1)  # price per 1.0 vol (not %) 

def _bisection_implied_vol(option_mid: float, S: float, K: float, tau: float, is_call: bool,
                           r: float = 0.0, q: float = 0.0, max_iter: int = 60) -> Optional[float]:
    """
    Minimal bisection IV solver using Black-Scholes (no dividends aside from q).
    Expects 'option_mid' per 1 underlying unit (divide by lot multiplier beforehand).
    """
    if option_mid <= 0 or S <= 0 or K <= 0 or tau <= 0:
        return None
    # very rough bounds
    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = _bs_price(S, K, tau, mid, is_call, r, q)
        if abs(price - option_mid) < 1e-6:
            return mid
        if price > option_mid:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def _cdf(x: float) -> float:
    # Abramowitz-Stegun approx via error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_price(S: float, K: float, tau: float, vol: float, is_call: bool,
              r: float = 0.0, q: float = 0.0) -> float:
    if tau <= 0 or vol <= 0:
        # intrinsic
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = _bs_d1(S, K, tau, vol, r, q)
    d2 = d1 - vol * math.sqrt(tau)
    if is_call:
        return S * math.exp(-q * tau) * _cdf(d1) - K * math.exp(-r * tau) * _cdf(d2)
    else:
        return K * math.exp(-r * tau) * _cdf(-d2) - S * math.exp(-q * tau) * _cdf(-d1)


# --------------------------- Strategy ----------------------------

class VolSmileArb(Strategy):
    """
    Volatility-smile arbitrage (educational / conservative):
      * Builds a smooth smile per expiry by regressing IV on log-moneyness x = ln(K/F).
      * Flags local mispricings via residual z-scores.
      * Enters small **vega-balanced verticals**: buy 'cheap' (negative residual), sell 'rich' (positive residual)
        within the same expiry and nearby strikes to neutralize primary vega.
    Feed tolerance for option ticks:
      {
        "type": "option",
        "symbol": "...",         # option contract symbol
        "underlying": "SPY",
        "right": "C"|"P",
        "strike": 440.0,
        "expiry_ts": 1737072000000,  # ms epoch
        "bid": 1.23, "ask": 1.31,    # per-contract price in quote currency
        "iv": 0.22,                  # optional – if absent, we back out IV from mid
        "venue": "IBKR"
      }
    Underlying tick (to keep latest S):
      { "symbol": "SPY", "price": 443.25 }  (or bid/ask/mid)
    """

    def __init__(self, name="alpha_vol_smile_arb", region=None, cfg: Optional[VolSmileArbConfig] = None):
        cfg = cfg or VolSmileArbConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty)
        self.cfg = cfg
        self.und = cfg.underlying.upper()

        # rolling state
        self.S: float = 0.0
        self._last_trade_ms: int = 0

        # per-expiry storage
        # chains[exp_ms] = list of option points
        self.chains: DefaultDict[int, Dict[str, Any]] = defaultdict(lambda: {
            "points": {},      # key by symbol -> OptionPoint
            "fit": None,       # (a,b,c,...) poly coefficients
            "resid_mu": 0.0,   # EWMA of residuals
            "resid_var": 1e-6  # EWMA of residual^2
        })

        # keep recent symbols per expiry to pair cheaply
        self._recent_by_exp: DefaultDict[int, deque] = defaultdict(lambda: deque(maxlen=256))

    # -------------------- lifecycle --------------------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["options", "vol", "smile", "stat_arb"],
            "region": self.ctx.region or "US",
            "notes": "Per-expiry smile fit on log-moneyness; pair rich/cheap options into vega-balanced verticals."
        })

    # ------------------- helpers -----------------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _mid_from_tick(t: Dict[str, Any]) -> Optional[float]:
        m = t.get("mid")
        if m is not None:
            try:
                return float(m)
            except Exception:
                return None
        b, a = t.get("bid"), t.get("ask")
        if b is not None and a is not None:
            try:
                b = float(b); a = float(a)
                if b > 0 and a > 0:
                    return 0.5 * (b + a)
            except Exception:
                return None
        p = t.get("price") or t.get("p")
        try:
            return float(p) if p is not None else None
        except Exception:
            return None

    @staticmethod
    def _years_until(expiry_ms: int, now_ms: int) -> float:
        return max(0.0, (expiry_ms - now_ms) / (365.0 * 24.0 * 3600.0 * 1000.0))

    @staticmethod
    def _poly_fit(x: List[float], y: List[float], deg: int) -> Optional[Tuple[float, ...]]:
        """
        Small OLS polynomial fit using normal equations (no numpy).
        Returns tuple of coefficients c0..c_deg for y ≈ Σ c_i x^i
        """
        n = len(x)
        if n == 0 or len(y) != n:
            return None
        deg = max(0, min(4, deg))
        m = deg + 1
        # Build Vandermonde sums
        # A = X^T X, b = X^T y
        pow_cache = [[1.0] * m for _ in range(n)]
        for i in range(n):
            for j in range(1, m):
                pow_cache[i][j] = pow_cache[i][j - 1] * x[i]
        A = [[0.0] * m for _ in range(m)]
        bvec = [0.0] * m
        for i in range(n):
            for r in range(m):
                bvec[r] += pow_cache[i][r] * y[i]
                for c in range(m):
                    A[r][c] += pow_cache[i][r] * pow_cache[i][c]
        # Solve A c = b via Gaussian elimination
        # augment
        for r in range(m):
            A[r].append(bvec[r])
        # elimination
        for col in range(m):
            # pivot
            piv = col
            piv_val = abs(A[piv][col])
            for r in range(col + 1, m):
                if abs(A[r][col]) > piv_val:
                    piv, piv_val = r, abs(A[r][col])
            if piv != col:
                A[col], A[piv] = A[piv], A[col]
            # normalize
            denom = A[col][col] if abs(A[col][col]) > 1e-12 else 1e-12
            for c in range(col, m + 1):
                A[col][c] /= denom
            # eliminate
            for r in range(m):
                if r == col:
                    continue
                factor = A[r][col]
                for c in range(col, m + 1):
                    A[r][c] -= factor * A[col][c]
        coeffs = tuple(A[r][m] for r in range(m))
        return coeffs

    @staticmethod
    def _poly_eval(coeffs: Tuple[float, ...], x: float) -> float:
        s = 0.0
        p = 1.0
        for c in coeffs:
            s += c * p
            p *= x
        return s

    def _fit_smile_for_expiry(self, exp_ms: int, now_ms: int) -> Optional[Tuple[float, ...]]:
        chain = self.chains[exp_ms]
        pts = [v for v in chain["points"].values() if v.get("ok")]
        if len(pts) < self.cfg.min_points_for_fit:
            chain["fit"] = None
            return None
        # forward proxy: just use spot S (rates small)
        F = max(1e-9, self.S)
        xs: List[float] = []
        ys: List[float] = []
        for p in pts:
            K = p["strike"]
            iv = p["iv"]
            if K <= 0 or iv is None or iv <= 0:
                continue
            x = math.log(K / F)
            xs.append(x)
            ys.append(iv)
        if len(xs) < self.cfg.min_points_for_fit:
            chain["fit"] = None
            return None
        coeffs = self._poly_fit(xs, ys, self.cfg.fit_degree)
        chain["fit"] = coeffs
        return coeffs

    # ------------------- core -------------------------

    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill:
            return

        typ = (tick.get("type") or "").lower()
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        ven = (tick.get("venue") or tick.get("v") or "").upper()
        now = self._now_ms()

        # 1) Underlying update
        if sym == self.und and typ != "option":
            S = self._mid_from_tick(tick)
            if S and S > 0:
                self.S = float(S)
            return

        # 2) Option tick handling
        if typ != "option":
            return
        if ven and self.cfg.venues and ven not in self.cfg.venues:
            return
        und = (tick.get("underlying") or "").upper()
        if und and und != self.und:
            return
        if self.S <= 0:
            return

        strike = tick.get("strike") or tick.get("k")
        expiry_ms = tick.get("expiry_ts")
        right = (tick.get("right") or tick.get("cp") or "").upper()  # "C" / "P"
        if not strike or not expiry_ms or right not in ("C", "P"):
            return
        try:
            K = float(strike)
            exp_ms = int(expiry_ms)
        except Exception:
            return

        tau = self._years_until(exp_ms, now)
        if tau < self.cfg.min_tau_yr or tau > self.cfg.max_tau_yr:
            return

        mid = self._mid_from_tick(tick)
        if mid is None or mid <= 0:
            return

        iv = tick.get("iv") or tick.get("sigma")
        if iv is None:
            # implied vol from per-unit price (divide by lot multiplier)
            iv = _bisection_implied_vol(
                option_mid=mid / self.cfg.lot_mult,
                S=self.S, K=K, tau=tau, is_call=(right == "C")
            )
        try:
            iv = float(iv) if iv is not None else None
        except Exception:
            iv = None
        if iv is None or iv <= 0:
            return

        # vega (per contract)
        vega_per_unit = _bs_vega(self.S, K, tau, iv)  # per 1 underlying unit
        vega_contract = vega_per_unit * self.cfg.lot_mult

        # store point
        chain = self.chains[exp_ms]
        chain["points"][sym] = {
            "symbol": sym,
            "right": right,
            "strike": K,
            "tau": tau,
            "iv": iv,
            "mid": mid,
            "vega": vega_contract,
            "ok": True
        }
        self._recent_by_exp[exp_ms].append(sym)

        # 3) Fit / residual for this expiry
        coeffs = chain.get("fit")
        if coeffs is None:
            coeffs = self._fit_smile_for_expiry(exp_ms, now)
            if coeffs is None:
                return

        # residual for this option
        x = math.log(max(1e-9, K / self.S))
        iv_fit = self._poly_eval(coeffs, x)
        resid = iv - iv_fit

        # EWMA of residuals for z-score
        mu = float(chain["resid_mu"])
        var = float(chain["resid_var"])
        a = 0.1
        mu = (1 - a) * mu + a * resid
        var = (1 - a) * var + a * (resid - mu) ** 2
        var = max(var, 1e-8)
        z = (resid - mu) / math.sqrt(var)
        chain["resid_mu"] = mu
        chain["resid_var"] = var

        # Publish a simple signal: sign of average residual surprise
        sig = max(-1.0, min(1.0, z / 3.0))
        self.emit_signal(sig)

        # 4) Trade logic (pair rich vs cheap inside same expiry)
        if now - self._last_trade_ms < self.cfg.cooldown_ms:
            return
        if abs(z) < self.cfg.enter_z:
            return

        # Identify counterpart with opposite residual sign and near strike
        # Compute residuals for recent pool
        recent_syms = list(self._recent_by_exp[exp_ms])
        candidates: List[Tuple[str, float, float, float]] = []  # (sym, resid, K, vega)
        for s2 in recent_syms:
            p = chain["points"].get(s2)
            if not p or not p.get("ok"):
                continue
            K2 = p["strike"]; iv2 = p["iv"]
            x2 = math.log(max(1e-9, K2 / self.S))
            r2 = iv2 - self._poly_eval(coeffs, x2)
            candidates.append((s2, r2, K2, p["vega"]))

        if not candidates:
            return

        # pick the best opposite-sign with strike proximity
        def bps(k_a: float, k_b: float) -> float:
            return abs(k_a - k_b) / max(1e-9, self.S) * 1e4

        target_sign = -1.0 if resid > 0 else +1.0   # if current is rich(+), pair with cheap(-)
        best_pair = None
        best_score = 1e9
        for s2, r2, K2, v2 in candidates:
            if math.copysign(1.0, r2) != target_sign:
                continue
            dist = bps(K, K2)
            if dist < self.cfg.min_strike_dist_bps or dist > self.cfg.max_strike_dist_bps:
                continue
            # smaller |dist| and larger |resid| contrast → better
            score = dist / max(1e-6, abs(r2)) 
            if score < best_score:
                best_score = score
                best_pair = (s2, r2, K2, v2)

        if not best_pair:
            return

        sym2, resid2, K2, vega2 = best_pair
        p1 = chain["points"][sym]
        p2 = chain["points"][sym2]

        # Build vega-balanced vertical: long cheap, short rich
        long_sym, short_sym = (sym, sym2) if resid < 0 else (sym2, sym)
        long_p, short_p = (p1, p2) if resid < 0 else (p2, p1)

        v1 = long_p["vega"]
        v2 = -short_p["vega"]  # short vega = -vega
        denom = abs(v1) + abs(v2)
        v_ratio_ok = True
        if self.cfg.target_vega_neutral and denom > 1e-9:
            mismatch = abs(v1 + v2) / denom
            v_ratio_ok = (mismatch <= self.cfg.max_vega_ratio_mismatch)

        # size by premium cap and vega heuristic
        # price per contract = mid; gross premium ~ mid_long*qty - mid_short*qty2 (credit/debit ignored here)
        # simple same-qty spread unless vega neutralization used
        qty_long = 1.0
        qty_short = 1.0
        if self.cfg.target_vega_neutral and v_ratio_ok and abs(v1) > 1e-6 and abs(v2) > 1e-6:
            # choose integer-ish ratio
            ratio = abs(v2 / v1)
            qty_long = max(1.0, round(ratio))
            qty_short = float(qty_long)  # 1:1 after rounding because both multiplied
        # premium cap
        gross = abs(long_p["mid"] * qty_long * self.cfg.lot_mult) + abs(short_p["mid"] * qty_short * self.cfg.lot_mult)
        scale = min(1.0, self.cfg.max_gross_premium / max(1.0, gross))
        qty_long *= scale
        qty_short *= scale
        # floor to >= 1 contract where possible
        qty_long = max(1.0, round(qty_long))
        qty_short = max(1.0, round(qty_short))

        # Prices with tiny improvement inside spread (bps)
        bps_px = self.cfg.slippage_bps / 1e4
        long_px = long_p["mid"] * (1.0 + bps_px)   # paying a touch more to get filled
        short_px = short_p["mid"] * (1.0 - bps_px) # receiving a touch less

        # Fire the legs (venue-agnostic; OMS handles route)
        self.order(long_sym,  "buy",  qty=qty_long,  order_type="limit", limit_price=long_px,
                   extra={"reason": "vol_smile_arb_long_cheap",  "exp": exp_ms, "resid_z": z, "pair": short_sym})
        self.order(short_sym, "sell", qty=qty_short, order_type="limit", limit_price=short_px,
                   extra={"reason": "vol_smile_arb_short_rich", "exp": exp_ms, "resid_z": z, "pair": long_sym})

        self._last_trade_ms = now


# ------------------- optional runner -------------------

if __name__ == "__main__":
    """
    Hook up in your orchestrator; example:
      strat = VolSmileArb()
      # strat.run(stream="ticks.options.us")  # wherever option ticks arrive
    """
    strat = VolSmileArb()
    # strat.run(stream="ticks.options.us")