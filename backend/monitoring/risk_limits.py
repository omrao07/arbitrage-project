# risk/risk_limits.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# Models
# ======================================================================================

@dataclass
class LimitsConfig:
    # Notional caps (base USD)
    max_gross_usd: float = 25_000_000.0
    max_net_usd: float   = 10_000_000.0
    max_name_usd: float  = 5_000_000.0
    # Flow caps
    max_daily_turnover_usd: float = 10_000_000.0
    max_ticket_usd: float = 5_000_000.0
    min_ticket_usd: float = 5_000.0
    # Concentration
    max_weight_per_name: float = 0.25        # as fraction of gross
    max_names_per_bucket: int = 20           # optional (sector/issuer)
    # Liquidity
    max_participation_rate: float = 0.15     # order notional / (ADV * px) (equities) or / DV01 bucket (credit)
    min_adv_usd: float = 1_000_000.0
    # Shorting & sides
    allow_short: bool = True
    # Greeks caps (for options/credit if available)
    max_gamma: float = 0.0                    # <= 0 means ignore
    max_vega: float  = 0.0
    max_theta: float = 0.0
    # Trading windows
    trade_start_hhmm: int = 930               # local exchange time; set None to ignore
    trade_end_hhmm: int   = 1555
    # VaR & Stress (portfolio level)
    var_horizon_days: int = 1
    var_conf: float = 0.99
    var_limit_usd: float = 500_000.0
    # Kill switch: cumulative daily loss
    max_daily_loss_usd: float = 750_000.0
    # Cooldown per base symbol
    cooldown_secs: int = 15
    # Scaling behavior when over budget
    enable_proportional_scaling: bool = True


@dataclass
class RiskState:
    ts_epoch: int = 0
    # signed notionals per base symbol
    positions: Dict[str, float] = field(default_factory=dict)
    # rolling counters
    traded_today_usd: float = 0.0
    daily_pnl_usd: float = 0.0
    # last sent per base (for cooldown)
    last_action_ts: Dict[str, int] = field(default_factory=dict)
    # time series for VaR (portfolio returns or PnL/Notional)
    ret_history: List[float] = field(default_factory=list)


# ======================================================================================
# Engine
# ======================================================================================

class RiskLimits:
    """
    Core pre-trade and on-book limits.
    Expects orders DataFrame with columns:
      ['ticker','trade_notional','side'] and optional
      ['adv_usd','bucket','gamma','vega','theta','px_hint','client_order_id']
    Positions tracked in signed USD notional at base-symbol granularity.
    """

    def __init__(self, cfg: LimitsConfig = LimitsConfig()):
        self.cfg = cfg
        self.state = RiskState()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        return {"cfg": asdict(self.cfg), "state": asdict(self.state)}

    def pretrade_check(self, orders: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Returns dict with DataFrames:
          - approved
          - rejected (with 'reason')
          - scaled (if proportional scaling applied)
        """
        if orders is None or len(orders) == 0:
            return {"approved": _empty(), "rejected": _empty(), "scaled": _empty()}
        df = _normalize_orders(orders)

        # 0) Trading window
        df, rej_time = self._filter_trading_window(df)

        # 1) Ticket size & shorting rules
        df, rej_ticket = self._filter_ticket_and_short(df)

        # 2) Cooldown
        df, rej_cd = self._filter_cooldown(df)

        # 3) Per-name cap
        df, rej_name = self._filter_per_name(df)

        # 4) Liquidity (ADV participation)
        df, rej_liq = self._filter_liquidity(df)

        # 5) Greeks caps (optional)
        df, rej_greeks = self._filter_greeks(df)

        # 6) Flow caps (daily turnover)
        df, scaled = self._apply_turnover_cap(df)

        # 7) Portfolio caps (gross/net); proportional scaling if enabled
        df, scaled2, rej_port = self._apply_portfolio_caps(df)

        # Assemble
        rejected = pd.concat([x for x in [rej_time, rej_ticket, rej_cd, rej_name, rej_liq, rej_greeks, rej_port] if len(x)], ignore_index=True) if any([len(x) for x in [rej_time, rej_ticket, rej_cd, rej_name, rej_liq, rej_greeks, rej_port]]) else _empty()
        scaled_all = pd.concat([x for x in [scaled, scaled2] if len(x)], ignore_index=True) if any([len(x) for x in [scaled, scaled2]]) else _empty()

        return {"approved": df.reset_index(drop=True), "rejected": rejected.reset_index(drop=True), "scaled": scaled_all.reset_index(drop=True)}

    def posttrade_apply_fills(self, fills: pd.DataFrame) -> None:
        """
        Update positions & turnover with executed fills.
        Expects columns: ['ticker','side','filled_usd','ts']
        """
        if fills is None or len(fills) == 0:
            return
        df = fills.copy()
        req = {"ticker","side","filled_usd"}
        if not req.issubset(df.columns):
            raise ValueError(f"fills must include {req}")
        for _, r in df.iterrows():
            signed = float(r["filled_usd"]) if "BUY" in str(r["side"]).upper() else -float(r["filled_usd"])
            base = base_symbol(str(r["ticker"]))
            self.state.positions[base] = float(self.state.positions.get(base, 0.0) + signed)
            self.state.traded_today_usd += abs(float(r["filled_usd"]))
            self.state.last_action_ts[base] = int(r.get("ts", time.time()))

    def record_pnl(self, pnl_increment_usd: float) -> None:
        self.state.daily_pnl_usd += float(pnl_increment_usd)

    # ------ Portfolio risk diagnostics ---------------------------------------

    def portfolio_gross(self) -> float:
        return float(sum(abs(v) for v in self.state.positions.values()))

    def portfolio_net(self) -> float:
        return float(abs(sum(v for v in self.state.positions.values())))

    def portfolio_concentration(self) -> pd.Series:
        gross = self.portfolio_gross()
        if gross <= 0:
            return pd.Series(dtype=float)
        return pd.Series({k: abs(v)/gross for k, v in self.state.positions.items()}).sort_values(ascending=False)

    def var_parametric(self, ret_series: Optional[pd.Series] = None, *, horizon_days: Optional[int] = None, alpha: Optional[float] = None, scale_with_gross: bool = True) -> float:
        """
        Simple parametric VaR using portfolio return history (or internal state.ret_history).
        Returns USD VaR (>0).
        """
        alpha = alpha or self.cfg.var_conf
        H = horizon_days or self.cfg.var_horizon_days
        if ret_series is None:
            ret = pd.Series(self.state.ret_history, dtype=float)
        else:
            ret = pd.Series(ret_series, dtype=float)
        if len(ret) < 50:
            return 0.0
        mu = float(ret.mean())
        sd = float(ret.std(ddof=1))
        z = _z_for(alpha)
        var_ret = -(mu * H + z * sd * math.sqrt(H))  # negative return quantile → positive VaR
        base = self.portfolio_gross() if scale_with_gross else 1.0
        return max(0.0, var_ret * base)

    def stress_pnl(self, shocks: Dict[str, float]) -> float:
        """
        shocks: dict base_symbol -> shock fraction (e.g., {'IG_A': -0.05} for -5% adverse)
        Returns USD PnL under shocks (negative = loss).
        """
        pnl = 0.0
        for name, shock in shocks.items():
            pos = float(self.state.positions.get(name, 0.0))
            pnl += pos * float(shock)
        return float(pnl)

    def breached(self) -> List[str]:
        """
        Check soft/hard breaches on the *book now* (not pretrade).
        """
        breaches = []
        if self.portfolio_gross() > self.cfg.max_gross_usd + 1e-6:
            breaches.append("MAX_GROSS")
        if self.portfolio_net()   > self.cfg.max_net_usd + 1e-6:
            breaches.append("MAX_NET")
        if self.state.daily_pnl_usd <= -abs(self.cfg.max_daily_loss_usd):
            breaches.append("DAILY_LOSS_LIMIT")
        # concentration
        conc = self.portfolio_concentration()
        if not conc.empty and conc.iloc[0] > self.cfg.max_weight_per_name + 1e-9:
            breaches.append("CONCENTRATION")
        # VaR
        v = self.var_parametric()
        if v > self.cfg.var_limit_usd + 1e-6:
            breaches.append("VAR")
        return breaches

    # ==================================================================================
    # Internals: pretrade filters
    # ==================================================================================

    def _filter_trading_window(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.cfg.trade_start_hhmm or not self.cfg.trade_end_hhmm:
            return df, _empty()
        # If orders have ts, use; else allow (window checked at caller)
        if "ts" not in df.columns:
            return df, _empty()
        hhmm = df["ts"].apply(lambda t: _to_hhmm(int(t)))
        m = (hhmm >= int(self.cfg.trade_start_hhmm)) & (hhmm <= int(self.cfg.trade_end_hhmm))
        rej = df.loc[~m].copy()
        if len(rej): rej["reason"] = "OUT_OF_WINDOW"
        return df.loc[m].copy(), rej

    def _filter_ticket_and_short(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        m1 = df["trade_notional"].abs() >= float(self.cfg.min_ticket_usd)
        m2 = df["trade_notional"].abs() <= float(self.cfg.max_ticket_usd)
        ok = m1 & m2
        rej = df.loc[~ok].copy()
        if len(rej):
            rej.loc[~m1, "reason"] = "BELOW_MIN_TICKET"
            rej.loc[~m2, "reason"] = "ABOVE_MAX_TICKET"
        if not self.cfg.allow_short:
            # reject SELL sides
            m3 = df["side"].str.upper().str.contains("BUY")
            rej2 = df.loc[~m3].copy(); 
            if len(rej2): rej2["reason"] = "SHORTING_DISABLED"
            return df.loc[ok & m3].copy(), pd.concat([rej, rej2], ignore_index=True) if len(rej) or len(rej2) else _empty()
        return df.loc[ok].copy(), rej

    def _filter_cooldown(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.cfg.cooldown_secs <= 0:
            return df, _empty()
        now = int(time.time())
        def ok(row) -> bool:
            base = base_symbol(row["ticker"])
            last = int(self.state.last_action_ts.get(base, 0))
            return (now - last) >= int(self.cfg.cooldown_secs)
        mask = df.apply(ok, axis=1)
        rej = df.loc[~mask].copy()
        if len(rej): rej["reason"] = "COOLDOWN"
        return df.loc[mask].copy(), rej

    def _filter_per_name(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def within_cap(row) -> bool:
            base = base_symbol(row["ticker"])
            cur = float(self.state.positions.get(base, 0.0))
            signed = float(row["trade_notional"]) if "BUY" in row["side"].upper() else -float(row["trade_notional"])
            return abs(cur + signed) <= (self.cfg.max_name_usd + 1e-6)
        mask = df.apply(within_cap, axis=1)
        rej = df.loc[~mask].copy()
        if len(rej): rej["reason"] = "PER_NAME_CAP"
        return df.loc[mask].copy(), rej

    def _filter_liquidity(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "adv_usd" not in df.columns:
            return df, _empty()
        adv = df["adv_usd"].fillna(0.0)
        prt = df["trade_notional"].abs() / adv.replace(0.0, np.nan)
        # enforce min ADV and max participation
        ok = (adv >= float(self.cfg.min_adv_usd)) & (prt <= float(self.cfg.max_participation_rate))
        rej = df.loc[~ok].copy()
        if len(rej):
            rej.loc[adv < self.cfg.min_adv_usd, "reason"] = "ADV_TOO_LOW"
            rej.loc[prt > self.cfg.max_participation_rate, "reason"] = "PARTICIPATION_HIGH"
        return df.loc[ok].copy(), rej

    def _filter_greeks(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if all(x <= 0 for x in [self.cfg.max_gamma, self.cfg.max_vega, self.cfg.max_theta]):
            return df, _empty()
        # sum proposed + current; if over cap, reject row
        # we expect per-order greeks columns already scaled to USD or per-1% (consistent units)
        gam = df.get("gamma", pd.Series(0.0, index=df.index)).astype(float)
        veg = df.get("vega",  pd.Series(0.0, index=df.index)).astype(float)
        the = df.get("theta", pd.Series(0.0, index=df.index)).astype(float)

        def ok(idx) -> bool:
            g_ok = (self.cfg.max_gamma <= 0) or (abs(gam.loc[idx]) <= self.cfg.max_gamma)
            v_ok = (self.cfg.max_vega  <= 0) or (abs(veg.loc[idx]) <= self.cfg.max_vega)
            t_ok = (self.cfg.max_theta <= 0) or (abs(the.loc[idx]) <= self.cfg.max_theta)
            return g_ok and v_ok and t_ok

        mask = df.index.to_series().apply(ok)
        rej = df.loc[~mask].copy()
        if len(rej): rej["reason"] = "GREEKS_CAP"
        return df.loc[mask].copy(), rej

    def _apply_turnover_cap(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df, _empty()
        budget = max(0.0, float(self.cfg.max_daily_turnover_usd) - float(self.state.traded_today_usd))
        want = float(df["trade_notional"].abs().sum())
        if want <= budget or budget <= 0.0:
            if budget <= 0.0:
                rej = df.copy(); rej["reason"] = "DAILY_TURNOVER_CAP"
                return df.iloc[0:0], rej
            return df, _empty()
        # scale down proportionally
        scale = budget / max(want, 1e-9)
        scaled = df.copy()
        scaled["scaled_from"] = scaled["trade_notional"]
        scaled["trade_notional"] = scaled["trade_notional"] * scale
        scaled["reason"] = f"SCALED_TURNOVER({scale:.2%})"
        return scaled, scaled

    def _apply_portfolio_caps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Check gross/net if we executed the batch. Scale if enabled, else reject overflow.
        """
        if df.empty:
            return df, _empty(), _empty()
        # Simulate resulting positions
        pos = self.state.positions.copy()
        for _, r in df.iterrows():
            signed = float(r["trade_notional"]) if "BUY" in r["side"].upper() else -float(r["trade_notional"])
            base = base_symbol(str(r["ticker"]))
            pos[base] = float(pos.get(base, 0.0) + signed)

        gross = sum(abs(v) for v in pos.values())
        net   = abs(sum(v for v in pos.values()))
        over_gross = gross > self.cfg.max_gross_usd + 1e-6
        over_net   = net   > self.cfg.max_net_usd   + 1e-6
        if not over_gross and not over_net:
            return df, _empty(), _empty()

        if not self.cfg.enable_proportional_scaling:
            rej = df.copy()
            rej["reason"] = "PORTFOLIO_CAP"
            return df.iloc[0:0], _empty(), rej

        # Compute scale factor so new gross equals cap (the tighter of gross/net)
        cur_gross = sum(abs(v) for v in self.state.positions.values())
        cur_net   = abs(sum(v for v in self.state.positions.values()))
        add_gross = gross - cur_gross
        add_net   = net   - cur_net
        # If add_* <= 0 (rare), just reject
        if add_gross <= 0 and add_net <= 0:
            rej = df.copy(); rej["reason"] = "PORTFOLIO_CAP"
            return df.iloc[0:0], _empty(), rej

        scale_gross = (self.cfg.max_gross_usd - cur_gross) / max(add_gross, 1e-9) if add_gross > 0 else 1.0
        scale_net   = (self.cfg.max_net_usd   - cur_net)   / max(add_net,   1e-9) if add_net   > 0 else 1.0
        scale = max(0.0, min(1.0, scale_gross, scale_net))

        if scale <= 0:
            rej = df.copy(); rej["reason"] = "PORTFOLIO_CAP"
            return df.iloc[0:0], _empty(), rej

        scaled = df.copy()
        scaled["scaled_from"] = scaled["trade_notional"]
        scaled["trade_notional"] = scaled["trade_notional"] * scale
        scaled["reason"] = f"SCALED_PORTFOLIO({scale:.2%})"
        return scaled, scaled, _empty()


# ======================================================================================
# Helpers
# ======================================================================================

def base_symbol(ticker: str) -> str:
    parts = str(ticker).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]

def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker","trade_notional","side","reason"])

def _normalize_orders(df: pd.DataFrame) -> pd.DataFrame:
    req = {"ticker","trade_notional","side"}
    if not req.issubset(df.columns):
        raise ValueError(f"orders must include {req}")
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    out["trade_notional"] = out["trade_notional"].astype(float)
    out["side"] = out["side"].astype(str)
    return out

def _to_hhmm(ts: int) -> int:
    t = time.gmtime(int(ts))
    return t.tm_hour * 100 + t.tm_min

def _z_for(conf: float) -> float:
    # approximate inverse CDF for standard normal for common confs
    # 0.95→1.645, 0.975→1.96, 0.99→2.326
    if conf >= 0.999: return 3.09
    if conf >= 0.99:  return 2.326
    if conf >= 0.975: return 1.960
    if conf >= 0.95:  return 1.645
    return 1.282


# ======================================================================================
# Example (smoke)
# ======================================================================================

if __name__ == "__main__":
    # Current book
    rl = RiskLimits(LimitsConfig(max_gross_usd=5_000_000, max_net_usd=2_000_000, max_name_usd=1_500_000,
                                 max_daily_turnover_usd=2_000_000, cooldown_secs=5,
                                 max_participation_rate=0.2, min_adv_usd=500_000))
    # Simulated positions
    rl.state.positions = {"IG_A": 800_000.0, "HY_B": -400_000.0}
    rl.state.traded_today_usd = 900_000.0
    rl.state.daily_pnl_usd = -120_000.0
    rl.state.ret_history = list(np.random.normal(0, 0.01, 250))

    batch = pd.DataFrame([
        {"ticker":"IG_A_5Y","trade_notional":600_000,"side":"BUY_PROTECTION","adv_usd":5_000_000},
        {"ticker":"HY_B_5Y","trade_notional":900_000,"side":"SELL_PROTECTION","adv_usd":2_000_000},
        {"ticker":"XYZ_C_5Y","trade_notional":50_000,"side":"SELL_PROTECTION","adv_usd":100_000},  # low ADV
        {"ticker":"ABC_D_5Y","trade_notional":25_000,"side":"BUY_PROTECTION","adv_usd":800_000},
    ])

    out = rl.pretrade_check(batch)
    print("\nAPPROVED\n", out["approved"])
    print("\nSCALED\n", out["scaled"])
    print("\nREJECTED\n", out["rejected"])
    print("\nBREACHES NOW:", rl.breached())