# backend/risk/counterparty_exposure.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Literal, Any

Side = Literal["buy", "sell"]
Asset = Literal["equity", "futures", "fx", "rates", "credit", "crypto", "option"]

_now_ms = lambda: int(time.time() * 1000)


# ------------------------------ Models ------------------------------

@dataclass
class CSATerms:
    threshold: float = 0.0          # unsecured threshold (+ you can owe without posting)
    min_transfer: float = 0.0       # MTA
    independent_amount: float = 0.0 # initial margin independent of variation
    haircut_bps: float = 0.0        # generic collateral haircut
    rehypothecation_allowed: bool = True


@dataclass
class Limits:
    ce_max: float = 5_000_000.0     # Current Exposure hard limit
    pfe_95_max: float = 10_000_000.0
    epe_max: float = 6_000_000.0
    single_name_conc_max: float = 0.30  # 30% of gross to one name
    sector_conc_max: float = 0.50
    ww_risk_on: bool = True


@dataclass
class Counterparty:
    cpty_id: str
    name: str
    rating: str = "BBB"
    sector: str = "Broker"
    csa: CSATerms = field(default_factory=CSATerms)
    limits: Limits = field(default_factory=Limits)
    pd_1y: float = 0.01             # 1y default probability (flat, crude)
    lgd: float = 0.6                # loss given default


@dataclass
class Trade:
    trade_id: str
    cpty_id: str
    asset: Asset
    side: Side
    notional: float                 # base currency notionals
    mtm: float = 0.0                # mark-to-market value (+ you are in-the-money)
    vega: float = 0.0               # optional for options
    delta: float = 0.0
    maturity_days: int = 2          # time to maturity (T)
    sigma_ann: float = 0.20         # annualized vol input (or implied vol for options)
    corr_to_cpty: float = 0.1       # wrong-way risk proxy (positive => WWR)
    netting_set: str = "DEFAULT"
    last_update_ms: int = field(default_factory=_now_ms)


@dataclass
class NettingSet:
    cpty_id: str
    code: str
    trades: Dict[str, Trade] = field(default_factory=dict)
    vm_posted: float = 0.0          # variation margin posted to you (+ you hold)
    vm_received: float = 0.0        # variation margin you posted (− your asset)
    im_posted: float = 0.0          # initial margin posted to you
    im_received: float = 0.0        # initial margin you posted


@dataclass
class ExposurePoint:
    ts_ms: int
    ce: float              # current exposure
    ee: float              # expected exposure at this observation horizon (instant proxy)
    pfe_95: float          # potential future exposure (95%)
    pfe_99: float          # potential future exposure (99%)
    epe: float             # running expected positive exposure
    cva: float             # crude CVA estimate for the book


# --------------------------- Engine ----------------------------

class CounterpartyExposureEngine:
    """
    Tracks counterparty credit exposure with netting sets and CSA/margin.
    - Current Exposure (CE) = max( Σ MTM - VM_received + VM_posted - threshold - IM_received + haircuts, 0 )
    - Future Exposure via simple parametric model (lognormal-ish) using sigma and time.
    - EPE integrates EE over a small set of anchor horizons.
    - PFE at 95/99, with WWR tilt via corr_to_cpty.
    - CVA-lite ≈ LGD * ∑ EE(t) * marginal_PD(t) (discrete buckets).
    """

    def __init__(self, base_ccy: str = "USD"):
        self.base_ccy = base_ccy
        self.cptys: Dict[str, Counterparty] = {}
        self.nets: Dict[Tuple[str, str], NettingSet] = {}   # (cpty_id, netting_code) -> set
        # rolling snapshots (keep short tail for dashboard)
        self.history: Dict[str, List[ExposurePoint]] = {}

        # horizons (days) used for EE/EPE/PFE
        self.horizons_d = [1, 5, 10, 20, 60, 120, 252]

        # haircuts per asset (very coarse)
        self.asset_haircut_bps: Dict[Asset, float] = {
            "equity": 300, "futures": 50, "fx": 100, "rates": 80, "credit": 400, "crypto": 1000, "option": 500
        }

    # -------- registry --------
    def upsert_counterparty(self, c: Counterparty) -> None:
        self.cptys[c.cpty_id] = c

    def get_or_create_set(self, cpty_id: str, netting_code: str) -> NettingSet:
        key = (cpty_id, netting_code)
        if key not in self.nets:
            self.nets[key] = NettingSet(cpty_id=cpty_id, code=netting_code)
        return self.nets[key]

    def upsert_trade(self, t: Trade) -> None:
        ns = self.get_or_create_set(t.cpty_id, t.netting_set)
        ns.trades[t.trade_id] = t

    def cancel_trade(self, cpty_id: str, trade_id: str, netting_code: str = "DEFAULT") -> None:
        ns = self.get_or_create_set(cpty_id, netting_code)
        ns.trades.pop(trade_id, None)

    def update_mtm(self, cpty_id: str, trade_id: str, mtm: float,
                   *, vega: Optional[float] = None, delta: Optional[float] = None) -> None:
        for key, ns in self.nets.items():
            if key[0] != cpty_id: 
                continue
            tr = ns.trades.get(trade_id)
            if tr:
                tr.mtm = float(mtm)
                if vega is not None: tr.vega = float(vega)
                if delta is not None: tr.delta = float(delta)
                tr.last_update_ms = _now_ms()
                break

    def post_margin(self, cpty_id: str, netting_code: str, *,
                    vm_posted: float = 0.0, vm_received: float = 0.0,
                    im_posted: float = 0.0, im_received: float = 0.0) -> None:
        ns = self.get_or_create_set(cpty_id, netting_code)
        ns.vm_posted += vm_posted
        ns.vm_received += vm_received
        ns.im_posted += im_posted
        ns.im_received += im_received

    # --------- exposure math (parametric) ---------
    @staticmethod
    def _norm_ppf(p: float) -> float:
        # rational approximation for inverse CDF (Acklam/Beasley-Springer)
        # good enough for 95/99 without numpy/scipy
        if p <= 0.0: return -1e9
        if p >= 1.0: return +1e9
        # coefficients
        a = [ -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
              1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 ]
        b = [ -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
              6.680131188771972e+01, -1.328068155288572e+01 ]
        c = [ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
              -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 ]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
              3.754408661907416e+00 ]
        plow  = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2*math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = math.sqrt(-2*math.log(1-p))
            return -((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

    def _portfolio_mtm(self, ns: NettingSet) -> float:
        return sum(t.mtm for t in ns.trades.values())

    def _haircut(self, ns: NettingSet) -> float:
        # apply conservative haircut based on mix of assets
        if not ns.trades:
            return 0.0
        w = 0.0
        for t in ns.trades.values():
            hb = self.asset_haircut_bps.get(t.asset, 300)
            w += abs(t.mtm) * hb / 1e4
        # normalize by gross |MTM| so haircut is a percent-like adjustment
        gross = sum(abs(t.mtm) for t in ns.trades.values()) or 1.0
        return w / gross * max(gross, 1.0)

    def _wwr_tilt(self, trades: List[Trade], horizon_days: float) -> float:
        # aggregate wrong-way risk multiplier based on exposure-weighted correlation to cpty
        # >0 → increase exposure tails
        if not trades:
            return 0.0
        wsum, num = 0.0, 0.0
        for t in trades:
            w = max(1e-9, abs(t.mtm) + 0.1 * abs(t.notional))
            wsum += w * t.corr_to_cpty
            num += w
        tilt = (wsum / max(num, 1e-9))
        # grow with horizon (diminishing)
        return max(0.0, tilt) * math.sqrt(min(1.0, horizon_days / 252.0))

    def _sigma_slice(self, t: Trade, horizon_days: float) -> float:
        # convert annual sigma to slice sigma with sqrt(time); ensure floor
        return max(1e-6, t.sigma_ann) * math.sqrt(max(1e-6, horizon_days / 252.0))

    # -------- EE / PFE for a netting set (parametric) --------
    def _ee_pfe_for_set(self, ns: NettingSet, csa: CSATerms, horizon_days: float) -> Tuple[float, float]:
        """
        Parametric: future MTM ~ N(m=Σ mtm, s = sqrt(Σ (|notional|*sigma_slice)^2))
        EE  = E[max(X - TH, 0)] via Black'76 call-on-exposure approximation
        PFE = TH + s * z_q (95/99), with WWR tilt.
        """
        trades = list(ns.trades.values())
        if not trades:
            return (0.0, 0.0)

        # aggregate current mark and volatility-of-exposure proxy
        m = sum(t.mtm for t in trades)
        s2 = 0.0
        for t in trades:
            s2 += (abs(t.notional) * self._sigma_slice(t, horizon_days)) ** 2
        s = math.sqrt(max(1e-12, s2))

        # unsecured threshold after VM/IM and CSA threshold
        unsecured = -ns.vm_received + ns.vm_posted - csa.threshold - ns.im_received
        # collateral haircut (very coarse)
        hc = self._haircut(ns) + (csa.haircut_bps / 1e4) * max(0.0, ns.vm_posted + ns.im_posted)
        th = unsecured - hc  # threshold we subtract before taking positive exposure

        # EE (Black'76 style: E[max(N(m,s)-th,0)] = (m-th)*Phi(d) + s*phi(d))
        if s <= 1e-9:
            ee = max(m - th, 0.0)
        else:
            d = (m - th) / s
            phi = math.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
            Phi = 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))
            ee = (m - th) * Phi + s * phi
            ee = max(0.0, ee)

        # PFE 95% (or 99%) with WWR tilt
        tilt = self._wwr_tilt(trades, horizon_days)
        z95 = self._norm_ppf(0.95 + 0.03 * tilt)   # push quantile outward if WWR
        z99 = self._norm_ppf(0.99 + 0.005 * tilt)
        pfe95 = max(0.0, (m - th) + s * z95)
        pfe99 = max(0.0, (m - th) + s * z99)

        return (ee, max(pfe95, pfe99))  # return largest for conservative set-level use

    # -------- CVA-lite over horizons (discrete) --------
    def _cva_for_set(self, ns: NettingSet, c: Counterparty) -> float:
        """
        Very crude CVA: sum over horizons of EE(t) * marginal_PD(t) * LGD
        Assume flat PD over 1y distributed across buckets.
        """
        if not ns.trades:
            return 0.0
        horizons = self.horizons_d
        pd_total = c.pd_1y
        # simple: equal marginal PD per bucket over 1y
        marginal = pd_total / len(horizons)
        lgd = c.lgd
        total = 0.0
        for h in horizons:
            ee, _ = self._ee_pfe_for_set(ns, c.csa, h)
            total += ee * marginal * lgd
        return max(0.0, total)

    # --------- public: compute per counterparty ---------
    def compute(self, cpty_id: str) -> Dict[str, Any]:
        c = self.cptys.get(cpty_id)
        if not c:
            return {"error": f"unknown counterparty {cpty_id}"}

        # gather sets
        sets = [ns for (cid, _), ns in self.nets.items() if cid == cpty_id]
        if not sets:
            snap = {"cpty_id": cpty_id, "name": c.name, "ce": 0.0, "epe": 0.0, "pfe_95": 0.0, "pfe_99": 0.0, "cva": 0.0,
                    "by_set": {}, "flags": []}
            self._store_hist(cpty_id, snap)
            return snap

        # current exposure (instant)
        ce_sets: Dict[str, float] = {}
        total_ce = 0.0
        for ns in sets:
            mtm = self._portfolio_mtm(ns)
            unsecured = -ns.vm_received + ns.vm_posted - c.csa.threshold - ns.im_received
            hc = self._haircut(ns) + (c.csa.haircut_bps / 1e4) * max(0.0, ns.vm_posted + ns.im_posted)
            ce_set = max(0.0, mtm + unsecured - hc)
            ce_sets[ns.code] = ce_set
            total_ce += ce_set

        # EPE + PFEs
        ee_list, pfe95_list, pfe99_list = [], [], []
        for h in self.horizons_d:
            ee_h, _ = 0.0, 0.0
            pfe95_h, pfe99_h = 0.0, 0.0
            for ns in sets:
                ee, pfe_conservative = self._ee_pfe_for_set(ns, c.csa, h)
                # Split conservative PFE into 95/99 proxies using a fixed ratio
                p95 = 0.8 * pfe_conservative
                p99 = 1.0 * pfe_conservative
                ee_h += ee
                pfe95_h += p95
                pfe99_h += p99
            ee_list.append(ee_h)
            pfe95_list.append(pfe95_h)
            pfe99_list.append(pfe99_h)

        epe = sum(ee_list) / max(1, len(ee_list))
        pfe_95 = max(pfe95_list) if pfe95_list else 0.0
        pfe_99 = max(pfe99_list) if pfe99_list else 0.0

        # CVA-lite
        cva = sum(self._cva_for_set(ns, c) for ns in sets)

        # flags / limits
        flags: List[str] = []
        if total_ce > c.limits.ce_max: flags.append(f"CE>{c.limits.ce_max:,.0f}")
        if pfe_95 > c.limits.pfe_95_max: flags.append(f"PFE95>{c.limits.pfe_95_max:,.0f}")
        if epe > c.limits.epe_max: flags.append(f"EPE>{c.limits.epe_max:,.0f}")

        # concentration (single-name)
        gross_by_name: Dict[str, float] = {}
        gross_total = 0.0
        for ns in sets:
            for t in ns.trades.values():
                g = abs(t.notional)
                gross_by_name[t.trade_id] = g
                gross_total += g
        if gross_total > 0:
            top = max(gross_by_name.values()) if gross_by_name else 0.0
            conc = top / gross_total
            if conc > c.limits.single_name_conc_max:
                flags.append(f"Concentration>{c.limits.single_name_conc_max:.0%}")

        snap = {
            "cpty_id": c.cpty_id,
            "name": c.name,
            "rating": c.rating,
            "sector": c.sector,
            "base_ccy": self.base_ccy,
            "ts_ms": _now_ms(),
            "ce": round(total_ce, 2),
            "epe": round(epe, 2),
            "pfe_95": round(pfe_95, 2),
            "pfe_99": round(pfe_99, 2),
            "cva": round(cva, 2),
            "by_set": {ns.code: {
                "ce": round(ce_sets.get(ns.code, 0.0), 2),
                "mtm": round(self._portfolio_mtm(ns), 2),
                "counts": len(ns.trades),
                "vm_posted": ns.vm_posted, "vm_received": ns.vm_received,
                "im_posted": ns.im_posted, "im_received": ns.im_received,
            } for ns in sets},
            "flags": flags,
            "horizons_d": self.horizons_d,
            "ee_path": [round(x, 2) for x in ee_list],
            "pfe95_path": [round(x, 2) for x in pfe95_list],
            "pfe99_path": [round(x, 2) for x in pfe99_list],
        }
        self._store_hist(cpty_id, snap)
        return snap

    def _store_hist(self, cpty_id: str, snap: Dict[str, Any]) -> None:
        arr = self.history.setdefault(cpty_id, [])
        ep = ExposurePoint(
            ts_ms=snap["ts_ms"], ce=snap["ce"], ee=snap["ee_path"][0] if snap.get("ee_path") else snap.get("epe", 0.0),
            pfe_95=snap.get("pfe_95", 0.0), pfe_99=snap.get("pfe_99", 0.0),
            epe=snap.get("epe", 0.0), cva=snap.get("cva", 0.0)
        )
        arr.append(ep)
        if len(arr) > 2000:
            self.history[cpty_id] = arr[-1200:]

    # ------------- stress tests (what-if shocks) -------------
    def stress(self, cpty_id: str, *,
               mtm_shock_pct: float = -0.1,
               vol_mult: float = 1.5,
               vm_call_abs: float = 0.0) -> Dict[str, Any]:
        """
        Apply coarse shocks:
          - mtm_shock_pct: multiply each trade MTM by (1+shock)
          - vol_mult: multiply each trade sigma_ann
          - vm_call_abs: add a variation margin call you must POST (reduces exposure)
        Returns stressed snapshot (does not mutate live state).
        """
        c = self.cptys.get(cpty_id)
        if not c:
            return {"error": f"unknown counterparty {cpty_id}"}
        # clone sets shallowly
        cloned_sets: List[NettingSet] = []
        for (cid, _), ns in self.nets.items():
            if cid != cpty_id: 
                continue
            clone = NettingSet(cpty_id=ns.cpty_id, code=ns.code,
                               vm_posted=ns.vm_posted, vm_received=ns.vm_received,
                               im_posted=ns.im_posted, im_received=ns.im_received)
            for t in ns.trades.values():
                clone.trades[t.trade_id] = Trade(
                    trade_id=t.trade_id, cpty_id=t.cpty_id, asset=t.asset, side=t.side,
                    notional=t.notional, mtm=t.mtm * (1.0 + mtm_shock_pct),
                    vega=t.vega, delta=t.delta, maturity_days=t.maturity_days,
                    sigma_ann=t.sigma_ann * vol_mult, corr_to_cpty=t.corr_to_cpty,
                    netting_set=t.netting_set
                )
            # apply VM call you POST (reduces your asset)
            clone.vm_received += vm_call_abs
            cloned_sets.append(clone)

        # compute metrics on clones
        ce_total, epe, pfe95, pfe99 = 0.0, 0.0, 0.0, 0.0
        for ns in cloned_sets:
            mtm = self._portfolio_mtm(ns)
            hc = self._haircut(ns) + (c.csa.haircut_bps / 1e4) * max(0.0, ns.vm_posted + ns.im_posted)
            unsecured = -ns.vm_received + ns.vm_posted - c.csa.threshold - ns.im_received
            ce_total += max(0.0, mtm + unsecured - hc)
        ee_list, p95_list, p99_list = [], [], []
        for h in self.horizons_d:
            ee_h, p95_h, p99_h = 0.0, 0.0, 0.0
            for ns in cloned_sets:
                ee, pfe_cons = self._ee_pfe_for_set(ns, c.csa, h)
                ee_h += ee
                p95_h += 0.8 * pfe_cons
                p99_h += 1.0 * pfe_cons
            ee_list.append(ee_h); p95_list.append(p95_h); p99_list.append(p99_h)
        epe = sum(ee_list)/len(ee_list) if ee_list else 0.0
        pfe95 = max(p95_list) if p95_list else 0.0
        pfe99 = max(p99_list) if p99_list else 0.0

        return {
            "cpty_id": c.cpty_id,
            "name": c.name,
            "stress": {"mtm_pct": mtm_shock_pct, "vol_mult": vol_mult, "vm_call_abs": vm_call_abs},
            "ce": round(ce_total, 2), "epe": round(epe, 2), "pfe_95": round(pfe95, 2), "pfe_99": round(pfe99, 2),
            "ee_path": [round(x, 2) for x in ee_list],
            "pfe95_path": [round(x, 2) for x in p95_list],
            "pfe99_path": [round(x, 2) for x in p99_list],
        }

    # ------------- exports -------------
    def snapshot(self, cpty_id: str) -> Dict[str, Any]:
        return self.compute(cpty_id)

    def snapshot_all(self) -> Dict[str, Any]:
        return {cid: self.compute(cid) for cid in self.cptys.keys()}