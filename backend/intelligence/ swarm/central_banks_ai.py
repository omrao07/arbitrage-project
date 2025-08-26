# backend/macro/central_bank_ai.py
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------
# Soft imports to keep this file standalone-friendly
# ----------------------------------------------------------------------
try:
    from backend.macro.economy import EconEvent, EconSeries # type: ignore
except Exception:
    @dataclass
    class EconEvent:
        code: str; name: str; ts: datetime
        actual: Optional[float] = None; consensus: Optional[float] = None
        previous: Optional[float] = None; region: str = ""; unit: str = ""; meta: Dict[str, Any] = field(default_factory=dict)
        def surprise(self) -> Optional[float]:
            if self.actual is None or self.consensus is None: return None
            return self.actual - self.consensus
    class EconSeries: ...  # placeholder

try:
    from backend.treasury.soverign_adapter import YieldCurve, CurvePoint # type: ignore
except Exception:
    @dataclass
    class CurvePoint: tenor_yrs: float; yld: float
    @dataclass
    class YieldCurve:
        curve_date: date; currency: str; points: List[CurvePoint]
        def sorted_points(self): return sorted(self.points, key=lambda p: p.tenor_yrs)

try:
    from backend.risk.policy_sim import RateShock # type: ignore
except Exception:
    @dataclass
    class RateShock:
        parallel_bp: float = 0.0
        steepen_bp: float = 0.0
        butterfly_bp: float = 0.0
        twist_pivot_yrs: float = 5.0
        custom_tenor_bp: Dict[float, float] = field(default_factory=dict)

# ----------------------------------------------------------------------
# Core models
# ----------------------------------------------------------------------

@dataclass
class PolicyState:
    """Minimal central bank state (nominal rates expressed as decimals, e.g., 0.065 = 6.5%)."""
    bank: str = "CB"
    currency: str = "USD"
    policy_rate: float = 0.05                 # current target short rate (policy)
    r_star: float = 0.006                     # neutral real rate
    infl_target: float = 0.02                 # inflation target (YoY, decimal)
    inflation_yoy: float = 0.03               # last YoY inflation
    output_gap: float = 0.0                   # % gap (decimal, + = overheating)
    unemployment_gap: float = 0.0             # actual - NAIRU (decimal)
    balance_sheet: float = 0.0                # notional size (for QE/QT signaling)
    hawkish_bias: float = 0.0                 # -1..+1 aggregate bias from comms & data
    term_premium_bps_10y: float = 60.0        # approximate 10y term premium in bps
    last_decision_bps: float = 0.0
    last_meeting: Optional[date] = None

    def taylor_rule_target(self, phi_pi: float = 1.5, phi_y: float = 0.5) -> float:
        """Taylor-style nominal target rate."""
        pi_gap = self.inflation_yoy - self.infl_target
        return max(0.0, self.r_star + self.inflation_yoy + phi_pi * pi_gap + phi_y * self.output_gap)

@dataclass
class RateDecision:
    asof: datetime
    bank: str
    currency: str
    delta_bps: float
    new_policy_rate: float
    probs: Dict[str, float]             # {"hike":p, "hold":p, "cut":p}
    rationale: Dict[str, Any]           # components used
    guidance_bps_3m: float              # forward guidance (rough)

# ----------------------------------------------------------------------
# NLP-lite: hawkish/dovish score from statement text
# ----------------------------------------------------------------------

# Simple lexicon (expand as needed). Values are in "bps of bias" contributions.
_LEXICON = {
    # hawkish
    "elevated": +5, "persistent": +6, "upside risks": +9, "strong": +4, "robust": +4, "tight": +5,
    "tightening": +6, "restrictive": +7, "above target": +7, "resilient": +3,
    # dovish
    "moderating": -5, "slowing": -5, "softening": -6, "downside risks": -8, "disinflation": -7,
    "accommodative": -8, "appropriate to cut": -12, "below target": -6,
}

def text_bias_bps(text: str) -> float:
    """
    Crude, fast bag-of-phrases scorer.
    Returns a signed number of "bps of bias" (positive = hawkish).
    """
    if not text:
        return 0.0
    t = " " + " ".join(text.lower().split()) + " "
    score = 0.0
    for phrase, w in _LEXICON.items():
        if f" {phrase} " in t:
            score += w
    # punctuation emphasis
    score += 2.0 * t.count("!") + 1.0 * t.count(";")
    # clamp to a sensible band
    return max(-25.0, min(25.0, score))

# ----------------------------------------------------------------------
# Surprise plumbing
# ----------------------------------------------------------------------

def standardized_surprise(ev: EconEvent, sigma: float = 0.2) -> float:
    """
    Scale surprise by an assumed std dev (fallback if we don't have historical).
    Positive surprise for inflation/earnings means hotter; for unemployment we invert.
    """
    if ev is None or ev.consensus is None or ev.actual is None:
        return 0.0
    s = (ev.actual - ev.consensus) / max(1e-9, sigma)
    # crude sign rules by code
    code = (ev.code or ev.name or "").upper()
    if any(k in code for k in ("UNEMP", "JOBLESS", "URATE")):
        s = -s
    return float(max(-5.0, min(5.0, s)))

# ----------------------------------------------------------------------
# Decision engine
# ----------------------------------------------------------------------

def decide_next_meeting(
    state: PolicyState,
    *,
    statement_text: str = "",
    events: Sequence[EconEvent] = (),
    weights: Dict[str, float] = None, # type: ignore
    hike_cut_step_bps: float = 25.0,
    asof: Optional[datetime] = None,
) -> RateDecision:
    """
    Blend Taylor target, comms bias, and macro surprises into a discrete decision.
    """
    asof = asof or datetime.utcnow()
    W = {"taylor": 0.6, "text": 0.25, "events": 0.15}
    if weights: W.update(weights)

    taylor = state.taylor_rule_target()
    gap_nominal = (taylor - state.policy_rate) * 1e4  # in bps
    text_bps = text_bias_bps(statement_text)

    # Aggregate standardized surprises
    ev_contrib = 0.0
    details = []
    for ev in events or []:
        s = standardized_surprise(ev)
        ev_contrib += s * 4.0  # 1σ surprise ≈ 4 bps bias
        details.append({"code": ev.code, "surprise_sigma": s, "bias_bps": s*4.0})

    # Combined "signal" in bps
    signal_bps = W["taylor"] * gap_nominal + W["text"] * text_bps + W["events"] * ev_contrib

    # Convert to probabilities (softmax over {-1,0,+1} steps)
    # We map signal (bps) → logits via a temperature.
    temp = 20.0  # higher = smoother
    logits = {
        "cut": max(-10.0, min(10.0, -signal_bps / temp)),
        "hold": -abs(signal_bps) / (temp * 2.0),
        "hike": max(-10.0, min(10.0, +signal_bps / temp)),
    }
    exps = {k: math.exp(v) for k, v in logits.items()}
    Z = sum(exps.values()) or 1.0
    probs = {k: float(exps[k] / Z) for k in ("hike", "hold", "cut")}

    # Pick the modal move in ±hike_cut_step_bps
    mode = max(probs.items(), key=lambda kv: kv[1])[0]
    delta = 0.0
    if mode == "hike": delta = +hike_cut_step_bps
    elif mode == "cut": delta = -hike_cut_step_bps

    new_rate = max(0.0, state.policy_rate + (delta / 1e4))
    guidance = 0.4 * signal_bps  # "verbal forward guidance" bps into ~3m horizon

    rationale = {
        "taylor_rate": taylor,
        "gap_nominal_bps": gap_nominal,
        "text_bias_bps": text_bps,
        "event_components": details,
        "signal_bps": signal_bps,
        "weights": W,
    }

    return RateDecision(
        asof=asof, bank=state.bank, currency=state.currency,
        delta_bps=delta, new_policy_rate=new_rate, probs=probs,
        rationale=rationale, guidance_bps_3m=guidance
    )

# ----------------------------------------------------------------------
# Path simulator & curve mapping
# ----------------------------------------------------------------------

def simulate_policy_path(
    state: PolicyState,
    *,
    months: int = 24,
    shock_sigma_bps: float = 10.0,
    mean_revert: float = 0.25,
    guidance_bps_3m: float = 0.0,
    seed: Optional[int] = None,
) -> List[Tuple[date, float]]:
    """
    Simulate a monthly short-rate path (nominal), mean-reverting to Taylor target,
    incorporating a short-lived forward guidance impulse.
    Returns list of (month_end_date, rate_decimal).
    """
    if seed is not None:
        random.seed(seed)
    path: List[Tuple[date, float]] = []
    r = float(state.policy_rate)
    taylor = state.taylor_rule_target()

    # translate guidance into an initial bias that decays over ~3 months
    g0 = guidance_bps_3m / 1e4
    for k in range(months):
        dt = _month_end(datetime.utcnow().date(), k)
        guide = g0 * max(0.0, 1.0 - k / 3.0)
        # AR(1) towards taylor + guide, with shock
        target = taylor + guide
        r = r + mean_revert * (target - r) + (shock_sigma_bps / 1e4) * random.gauss(0.0, 1.0)
        r = max(0.0, r)
        path.append((dt, r))
    return path

def path_to_yield_curve(
    *,
    asof: date,
    currency: str,
    short_path: List[Tuple[date, float]],
    tenors_yrs: Sequence[float] = (0.25, 0.5, 1, 2, 3, 5, 7, 10),
    term_premium_bps_10y: float = 60.0,
) -> YieldCurve:
    """
    Expectations-hypothesis style: yield(T) ≈ average expected short rate over T + term premium tilt.
    """
    pts: List[CurvePoint] = []
    # build a simple monthly grid of forward short rates
    grid = [(d, r) for (d, r) in short_path]
    if not grid:
        grid = [(asof, 0.0)]
    # map term premium as a convex shape peaking around 10Y
    def term_prem(tenor):
        return (term_premium_bps_10y * min(1.0, tenor / 10.0)) / 1e4
    for T in tenors_yrs:
        months = max(1, int(T * 12))
        rates = [grid[min(i, len(grid)-1)][1] for i in range(months)]
        avg = sum(rates) / len(rates)
        y = max(0.0, avg + term_prem(T))
        pts.append(CurvePoint(T, y))
    return YieldCurve(curve_date=asof, currency=currency.upper(), points=pts)

def curve_vs_base_to_shock(base: YieldCurve, proposed: YieldCurve) -> RateShock:
    """
    Create a RateShock with custom tenor bumps = (proposed - base) in bps.
    """
    d: Dict[float, float] = {}
    base_map = {round(p.tenor_yrs, 6): p.yld for p in base.sorted_points()}
    for p in proposed.sorted_points():
        t = round(p.tenor_yrs, 6)
        if t in base_map:
            d[t] = (p.yld - base_map[t]) * 1e4
    return RateShock(custom_tenor_bp=d, twist_pivot_yrs=5.0)

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def update_state_from_events(state: PolicyState, events: Sequence[EconEvent]) -> PolicyState:
    """
    Adjust inflation/output/unemployment gaps from recent releases (very simple rules).
    """
    pi_adj = 0.0; y_adj = 0.0; u_adj = 0.0
    for ev in events or []:
        s = standardized_surprise(ev)
        code = (ev.code or ev.name or "").upper()
        if any(k in code for k in ("CPI", "INFL", "PCE")):
            pi_adj += 0.001 * s  # 0.1ppt per 1σ surprise
        if any(k in code for k in ("PMI", "IIP", "GDP", "NFP", "PAYROLL")):
            y_adj += 0.001 * s
        if any(k in code for k in ("UNEMP", "URATE", "JOBLESS")):
            u_adj += -0.001 * s
    state.inflation_yoy = max(0.0, state.inflation_yoy + pi_adj)
    state.output_gap = state.output_gap + y_adj
    state.unemployment_gap = state.unemployment_gap + u_adj
    return state

def _month_end(start: date, add_months: int) -> date:
    y = start.year; m = start.month + add_months
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    # last day of month
    if m == 12: nxt = date(y+1, 1, 1)
    else: nxt = date(y, m+1, 1)
    return nxt - timedelta(days=1)

# ----------------------------------------------------------------------
# JSON I/O
# ----------------------------------------------------------------------

def decision_to_json(d: RateDecision) -> str:
    obj = asdict(d); obj["asof"] = d.asof.isoformat()
    return json.dumps(obj, indent=2)

def state_to_json(s: PolicyState) -> str:
    obj = asdict(s)
    if isinstance(s.last_meeting, date):
        obj["last_meeting"] = s.last_meeting.isoformat()
    return json.dumps(obj, indent=2)

# ----------------------------------------------------------------------
# Tiny demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Example: RBI-like setup (INR), edit as needed
    st = PolicyState(
        bank="RBI", currency="INR", policy_rate=0.065,
        r_star=0.01, infl_target=0.04, inflation_yoy=0.051, output_gap=0.002, unemployment_gap=-0.002,
        term_premium_bps_10y=75.0,
    )
    # Mock events
    today = datetime.utcnow()
    evs = [
        EconEvent(code="IN_CPI_HEADLINE", name="India CPI YoY", ts=today, actual=5.5, consensus=5.2, unit="%"),
        EconEvent(code="IN_IIP", name="India IIP YoY", ts=today, actual=4.0, consensus=3.2, unit="%"),
        EconEvent(code="IN_URATE", name="Unemployment Rate", ts=today, actual=7.5, consensus=7.2, unit="%"),
    ]
    st = update_state_from_events(st, evs)
    statement = "Inflation remains elevated with upside risks; growth is resilient. Further tightening may be warranted."

    dec = decide_next_meeting(st, statement_text=statement, events=evs, hike_cut_step_bps=25.0)
    print("Decision:", decision_to_json(dec))

    # Build a short-rate path w/ guidance and map to curve
    path = simulate_policy_path(st, months=36, guidance_bps_3m=dec.guidance_bps_3m)
    curve = path_to_yield_curve(asof=date.today(), currency=st.currency, short_path=path)
    for p in curve.sorted_points():
        print(f"YC {p.tenor_yrs:>4.1f}y = {p.yld*100:5.2f}%")