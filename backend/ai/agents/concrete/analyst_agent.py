# backend/ai/agents/concrete/analyst_agent.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------
# Optional imports from your framework (with safe fallbacks)
# ------------------------------------------------------------
try:
    # expected paths based on your structure:
    # backend/ai/agents/core/base_agent.py
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:  # pragma: no cover
    class BaseAgent:  # minimal shim so this file runs standalone
        name: str = "analyst_agent"
        def plan(self, *_, **__): ...
        def act(self, *_, **__): ...
        def explain(self, *_, **__): ...
        def heartbeat(self, *_, **__): return {"ok": True}

try:
    # backend/ai/agents/skills/market/quotes.py
    from ..skills.market.quotes import get_candles  # type: ignore
except Exception:  # pragma: no cover
    def get_candles(symbol: str, *, interval: str = "1d", lookback: int = 120) -> List[Dict[str, Any]]:
        """
        Fallback: synthetic random-walk candles so the agent still runs.
        Output schema: [{ts: int(ms), o,h,l,c,v}, ...]
        """
        random.seed(hash(symbol) & 0xFFFF)
        price = 100.0 + (hash(symbol) % 200) / 10.0
        out: List[Dict[str, Any]] = []
        now = datetime.utcnow()
        for i in range(lookback):
            # backwards in time (newest last)
            ts = int((now - timedelta(days=lookback - i)).timestamp() * 1000)
            ret = random.gauss(0.0002, 0.012)
            price = max(0.5, price * (1.0 + ret))
            rng = abs(random.gauss(0, 0.01)) * price
            o = price / (1.0 + ret) if i > 0 else price
            h = price + rng
            l = max(0.5, price - rng)
            c = price
            v = abs(random.gauss(0, 1)) * 1_000_000
            out.append({"ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v})
        return out

try:
    # backend/ai/agents/skills/risk/var_engine.py
    from ..skills.risk.var_engine import VaREngine  # type: ignore
except Exception:  # pragma: no cover
    class VaREngine:
        @staticmethod
        def gaussian(returns: List[float], alpha: float = 0.99, horizon_days: int = 1):
            mu = sum(returns) / max(1, len(returns))
            sd = (sum((x - mu) ** 2 for x in returns) / max(1, len(returns) - 1)) ** 0.5
            z = -2.3263478740408408  # ~ N^-1(0.01)
            var_h = mu * horizon_days + sd * (horizon_days ** 0.5) * z
            # tiny struct mimic
            return type("VarEstimate", (), {"var": var_h, "mean": mu, "stdev": sd, "alpha": alpha, "horizon_days": horizon_days})


# ------------------------------------------------------------
# Request/Response models
# ------------------------------------------------------------

@dataclass
class AnalystRequest:
    symbols: List[str]
    interval: str = "1d"              # "1m","5m","1h","1d"
    lookback: int = 120               # number of candles to fetch
    tasks: List[str] = field(default_factory=lambda: ["overview", "signals", "risk"])
    risk_alpha: float = 0.99          # VaR confidence
    horizon_days: int = 1             # VaR horizon
    notes: Optional[str] = None       # free-text instruction

@dataclass
class SignalBlock:
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    sma_fast: Optional[float] = None
    sma_slow: Optional[float] = None
    trend: Optional[str] = None       # "bullish"/"bearish"/"neutral"

@dataclass
class RiskBlock:
    var_return: float                 # negative number (e.g., -0.025)
    var_cash_per_100: float           # cash loss per 100 notional
    mean: float
    stdev: float
    alpha: float
    horizon_days: int

@dataclass
class Insight:
    symbol: str
    last_price: float
    signals: Optional[SignalBlock] = None
    risk: Optional[RiskBlock] = None
    commentary: Optional[str] = None

@dataclass
class AnalystResponse:
    generated_at: int
    interval: str
    lookback: int
    insights: List[Insight]
    summary: str


# ------------------------------------------------------------
# Math helpers (no deps)
# ------------------------------------------------------------

def _sma(xs: List[float], n: int) -> Optional[float]:
    if len(xs) < n or n <= 0: return None
    return sum(xs[-n:]) / float(n)

def _ema(xs: List[float], n: int) -> Optional[float]:
    if len(xs) < n or n <= 0: return None
    k = 2.0 / (n + 1.0)
    ema = sum(xs[:n]) / n
    for x in xs[n:]:
        ema = x * k + ema * (1 - k)
    return ema

def _rsi(closes: List[float], n: int = 14) -> Optional[float]:
    if len(closes) < n + 1: return None
    gains = []; losses = []
    for i in range(-n, 0):
        ch = closes[i] - closes[i - 1]
        if ch >= 0: gains.append(ch)
        else: losses.append(-ch)
    avg_gain = sum(gains) / max(1, n)
    avg_loss = sum(losses) / max(1, n)
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < slow + signal: return (None, None, None)
    def ema(series: List[float], n: int) -> float:
        k = 2.0 / (n + 1.0)
        e = series[0]
        for x in series[1:]:
            e = x * k + e * (1 - k)
        return e
    macd_line = ema(closes[-slow:], fast) - ema(closes[-slow:], slow)
    signal_line = ema([macd_line] * (signal + 1), signal)  # simple smooth
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _log_returns(closes: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(closes)):
        p0, p1 = max(1e-9, closes[i - 1]), max(1e-9, closes[i])
        out.append(math.log(p1 / p0))
    return out


# ------------------------------------------------------------
# Analyst Agent
# ------------------------------------------------------------

class AnalystAgent(BaseAgent): # type: ignore
    """
    Research/overview agent:
      - fetches candles
      - computes basic signals (RSI/SMA/MACD)
      - runs VaR on log returns (uses your VaR engine if available)
      - emits concise commentary per symbol + portfolio-style summary
    """

    name = "analyst_agent"

    def plan(self, req: AnalystRequest | Dict[str, Any]) -> AnalystRequest:
        if isinstance(req, AnalystRequest):
            return req
        # tolerate loose dict inputs
        return AnalystRequest(
            symbols=list(req.get("symbols", [])),
            interval=req.get("interval", "1d"),
            lookback=int(req.get("lookback", 120)),
            tasks=list(req.get("tasks", ["overview", "signals", "risk"])),
            risk_alpha=float(req.get("risk_alpha", 0.99)),
            horizon_days=int(req.get("horizon_days", 1)),
            notes=req.get("notes"),
        )

    def act(self, request: AnalystRequest | Dict[str, Any]) -> AnalystResponse:
        req = self.plan(request)
        insights: List[Insight] = []

        for sym in req.symbols:
            candles = get_candles(sym, interval=req.interval, lookback=req.lookback)
            if not candles:
                insights.append(Insight(symbol=sym, last_price=float("nan"), commentary="No data"))
                continue

            closes = [float(x["c"]) for x in candles]
            last_px = closes[-1]

            signals: Optional[SignalBlock] = None
            if "signals" in req.tasks or "overview" in req.tasks:
                rsi = _rsi(closes, 14)
                sma_fast = _sma(closes, 20)
                sma_slow = _sma(closes, 50)
                macd_line, macd_sig, macd_hist = _macd(closes, 12, 26, 9)
                trend = None
                if sma_fast and sma_slow:
                    if sma_fast > sma_slow * 1.002:
                        trend = "bullish"
                    elif sma_fast < sma_slow * 0.998:
                        trend = "bearish"
                    else:
                        trend = "neutral"
                signals = SignalBlock(
                    rsi=rsi, macd=macd_line, macd_signal=macd_sig, macd_hist=macd_hist,
                    sma_fast=sma_fast, sma_slow=sma_slow, trend=trend
                )

            risk: Optional[RiskBlock] = None
            if "risk" in req.tasks:
                rets = _log_returns(closes)
                est = VaREngine.gaussian(rets, alpha=req.risk_alpha, horizon_days=req.horizon_days)
                var_cash_per_100 = max(0.0, -est.var * 100.0)  # type: ignore # cash loss per 100 notional
                risk = RiskBlock(
                    var_return=est.var, # type: ignore
                    var_cash_per_100=var_cash_per_100,
                    mean=getattr(est, "mean", sum(rets) / max(1, len(rets))) if rets else 0.0,
                    stdev=getattr(est, "stdev", (sum((x - (sum(rets)/max(1,len(rets))))**2 for x in rets)/max(1,len(rets)-1))**0.5) if rets else 0.0,
                    alpha=req.risk_alpha,
                    horizon_days=req.horizon_days
                )

            commentary = self._commentary(sym, last_px, signals, risk)
            insights.append(Insight(symbol=sym, last_price=last_px, signals=signals, risk=risk, commentary=commentary))

        summary = self._summarize(insights)

        return AnalystResponse(
            generated_at=int(datetime.utcnow().timestamp() * 1000),
            interval=req.interval,
            lookback=req.lookback,
            insights=insights,
            summary=summary,
        )

    # -------- human-friendly text helpers --------

    def _commentary(self, sym: str, px: float, sig: Optional[SignalBlock], risk: Optional[RiskBlock]) -> str:
        parts: List[str] = [f"{sym}: last {px:,.2f}."]
        if sig:
            if sig.trend:
                parts.append(f"Trend looks {sig.trend}.")
            if sig.rsi is not None:
                if sig.rsi >= 70:
                    parts.append(f"RSI {sig.rsi:.1f} (overbought risk).")
                elif sig.rsi <= 30:
                    parts.append(f"RSI {sig.rsi:.1f} (oversold bounce?).")
                else:
                    parts.append(f"RSI {sig.rsi:.1f}.")
            if sig.macd is not None and sig.macd_signal is not None:
                cross = "↑" if sig.macd > sig.macd_signal else "↓"
                parts.append(f"MACD {sig.macd:.3f}/{sig.macd_signal:.3f} ({cross})")
        if risk:
            parts.append(f"VaR{int(risk.alpha*100)} (h={risk.horizon_days}d): ~{risk.var_cash_per_100:.2f} per $100.")
        return " ".join(parts)

    def _summarize(self, insights: List[Insight]) -> str:
        if not insights:
            return "No symbols analyzed."
        bulls = [i.symbol for i in insights if i.signals and i.signals.trend == "bullish"]
        bears = [i.symbol for i in insights if i.signals and i.signals.trend == "bearish"]
        txt: List[str] = []
        if bulls: txt.append("Bullish tilt: " + ", ".join(bulls) + ".")
        if bears: txt.append("Bearish tilt: " + ", ".join(bears) + ".")
        worst = None
        worst_abs = -1.0
        for i in insights:
            if i.risk and i.risk.var_cash_per_100 > worst_abs:
                worst_abs = i.risk.var_cash_per_100
                worst = i.symbol
        if worst:
            txt.append(f"Highest risk by VaR per $100: {worst} (~{worst_abs:.2f}).")
        return " ".join(txt) or "Neutral read."

    # -------- optional: rationale for UI --------

    def explain(self) -> str:
        return (
            "AnalystAgent fetches candles (via skills.market.quotes), computes RSI/SMA/MACD, "
            "runs VaR on log-returns (skills.risk.var_engine) and outputs symbol-level insights + a summary. "
            "Fallbacks generate synthetic candles and a Gaussian VaR if skills are unavailable."
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "time": int(datetime.utcnow().timestamp())}


# ------------------------------------------------------------
# Quick CLI smoke test (optional)
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    agent = AnalystAgent()
    req = AnalystRequest(symbols=["AAPL", "MSFT", "SPY"], interval="1d", lookback=120)
    resp = agent.act(req)
    print("Summary:", resp.summary)
    for ins in resp.insights:
        print(f"- {ins.symbol}: px={ins.last_price:.2f} | {ins.commentary}")