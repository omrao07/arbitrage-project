# bus/python/events/risk.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Literal


# =========================
# Base Event
# =========================
@dataclass
class RiskEvent:
    event_type: str
    ts_event: int
    ts_ingest: int
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def _now_ms(cls) -> int:
        return int(time.time() * 1000)

    @classmethod
    def _base(cls, event_type: str, source: str, ts_event: Optional[int] = None) -> Dict[str, Any]:
        now = cls._now_ms()
        return {
            "event_type": event_type,
            "ts_event": ts_event if ts_event is not None else now,
            "ts_ingest": now,
            "source": source,
        }


# =========================
# VaR / ES
# =========================
VaRMethod = Literal["HISTORICAL", "PARAMETRIC", "MONTE_CARLO"]

@dataclass
class VaRUpdate(RiskEvent):
    portfolio_id: str
    horizon_days: int
    confidence: float
    method: VaRMethod = "HISTORICAL"
    var: float = 0.0
    es: Optional[float] = None
    nav: Optional[float] = None

    @classmethod
    def create(
        cls,
        portfolio_id: str,
        var: float,
        confidence: float = 0.95,
        horizon_days: int = 1,
        source: str = "var",
        ts_event: Optional[int] = None,
        es: Optional[float] = None,
        nav: Optional[float] = None,
        method: VaRMethod = "HISTORICAL",
    ) -> "VaRUpdate":
        base = cls._base("var", source, ts_event)
        return cls(
            portfolio_id=portfolio_id,
            horizon_days=horizon_days,
            confidence=confidence,
            method=method,
            var=var,
            es=es,
            nav=nav,
            **base,
        )


# =========================
# Stress Scenario
# =========================
@dataclass
class StressScenario(RiskEvent):
    portfolio_id: str
    scenario_id: str
    pnl_impact: float
    pnl_pct_nav: Optional[float] = None
    shocks: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        portfolio_id: str,
        scenario_id: str,
        pnl_impact: float,
        source: str = "stress",
        ts_event: Optional[int] = None,
        pnl_pct_nav: Optional[float] = None,
        shocks: Optional[Dict[str, float]] = None,
    ) -> "StressScenario":
        base = cls._base("stress", source, ts_event)
        return cls(
            portfolio_id=portfolio_id,
            scenario_id=scenario_id,
            pnl_impact=pnl_impact,
            pnl_pct_nav=pnl_pct_nav,
            shocks=shocks or {},
            **base,
        )


# =========================
# Limit Breach
# =========================
Severity = Literal["INFO", "MINOR", "MAJOR", "CRITICAL"]

@dataclass
class LimitBreach(RiskEvent):
    limit_id: str
    portfolio_id: str
    metric: str
    value: float
    threshold: Optional[float] = None
    severity: Severity = "MAJOR"
    message: Optional[str] = None

    @classmethod
    def create(
        cls,
        limit_id: str,
        portfolio_id: str,
        metric: str,
        value: float,
        source: str = "guard",
        ts_event: Optional[int] = None,
        threshold: Optional[float] = None,
        severity: Severity = "MAJOR",
        message: Optional[str] = None,
    ) -> "LimitBreach":
        base = cls._base("limit_breach", source, ts_event)
        return cls(
            limit_id=limit_id,
            portfolio_id=portfolio_id,
            metric=metric,
            value=value,
            threshold=threshold,
            severity=severity,
            message=message,
            **base,
        )


# =========================
# Drawdown
# =========================
@dataclass
class DrawdownUpdate(RiskEvent):
    portfolio_id: str
    dd_day_pct: float
    dd_peak_to_trough_pct: float
    nav: Optional[float] = None

    @classmethod
    def create(
        cls,
        portfolio_id: str,
        dd_day_pct: float,
        dd_peak_to_trough_pct: float,
        source: str = "risk",
        ts_event: Optional[int] = None,
        nav: Optional[float] = None,
    ) -> "DrawdownUpdate":
        base = cls._base("drawdown", source, ts_event)
        return cls(
            portfolio_id=portfolio_id,
            dd_day_pct=dd_day_pct,
            dd_peak_to_trough_pct=dd_peak_to_trough_pct,
            nav=nav,
            **base,
        )


# =========================
# Exposure
# =========================
@dataclass
class ExposureUpdate(RiskEvent):
    portfolio_id: str
    gross_exposure: float
    net_exposure: float

    @classmethod
    def create(
        cls,
        portfolio_id: str,
        gross_exposure: float,
        net_exposure: float,
        source: str = "risk",
        ts_event: Optional[int] = None,
    ) -> "ExposureUpdate":
        base = cls._base("exposure", source, ts_event)
        return cls(
            portfolio_id=portfolio_id,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            **base,
        )


# =========================
# Example
# =========================
if __name__ == "__main__":
    var_ev = VaRUpdate.create("PORT-1", var=0.036, es=0.052, nav=10_000_000)
    print(var_ev.to_json())

    stress_ev = StressScenario.create("PORT-1", "OIL+10_USD-2", pnl_impact=-350_000, pnl_pct_nav=-0.035, shocks={"OIL": 0.1, "USD": -0.02})
    print(stress_ev.to_json())

    breach_ev = LimitBreach.create("var_limit", "PORT-1", "var95", value=0.048, threshold=0.04, severity="CRITICAL")
    print(breach_ev.to_json())

    dd_ev = DrawdownUpdate.create("PORT-1", dd_day_pct=0.031, dd_peak_to_trough_pct=0.087, nav=9_500_000)
    print(dd_ev.to_json())

    expo_ev = ExposureUpdate.create("PORT-1", gross_exposure=24_500_000, net_exposure=1_200_000)
    print(expo_ev.to_json())