# bus/python/events/system.py
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Literal


# =========================
# Base event
# =========================
@dataclass
class SystemEvent:
    event_type: str                       # "heartbeat" | "health" | "audit" | "status" | "error" | "deploy" | "config" | "metric" | "alert"
    ts_event: int                         # business event time (ms epoch, UTC)
    ts_ingest: int                        # ingest time (ms epoch, UTC)
    source: str                           # emitter, e.g., "api", "risk", "ems", "scheduler"
    service: str                          # logical service name
    instance_id: str                      # pod/container/host id
    env: str = os.getenv("APP_ENV", "prod")
    region: Optional[str] = os.getenv("REGION") or None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @classmethod
    def _base(
        cls,
        event_type: str,
        service: str,
        source: str = "system",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = cls._now_ms()
        return {
            "event_type": event_type,
            "ts_event": ts_event if ts_event is not None else now,
            "ts_ingest": now,
            "source": source,
            "service": service,
            "instance_id": instance_id or socket.gethostname(),
            "env": os.getenv("APP_ENV", "prod"),
            "region": os.getenv("REGION") or None,
        }


# =========================
# Heartbeat (liveness)
# =========================
@dataclass
class Heartbeat(SystemEvent):
    uptime_sec: float # type: ignore
    pid: int = os.getpid()
    version: Optional[str] = None         # app build/git sha
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        service: str,
        uptime_sec: float,
        source: str = "system",
        ts_event: Optional[int] = None,
        version: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
        instance_id: Optional[str] = None,
    ) -> "Heartbeat":
        base = cls._base("heartbeat", service, source, ts_event, instance_id)
        return cls(uptime_sec=float(uptime_sec), version=version, extras=extras or {}, **base)


# =========================
# Health (readiness)
# =========================
HealthStatus = Literal["OK", "DEGRADED", "FAIL"]

@dataclass
class HealthCheck(SystemEvent):
    status: HealthStatus # type: ignore
    checks: Dict[str, HealthStatus] = field(default_factory=dict)  # e.g., {"db":"OK","kafka":"DEGRADED"}
    latency_ms: Optional[float] = None

    @classmethod
    def create(
        cls,
        service: str,
        status: HealthStatus,
        checks: Optional[Dict[str, HealthStatus]] = None,
        latency_ms: Optional[float] = None,
        source: str = "health",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> "HealthCheck":
        base = cls._base("health", service, source, ts_event, instance_id)
        return cls(status=status, checks=checks or {}, latency_ms=latency_ms, **base)


# =========================
# Service status (state)
# =========================
ServiceState = Literal["STARTING", "RUNNING", "PAUSED", "DRAINING", "STOPPING", "STOPPED"]

@dataclass
class ServiceStatus(SystemEvent):
    state: ServiceState # type: ignore
    reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        service: str,
        state: ServiceState,
        reason: Optional[str] = None,
        source: str = "system",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> "ServiceStatus":
        base = cls._base("status", service, source, ts_event, instance_id)
        return cls(state=state, reason=reason, **base)


# =========================
# Errors / incidents
# =========================
Severity = Literal["INFO", "MINOR", "MAJOR", "CRITICAL"]

@dataclass
class ErrorEvent(SystemEvent):
    severity: Severity # type: ignore
    message: str # type: ignore
    code: Optional[str] = None
    stack: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        service: str,
        message: str,
        severity: Severity = "MAJOR",
        code: Optional[str] = None,
        stack: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        source: str = "app",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> "ErrorEvent":
        base = cls._base("error", service, source, ts_event, instance_id)
        return cls(severity=severity, message=message, code=code, stack=stack, context=context or {}, **base)


# =========================
# Audit (GRC)
# =========================
@dataclass
class AuditEvent(SystemEvent):
    actor: str                         # type: ignore # user/service principal
    action: str                        # type: ignore # e.g., "LOGIN", "ORDER_SUBMIT", "POLICY_CHANGE"
    resource: str                      # type: ignore # ARN/path/id
    outcome: Literal["ALLOW", "DENY", "ERROR"] = "ALLOW"
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        service: str,
        actor: str,
        action: str,
        resource: str,
        outcome: Literal["ALLOW", "DENY", "ERROR"] = "ALLOW",
        details: Optional[Dict[str, Any]] = None,
        source: str = "audit",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> "AuditEvent":
        base = cls._base("audit", service, source, ts_event, instance_id)
        return cls(actor=actor, action=action, resource=resource, outcome=outcome, details=details or {}, **base)


# =========================
# Config changes
# =========================
@dataclass
class ConfigChange(SystemEvent):
    key: str # type: ignore
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    changer: Optional[str] = None        # who/what changed it (user/service)
    reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        service: str,
        key: str,
        new_value: Any,
        old_value: Optional[Any] = None,
        changer: Optional[str] = None,
        reason: Optional[str] = None,
        source: str = "config",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> "ConfigChange":
        base = cls._base("config", service, source, ts_event, instance_id)
        return cls(key=key, old_value=old_value, new_value=new_value, changer=changer, reason=reason, **base)


# =========================
# Deployments / releases
# =========================
@dataclass
class DeploymentEvent(SystemEvent):
    version: str                        # type: ignore # git sha / semver
    image: Optional[str] = None         # container image ref
    change_log: Optional[str] = None
    initiator: Optional[str] = None     # CI job/user
    status: Literal["STARTED", "SUCCEEDED", "FAILED"] = "STARTED"

    @classmethod
    def create(
        cls,
        service: str,
        version: str,
        status: Literal["STARTED", "SUCCEEDED", "FAILED"] = "STARTED",
        image: Optional[str] = None,
        change_log: Optional[str] = None,
        initiator: Optional[str] = None,
        source: str = "deploy",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> "DeploymentEvent":
        base = cls._base("deploy", service, source, ts_event, instance_id)
        return cls(version=version, status=status, image=image, change_log=change_log, initiator=initiator, **base)


# =========================
# Metrics (lightweight)
# =========================
@dataclass
class MetricPoint(SystemEvent):
    name: str                           # type: ignore # e.g., "orders_per_min", "latency_ms"
    value: float # type: ignore
    tags: Dict[str, str] = field(default_factory=dict)  # {"portfolio":"PORT-1","topic":"market.ticks.v1"}

    @classmethod
    def create(
        cls,
        service: str,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        source: str = "telemetry",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> "MetricPoint":
        base = cls._base("metric", service, source, ts_event, instance_id)
        return cls(name=name, value=float(value), tags=tags or {}, **base)


# =========================
# Alerts (policy/runtime)
# =========================
@dataclass
class AlertEvent(SystemEvent):
    severity: Severity # type: ignore
    title: str # type: ignore
    description: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)     # e.g., {"rule":"var_breach","portfolio":"PORT-1"}
    annotations: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        service: str,
        title: str,
        severity: Severity = "MAJOR",
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        source: str = "alert",
        ts_event: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> "AlertEvent":
        base = cls._base("alert", service, source, ts_event, instance_id)
        return cls(severity=severity, title=title, description=description, labels=labels or {}, annotations=annotations or {}, **base)


# =========================
# Helpers / example
# =========================
def to_json(obj: Any) -> str:
    if hasattr(obj, "to_json"):
        return obj.to_json()
    if hasattr(obj, "to_dict"):
        return json.dumps(obj.to_dict(), ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    hb = Heartbeat.create(service="risk", uptime_sec=12345.6, version="1.2.3+abc")
    print(hb.to_json())

    hc = HealthCheck.create(service="api", status="DEGRADED", checks={"db": "OK", "kafka": "DEGRADED"}, latency_ms=42.0)
    print(hc.to_json())

    st = ServiceStatus.create(service="ems", state="RUNNING")
    print(st.to_json())

    err = ErrorEvent.create(service="nlp", message="Model load failed", severity="CRITICAL", code="E-ML-001")
    print(err.to_json())

    aud = AuditEvent.create(service="api", actor="user:alice", action="ORDER_SUBMIT", resource="order:o-123", outcome="ALLOW")
    print(aud.to_json())

    cfg = ConfigChange.create(service="risk", key="var.confidence", old_value=0.95, new_value=0.975, changer="risk:cron", reason="Quarterly review")
    print(cfg.to_json())

    dep = DeploymentEvent.create(service="backtester", version="v2025.09.14", status="SUCCEEDED", image="ghcr.io/fund/backtester:sha-123")
    print(dep.to_json())

    m = MetricPoint.create(service="gateway", name="orders_per_min", value=142, tags={"portfolio": "PORT-1"})
    print(m.to_json())

    al = AlertEvent.create(service="guard", title="Pre-trade VaR breach", severity="CRITICAL", labels={"portfolio":"PORT-1","rule":"var_limit"})
    print(al.to_json())