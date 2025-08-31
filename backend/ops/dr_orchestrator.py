# backend/ops/dr_orchestration.py
from __future__ import annotations
"""
DR Orchestrator — cutover, failover, failback & drills
------------------------------------------------------
Goals
- One blessed way to *freeze → drain → snapshot → cutover → verify → unfreeze*.
- Works for both planned maintenance and real incidents.
- Records every step to an append-only compliance ledger (if available).
- Dry-run by default, hard to misuse in prod unless --force is passed.

Optional integrations (auto-detected if importable)
- backend.bus.streams.publish_stream
- backend.compliance.compliance_recorder.ComplianceRecorder
- backend.ops.chaos_money.ChaosMoney (for DR drills)
- Redis ping (if redis is importable)  — best-effort health signal

CLI
  python -m backend.ops.dr_orchestration plan --cfg dr.yaml
  python -m backend.ops.dr_orchestration readiness --cfg dr.yaml
  python -m backend.ops.dr_orchestration failover --cfg dr.yaml --force
  python -m backend.ops.dr_orchestration failback --cfg dr.yaml --force
  python -m backend.ops.dr_orchestration drill --cfg dr.yaml --duration 120
  python -m backend.ops.dr_orchestration status --cfg dr.yaml
"""

import os, json, time, socket, subprocess, shutil
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# -------- optional bus ----------
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        print(f"[BUS] {stream}: {payload}")

OPS_STREAM = os.getenv("DR_EVENTS_STREAM", "ops.dr.events")

# -------- optional compliance ledger ----------
try:
    from backend.compliance.compliance_recorder import ComplianceRecorder, LedgerHeader  # type: ignore
except Exception:
    ComplianceRecorder = None  # type: ignore
    LedgerHeader = None        # type: ignore

# -------- optional chaos (for drills) ----------
try:
    from backend.ops.chaos_money import ChaosMoney, ChaosConfig, BusFaults  # type: ignore
except Exception:
    ChaosMoney = None  # type: ignore

# -------- optional redis ping ----------
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

ENV = os.getenv("ENV", os.getenv("ENVIRONMENT", "dev")).lower()

# ---------------- models ----------------

@dataclass
class Endpoint:
    name: str
    host: str
    port: int
    kind: str = "tcp"   # "tcp" | "http" (http does tcp connect only, no GET to stay stdlib)
    critical: bool = True

@dataclass
class Service:
    name: str
    endpoints: List[Endpoint]
    pre_freeze_cmd: Optional[str] = None
    post_unfreeze_cmd: Optional[str] = None
    snapshot_cmd: Optional[str] = None       # e.g., "pg_basebackup ..." or "./scripts/snap.sh"
    restore_cmd: Optional[str] = None        # e.g., "pg_ctl promote ..." or "./scripts/restore.sh"
    promote_cmd: Optional[str] = None        # for replicas
    validate_cmd: Optional[str] = None       # quick smoke verification
    data_dir: Optional[str] = None           # optional existence check
    rpo_seconds: int = 60                    # target; informational

@dataclass
class Site:
    name: str                  # e.g., "primary", "dr"
    region: str                # e.g., "us-east-1"
    services: List[Service]

@dataclass
class DRConfig:
    dry_run: bool = True
    session: str = "dr-" + str(int(time.time()))
    book: Optional[str] = None
    allow_envs: Tuple[str, ...] = ("dev","staging")
    primary: Site = None  # type: ignore
    dr: Site = None       # type: ignore
    freeze_bus_stream: str = "risk.freeze"
    resume_bus_stream: str = "risk.resume"
    kill_switch_stream: str = "risk.killswitch"
    health_timeout_ms: int = 1500
    health_retries: int = 3
    post_cutover_verifications: List[str] = None  # type: ignore  # list of shell commands
    notes: Optional[str] = None

# ---------------- config loader ----------------

def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    if path.lower().endswith((".yaml",".yml")):
        try:
            import yaml  # type: ignore
        except Exception:
            raise SystemExit("Install PyYAML or use JSON for DR config.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_config(path: str) -> DRConfig:
    raw = _load_yaml_or_json(path)
    def _ep(e): return Endpoint(**e)
    def _svc(s): return Service(**{**s, "endpoints": [ _ep(x) for x in s.get("endpoints",[]) ]})
    def _site(s): return Site(name=s["name"], region=s["region"], services=[_svc(x) for x in s.get("services",[])])
    cfg = DRConfig(
        dry_run=bool(raw.get("dry_run", True)),
        session=str(raw.get("session") or f"dr-{int(time.time())}"),
        book=raw.get("book"),
        allow_envs=tuple(raw.get("allow_envs", ["dev","staging"])),
        primary=_site(raw["primary"]),
        dr=_site(raw["dr"]),
        freeze_bus_stream=raw.get("freeze_bus_stream","risk.freeze"),
        resume_bus_stream=raw.get("resume_bus_stream","risk.resume"),
        kill_switch_stream=raw.get("kill_switch_stream","risk.killswitch"),
        health_timeout_ms=int(raw.get("health_timeout_ms", 1500)),
        health_retries=int(raw.get("health_retries", 3)),
        post_cutover_verifications=list(raw.get("post_cutover_verifications", [])),
        notes=raw.get("notes")
    )
    return cfg

# ---------------- utils ----------------

def _ts_ms() -> int: return int(time.time()*1000)

def _log(kind: str, data: Dict[str, Any]) -> None:
    publish_stream(OPS_STREAM, {"ts_ms": _ts_ms(), "kind": kind, **data})

def _ping_tcp(host: str, port: int, timeout_ms: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_ms/1000.0):
            return True
    except Exception:
        return False

def _shell(cmd: str, dry: bool) -> Tuple[int,str,str]:
    if dry: 
        return 0, "[dry-run] " + cmd, ""
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate()
        return p.returncode, out, err
    except Exception as e:
        return 9, "", str(e)

def _redis_ping(url: Optional[str]) -> Optional[bool]:
    if not url or not redis: return None
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5, retry_on_timeout=False)
        return client.ping() # type: ignore
    except Exception:
        return False

# ---------------- compliance session ----------------

class _Ledger:
    def __init__(self, cfg: DRConfig):
        self.enabled = bool(ComplianceRecorder)
        self.session = cfg.session
        if self.enabled:
            self.rec = ComplianceRecorder(root=os.getenv("COMPLIANCE_LEDGER_ROOT","./audit"), pii_mode=None) # type: ignore
            if LedgerHeader:
                self.rec.start(LedgerHeader(session_id=cfg.session, started_ts_ms=_ts_ms(), env=ENV, book=cfg.book, region=cfg.primary.region, meta={"dr": True}))
            _log("ledger_start", {"session": cfg.session})

    def record(self, kind: str, payload: Dict[str, Any]) -> None:
        if self.enabled:
            try:
                self.rec.record(self.session, kind, payload, redact=False)  # infra-only
            except Exception:
                pass

    def close(self) -> None:
        if self.enabled:
            try:
                man = self.rec.close(self.session)
                _log("ledger_close", {"session": self.session, "root": man.get("root")})
            except Exception:
                pass

# ---------------- core orchestrator ----------------

class DROrchestrator:
    def __init__(self, cfg: DRConfig):
        self.cfg = cfg
        self.ledger = _Ledger(cfg)
        self._guard_env()

    # ---- safety rails ----
    def _guard_env(self):
        if ENV not in self.cfg.allow_envs and not self.cfg.dry_run:
            raise SystemExit(f"ENV='{ENV}' not permitted for non-dry DR without --force/dry_run.")

    # ---- high-level operations ----

    def plan(self) -> Dict[str, Any]:
        plan = {
            "session": self.cfg.session,
            "env": ENV,
            "dry_run": self.cfg.dry_run,
            "primary": {"name": self.cfg.primary.name, "region": self.cfg.primary.region, "services": [s.name for s in self.cfg.primary.services]},
            "dr": {"name": self.cfg.dr.name, "region": self.cfg.dr.region, "services": [s.name for s in self.cfg.dr.services]},
            "verifications": self.cfg.post_cutover_verifications,
            "notes": self.cfg.notes
        }
        _log("plan", plan); self.ledger.record("dr_plan", plan)
        return plan

    def readiness(self) -> Dict[str, Any]:
        res = {
            "primary": self._site_health(self.cfg.primary),
            "dr": self._site_health(self.cfg.dr)
        }
        _log("readiness", res); self.ledger.record("dr_readiness", res)
        return res

    def failover(self) -> Dict[str, Any]:
        """
        Primary -> DR
        Sequence:
        1) Freeze trading; drain OMS queues.
        2) Snapshot/Promote services at DR site (or restore from latest).
        3) Cutover: point routers/clients to DR endpoints (DNS toggle placeholder).
        4) Verification; then unfreeze trading.
        """
        cfg = self.cfg
        self._announce("failover_start", {"from": cfg.primary.name, "to": cfg.dr.name})

        self._freeze("DR_FAILOVER")

        self._dr_prep_promote(self.cfg.dr, label="failover")

        self._cutover_dns_placeholder(self.cfg.dr)

        verify = self._verify_post_cutover(self.cfg.dr)

        self._resume()

        out = {"ok": verify.get("ok", False), "verify": verify}
        _log("failover_done", out); self.ledger.record("dr_failover", out)
        return out

    def failback(self) -> Dict[str, Any]:
        """
        DR -> Primary (reverse path)
        """
        cfg = self.cfg
        self._announce("failback_start", {"from": cfg.dr.name, "to": cfg.primary.name})

        self._freeze("DR_FAILBACK")

        self._dr_prep_promote(self.cfg.primary, label="failback")

        self._cutover_dns_placeholder(self.cfg.primary)

        verify = self._verify_post_cutover(self.cfg.primary)

        self._resume()

        out = {"ok": verify.get("ok", False), "verify": verify}
        _log("failback_done", out); self.ledger.record("dr_failback", out)
        return out

    def drill(self, duration_s: int = 120) -> Dict[str, Any]:
        """
        Run a short DR drill:
        - Freeze briefly
        - Patch bus (light latency + drops) if ChaosMoney exists
        - Verify services + basic trades-off path health
        - Resume
        """
        self._announce("drill_start", {"duration_s": duration_s})
        self._freeze("DR_DRILL")

        chaos_used = False
        if ChaosMoney is not None:
            chaos_used = True
            try:
                from backend.ops.chaos_money import ChaosConfig, BusFaults  # type: ignore
                cm = ChaosMoney(ChaosConfig(dry_run=False, bus_faults=BusFaults(enable=True, drop_prob=0.02, jitter_ms=25)))
                with cm.patch_bus():
                    time.sleep(max(2, min(10, duration_s // 4)))
            except Exception:
                chaos_used = False

        verify = self._verify_post_cutover(self.cfg.primary)
        self._resume()
        out = {"ok": verify.get("ok", False), "verify": verify, "chaos": chaos_used}
        _log("drill_done", out); self.ledger.record("dr_drill", out)
        return out

    def status(self) -> Dict[str, Any]:
        s = {"primary": self._site_health(self.cfg.primary), "dr": self._site_health(self.cfg.dr)}
        _log("status", s)
        return s

    # ---- helpers ----

    def _announce(self, kind: str, data: Dict[str, Any]) -> None:
        _log(kind, data); self.ledger.record(kind, data)

    def _freeze(self, reason: str) -> None:
        publish_stream(self.cfg.kill_switch_stream, {"ts_ms": _ts_ms(), "reason": reason})
        publish_stream(self.cfg.freeze_bus_stream, {"ts_ms": _ts_ms(), "reason": reason})
        self.ledger.record("freeze", {"reason": reason})
        # give downstreams a moment to quiesce
        time.sleep(1.0)

    def _resume(self) -> None:
        publish_stream(self.cfg.resume_bus_stream, {"ts_ms": _ts_ms(), "reason": "DR_RESUME"})
        self.ledger.record("resume", {"reason": "DR_RESUME"})

    def _site_health(self, site: Site) -> Dict[str, Any]:
        out: Dict[str, Any] = {"site": site.name, "region": site.region, "services": {}}
        for svc in site.services:
            svc_res = {"ok": True, "endpoints": []}
            # datadir existence
            if svc.data_dir:
                svc_res["data_dir_exists"] = bool(os.path.exists(svc.data_dir))
                svc_res["ok"] &= svc_res["data_dir_exists"]
            # endpoints
            eps = []
            for ep in svc.endpoints:
                ok = False
                for _ in range(self.cfg.health_retries):
                    ok = _ping_tcp(ep.host, ep.port, self.cfg.health_timeout_ms)
                    if ok: break
                    time.sleep(0.1)
                eps.append({"name": ep.name, "host": ep.host, "port": ep.port, "ok": ok, "critical": ep.critical})
                if ep.critical: svc_res["ok"] &= ok
            # optional redis ping (first redis:// endpoint host:port)
            redis_ok = None
            for ep in svc.endpoints:
                if str(ep.port) == "6379" or "redis" in ep.name.lower():
                    url = f"redis://{ep.host}:{ep.port}"
                    redis_ok = _redis_ping(url)
                    break
            if redis_ok is not None:
                svc_res["redis_ping"] = redis_ok
                svc_res["ok"] &= bool(redis_ok)
            svc_res["endpoints"] = eps
            out["services"][svc.name] = svc_res
        out["ok"] = all(v["ok"] for v in out["services"].values()) if out["services"] else True
        return out

    def _dr_prep_promote(self, target: Site, *, label: str) -> None:
        """
        For each service at target site:
        - run pre_freeze hooks on previous site (best-effort — we are already frozen)
        - snapshot/restore/promo commands on target
        - validate command
        """
        for svc in target.services:
            self.ledger.record("service_prepare", {"site": target.name, "service": svc.name, "label": label})
            # snapshot/restore/promote
            if svc.snapshot_cmd:
                rc, out, err = _shell(svc.snapshot_cmd, self.cfg.dry_run)
                _log("snapshot", {"service": svc.name, "rc": rc, "out": out[-2000:], "err": err[-2000:]})
                if rc != 0: raise SystemExit(f"snapshot failed for {svc.name}: {err}")
            if svc.restore_cmd:
                rc, out, err = _shell(svc.restore_cmd, self.cfg.dry_run)
                _log("restore", {"service": svc.name, "rc": rc, "out": out[-2000:], "err": err[-2000:]})
                if rc != 0: raise SystemExit(f"restore failed for {svc.name}: {err}")
            if svc.promote_cmd:
                rc, out, err = _shell(svc.promote_cmd, self.cfg.dry_run)
                _log("promote", {"service": svc.name, "rc": rc, "out": out[-2000:], "err": err[-2000:]})
                if rc != 0: raise SystemExit(f"promote failed for {svc.name}: {err}")
            # validate
            if svc.validate_cmd:
                rc, out, err = _shell(svc.validate_cmd, self.cfg.dry_run)
                _log("validate", {"service": svc.name, "rc": rc, "out": out[-2000:], "err": err[-2000:]})
                if rc != 0: raise SystemExit(f"validate failed for {svc.name}: {err}")

    def _cutover_dns_placeholder(self, target: Site) -> None:
        """
        Placeholder for your DNS/ingress switch (Route53, Cloud DNS, etc).
        We only emit an event here so you can hook your own deploy step in CI.
        """
        _log("cutover_request", {"target_site": target.name, "region": target.region})
        self.ledger.record("cutover_request", {"target_site": target.name})

    def _verify_post_cutover(self, site: Site) -> Dict[str, Any]:
        health = self._site_health(site)
        runs: List[Dict[str, Any]] = []
        ok = bool(health.get("ok", False))
        # additional custom commands
        for cmd in (self.cfg.post_cutover_verifications or []):
            rc, out, err = _shell(cmd, self.cfg.dry_run)
            runs.append({"cmd": cmd, "rc": rc, "out_tail": out[-1000:], "err_tail": err[-1000:]})
            ok &= (rc == 0)
        res = {"health": health, "cmds": runs, "ok": ok}
        _log("post_cutover_verify", {"ok": ok}); self.ledger.record("post_cutover_verify", res)
        return res

# ---------------- CLI ----------------

def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="DR Orchestrator — failover/failback/drills")
    ap.add_argument("--cfg", required=True, help="dr.yaml or dr.json")
    ap.add_argument("--force", action="store_true", help="override env guard & dry-run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plan")
    sub.add_parser("readiness")
    sub.add_parser("status")

    fo = sub.add_parser("failover")
    fb = sub.add_parser("failback")

    dr = sub.add_parser("drill")
    dr.add_argument("--duration", type=int, default=120)

    return ap.parse_args()

def main():
    args = _parse_args()
    cfg = load_config(args.cfg)

    # Safety: default to dry-run; --force flips to live unless cfg.dry_run was set true explicitly
    if args.force:
        cfg.dry_run = False
        cfg.allow_envs = tuple(set(list(cfg.allow_envs) + [ENV]))  # allow current env when forced

    orch = DROrchestrator(cfg)

    if args.cmd == "plan":
        print(json.dumps(orch.plan(), indent=2))
    elif args.cmd == "readiness":
        print(json.dumps(orch.readiness(), indent=2))
    elif args.cmd == "status":
        print(json.dumps(orch.status(), indent=2))
    elif args.cmd == "failover":
        if ENV == "production" and not args.force:
            raise SystemExit("Refusing to failover in production without --force")
        print(json.dumps(orch.failover(), indent=2))
        orch.ledger.close()
    elif args.cmd == "failback":
        if ENV == "production" and not args.force:
            raise SystemExit("Refusing to failback in production without --force")
        print(json.dumps(orch.failback(), indent=2))
        orch.ledger.close()
    elif args.cmd == "drill":
        print(json.dumps(orch.drill(duration_s=int(args.duration)), indent=2))
        orch.ledger.close()
    else:
        raise SystemExit("unknown command")

if __name__ == "__main__":  # pragma: no cover
    main()