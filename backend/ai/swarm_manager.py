# backend/ai/swarm_manager.py
from __future__ import annotations

"""
Swarm Manager
-------------
A lightweight orchestrator for AI/quant agents.

Capabilities
- Agent registry (name, kind, capabilities, endpoint or callable)
- Task queue (Redis Streams): submit -> lease -> run -> ack/fail -> result stream
- Heartbeats & health (Redis Hash + Stream)
- Work-stealing: idle agents can pick up leased-but-expired tasks
- Broadcast vs directed tasks
- Pluggable runners: Python callables (in-proc), HTTP endpoints, CLI commands

Streams / Keys (override via env):
  swarm.tasks         : inbound tasks (json={id,kind,target,capability,params,ttl_ms})
  swarm.results       : results stream (json={task_id,status,output,meta})
  swarm.heartbeats    : agent heartbeats (json={agent,ts_ms,load,ok})
  swarm.logs          : optional logs/debug
  swarm.registry      : HSET -> agent:{name} = json(spec)
  swarm.leases        : HSET -> task_id = {"agent":..., "lease_ts":...}

Examples:
  Submit a task (RPC-ish):
    XADD swarm.tasks * json '{"id":"t-1","kind":"rpc","capability":"summarize","params":{"text":"..."}}'

  Add an agent (Python callable):
    manager.register_callable("analyst_summarizer", ["summarize"], fn=my_summarize)

  Add an external agent (HTTP):
    manager.register_http("risk_explainer", ["explain"], url="http://localhost:8010/run")
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# -------- Optional Redis -----------------------------------------------------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    AsyncRedis = None  # type: ignore
    USE_REDIS = False

# -------- Env / defaults -----------------------------------------------------
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TASKS_STREAM     = os.getenv("SWARM_TASKS_STREAM", "swarm.tasks")
RESULTS_STREAM   = os.getenv("SWARM_RESULTS_STREAM", "swarm.results")
HEARTBEAT_STREAM = os.getenv("SWARM_HEARTBEATS", "swarm.heartbeats")
LOGS_STREAM      = os.getenv("SWARM_LOGS", "swarm.logs")

REGISTRY_HASH    = os.getenv("SWARM_REGISTRY_HASH", "swarm.registry")
LEASES_HASH      = os.getenv("SWARM_LEASES_HASH", "swarm.leases")

MAXLEN           = int(os.getenv("SWARM_MAXLEN", "20000"))
LEASE_MS         = int(os.getenv("SWARM_LEASE_MS", "15000"))     # task lease time
HB_INTERVAL_MS   = int(os.getenv("SWARM_HB_INTERVAL_MS", "3000"))# heartbeat cadence
POLL_BLOCK_MS    = int(os.getenv("SWARM_POLL_BLOCK_MS", "5000")) # xread block

# -------- Data models --------------------------------------------------------
@dataclass
class AgentSpec:
    name: str
    kind: str                     # "python" | "http" | "cli"
    capabilities: List[str]       # e.g., ["summarize","rank","explain"]
    endpoint: Optional[str] = None# URL or CLI command template
    meta: Dict[str, Any] = None # type: ignore

@dataclass
class Task:
    id: str
    kind: str                     # "rpc" | "broadcast"
    capability: str               # which agent skill to use
    params: Dict[str, Any]
    target: Optional[str] = None  # if provided, direct to agent by name
    ttl_ms: int = 60000           # kill if older than this
    submit_ts: int = 0

@dataclass
class Result:
    task_id: str
    status: str                   # "ok" | "error" | "timeout"
    output: Any
    meta: Dict[str, Any]

# -------- Swarm Manager ------------------------------------------------------
class SwarmManager:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.r: Optional[AsyncRedis] = None # type: ignore
        self._callables: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {}
        self._spec: Optional[AgentSpec] = None
        self._running = False

    # -- Setup ----------------------------------------------------------------
    async def connect(self):
        if not USE_REDIS:
            raise RuntimeError("Redis required for swarm manager")
        self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
        await self.r.ping() # type: ignore

    async def announce(self, spec: AgentSpec):
        """Register/refresh this agent in the registry."""
        self._spec = spec
        await self._hset(REGISTRY_HASH, f"agent:{spec.name}", asjson(spec))
        # initial heartbeat
        await self._xadd(HEARTBEAT_STREAM, {"agent": spec.name, "ts_ms": now_ms(), "load": 0.0, "ok": True})

    def register_callable(self, name: str, capabilities: List[str], fn: Callable[[Dict[str, Any]], Awaitable[Any]],
                          meta: Optional[Dict[str, Any]] = None):
        """Register a local async function as an agent capability."""
        for cap in capabilities:
            self._callables[cap] = fn
        self._spec = AgentSpec(name=name, kind="python", capabilities=capabilities, endpoint=None, meta=meta or {})

    def register_http(self, name: str, capabilities: List[str], url: str, meta: Optional[Dict[str, Any]] = None):
        self._spec = AgentSpec(name=name, kind="http", capabilities=capabilities, endpoint=url, meta=meta or {})

    def register_cli(self, name: str, capabilities: List[str], command: str, meta: Optional[Dict[str, Any]] = None):
        self._spec = AgentSpec(name=name, kind="cli", capabilities=capabilities, endpoint=command, meta=meta or {})

    # -- Core loops -----------------------------------------------------------
    async def run_forever(self):
        assert self._spec, "Call announce()/register_*() first"
        if self.r is None:
            await self.connect()
        self._running = True
        asyncio.create_task(self._heartbeat_loop())
        await self._consume_loop()

    async def _consume_loop(self):
        """Consume tasks, lease and execute matching ones."""
        r = self.r
        last_id = "$"
        while self._running:
            try:
                resp = await r.xread({TASKS_STREAM: last_id}, count=200, block=POLL_BLOCK_MS)  # type: ignore
                if not resp:
                    # periodic cleanup of expired leases (work-steal)
                    await self._steal_expired_leases()
                    continue

                _, entries = resp[0]
                for _id, fields in entries:
                    last_id = _id
                    t = parse_task(fields)
                    if not t:
                        continue
                    if not self._should_accept(t):
                        continue
                    if not await self._lease_task(t.id):
                        continue  # another agent grabbed it

                    # run in background (do not block stream)
                    asyncio.create_task(self._execute_task(t))
            except Exception as e:
                await self._log({"lvl":"error","msg":str(e)})
                await asyncio.sleep(0.5)

    async def _execute_task(self, t: Task):
        start = now_ms()
        try:
            out = await self._dispatch(t)
            status = "ok"
            output = out
        except asyncio.TimeoutError:
            status = "timeout"
            output = {"error": "timeout"}
        except Exception as e:
            status = "error"
            output = {"error": str(e)}

        await self._xadd(RESULTS_STREAM, {
            "task_id": t.id, "status": status, "output": output,
            "meta": {"agent": self.agent_name, "lat_ms": now_ms() - start}
        })
        await self._release_task(t.id)

    async def _dispatch(self, t: Task) -> Any:
        """Route to Python callable / HTTP / CLI per agent kind."""
        assert self._spec
        if self._spec.kind == "python":
            fn = self._callables.get(t.capability)
            if not fn:
                raise RuntimeError(f"capability {t.capability} not registered on {self.agent_name}")
            # Allow simple timeout via params.timeout_ms
            tout = int(t.params.get("timeout_ms", 10000))
            return await asyncio.wait_for(fn(t.params), timeout=tout/1000.0)

        elif self._spec.kind == "http":
            import aiohttp  # type: ignore # lazy import
            url = self._spec.endpoint or ""
            payload = {"capability": t.capability, "params": t.params, "task_id": t.id}
            tout = aiohttp.ClientTimeout(total=t.params.get("timeout_sec", 10))
            async with aiohttp.ClientSession(timeout=tout) as sess:
                async with sess.post(url, json=payload) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}")
                    return await resp.json()

        elif self._spec.kind == "cli":
            # endpoint is a command template, e.g., "python run_tool.py --cap {capability}"
            import shlex, subprocess
            cmd = (self._spec.endpoint or "").format(capability=t.capability)
            proc = await asyncio.create_subprocess_exec(*shlex.split(cmd),
                                                        stdin=asyncio.subprocess.PIPE,
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
            stdin = json.dumps(t.params).encode()
            out, err = await proc.communicate(stdin)
            if proc.returncode != 0:
                raise RuntimeError(err.decode()[:400])
            try:
                return json.loads(out.decode())
            except Exception:
                return out.decode()

        else:
            raise RuntimeError(f"unknown agent kind {self._spec.kind}")

    # -- Leasing --------------------------------------------------------------
    async def _lease_task(self, task_id: str) -> bool:
        """Try to lease; succeed if TTL empty or expired."""
        r = self.r
        now = now_ms()
        raw = await r.hget(LEASES_HASH, task_id)  # type: ignore
        if raw:
            try:
                j = json.loads(raw)
                if now - int(j.get("lease_ts", 0)) < LEASE_MS:
                    return False  # still leased
            except Exception:
                pass
        await r.hset(LEASES_HASH, task_id, json.dumps({"agent": self.agent_name, "lease_ts": now}))  # type: ignore
        return True

    async def _release_task(self, task_id: str):
        try:
            await self.r.hdel(LEASES_HASH, task_id)  # type: ignore
        except Exception:
            pass

    async def _steal_expired_leases(self):
        """Cleanup/steal leases that expired (so idle agents can help)."""
        r = self.r
        now = now_ms()
        try:
            h = await r.hgetall(LEASES_HASH)  # type: ignore
        except Exception:
            return
        stale: List[str] = []
        for k, v in h.items():
            try:
                j = json.loads(v)
                if now - int(j.get("lease_ts", 0)) > LEASE_MS:
                    stale.append(k)
            except Exception:
                stale.append(k)
        if stale:
            await r.hdel(LEASES_HASH, *stale)  # type: ignore

    # -- Heartbeats & selection ----------------------------------------------
    async def _heartbeat_loop(self):
        while self._running:
            try:
                await self._xadd(HEARTBEAT_STREAM, {"agent": self.agent_name, "ts_ms": now_ms(), "ok": True, "load": 0.2})
                if self._spec:
                    await self._hset(REGISTRY_HASH, f"agent:{self._spec.name}", asjson(self._spec))
                await asyncio.sleep(HB_INTERVAL_MS / 1000.0)
            except Exception:
                await asyncio.sleep(1.0)

    def _should_accept(self, t: Task) -> bool:
        if not self._spec:
            return False
        # discard too old
        if t.submit_ts and (now_ms() - t.submit_ts) > t.ttl_ms:
            return False
        # directed tasks
        if t.target and t.target != self.agent_name:
            return False
        # capability match
        return t.capability in (self._spec.capabilities or [])

    # -- Redis helpers --------------------------------------------------------
    async def _xadd(self, stream: str, obj: Dict[str, Any]):
        try:
            await self.r.xadd(stream, {"json": json.dumps(obj)}, maxlen=MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

    async def _hset(self, key: str, field: str, val: str):
        try:
            await self.r.hset(key, field, val)  # type: ignore
        except Exception:
            pass

    async def _log(self, obj: Dict[str, Any]):
        obj = {"ts_ms": now_ms(), "agent": self.agent_name, **obj}
        await self._xadd(LOGS_STREAM, obj)

# -------- Utilities ----------------------------------------------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def asjson(x: Any) -> str:
    return json.dumps(x if isinstance(x, dict) else asdict(x), ensure_ascii=False)

def parse_task(fields: Dict[str, Any]) -> Optional[Task]:
    try:
        j = json.loads(fields.get("json", "{}"))
        return Task(
            id=str(j.get("id") or uuid.uuid4().hex),
            kind=str(j.get("kind") or "rpc"),
            capability=str(j["capability"]),
            params=dict(j.get("params") or {}),
            target=j.get("target"),
            ttl_ms=int(j.get("ttl_ms") or 60000),
            submit_ts=int(j.get("submit_ts") or now_ms()),
        )
    except Exception:
        return None

# -------- Example built-in callable (you can remove) -------------------------
async def demo_summarize(params: Dict[str, Any]) -> Dict[str, Any]:
    """Toy capability: return first N words."""
    text = str(params.get("text") or "")
    n = int(params.get("n", 40))
    words = [w for w in text.split() if w.strip()]
    return {"summary": " ".join(words[:n]), "len": len(words)}

# -------- CLI entrypoints ----------------------------------------------------
async def _amain():
    """
    Run:
      python -m backend.ai.swarm_manager  summarize
    or:
      python -m backend.ai.swarm_manager  http http://localhost:8010/run summarize,rank
    """
    import sys
    agent = f"agent-{os.getenv('HOSTNAME', 'local')}"
    mgr = SwarmManager(agent_name=agent)

    # parse quick mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "summarize"
    if mode == "http":
        url = sys.argv[2]
        caps = sys.argv[3].split(",")
        mgr.register_http(agent, caps, url)
    elif mode == "cli":
        cmd = sys.argv[2]
        caps = sys.argv[3].split(",")
        mgr.register_cli(agent, caps, cmd)
    else:
        # default: local summarize capability
        mgr.register_callable(agent, ["summarize"], demo_summarize)

    await mgr.connect()
    await mgr.announce(mgr._spec)  # type: ignore
    print(f"[swarm] {agent} up with caps={mgr._spec.capabilities}")  # type: ignore
    await mgr.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass