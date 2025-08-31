# backend/telemetry/session_recorder.py
from __future__ import annotations

import os, sys, json, time, asyncio, signal, shutil, platform, subprocess
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Iterable
from contextlib import contextmanager

# -------- Optional deps (graceful) -------------------------------------------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

HAVE_PSUTIL = True
try:
    import psutil  # type: ignore
except Exception:
    HAVE_PSUTIL = False
    psutil = None  # type: ignore

# -------- Defaults ------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_STREAMS = os.getenv("RECORDER_STREAMS", "prices.bars,orders.incoming,orders.filled,orders.rejected,positions.snapshots,features.alt.news").split(",")

def now_ms() -> int: return int(time.time() * 1000)

# -------- Filesystem helpers -------------------------------------------------
def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------- Stdout/Stderr tee --------------------------------------------------
class _TeeIO:
    def __init__(self, real, path: str):
        self.real = real
        self.f = open(path, "a", buffering=1, encoding="utf-8")
    def write(self, s):
        try:
            self.real.write(s)
        except Exception:
            pass
        try:
            self.f.write(s)
        except Exception:
            pass
    def flush(self):
        try: self.real.flush()
        except Exception: pass
        try: self.f.flush()
        except Exception: pass
    def close(self):
        try: self.f.close()
        except Exception: pass

@contextmanager
def tee_stdio(out_path: str, err_path: Optional[str] = None):
    err_path = err_path or out_path
    old_out, old_err = sys.stdout, sys.stderr
    tee_out = _TeeIO(sys.stdout, out_path)
    tee_err = _TeeIO(sys.stderr, err_path)
    sys.stdout, sys.stderr = tee_out, tee_err
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout, sys.stderr = old_out, old_err
        tee_out.close(); tee_err.close()

# -------- Datatypes ----------------------------------------------------------
@dataclass
class RecorderLimits:
    duration_sec: Optional[int] = None     # stop after N seconds
    max_events: Optional[int] = None       # stop after total N events across streams

@dataclass
class RecorderConfig:
    run_id: str
    out_dir: str = "artifacts/sessions"
    streams: List[str] = field(default_factory=lambda: [s.strip() for s in DEFAULT_STREAMS if s.strip()])
    redis_url: str = REDIS_URL
    from_beginning: bool = False           # xread from "0-0" if True, else "$"
    attach_stdio: bool = True              # tee stdout/stderr to logs
    save_env: bool = True                  # include process env in manifest
    save_git: bool = True                  # include git hash if available
    tags: Dict[str, Any] = field(default_factory=dict)
    extra_meta: Dict[str, Any] = field(default_factory=dict)
    limits: RecorderLimits = field(default_factory=RecorderLimits)

# -------- Recorder -----------------------------------------------------------
class SessionRecorder:
    """
    Captures Redis streams into JSONL files + manifest + logs.
    """
    def __init__(self, cfg: RecorderConfig):
        self.cfg = cfg
        self.root = os.path.join(cfg.out_dir, cfg.run_id)
        self.dir_streams = os.path.join(self.root, "streams")
        self.dir_logs = os.path.join(self.root, "logs")
        self.dir_meta = os.path.join(self.root, "meta")
        self.dir_custom = os.path.join(self.root, "custom")
        for d in (self.root, self.dir_streams, self.dir_logs, self.dir_meta, self.dir_custom):
            safe_mkdir(d)

        self.r: Optional[AsyncRedis] = None # type: ignore
        self._running = False
        self._counts: Dict[str, int] = {}
        self._total_events = 0
        self._last_ids: Dict[str, str] = {}

        # files
        self.path_manifest = os.path.join(self.root, "manifest.json")
        self.path_stdout = os.path.join(self.dir_logs, "stdout.log")
        self.path_stderr = os.path.join(self.dir_logs, "stderr.log")
        self.path_events = os.path.join(self.dir_custom, "events.jsonl")

    # ---- Manifest -----------------------------------------------------------
    def _collect_manifest(self) -> Dict[str, Any]:
        man: Dict[str, Any] = {
            "run_id": self.cfg.run_id,
            "created_ts_ms": now_ms(),
            "streams": self.cfg.streams,
            "redis_url": self.cfg.redis_url,
            "from_beginning": self.cfg.from_beginning,
            "tags": self.cfg.tags,
            "extra": self.cfg.extra_meta,
            "limits": asdict(self.cfg.limits),
            "system": {
                "platform": platform.platform(),
                "python": sys.version,
                "pid": os.getpid(),
            },
        }
        if HAVE_PSUTIL:
            try:
                p = psutil.Process() # type: ignore
                man["system"]["cpu_count"] = psutil.cpu_count(logical=True) # type: ignore
                man["system"]["mem_total_bytes"] = psutil.virtual_memory().total # type: ignore
                man["system"]["cmdline"] = p.cmdline()
            except Exception:
                pass
        if self.cfg.save_env:
            # Avoid dumping secrets wholesale — keep small and filter obvious secrets
            try:
                env = {k: ("***" if any(s in k.lower() for s in ("secret","token","key","passwd","password")) else v) 
                       for k,v in os.environ.items()}
                man["env"] = env
            except Exception:
                pass
        if self.cfg.save_git:
            try:
                sha = subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
                branch = subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
                man["git"] = {"commit": sha, "branch": branch, "dirty": bool(subprocess.call(["git","diff","--quiet"])) == False}
            except Exception:
                man["git"] = None
        return man

    def _write_manifest(self, status: str = "starting"):
        man = self._collect_manifest()
        man["status"] = status
        man["counts"] = dict(self._counts)
        man["total_events"] = self._total_events
        man["updated_ts_ms"] = now_ms()
        with open(self.path_manifest, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)

    # ---- Public API ---------------------------------------------------------
    async def start(self):
        self._running = True
        # init last ids
        start_id = "0-0" if self.cfg.from_beginning else "$"
        self._last_ids = {s: start_id for s in self.cfg.streams}

        # connect redis
        if HAVE_REDIS:
            try:
                self.r = AsyncRedis.from_url(self.cfg.redis_url, decode_responses=True)  # type: ignore
                await self.r.ping() # type: ignore
            except Exception:
                self.r = None
        self._write_manifest("starting")

        # handle signals for graceful stop
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.stop(reason=f"signal:{s.name}")))
        except Exception:
            pass

        # launch tasks
        tasks = []
        if self.r:
            tasks.append(asyncio.create_task(self._pump_streams()))
        if self.cfg.limits.duration_sec:
            tasks.append(asyncio.create_task(self._timer_stop(self.cfg.limits.duration_sec)))
        await asyncio.gather(*tasks) if tasks else asyncio.sleep(0)

    async def stop(self, reason: str = "requested"):
        if not self._running:
            return
        self._running = False
        # write final manifest
        self._write_manifest(status=f"stopped:{reason}")

    async def record_custom(self, topic: str, payload: Dict[str, Any]):
        """
        Write a custom event (your code can call this).
        """
        obj = {"ts_ms": now_ms(), "topic": topic, "payload": payload}
        jsonl_append(self.path_events, obj)

    # ---- Internals ----------------------------------------------------------
    async def _timer_stop(self, sec: int):
        await asyncio.sleep(max(0, int(sec)))
        await self.stop(reason="duration")

    async def _pump_streams(self):
        assert self.r is not None
        # open file handles per stream (append mode)
        writers: Dict[str, str] = {}
        try:
            for s in self.cfg.streams:
                writers[s] = os.path.join(self.dir_streams, f"{_safe_stream_name(s)}.jsonl")
            while self._running:
                try:
                    resp = await self.r.xread(self._last_ids, count=500, block=1000)  # type: ignore
                except Exception:
                    resp = []
                if not resp:
                    continue
                for stream, entries in resp:
                    self._last_ids[stream] = entries[-1][0]
                    wpath = writers.get(stream) or os.path.join(self.dir_streams, f"{_safe_stream_name(stream)}.jsonl")
                    for _id, fields in entries:
                        obj: Dict[str, Any] = {"_id": _id, "stream": stream, "ts_recv_ms": now_ms()}
                        # best-effort parse "json" field; otherwise dump full fields
                        raw = fields.get("json")
                        if raw:
                            try:
                                obj["data"] = json.loads(raw)
                            except Exception:
                                obj["data"] = {"json": raw}
                        else:
                            obj["data"] = fields
                        jsonl_append(wpath, obj)
                        # counts
                        self._counts[stream] = self._counts.get(stream, 0) + 1
                        self._total_events += 1

                        # stop if over max_events
                        if self.cfg.limits.max_events and self._total_events >= self.cfg.limits.max_events:
                            await self.stop(reason="max_events")
                            return
                # refresh manifest occasionally
                if (self._total_events % 1000) == 1:
                    self._write_manifest(status="running")
        finally:
            self._write_manifest(status="stopped:done")

# -------- Helpers ------------------------------------------------------------
def _safe_stream_name(s: str) -> str:
    return s.replace(":", "_").replace(".", "_").replace("/", "_")

# -------- High-level convenience --------------------------------------------
@contextmanager
def record_session(cfg: RecorderConfig):
    """
    Context manager that tees stdout/stderr (if enabled) and runs the recorder until exit.
    """
    rec = SessionRecorder(cfg)
    if cfg.attach_stdio:
        ctx = tee_stdio(rec.path_stdout, rec.path_stderr)
    else:
        @contextmanager
        def _noop():
            yield
        ctx = _noop()

    async def _run():
        await rec.start()

    try:
        with ctx:
            # run the recorder concurrently with user code when used as context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            t = loop.create_task(_run())
            yield rec  # user can call rec.record_custom(...) inside
            # when leaving the context, stop
            loop.run_until_complete(rec.stop("context_exit"))
            # ensure task completes
            loop.run_until_complete(asyncio.sleep(0.05))
            t.cancel()
            with contextmanager(lambda: (yield))():
                pass
    finally:
        try:
            loop.close()
        except Exception:
            pass

# -------- CLI ----------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("session_recorder")
    ap.add_argument("--run-id", required=True, help="Unique ID (e.g., 2025-08-28T12-00Z_demo)")
    ap.add_argument("--streams", type=str, default=",".join(DEFAULT_STREAMS), help="Comma-separated Redis streams")
    ap.add_argument("--redis-url", type=str, default=REDIS_URL)
    ap.add_argument("--from-beginning", action="store_true", help="Start reading from 0-0")
    ap.add_argument("--out", type=str, default="artifacts/sessions")
    ap.add_argument("--no-stdio", action="store_true", help="Do not tee stdout/stderr")
    ap.add_argument("--duration-sec", type=int, default=None, help="Stop after N seconds")
    ap.add_argument("--max-events", type=int, default=None, help="Stop after total N events")
    ap.add_argument("--tag", action="append", help="key=value tag (repeatable)")
    args = ap.parse_args()

    tags: Dict[str, Any] = {}
    if args.tag:
        for kv in args.tag:
            if "=" in kv:
                k, v = kv.split("=", 1)
                tags[k] = v

    cfg = RecorderConfig(
        run_id=args.run_id,
        out_dir=args.out,
        streams=[s.strip() for s in args.streams.split(",") if s.strip()],
        redis_url=args.redis_url,
        from_beginning=bool(args.from_beginning),
        attach_stdio=not args.no_stdio,
        tags=tags,
        limits=RecorderLimits(duration_sec=args.duration_sec, max_events=args.max_events),
    )

    async def _run():
        rec = SessionRecorder(cfg)
        # Optionally tee stdio here too (CLI path)
        ctx = tee_stdio(rec.path_stdout, rec.path_stderr) if cfg.attach_stdio else contextmanager(lambda: (yield))()
        with ctx:
            await rec.start()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()