# backend/sim/replay_with_params.py
from __future__ import annotations
import os, csv, json, time, asyncio, random, math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---- optional deps (graceful) ----------------------------------------------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

HAVE_YAML = True
try:
    import yaml  # type: ignore
except Exception:
    HAVE_YAML = False

# ---- env / streams ----------------------------------------------------------
REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379/0")
S_BARS      = os.getenv("PRICES_STREAM", "prices.bars")
S_NEWS      = os.getenv("NEWS_STREAM", "features.alt.news")
S_POS       = os.getenv("POS_SNAPSHOTS", "positions.snapshots")  # optional placeholder
S_ALERTS    = os.getenv("ALERTS_STREAM", "alerts.events")
MAXLEN      = int(os.getenv("REPLAY_MAXLEN", "10000"))

def now_ms() -> int: return int(time.time() * 1000)

# ---- utilities --------------------------------------------------------------
def load_any(path: str) -> List[Dict[str, Any]]:
    if path is None: return []
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    # CSV
    rows = []
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def to_int(x: Any, default: int = 0) -> int:
    try: return int(float(x))
    except Exception: return default

def to_float(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

# ---- publisher --------------------------------------------------------------
class Bus:
    def __init__(self, url: str = REDIS_URL):
        self.url = url
        self.r: Optional[AsyncRedis] = None # type: ignore

    async def connect(self):
        if not HAVE_REDIS: return
        try:
            self.r = AsyncRedis.from_url(self.url, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def xadd(self, stream: str, obj: Dict[str, Any]):
        if not self.r:
            # graceful fallback
            print(f"[{stream}] {json.dumps(obj, ensure_ascii=False)[:200]}")
            return
        try:
            await self.r.xadd(stream, {"json": json.dumps(obj, ensure_ascii=False)}, maxlen=MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

# ---- config dataclasses -----------------------------------------------------
@dataclass
class RunParams:
    run_id: str
    # pacing
    speed: float = 1.0           # 1.0 = real time (based on bar timestamp gaps), 0 = firehose, 2.0 = 2x faster (half sleeps)
    min_sleep_ms: int = 0        # lower bound per step
    jitter_ms: int = 0
    # filters & transforms
    symbols: Optional[List[str]] = None       # whitelist (upper)
    start_ts_ms: Optional[int] = None
    end_ts_ms: Optional[int] = None
    price_mult: float = 1.0                   # multiply all prices (what-if)
    price_offset: float = 0.0                 # add offset to prices (what-if)
    # chaos
    drop_prob: float = 0.0                    # chance to drop an event
    dupe_prob: float = 0.0                    # chance to duplicate an event
    # relabeling
    tag: Optional[str] = None                 # extra tag for dashboards

@dataclass
class ReplayConfig:
    bars_path: str
    news_path: Optional[str] = None
    align_by_symbol: bool = True
    runs: List[RunParams] = field(default_factory=list)

# ---- core replayer ----------------------------------------------------------
class Replayer:
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.bus = Bus()

    async def run_all(self):
        await self.bus.connect()
        # Preload data
        raw_bars = load_any(self.cfg.bars_path)
        bars = []
        for r in raw_bars:
            ts = to_int(r.get("ts_ms") or r.get("timestamp") or r.get("time"))
            sym = str(r.get("symbol") or r.get("sym") or "").upper()
            if not ts or not sym: continue
            bars.append({
                "ts_ms": ts,
                "symbol": sym,
                "open":  to_float(r.get("open")  or r.get("o") or r.get("close")),
                "high":  to_float(r.get("high")  or r.get("h") or r.get("close")),
                "low":   to_float(r.get("low")   or r.get("l") or r.get("close")),
                "close": to_float(r.get("close") or r.get("c") or r.get("price") or r.get("px")),
                "volume":to_float(r.get("volume") or r.get("v") or 0.0),
            })
        bars.sort(key=lambda x: (x["ts_ms"], x["symbol"]))

        news = []
        if self.cfg.news_path:
            for r in load_any(self.cfg.news_path):
                ts = to_int(r.get("ts_ms") or r.get("timestamp") or r.get("time"))
                if not ts: continue
                news.append({
                    "ts_ms": ts,
                    "title": str(r.get("title") or r.get("text") or ""),
                    "symbols": [str(s).upper() for s in (r.get("symbols") or r.get("tickers") or [])],
                    "score": r.get("score") if r.get("score") is not None else r.get("sentiment"),
                    "source": r.get("source"),
                    "url": r.get("url"),
                })
            news.sort(key=lambda x: x["ts_ms"])

        # Run each param set sequentially (easy to parallelize if you want)
        for rp in self.cfg.runs:
            print(f"[replay] starting run: {rp.run_id}")
            await self._run_one(bars, news, rp)
            print(f"[replay] finished run: {rp.run_id}")

    async def _run_one(self, bars: List[Dict[str, Any]], news: List[Dict[str, Any]], rp: RunParams):
        # Filter bars/news
        def bar_ok(b):
            if rp.symbols and b["symbol"] not in rp.symbols: return False
            if rp.start_ts_ms and b["ts_ms"] < rp.start_ts_ms: return False
            if rp.end_ts_ms   and b["ts_ms"] > rp.end_ts_ms:   return False
            return True

        def news_ok(n):
            if rp.start_ts_ms and n["ts_ms"] < rp.start_ts_ms: return False
            if rp.end_ts_ms   and n["ts_ms"] > rp.end_ts_ms:   return False
            if rp.symbols:
                if not n["symbols"]: return False
                if not any(s in rp.symbols for s in n["symbols"]): return False
            return True

        bseq = [b for b in bars if bar_ok(b)]
        nseq = [n for n in news if news_ok(n)]
        i_news = 0

        # Compute pacing from timestamps
        # If speed==0 -> no sleep; else sleep scaled by (Δt / speed)
        prev_ts = bseq[0]["ts_ms"] if bseq else None

        for b in bseq:
            # pacing sleep
            if prev_ts is not None and rp.speed > 0:
                dt_ms = max(0, b["ts_ms"] - prev_ts)
                sleep_ms = int(dt_ms / max(1e-9, rp.speed))
                sleep_ms = max(rp.min_sleep_ms, sleep_ms)
                if rp.jitter_ms:
                    sleep_ms = max(0, sleep_ms + random.randint(-rp.jitter_ms, rp.jitter_ms))
                if sleep_ms > 0:
                    await asyncio.sleep(sleep_ms / 1000.0)
            prev_ts = b["ts_ms"]

            # transform prices
            b2 = dict(b)
            for k in ("open","high","low","close"):
                b2[k] = b2[k] * rp.price_mult + rp.price_offset

            # chaos: drop/dupe
            if random.random() < rp.drop_prob:
                pass  # drop
            else:
                obj = {**b2, "run_id": rp.run_id}
                if rp.tag: obj["tag"] = rp.tag
                await self.bus.xadd(S_BARS, obj)
                if random.random() < rp.dupe_prob:
                    await self.bus.xadd(S_BARS, obj)

            # flush news up to bar ts
            while i_news < len(nseq) and nseq[i_news]["ts_ms"] <= b["ts_ms"]:
                n = dict(nseq[i_news]); i_news += 1
                if random.random() < rp.drop_prob:
                    continue
                n["run_id"] = rp.run_id
                if rp.tag: n["tag"] = rp.tag
                await self.bus.xadd(S_NEWS, n)
                if random.random() < rp.dupe_prob:
                    await self.bus.xadd(S_NEWS, n)

        # tail any remaining news
        while i_news < len(nseq):
            n = dict(nseq[i_news]); i_news += 1
            if random.random() < rp.drop_prob:
                continue
            n["run_id"] = rp.run_id
            if rp.tag: n["tag"] = rp.tag
            await self.bus.xadd(S_NEWS, n)

# ---- loaders for params file -----------------------------------------------
def load_params(path: str) -> ReplayConfig:
    if path.endswith((".yml",".yaml")):
        if not HAVE_YAML:
            raise RuntimeError("PyYAML not installed but YAML params given.")
        with open(path, "r") as f:
            conf = yaml.safe_load(f) or {}
    else:
        with open(path, "r") as f:
            conf = json.load(f)

    runs = []
    for r in conf.get("runs", []):
        runs.append(RunParams(
            run_id = str(r.get("run_id")),
            speed = float(r.get("speed", 1.0)),
            min_sleep_ms = int(r.get("min_sleep_ms", 0)),
            jitter_ms = int(r.get("jitter_ms", 0)),
            symbols = [s.upper() for s in (r.get("symbols") or [])] or None,
            start_ts_ms = r.get("start_ts_ms"),
            end_ts_ms = r.get("end_ts_ms"),
            price_mult = float(r.get("price_mult", 1.0)),
            price_offset = float(r.get("price_offset", 0.0)),
            drop_prob = float(r.get("drop_prob", 0.0)),
            dupe_prob = float(r.get("dupe_prob", 0.0)),
            tag = r.get("tag")
        ))

    return ReplayConfig(
        bars_path = conf["bars_path"],
        news_path = conf.get("news_path"),
        align_by_symbol = bool(conf.get("align_by_symbol", True)),
        runs = runs
    )

# ---- CLI --------------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("replay_with_params")
    ap.add_argument("--params", type=str, required=True, help="YAML/JSON describing bars/news and runs")
    args = ap.parse_args()

    cfg = load_params(args.params)

    async def _run():
        rp = Replayer(cfg)
        await rp.run_all()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()