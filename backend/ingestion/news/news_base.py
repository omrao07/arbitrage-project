# backend/ingestion/news/news_base.py
from __future__ import annotations

import asyncio
import hashlib
import html
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Union


# ---------- Event model ----------

@dataclass(slots=True)
class NewsEvent:
    id: str                 # stable unique id (hash)
    source: str             # e.g., "yahoo", "moneycontrol"
    headline: str
    url: str
    published_at: float     # unix seconds
    summary: str = ""
    symbol: Optional[str] = None   # optional: "RELIANCE.NS", "AAPL", etc.
    raw: Optional[Dict[str, Any]] = None  # original payload (optional)


# ---------- Helpers ----------

def hash_key(*parts: str) -> str:
    s = "||".join(parts)
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return html.unescape(" ".join(s.split()))

def to_unix(ts: Union[int, float, None]) -> float:
    if ts is None:
        return time.time()
    try:
        f = float(ts)
        # if it looks like ms, convert
        if f > 10_000_000_000:
            return f / 1000.0
        return f
    except Exception:
        return time.time()


# ---------- Dedupe ----------

class DedupeCache:
    """
    Keeps a moving window of seen event ids.
    Usage:
        cache = DedupeCache(ttl_seconds=7200)
        fresh = cache.filter(events)
    """
    def __init__(self, ttl_seconds: int = 7200):
        self.ttl = int(ttl_seconds)
        self._seen: Dict[str, float] = {}

    def _sweep(self, now: float) -> None:
        # keep only recent ids
        self._seen = {k: v for k, v in self._seen.items() if (now - v) < self.ttl}

    def filter(self, events: Iterable[NewsEvent]) -> List[NewsEvent]:
        now = time.time()
        self._sweep(now)
        out: List[NewsEvent] = []
        for e in events:
            if e.id in self._seen:
                continue
            self._seen[e.id] = now
            out.append(e)
        return out


# ---------- Push interfaces ----------

class NewsSink(Protocol):
    def __call__(self, event: NewsEvent) -> None: ...


# ---------- Base class for sources ----------

class NewsSource:
    """
    Base class you subclass for each provider.
    Implement ONE of:
        - poll_sync() -> List[NewsEvent]
        - async poll_async() -> List[NewsEvent]

    Then call:
        await source.run(sink=queue.put_nowait, interval=60, stop=stop_event)

    Where `sink` is either:
      - an asyncio.Queue().put_nowait
      - a callable: sink(event)
    """
    source_name: str = "base"

    def __init__(self, dedupe_ttl_seconds: int = 7200):
        self._dedupe = DedupeCache(ttl_seconds=dedupe_ttl_seconds)

    # ---- override one of these in subclasses ----
    def poll_sync(self) -> List[NewsEvent]:
        raise NotImplementedError  # optional to implement

    async def poll_async(self) -> List[NewsEvent]:
        # default path calls sync version in a thread
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.poll_sync)

    # ---- public runner ----
    async def run(
        self,
        sink: Callable[[NewsEvent], None] | Callable[[NewsEvent], Any],
        interval: int = 60,
        stop: Optional[asyncio.Event] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Periodically polls the source, dedupes, and pushes events to sink.
        - sink: function or queue.put_nowait
        - interval: seconds between polls
        - stop: asyncio.Event to end loop
        """
        if stop is None:
            stop = asyncio.Event()

        def _log(msg: str):
            if log:
                log(f"[news:{self.source_name}] {msg}")

        _log(f"runner started (interval={interval}s)")

        try:
            while not stop.is_set():
                try:
                    events = await self.poll_async()
                    events = self._dedupe.filter(sorted(events, key=lambda e: e.published_at, reverse=True))
                    for e in events:
                        try:
                            sink(e)
                        except Exception as push_err:
                            _log(f"sink error: {push_err}")
                except Exception as e:
                    _log(f"poll error: {e}")

                try:
                    await asyncio.wait_for(stop.wait(), timeout=max(1, int(interval)))
                except asyncio.TimeoutError:
                    pass
        finally:
            _log("runner stopped")