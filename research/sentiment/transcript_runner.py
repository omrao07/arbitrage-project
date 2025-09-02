# services/transcript_runner.py
"""
Transcript Runner:
- Scans configured directory (or API endpoint) for new transcript files
- Cleans and segments into paragraphs
- Publishes sentiment jobs to STREAM_SENTIMENT_REQUESTS
- Deduplicates by hash in Redis
"""

from __future__ import annotations

import glob
import hashlib
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from platform import bootstrap # type: ignore
from platform import envelope as env # type: ignore

SERVICE = "transcript-runner"

# Streams
OUT_STREAM = os.getenv("OUT_STREAM", "STREAM_SENTIMENT_REQUESTS")

# Config
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR", "data/transcripts")
POLL_SECONDS = int(os.getenv("TRANSCRIPT_POLL_SECONDS", "120"))
BATCH_MAX = int(os.getenv("TRANSCRIPT_BATCH_MAX", "20"))
LANG = os.getenv("SENTIMENT_LANG", "en")


@dataclass(frozen=True)
class TranscriptDoc:
    path: str
    text: str
    symbol: Optional[str] = None
    meta: Optional[dict] = None


def _file_fingerprint(path: str, text: str) -> str:
    return hashlib.sha256((path + "::" + text[:200]).encode("utf-8")).hexdigest()


def _split_paragraphs(text: str, min_len: int = 30) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [p for p in paras if len(p) >= min_len]


def _load_transcripts(dir_path: str) -> List[TranscriptDoc]:
    docs: List[TranscriptDoc] = []
    for file in glob.glob(os.path.join(dir_path, "*.txt")):
        try:
            raw = Path(file).read_text(encoding="utf-8")
            docs.append(TranscriptDoc(path=file, text=raw))
        except Exception:
            continue
    return docs


class TranscriptRunner:
    def __init__(self, directory: str):
        self.ctx = bootstrap.init(SERVICE)
        self.tracer = self.ctx["tracer"]
        self.METRICS = self.ctx["metrics"]
        self.r = self.ctx["redis"]
        self.audit = self.ctx["audit"]

        self.dir = directory
        self._stop = False

        # Deduplication in Redis
        self.hash_prefix = os.getenv("TRANSCRIPT_HASH_PREFIX", "transcript:seen:")
        self.hash_ttl = int(os.getenv("TRANSCRIPT_HASH_TTL", "604800"))  # 7 days

    def _seen(self, fp: str) -> bool:
        key = self.hash_prefix + fp
        try:
            if self.r.setnx(key, 1):
                self.r.expire(key, self.hash_ttl)
                return False
            return True
        except Exception:
            return False

    def _publish_sentiment(self, paras: List[str], meta: dict) -> int:
        if not paras:
            return 0
        e = env.new(
            schema_name="sentiment.request",
            payload={
                "texts": paras[:BATCH_MAX],
                "language": LANG,
                "meta": meta,
            },
            producer={"svc": SERVICE, "roles": ["research"]},
        )
        self.r.xadd(OUT_STREAM, e.flatten_for_stream())
        return len(paras[:BATCH_MAX])

    def run_once(self) -> int:
        total_sent = 0
        docs = _load_transcripts(self.dir)
        for d in docs:
            fp = _file_fingerprint(d.path, d.text)
            if self._seen(fp):
                continue

            paras = _split_paragraphs(d.text)
            if not paras:
                continue

            sent = self._publish_sentiment(paras, meta={"path": d.path, "symbol": d.symbol})
            total_sent += sent

            self.audit.record(
                action="transcript_publish",
                resource="sentiment/analyze",
                user=None,
                corr_id=None,
                region=os.getenv("REGION", "US"),
                policy_hash=os.getenv("POLICY_HASH"),
                details={"file": d.path, "n_paras": len(paras), "sent_to_sentiment": sent},
                input_for_hash={"path": d.path, "n": len(paras)},
            )

        return total_sent

    def run_forever(self):
        log = __import__("logging").getLogger(SERVICE)
        log.info("Starting transcript runner, dir=%s, poll=%ss", self.dir, POLL_SECONDS)
        while not self._stop:
            try:
                n = self.run_once()
                log.info("Cycle complete: published=%d", n)
            except Exception:
                log.exception("Error in run_once")
            for _ in range(POLL_SECONDS):
                if self._stop:
                    break
                time.sleep(1)

    def stop(self, *_):
        self._stop = True


def main():
    tr = TranscriptRunner(TRANSCRIPTS_DIR)
    signal.signal(signal.SIGINT, tr.stop)
    signal.signal(signal.SIGTERM, tr.stop)
    tr.run_forever()


if __name__ == "__main__":
    main()