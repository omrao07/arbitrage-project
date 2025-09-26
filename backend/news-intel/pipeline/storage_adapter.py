# news-intel/pipeline/storage_adapter.py
"""
Unified storage adapter for enriched news rows.

Backends
- jsonl: append JSON lines to a local file (optionally gzip)
- sqlite: local DB with UPSERT by id
- postgres: UPSERT by id (requires psycopg2)
- s3: append-like puts (writes newline-delimited JSON object; requires boto3)
- kafka: publish to a topic (requires confluent-kafka)

API
----
cfg = StorageConfig(kind="sqlite", dsn="file:news.sqlite")
store = StorageAdapter(cfg)
store.write_many(rows)
store.close()

Each `row` is a dict produced by pipeline_runner (has keys like id, url, title, topics, sentiment, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import contextlib
import gzip
import io
import json
import os
import sqlite3
import time

# Optional deps
try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
except Exception:
    psycopg2 = None  # type: ignore

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None  # type: ignore

try:
    from confluent_kafka import Producer  # type: ignore
except Exception:
    Producer = None  # type: ignore


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class StorageConfig:
    kind: str  # "jsonl" | "sqlite" | "postgres" | "s3" | "kafka"
    # common
    batch_size: int = 500
    # jsonl
    path: Optional[str] = None
    gzip: bool = False
    # sqlite
    dsn: Optional[str] = None  # file path for sqlite or 'file:...'
    # postgres
    pg_dsn: Optional[str] = None  # e.g. "postgresql://user:pass@host:5432/dbname"
    pg_table: str = "articles"
    # s3
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None          # e.g. "news/enriched-%Y%m%d.jsonl"
    s3_region: Optional[str] = None
    s3_gzip: bool = False
    # kafka
    kafka_bootstrap: Optional[str] = None
    kafka_topic: Optional[str] = None
    kafka_config: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------

class StorageAdapter:
    def __init__(self, cfg: StorageConfig):
        self.cfg = cfg
        self._kind = cfg.kind.lower()
        self._buf: List[str] = []

        if self._kind == "jsonl":
            if not cfg.path:
                raise ValueError("jsonl backend requires cfg.path")
            os.makedirs(os.path.dirname(cfg.path) or ".", exist_ok=True)
            # open lazily on first flush

        elif self._kind == "sqlite":
            dsn = cfg.dsn or "news.sqlite"
            uri = dsn.startswith("file:")
            self._sq = sqlite3.connect(dsn, uri=uri, check_same_thread=False)
            self._sq.execute("PRAGMA journal_mode=WAL;")
            self._ensure_sqlite_schema()

        elif self._kind == "postgres":
            if psycopg2 is None:
                raise RuntimeError("postgres backend requires psycopg2")
            if not cfg.pg_dsn:
                raise ValueError("postgres backend requires cfg.pg_dsn")
            self._pg = psycopg2.connect(cfg.pg_dsn)
            self._pg.autocommit = True
            self._ensure_postgres_schema()

        elif self._kind == "s3":
            if boto3 is None:
                raise RuntimeError("s3 backend requires boto3")
            if not cfg.s3_bucket or not cfg.s3_key:
                raise ValueError("s3 backend requires s3_bucket and s3_key")
            self._s3 = boto3.client("s3", region_name=cfg.s3_region)

        elif self._kind == "kafka":
            if Producer is None:
                raise RuntimeError("kafka backend requires confluent-kafka")
            if not cfg.kafka_bootstrap or not cfg.kafka_topic:
                raise ValueError("kafka backend requires kafka_bootstrap and kafka_topic")
            conf = {"bootstrap.servers": cfg.kafka_bootstrap}
            conf.update(cfg.kafka_config or {})
            self._kprod = Producer(conf)

        else:
            raise ValueError(f"unknown storage kind: {cfg.kind}")

    # ---------------------- Public API ----------------------

    def write_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        if self._kind in ("jsonl", "s3", "kafka"):
            for r in rows:
                line = json.dumps(r, ensure_ascii=False)
                self._buf.append(line)
                if len(self._buf) >= self.cfg.batch_size:
                    self._flush_streaming()
            # leave remaining buffered lines for next call (or explicit close)
            return

        if self._kind == "sqlite":
            self._sqlite_upsert(rows)
            return

        if self._kind == "postgres":
            self._postgres_upsert(rows)
            return

    def close(self) -> None:
        # flush streaming buffers
        if self._kind in ("jsonl", "s3", "kafka"):
            self._flush_streaming(final=True)

        # close DBs
        with contextlib.suppress(Exception):
            if getattr(self, "_sq", None):
                self._sq.commit()
                self._sq.close()
        with contextlib.suppress(Exception):
            if getattr(self, "_pg", None):
                self._pg.close()
        with contextlib.suppress(Exception):
            if getattr(self, "_kprod", None):
                self._kprod.flush(5.0)

    # ---------------------- Streaming backends ----------------------

    def _flush_streaming(self, final: bool = False) -> None:
        if not self._buf:
            return
        lines = self._buf
        self._buf = []

        if self._kind == "jsonl":
            mode = "ab" if self.cfg.gzip else "a"
            if self.cfg.gzip:
                # append gzip block; many tools can read concatenated gzip members
                with gzip.open(self.cfg.path, mode) as f:  # type: ignore[arg-type]
                    f.write(("\n".join(lines) + "\n").encode("utf-8"))
            else:
                with open(self.cfg.path, mode, encoding=None if self.cfg.gzip else "utf-8") as f:  # type: ignore[arg-type]
                    data = ("\n".join(lines) + "\n").encode("utf-8")
                    f.write(data)  # type: ignore[arg-type]

        elif self._kind == "s3":
            # Write a chunk as a new object part (simple: put new object with timestamp suffix),
            # or append to the same key by reading+append (not great). We'll use time-sliced keys.
            key = time.strftime(self.cfg.s3_key)  # type: ignore # supports %Y%m%d etc.
            payload = "\n".join(lines) + "\n"
            if self.cfg.s3_gzip:
                bio = io.BytesIO()
                with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
                    gz.write(payload.encode("utf-8"))
                body = bio.getvalue()
                self._s3.put_object(Bucket=self.cfg.s3_bucket, Key=key, Body=body, ContentEncoding="gzip", ContentType="application/json")
            else:
                self._s3.put_object(Bucket=self.cfg.s3_bucket, Key=key, Body=payload.encode("utf-8"), ContentType="application/json")

        elif self._kind == "kafka":
            topic = self.cfg.kafka_topic or ""
            for ln in lines:
                self._kprod.produce(topic, value=ln.encode("utf-8"))
            # throttle large batches gently
            self._kprod.poll(0)

    # ---------------------- SQLite ----------------------

    def _ensure_sqlite_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            url TEXT,
            published_at TEXT,
            source TEXT,
            enriched_at TEXT,
            payload JSON
        );
        CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
        CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
        """
        cur = self._sq.cursor()
        cur.executescript(ddl)
        self._sq.commit()

    def _sqlite_upsert(self, rows: Iterable[Dict[str, Any]]) -> None:
        cur = self._sq.cursor()
        data = []
        for r in rows:
            rid = str(r.get("id") or "")
            url = str(r.get("url") or "")
            pub = str(r.get("published_at") or "")
            src = str(r.get("source") or "")
            enr = str(r.get("enriched_at") or "")
            payload = json.dumps(r, ensure_ascii=False)
            data.append((rid, url, pub, src, enr, payload))
            if len(data) >= self.cfg.batch_size:
                self._sqlite_upsert_batch(cur, data)
                data = []
        if data:
            self._sqlite_upsert_batch(cur, data)
        self._sq.commit()

    @staticmethod
    def _sqlite_upsert_batch(cur: sqlite3.Cursor, rows: List[tuple]) -> None:
        cur.executemany(
            """
            INSERT INTO articles (id, url, published_at, source, enriched_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                url=excluded.url,
                published_at=excluded.published_at,
                source=excluded.source,
                enriched_at=excluded.enriched_at,
                payload=excluded.payload
            """,
            rows,
        )

    # ---------------------- Postgres ----------------------

    def _ensure_postgres_schema(self) -> None:
        table = self.cfg.pg_table
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            url TEXT,
            published_at TIMESTAMPTZ NULL,
            source TEXT,
            enriched_at TIMESTAMPTZ NULL,
            payload JSONB
        );
        CREATE INDEX IF NOT EXISTS {table}_pub_idx ON {table} (published_at);
        CREATE INDEX IF NOT EXISTS {table}_src_idx ON {table} (source);
        """
        with self._pg.cursor() as cur:
            cur.execute(ddl)

    def _postgres_upsert(self, rows: Iterable[Dict[str, Any]]) -> None:
        table = self.cfg.pg_table
        batch: List[tuple] = []
        for r in rows:
            rid = str(r.get("id") or "")
            url = str(r.get("url") or "")
            pub = r.get("published_at") or None
            src = str(r.get("source") or "")
            enr = r.get("enriched_at") or None
            payload = json.dumps(r, ensure_ascii=False)
            batch.append((rid, url, pub, src, enr, payload))
            if len(batch) >= self.cfg.batch_size:
                self._postgres_upsert_batch(table, batch)
                batch = []
        if batch:
            self._postgres_upsert_batch(table, batch)

    def _postgres_upsert_batch(self, table: str, rows: List[tuple]) -> None:
        assert psycopg2 is not None
        with self._pg.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO {table} (id, url, published_at, source, enriched_at, payload)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    url=EXCLUDED.url,
                    published_at=EXCLUDED.published_at,
                    source=EXCLUDED.source,
                    enriched_at=EXCLUDED.enriched_at,
                    payload=EXCLUDED.payload
                """,
                rows,
                template="(%s,%s,%s,%s,%s,%s)",
                page_size=max(100, min(10_000, len(rows))),
            )