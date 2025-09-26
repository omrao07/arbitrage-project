# analytics-engine/vector-ai/pipelines/stream_index.py
"""
Streaming indexer: Kafka/NATS -> chunk -> embed -> (FAISS/Pinecone) + metadata parquet

Event shape (JSON)
------------------
# minimum
{ "id": "doc-123", "text": "…content…", "source": "news:reuters", "tags": ["macro","JPY"] }

# optional metadata passes through to parquet (NOT to vector dims)
{
  "id": "doc-456",
  "text": "…",
  "source": "filing:10Q",
  "tags": ["equities","AAPL"],
  "meta": {"ticker":"AAPL","date":"2025-09-01","lang":"en"}
}

Transports
----------
- Kafka (aiokafka)
- NATS (nats-py)

Embedders
---------
- OpenAI (text-embedding-3-*)  -> set OPENAI_API_KEY
- HuggingFace sentence-transformers       -> e.g. "sentence-transformers/all-MiniLM-L6-v2"

Indexes
-------
- FAISS local (.faiss) + metadata parquet
- Pinecone (optional; set PINECONE_API_KEY + index name)

Usage
-----
# NATS example
python stream_index.py \
  --nats nats://localhost:4222 --subject vector.docs \
  --embedder hf --hf-model sentence-transformers/all-MiniLM-L6-v2 \
  --faiss ./indices/live.faiss --meta ./indices/live.parquet \
  --chunk-tokens 350 --chunk-overlap 50 --flush-interval 1.0 --max-batch 256

# Kafka example
python stream_index.py \
  --kafka localhost:9092 --topic vector.docs \
  --embedder openai --openai-model text-embedding-3-small \
  --faiss ./indices/live.faiss --meta ./indices/live.parquet \
  --pinecone-index my-index --pinecone-namespace live

Dependencies
------------
pip install pandas pyarrow sentence-transformers faiss-cpu nats-py aiokafka openai pinecone-client
"""

from __future__ import annotations

import os
import re
import json
import uuid
import time
import asyncio
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ======================= Embedders =======================

class BaseEmbedder:
    dim: int
    name: str
    def encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 256):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("pip install openai") from e
        self.client = OpenAI()
        self.model = model
        self.batch = batch_size
        self.name = f"openai:{model}"
        self.dim = -1

    def encode(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i : i + self.batch]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vecs = [d.embedding for d in resp.data]
            out.extend(vecs)
            if self.dim < 0 and vecs:
                self.dim = len(vecs[0])
        return out

class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 256, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("pip install sentence-transformers") from e
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        self.batch = batch_size
        self.name = f"hf:{model_name}"
        self.dim = self.model.get_sentence_embedding_dimension() # type: ignore

    def encode(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        arr = self.model.encode(
            texts, batch_size=self.batch, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True
        )
        return arr.astype("float32").tolist()


# ======================= Index sinks =======================

class FAISSSink:
    def __init__(self, faiss_path: Path, dim: int, metric: str = "ip"):
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError("pip install faiss-cpu") from e
        self.faiss = faiss
        self.path = faiss_path
        self.dim = dim
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
            if self.index.d != dim:
                raise ValueError(f"Existing FAISS dim={self.index.d} != {dim}")
        else:
            self.index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)

    def add(self, vectors: List[List[float]]):
        import numpy as np
        self.index.add(np.asarray(vectors, dtype="float32"))

    def save(self):
        self.faiss.write_index(self.index, str(self.path))

class PineconeSink:
    def __init__(self, index_name: str, namespace: Optional[str] = None):
        try:
            from pinecone import Pinecone
        except Exception as e:
            raise RuntimeError("pip install pinecone-client") from e
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("Set PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.ns = namespace

    def add(self, vectors: List[List[float]], ids: List[str], metadata: List[Dict[str, Any]]):
        items = [{"id": ids[i], "values": vectors[i], "metadata": metadata[i]} for i in range(len(ids))]
        self.index.upsert(vectors=items, namespace=self.ns) # type: ignore

    def save(self):  # no-op
        pass


# ======================= Chunking =======================

def _tokenize(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s, re.UNICODE)

def chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    toks = _tokenize(text or "")
    if not toks: return []
    step = max(1, max_tokens - overlap)
    chunks: List[str] = []
    for i in range(0, len(toks), step):
        window = toks[i:i + max_tokens]
        if not window: break
        chunks.append(" ".join(window))
        if i + max_tokens >= len(toks): break
    return chunks


# ======================= Settings / Routes =======================

@dataclass
class Settings:
    # transports
    kafka_bootstrap: Optional[str] = None
    kafka_topics: List[str] = field(default_factory=list)
    nats_url: Optional[str] = None
    nats_subjects: List[str] = field(default_factory=list)
    # embed/index
    embedder: str = "hf"
    openai_model: str = "text-embedding-3-small"
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch: int = 256
    faiss_path: Optional[str] = None
    pinecone_index: Optional[str] = None
    pinecone_namespace: Optional[str] = None
    meta_path: Optional[str] = None
    # chunking/batching
    chunk_tokens: int = 350
    chunk_overlap: int = 50
    max_batch: int = 512
    flush_interval: float = 1.0
    log_every: int = 500
    device: Optional[str] = None


# ======================= Buffer/Writer =======================

class StreamIndexer:
    """
    Collects events -> chunks -> embeds -> writes to sinks (FAISS/Pinecone) and metadata parquet.
    """
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        # embedder
        if cfg.embedder == "openai":
            self.embedder = OpenAIEmbedder(cfg.openai_model, cfg.batch)
        else:
            self.embedder = HFEmbedder(cfg.hf_model, cfg.batch, device=cfg.device)

        self.faiss = FAISSSink(Path(cfg.faiss_path), dim=(self.embedder.dim if self.embedder.dim > 0 else 384)) if cfg.faiss_path else None
        self.pine = PineconeSink(cfg.pinecone_index, cfg.pinecone_namespace) if cfg.pinecone_index else None
        self.meta_path = Path(cfg.meta_path) if cfg.meta_path else None

        self._buf: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._rows = 0

    def add_event(self, ev: Dict[str, Any]):
        """
        Accepts JSON event; must contain at least {'id','text'}.
        """
        if not ev or "text" not in ev:
            return
        # Normalize shape
        row = {
            "doc_id": str(ev.get("id") or uuid.uuid4()),
            "source": str(ev.get("source") or ""),
            "tags": ev.get("tags") or [],
            "meta": ev.get("meta") or {},
            "text": str(ev["text"]),
        }
        self._buf.append(row)

    def _should_flush(self) -> bool:
        if len(self._buf) >= self.cfg.max_batch:
            return True
        if (time.time() - self._last_flush) >= self.cfg.flush_interval:
            return True
        return False

    async def maybe_flush(self):
        if self._should_flush():
            await self.flush()

    def _chunk_rows(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        out: List[Dict[str, Any]] = []
        for r in rows:
            pieces = chunk_text(r["text"], self.cfg.chunk_tokens, self.cfg.chunk_overlap) if self.cfg.chunk_tokens > 0 else [r["text"]]
            for j, ch in enumerate(pieces):
                out.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": r["doc_id"],
                    "source": r["source"],
                    "text": ch,
                    "chunk_idx": j,
                    "tags": r["tags"],
                    "meta": r["meta"],
                })
        return pd.DataFrame(out)

    def _write_meta(self, meta_rows: List[Dict[str, Any]]):
        if not self.meta_path:
            return
        df = pd.DataFrame(meta_rows)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            old = pd.read_parquet(self.meta_path)
            df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(self.meta_path, index=False)

    async def flush(self):
        if not self._buf:
            self._last_flush = time.time()
            return

        batch = self._buf
        self._buf = []

        # chunk
        chunks = self._chunk_rows(batch)
        if chunks.empty:
            self._last_flush = time.time()
            return

        texts = chunks["text"].tolist()
        ids   = chunks["chunk_id"].tolist()
        srcs  = chunks["source"].tolist()
        dids  = chunks["doc_id"].tolist()
        cidx  = chunks["chunk_idx"].tolist()
        tags  = chunks["tags"].tolist()
        metas = chunks["meta"].tolist()

        # embed
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.cfg.batch):
            vectors.extend(self.embedder.encode(texts[i:i+self.cfg.batch]))

        # sinks
        if self.faiss:
            self.faiss.add(vectors)
            self.faiss.save()

        if self.pine:
            meta_rows = [{
                "chunk_id": ids[i],
                "doc_id": dids[i],
                "source": srcs[i],
                "chunk_idx": int(cidx[i]),
                "tags": tags[i],
                **(metas[i] if isinstance(metas[i], dict) else {}),
                "embedder": self.embedder.name,
            } for i in range(len(ids))]
            self.pine.add(vectors, ids, meta_rows)
            self.pine.save()
        else:
            # still persist metadata parquet even if only FAISS is used
            meta_rows = [{
                "chunk_id": ids[i],
                "doc_id": dids[i],
                "source": srcs[i],
                "chunk_idx": int(cidx[i]),
                "tags": tags[i],
                **(metas[i] if isinstance(metas[i], dict) else {}),
                "embedder": self.embedder.name,
            } for i in range(len(ids))]

        self._write_meta(meta_rows)

        self._rows += len(texts)
        self._last_flush = time.time()
        if self._rows % self.cfg.log_every == 0 or len(texts) >= self.cfg.log_every:
            print(f"[STREAM-INDEX] wrote {len(texts)} chunks (cumulative {self._rows})")

    async def close(self):
        await self.flush()


# ======================= Transports =======================

async def run_kafka(cfg: Settings, indexer: StreamIndexer):
    try:
        from aiokafka import AIOKafkaConsumer
    except Exception:
        raise SystemExit("Kafka selected but aiokafka not installed. pip install aiokafka")

    consumer = AIOKafkaConsumer(
        *cfg.kafka_topics,
        bootstrap_servers=cfg.kafka_bootstrap, # type: ignore
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer=lambda v: v,  # bytes in; we'll decode
        key_deserializer=lambda v: v,
        max_poll_records=cfg.max_batch,
    )
    await consumer.start()
    print(f"[KAFKA] connected → {cfg.kafka_bootstrap} topics={cfg.kafka_topics}")
    try:
        while True:
            msgs = await consumer.getmany(timeout_ms=int(cfg.flush_interval * 1000))
            for tp, batch in msgs.items():
                for msg in batch:
                    try:
                        b = msg.value
                        if isinstance(b, (bytes, bytearray)):
                            event = json.loads(b.decode("utf-8"))
                        else:
                            event = json.loads(bytes(b)) # type: ignore
                        indexer.add_event(event)
                    except Exception as e:
                        print(f"[KAFKA][{tp.topic}] parse error: {e}")
            await indexer.maybe_flush()
    except asyncio.CancelledError:
        pass
    finally:
        await indexer.close()
        await consumer.stop()
        print("[KAFKA] stopped.")

async def run_nats(cfg: Settings, indexer: StreamIndexer):
    try:
        import nats
    except Exception:
        raise SystemExit("NATS selected but nats-py not installed. pip install nats-py")

    nc = await nats.connect(cfg.nats_url) # type: ignore
    print(f"[NATS] connected → {cfg.nats_url} subjects={cfg.nats_subjects}")

    async def handler(msg):
        try:
            event = json.loads(msg.data.decode("utf-8"))
            indexer.add_event(event)
        except Exception as e:
            print(f"[NATS][{msg.subject}] parse error: {e}")

    subs = [await nc.subscribe(subj, cb=handler) for subj in cfg.nats_subjects]

    try:
        while True:
            await asyncio.sleep(cfg.flush_interval / 2)
            await indexer.maybe_flush()
    except asyncio.CancelledError:
        pass
    finally:
        for s in subs:
            await s.unsubscribe()
        await indexer.close()
        await nc.drain()
        print("[NATS] stopped.")


# ======================= CLI / Main =======================

def parse_args() -> Settings:
    import argparse
    p = argparse.ArgumentParser("Streaming vector indexer (Kafka/NATS)")
    # transports
    p.add_argument("--kafka", dest="kafka_bootstrap", help="host:port for Kafka")
    p.add_argument("--topic", dest="kafka_topics", action="append", default=[], help="Kafka topic (repeatable)")
    p.add_argument("--nats", dest="nats_url", help="nats://host:4222")
    p.add_argument("--subject", dest="nats_subjects", action="append", default=[], help="NATS subject (repeatable)")
    # embed/index
    p.add_argument("--embedder", choices=["openai", "hf"], default="hf")
    p.add_argument("--openai-model", default="text-embedding-3-small")
    p.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default=None, help="cuda|cpu (HF only)")
    p.add_argument("--faiss", dest="faiss_path", help="Path to .faiss index")
    p.add_argument("--pinecone-index")
    p.add_argument("--pinecone-namespace")
    p.add_argument("--meta", dest="meta_path", help="Path to metadata parquet")
    # chunking/batching
    p.add_argument("--chunk-tokens", type=int, default=350)
    p.add_argument("--chunk-overlap", type=int, default=50)
    p.add_argument("--max-batch", type=int, default=512)
    p.add_argument("--flush-interval", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=500)
    a = p.parse_args()

    if not a.kafka_bootstrap and not a.nats_url:
        raise SystemExit("Select a transport: --kafka ... or --nats ...")
    if a.kafka_bootstrap and not a.kafka_topics:
        raise SystemExit("Kafka selected but no --topic provided.")
    if a.nats_url and not a.nats_subjects:
        raise SystemExit("NATS selected but no --subject provided.")
    if not a.faiss_path and not a.pinecone_index:
        print("[WARN] No FAISS or Pinecone selected; you'll only write metadata parquet.")

    return Settings(
        kafka_bootstrap=a.kafka_bootstrap,
        kafka_topics=a.kafka_topics,
        nats_url=a.nats_url,
        nats_subjects=a.nats_subjects,
        embedder=a.embedder,
        openai_model=a.openai_model,
        hf_model=a.hf_model,
        device=a.device,
        batch=256,
        faiss_path=a.faiss_path,
        pinecone_index=a.pinecone_index,
        pinecone_namespace=a.pinecone_namespace,
        meta_path=a.meta_path,
        chunk_tokens=a.chunk_tokens,
        chunk_overlap=a.chunk_overlap,
        max_batch=a.max_batch,
        flush_interval=a.flush_interval,
        log_every=a.log_every,
    )

async def main_async():
    cfg = parse_args()
    indexer = StreamIndexer(cfg)

    stop = asyncio.Event()
    def _graceful(*_): stop.set()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(sig, _graceful)
        except NotImplementedError: pass

    if cfg.kafka_bootstrap:
        task = asyncio.create_task(run_kafka(cfg, indexer))
    else:
        task = asyncio.create_task(run_nats(cfg, indexer))

    await stop.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()