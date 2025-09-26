# analytics-engine/vector-ai/pipelines/batch_index.py
"""
Batch indexer: documents -> chunks -> embeddings -> vector index (+ metadata)

Sources
  - Directory with .txt/.md/.pdf/.csv/.parquet/.jsonl
  - Single CSV/Parquet/JSONL file (use --text-col)
Embedders
  - OpenAI (requires OPENAI_API_KEY)
  - HuggingFace sentence-transformers

Indexes
  - FAISS local (.faiss) + metadata parquet
  - Pinecone (optional; PINECONE_API_KEY)

Examples
  python batch_index.py --source ./data/docs \
    --faiss ./indices/news.faiss --meta ./indices/news.parquet \
    --embedder openai --openai-model text-embedding-3-small \
    --chunk-tokens 350 --chunk-overlap 50 --tags news global

  python batch_index.py --source ./data/articles.parquet --text-col body \
    --embedder hf --hf-model sentence-transformers/all-MiniLM-L6-v2 \
    --faiss ./indices/articles.faiss --meta ./indices/articles.parquet
"""

from __future__ import annotations
import os, re, json, uuid, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

# -------------------- Embedders --------------------

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
        self.dim = -1  # will set on first call

    def encode(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i:i + self.batch]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vecs = [d.embedding for d in resp.data]
            out.extend(vecs)
            if self.dim < 0 and vecs:
                self.dim = len(vecs[0])
        return out

class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 256, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer # type: ignore
        except Exception as e:
            raise RuntimeError("pip install sentence-transformers") from e
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        self.batch = batch_size
        self.name = f"hf:{model_name}"
        self.dim = self.model.get_sentence_embedding_dimension() # type: ignore

    def encode(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        embs = self.model.encode(
            texts, batch_size=self.batch, convert_to_numpy=True,
            show_progress_bar=True, normalize_embeddings=True
        )
        return embs.astype("float32").tolist()

# -------------------- Chunking --------------------

def simple_tokenizer(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s, re.UNICODE)

def chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    toks = simple_tokenizer(text)
    if not toks: return []
    step = max(1, max_tokens - overlap)
    chunks = []
    for i in range(0, len(toks), step):
        window = toks[i:i + max_tokens]
        if not window: break
        chunks.append(" ".join(window))
        if i + max_tokens >= len(toks): break
    return chunks

# -------------------- Loaders --------------------

TEXT_EXT = {".txt", ".md", ".markdown"}

def from_directory(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in path.rglob("*"):
        suf = p.suffix.lower()
        if suf in TEXT_EXT:
            rows.append({"id": str(uuid.uuid4()), "source": str(p), "text": p.read_text(errors="ignore")})
        elif suf == ".jsonl":
            with p.open() as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            rows.append({"id": obj.get("id", str(uuid.uuid4())), "source": str(p), **obj})
                    except Exception:
                        continue
        elif suf == ".csv":
            df = pd.read_csv(p); col = "text" if "text" in df.columns else df.columns[-1]
            for _, r in df.iterrows():
                rows.append({"id": r.get("id", str(uuid.uuid4())), "source": str(p), "text": str(r[col])})
        elif suf in {".parquet", ".pq"}:
            df = pd.read_parquet(p); col = "text" if "text" in df.columns else df.columns[-1]
            for _, r in df.iterrows():
                rows.append({"id": r.get("id", str(uuid.uuid4())), "source": str(p), "text": str(r[col])})
        elif suf == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore # pip install pypdf
                reader = PdfReader(str(p))
                txt = "\n".join([pg.extract_text() or "" for pg in reader.pages])
                rows.append({"id": str(uuid.uuid4()), "source": str(p), "text": txt})
            except Exception:
                pass
    return pd.DataFrame(rows)

def load_table(path: Path, text_col: str) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suf == ".csv":
        df = pd.read_csv(path)
    elif suf == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported table type: {suf}")
    if text_col not in df.columns:
        raise ValueError(f"--text-col '{text_col}' not in columns: {list(df.columns)}")
    if "id" not in df.columns: df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    if "source" not in df.columns: df["source"] = str(path)
    df["text"] = df[text_col].astype(str)
    return df[["id", "source", "text"] + [c for c in df.columns if c not in {"id", "source", "text"}]]

# -------------------- Index sinks --------------------

class FAISSSink:
    def __init__(self, faiss_path: Path, dim: int, metric: str = "ip"):
        try: import faiss  # type: ignore
        except Exception as e: raise RuntimeError("pip install faiss-cpu") from e
        self.faiss = faiss
        self.path = faiss_path
        self.dim = dim
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
            if self.index.d != dim:
                raise ValueError(f"Existing FAISS index dim={self.index.d} != {dim}")
        else:
            self.index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)

    def add(self, vectors: List[List[float]]):
        import numpy as np
        self.index.add(np.asarray(vectors, dtype="float32"))

    def save(self): self.faiss.write_index(self.index, str(self.path))

class PineconeSink:
    def __init__(self, index_name: str, namespace: Optional[str] = None):
        try: from pinecone import Pinecone
        except Exception as e: raise RuntimeError("pip install pinecone-client") from e
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key: raise RuntimeError("Set PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.ns = namespace

    def add(self, vectors: List[List[float]], ids: List[str], metadata: List[Dict[str, Any]]):
        items = [{"id": ids[i], "values": vectors[i], "metadata": metadata[i]} for i in range(len(ids))]
        self.index.upsert(vectors=items, namespace=self.ns) # type: ignore

    def save(self): pass

# -------------------- Orchestration --------------------

def chunk_df(df: pd.DataFrame, max_tokens: int, overlap: int) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        text = (r.get("text") or "").strip()
        if not text: continue
        pieces = chunk_text(text, max_tokens, overlap) if max_tokens > 0 else [text]
        for j, ch in enumerate(pieces):
            rows.append({
                "chunk_id": str(uuid.uuid4()),
                "doc_id": r["id"],
                "source": r.get("source", ""),
                "text": ch,
                "chunk_idx": j
            })
    return pd.DataFrame(rows)

def write_outputs(
    chunks: pd.DataFrame,
    embedder: BaseEmbedder,
    faiss_sink: Optional[FAISSSink],
    pine_sink: Optional[PineconeSink],
    meta_path: Optional[Path],
    tags: List[str],
    batch: int,
):
    texts   = chunks["text"].tolist()
    ids     = chunks["chunk_id"].tolist()
    doc_ids = chunks["doc_id"].tolist()
    srcs    = chunks["source"].tolist()
    idxs    = chunks["chunk_idx"].tolist()

    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch):
        vectors.extend(embedder.encode(texts[i:i+batch]))

    meta_rows = [{
        "chunk_id": ids[i], "doc_id": doc_ids[i], "source": srcs[i],
        "chunk_idx": int(idxs[i]), "tags": tags, "embedder": embedder.name
    } for i in range(len(texts))]

    if faiss_sink:
        faiss_sink.add(vectors); faiss_sink.save()
    if pine_sink:
        pine_sink.add(vectors, ids, meta_rows); pine_sink.save()
    if meta_path:
        md = pd.DataFrame(meta_rows)
        if meta_path.exists():
            old = pd.read_parquet(meta_path)
            md = pd.concat([old, md], ignore_index=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        md.to_parquet(meta_path, index=False)

@dataclass
class Args:
    source: str
    text_col: Optional[str]
    embedder: str
    openai_model: str
    hf_model: str
    batch: int
    chunk_tokens: int
    chunk_overlap: int
    faiss: Optional[str]
    meta: Optional[str]
    pinecone_index: Optional[str]
    pinecone_namespace: Optional[str]
    tags: List[str]

def parse_args() -> Args:
    ap = argparse.ArgumentParser("Batch embed & index")
    ap.add_argument("--source", required=True)
    ap.add_argument("--text-col")
    ap.add_argument("--embedder", choices=["openai","hf"], default="hf")
    ap.add_argument("--openai-model", default="text-embedding-3-small")
    ap.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk-tokens", type=int, default=350)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--faiss")
    ap.add_argument("--meta")
    ap.add_argument("--pinecone-index")
    ap.add_argument("--pinecone-namespace")
    ap.add_argument("--tags", nargs="*", default=[])
    a = ap.parse_args()
    return Args(
        source=a.source, text_col=a.text_col, embedder=a.embedder,
        openai_model=a.openai_model, hf_model=a.hf_model, batch=a.batch,
        chunk_tokens=a.chunk_tokens, chunk_overlap=a.chunk_overlap,
        faiss=a.faiss, meta=a.meta, pinecone_index=a.pinecone_index,
        pinecone_namespace=a.pinecone_namespace, tags=a.tags
    )

def build_embedder(a: Args) -> BaseEmbedder:
    return OpenAIEmbedder(a.openai_model, a.batch) if a.embedder == "openai" else HFEmbedder(a.hf_model, a.batch)

def main():
    a = parse_args()
    src = Path(a.source)

    if src.is_dir():
        df = from_directory(src)
    else:
        if not a.text_col:
            raise SystemExit("--text-col required when --source is a table file")
        df = load_table(src, a.text_col)

    if df.empty: raise SystemExit("No documents found")

    chunks = chunk_df(df, a.chunk_tokens, a.chunk_overlap)
    if chunks.empty: raise SystemExit("No chunks produced")

    embedder = build_embedder(a)
    faiss_sink = FAISSSink(Path(a.faiss), embedder.dim if embedder.dim > 0 else 384) if a.faiss else None
    pine_sink = PineconeSink(a.pinecone_index, a.pinecone_namespace) if a.pinecone_index else None
    meta_path = Path(a.meta) if a.meta else None

    write_outputs(chunks, embedder, faiss_sink, pine_sink, meta_path, a.tags, a.batch)

    print(f"✅ Indexed {len(chunks)} chunks")
    if a.faiss: print(f"   • FAISS: {a.faiss}")
    if a.meta:  print(f"   • Meta : {a.meta}")
    if a.pinecone_index: print(f"   • Pinecone: {a.pinecone_index} (ns={a.pinecone_namespace or 'default'})")

if __name__ == "__main__":
    main()