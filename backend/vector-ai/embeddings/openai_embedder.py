# analytics-engine/vector-ai/embeddings/openai_embedder.py
"""
OpenAIEmbedder
--------------
Batch text embedding via OpenAI (or Azure OpenAI).

Defaults
--------
- model: text-embedding-3-small  (good quality, cheap)
- detects vector dim lazily on first call

Install
-------
pip install openai backoff

Env
---
# OpenAI:
OPENAI_API_KEY=sk-...

# Azure OpenAI (optional):
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview  # or your version
AZURE_OPENAI_DEPLOYMENT=text-embedding-3-small  # deployment name

Notes
-----
- If AZURE_* envs are present, Azure endpoint is used automatically.
- Returns float32 lists for FAISS/Pinecone compatibility.
"""

from __future__ import annotations
import os
from typing import List, Optional
import backoff

# ---------------------- Backend Selection ----------------------

def _is_azure() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))

def _openai_client():
    if _is_azure():
        # Azure OpenAI (uses the same SDK, different base URL)
        from openai import AzureOpenAI  # type: ignore
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        if not endpoint or not api_key:
            raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for Azure OpenAI.")
        return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    else:
        from openai import OpenAI  # type: ignore
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY for OpenAI.")
        return OpenAI()

# --------------------------- Embedder ---------------------------

class OpenAIEmbedder:
    """
    Drop-in embedder for your batch/stream indexers.

    Attributes
    ----------
    name : str   -> "openai:<model>" or "azure-openai:<deployment>"
    dim  : int   -> embedding dimension (resolved on first encode)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = 256,
        timeout: Optional[float] = 60.0,
    ):
        self.client = _openai_client()
        self.is_azure = _is_azure()
        # For Azure, "model" is the deployment name. For OpenAI, it's the model string.
        self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT") if self.is_azure else (model or "text-embedding-3-small")
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.dim = -1
        self.name = f"{'azure-openai' if self.is_azure else 'openai'}:{self.model}"

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        # Azure and OpenAI both use .embeddings.create
        resp = self.client.embeddings.create(model=self.model, input=batch, timeout=self.timeout) # type: ignore
        vecs = [d.embedding for d in resp.data]
        if self.dim < 0 and vecs:
            self.dim = len(vecs[0])
        return vecs

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self._embed_batch(batch)
            out.extend(vecs)
        # Convert to float32 lists to keep memory + ANN libs happy
        try:
            import numpy as np
            return np.asarray(out, dtype="float32").tolist()
        except Exception:
            return out


# -------------------------- Quick self-test --------------------------

if __name__ == "__main__":
    emb = OpenAIEmbedder()  # uses OpenAI by default unless AZURE_* envs set
    vectors = emb.encode(["Yen carry trade unwind risk", "US CPI surprise lifts yields"])
    print("backend:", emb.name, "dim:", emb.dim, "nvecs:", len(vectors))
    print("first vector (10):", vectors[0][:10])