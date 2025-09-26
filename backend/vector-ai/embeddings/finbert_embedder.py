# analytics-engine/vector-ai/embeddings/finbert_embedder.py
"""
FinBERT Embedder
----------------
Financial-domain sentence embedding wrapper for FinBERT.

Use Cases
---------
- Better domain adaptation for finance text (news, filings, transcripts)
- Plug-and-play alternative to generic HF or OpenAI embeddings

Requirements
------------
pip install transformers torch sentence-transformers

Env (optional)
--------------
FINBERT_MODEL=ProsusAI/finbert
DEVICE=cuda   # or cpu
"""

from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class FinBERTEmbedder:
    """
    Wraps FinBERT (via sentence-transformers or HuggingFace) to produce dense embeddings.
    """

    def __init__(
        self,
        model_name: str = None, # type: ignore
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
        self.batch_size = batch_size
        self.device = device or os.getenv("DEVICE", None)

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.max_seq_length = 512
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FinBERT model ({self.model_name}). "
                f"Ensure transformers + sentence-transformers installed."
            ) from e

        self.name = f"finbert:{self.model_name}"
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        if not texts:
            return []
        arr = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return arr.astype("float32").tolist()


# -------------------------- Quick Test --------------------------

if __name__ == "__main__":
    embedder = FinBERTEmbedder()
    docs = [
        "The Federal Reserve raised interest rates by 25bps.",
        "Apple reported strong earnings growth in Q2.",
    ]
    vecs = embedder.encode(docs)
    print("Embedding dim:", embedder.dim)
    print("First vector snippet:", vecs[0][:10])