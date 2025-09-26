# analytics-engine/vector-ai/embeddings/hf_embedder.py
"""
HFEmbedder
----------
Generic HuggingFace sentence embedding wrapper.

Default model
-------------
- sentence-transformers/all-MiniLM-L6-v2  (384-dim, fast, strong baseline)

Install
-------
pip install sentence-transformers torch

Env (optional)
--------------
HF_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEVICE=cuda   # or cpu
"""

from __future__ import annotations
import os
from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Missing dependency: pip install sentence-transformers") from e


class HFEmbedder:
    """
    Drop-in embedder for batch/stream indexers.

    Attributes
    ----------
    name : str   -> "hf:<model_name>"
    dim  : int   -> embedding dimension
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 256,
        device: Optional[str] = None,
        max_seq_length: int = 512,
        normalize: bool = True,
    ):
        self.model_name = model_name or os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = batch_size
        self.device = device or os.getenv("DEVICE", None)
        self.normalize = normalize

        # load model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = max_seq_length

        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = f"hf:{self.model_name}"

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of strings into dense vectors.

        Returns
        -------
        List[List[float]]  # float32 vectors
        """
        if not texts:
            return []
        arr = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return arr.astype("float32").tolist()


# -------------------------- Quick self-test --------------------------

if __name__ == "__main__":
    enc = HFEmbedder()
    vecs = enc.encode(["yen carry trade unwinds", "US CPI surprise lifts yields"])
    print("model:", enc.name, "dim:", enc.dim, "nvecs:", len(vecs))
    print("first vector (10):", vecs[0][:10])