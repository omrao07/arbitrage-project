# analytics-engine/vector-ai/pipelines/rerank.py
"""
Rerank candidates for a query using a cross-encoder or LLM scoring.

Inputs (library)
---------------
Reranker.rerank(
    query: str,
    candidates: List[Dict[str, Any]],  # each: {"id","text","vector_score", ...}
    top_k: int = 20,
    alpha: float = 0.2,                # fusion: alpha*normalized_vector + (1-alpha)*normalized_ce
) -> List[Dict]

Inputs (CLI)
------------
JSONL file where each line is a candidate object with:
  { "id": "...", "text": "...", "vector_score": 0.73, "meta": {...} }

Example:
  python rerank.py \
    --query "market outlook for Japanese inflation breakevens" \
    --in ./runs/candidates.jsonl \
    --out ./runs/reranked.jsonl \
    --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --topk 25 --alpha 0.3

LLM fallback (optional):
  python rerank.py --query "..." --in ... --out ... \
    --backend openai --openai-model gpt-4o-mini
  (requires OPENAI_API_KEY)

Install
-------
pip install sentence-transformers numpy tqdm
# (optional) pip install openai

Notes
-----
- Vector-score fusion uses min-max normalization per batch.
- Cross-encoder scores are model-dependent (often 0..1 or unbounded logits) and are normalized before fusion.
- If you don't have vector scores, set alpha=0 so cross-encoder ranks alone.
"""

from __future__ import annotations
import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Tuple

import numpy as np
from tqdm import tqdm


# ============================================================
# Utilities
# ============================================================

def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# Backends
# ============================================================

class BaseRerankBackend:
    name: str
    def score(self, query: str, texts: List[str]) -> List[float]:
        """Return relevance scores per text for given query (higher = better)."""
        raise NotImplementedError


class CrossEncoderBackend(BaseRerankBackend):
    """
    HuggingFace Cross Encoder (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").
    Fast and local. Produces comparable scores per batch; we still min-max normalize.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512, device: Optional[str] = None):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore # noqa: F401
        except Exception as e:
            raise RuntimeError("pip install sentence-transformers") from e
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, texts: List[str]) -> List[float]:
        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


class OpenAILLMBackend(BaseRerankBackend):
    """
    LLM scoring fallback via OpenAI: uses a structured prompt to rate 0-1.
    Slower and $$$; keep top_k small for this backend.
    """
    def __init__(self, model: str = "gpt-4o-mini", system_prompt: Optional[str] = None):
        try:
            from openai import OpenAI  # noqa: F401
        except Exception as e:
            raise RuntimeError("pip install openai") from e
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY")
        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"
        self.system_prompt = system_prompt or (
            "You are a ranking model. Given a user query and a candidate passage, "
            "return a single JSON number between 0 and 1 representing relevance "
            "(0=irrelevant, 1=perfect match). Reply ONLY with the number."
        )

    def _score_one(self, query: str, text: str) -> float:
        prompt = (
            f"Query: {query}\n"
            f"Passage:\n{text}\n\n"
            "Relevance score (0..1):"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip() # type: ignore
        try:
            # Accept plain number or small JSON-like outputs
            if raw.startswith("{") or raw.startswith("["):
                raw = json.loads(raw)
                if isinstance(raw, list):
                    raw = raw[0]
                if isinstance(raw, dict):
                    # try common keys
                    raw = raw.get("score", raw.get("relevance", 0))
            return float(raw)
        except Exception:
            return 0.0

    def score(self, query: str, texts: List[str]) -> List[float]:
        scores: List[float] = []
        for t in tqdm(texts, desc="LLM rerank", leave=False):
            scores.append(self._score_one(query, t))
        return scores


# ============================================================
# Reranker
# ============================================================

@dataclass
class RerankConfig:
    backend: str = "cross-encoder"  # "cross-encoder" or "openai"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    device: Optional[str] = None
    alpha: float = 0.2              # fusion: alpha*vector + (1-alpha)*ce
    top_k: int = 20                 # final cut after rerank


class Reranker:
    """
    Orchestrates reranking with score fusion:
        fused = alpha*norm(vector_score) + (1-alpha)*norm(ce_score)
    """
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        self.backend = self._init_backend(cfg)

    def _init_backend(self, cfg: RerankConfig) -> BaseRerankBackend:
        if cfg.backend.lower() in {"cross-encoder", "ce"}:
            return CrossEncoderBackend(cfg.model, cfg.max_length, cfg.device)
        elif cfg.backend.lower() in {"openai", "llm"}:
            return OpenAILLMBackend(cfg.model)
        else:
            raise ValueError("Unknown backend: use 'cross-encoder' or 'openai'")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        top_k = top_k if top_k is not None else self.cfg.top_k
        alpha = alpha if alpha is not None else self.cfg.alpha

        texts = [c.get("text", "") for c in candidates]
        ce_scores = np.array(self.backend.score(query, texts), dtype=np.float32)

        # Normalize CE scores per batch
        ce_norm = _minmax(ce_scores)

        # Vector scores (optional)
        vec_raw = np.array(
            [float(c.get("vector_score", 0.0)) for c in candidates], dtype=np.float32
        )
        vec_norm = _minmax(vec_raw)

        fused = alpha * vec_norm + (1.0 - alpha) * ce_norm

        out: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            r = dict(c)
            r["ce_score"] = float(ce_scores[i])
            r["ce_score_norm"] = float(ce_norm[i])
            r["vector_score_norm"] = float(vec_norm[i])
            r["fused_score"] = float(fused[i])
            r["reranker"] = self.backend.name
            out.append(r)

        out.sort(key=lambda x: x["fused_score"], reverse=True)
        return out[: top_k]


# ============================================================
# CLI
# ============================================================

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Rerank candidates for a query")
    p.add_argument("--query", required=True, help="User query")
    p.add_argument("--in", dest="inp", required=True, help="Path to candidates JSONL")
    p.add_argument("--out", required=True, help="Path to write reranked JSONL")
    # backend/model
    p.add_argument("--backend", choices=["cross-encoder", "openai"], default="cross-encoder")
    p.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--device", default=None, help="cuda|cpu (optional)")
    p.add_argument("--max-length", type=int, default=512)
    # fusion / cut
    p.add_argument("--alpha", type=float, default=0.2, help="fusion weight on vector score (0..1)")
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_cli()
    candidates = _read_jsonl(args.inp)
    if not candidates:
        raise SystemExit(f"No candidates found in {args.inp}")

    cfg = RerankConfig(
        backend=args.backend,
        model=args.model,
        max_length=args.max_length,
        device=args.device,
        alpha=args.alpha,
        top_k=args.topk,
    )
    rr = Reranker(cfg)
    ranked = rr.rerank(query=args.query, candidates=candidates)

    _write_jsonl(args.out, ranked)
    print(f"✅ Wrote {len(ranked)} reranked candidates → {args.out}")
    print(f"   • backend = {rr.backend.name}")
    print(f"   • alpha   = {cfg.alpha} (vector weight)")
    print(f"   • top_k   = {cfg.top_k}")


if __name__ == "__main__":
    main()