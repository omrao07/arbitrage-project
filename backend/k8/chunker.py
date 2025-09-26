#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunker.py
----------
Universal chunker for KB ingestion (OCR/ASR docs, PDFs after text extraction, code, markdown, CSV/TSV).
- Token-aware (uses tiktoken if available, else char→token heuristic)
- Markdown-aware (splits on headings, preserves local context)
- Code-aware (splits on def/class/function blocks with fallback to line windows)
- Table-aware (splits CSV/TSV by row batches)
- Adds overlap for retrieval context
- Emits stable chunk IDs and rich metadata

Return format: List[Dict[str, Any]] with keys:
  - id: stable hex id
  - text: chunk text
  - meta: {source, kind, idx, start_char, end_char, n_tokens, section, language, ...}

Typical usage:
  chunks = Chunker(max_tokens=800, overlap=120).chunk_markdown(md_text, source="note.md")
"""

from __future__ import annotations
import re
import csv
import io
import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------- Optional tokenizer ----------
try:
    import tiktoken  # type: ignore
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(s: str) -> int:
        return len(_enc.encode(s))
except Exception:
    def _count_tokens(s: str) -> int:
        # ~4 chars/token heuristic; clamp to >=1
        return max(1, (len(s) + 3) // 4)


# ---------- Utilities ----------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```(\w+)?\s*$", re.MULTILINE)
_SENT_RE = re.compile(r"(?<=\S)([.!?])\s+(?=[A-Z(0-9\"\'])")

def _hash_id(*parts: str) -> str:
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()

def _clip_tokens(text: str, max_tokens: int) -> str:
    # fast clip by chars, then shrink tokenwise if needed
    approx_ratio = 4  # chars per token heuristic
    if len(text) // approx_ratio <= max_tokens:
        return text
    limit = max_tokens * approx_ratio
    s = text[:limit]
    # trim to sentence end if possible
    m = list(_SENT_RE.finditer(s))
    if m:
        s = s[: m[-1].end()]
    # ensure real token limit
    while _count_tokens(s) > max_tokens and len(s) > 0:
        s = s[:-min(80, len(s))]
    return s

def _normalize_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\r\n?", "\n", s)).strip()

def _window_tokens(parts: List[str], max_tokens: int, overlap: int) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    buf_tok = 0

    for p in parts:
        t = _count_tokens(p)
        if t > max_tokens:
            # hard split long paragraph/sentence
            out.extend(_split_long_text(p, max_tokens, overlap))
            continue
        if buf_tok + t <= max_tokens:
            buf.append(p)
            buf_tok += t
        else:
            if buf:
                out.append("\n".join(buf).strip())
            # build next buffer with overlap from previous end
            if overlap > 0 and out:
                tail = _right_overlap(out[-1], overlap)
                buf = [tail, p] if tail else [p]
                buf_tok = _count_tokens("\n".join(buf))
            else:
                buf = [p]; buf_tok = t
    if buf:
        out.append("\n".join(buf).strip())
    return [o for o in out if o.strip()]

def _right_overlap(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    toks_est = _count_tokens(text)
    if toks_est <= overlap_tokens:
        return text
    # approximate by chars
    ratio = max(1, len(text) // toks_est)
    need_chars = overlap_tokens * ratio
    return text[-need_chars:]

def _split_long_text(s: str, max_tokens: int, overlap: int) -> List[str]:
    # progressively split by sentences then hard clip
    sentences = _split_sentences(s)
    if not sentences:
        return [_clip_tokens(s, max_tokens)]
    return _window_tokens(sentences, max_tokens, overlap)

def _split_sentences(s: str) -> List[str]:
    # simple sentence splitter that preserves punctuation
    parts: List[str] = []
    last = 0
    for m in _SENT_RE.finditer(s):
        parts.append(s[last:m.end()])
        last = m.end()
    parts.append(s[last:])
    return [p.strip() for p in parts if p.strip()]

def _lang_from_ext(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    ext = os.path.splitext(source)[1].lower().lstrip(".")
    return {
        "py": "python", "js": "javascript", "ts": "typescript",
        "tsx": "tsx", "jsx": "jsx", "java": "java", "rb": "ruby",
        "go": "go", "rs": "rust", "cpp": "cpp", "cc": "cpp", "c": "c",
        "sql": "sql", "md": "markdown", "sh": "bash"
    }.get(ext, None)


# ---------- Core class ----------

@dataclass
class Chunker:
    max_tokens: int = 800
    overlap: int = 120
    min_tokens: int = 40  # discard ultra-short fragments unless they carry a heading

    # ===== Plain text =====
    def chunk_text(self, text: str, source: Optional[str] = None, kind: str = "text") -> List[Dict[str, Any]]:
        raw = _normalize_ws(text)
        # split into paragraphs then sentence-pack them token-aware
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
        windows = _window_tokens(paragraphs, self.max_tokens, self.overlap)
        out: List[Dict[str, Any]] = []
        start = 0
        for i, w in enumerate(windows):
            end = start + len(w)
            if _count_tokens(w) < self.min_tokens and i != 0:
                start = end
                continue
            out.append(self._make_chunk(w, source, kind, idx=i, start=start, end=end))
            start = end
        return out

    # ===== Markdown =====
    def chunk_markdown(self, md: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        # split by headings while respecting fenced code blocks
        blocks = self._split_markdown_sections(md)
        # within each section, pack paragraphs
        out: List[Dict[str, Any]] = []
        for sec_idx, (heading, body, span) in enumerate(blocks):
            base = f"{heading}\n\n{body}".strip() if heading else body.strip()
            paras = [p.strip() for p in re.split(r"\n{2,}", base) if p.strip()]
            windows = _window_tokens(paras, self.max_tokens, self.overlap)
            for i, w in enumerate(windows):
                meta = {"section": heading.strip("# ").strip() if heading else None}
                chunk = self._make_chunk(w, source, "markdown", idx=len(out), start=span[0], end=span[1], extra_meta=meta)
                if _count_tokens(chunk["text"]) >= self.min_tokens or heading:
                    out.append(chunk)
        return out

    def _split_markdown_sections(self, md: str) -> List[Tuple[Optional[str], str, Tuple[int,int]]]:
        """
        Returns list of (heading_line or None, section_body, (start_char, end_char))
        Respects fenced code blocks so we don't split mid-fence.
        """
        text = md if md.endswith("\n") else md + "\n"
        # Mark fence regions to avoid heading splits inside code
        fences: List[Tuple[int,int]] = []
        stack: List[int] = []
        for m in _CODE_FENCE_RE.finditer(text):
            if stack:
                start = stack.pop()
                fences.append((start, m.end()))
            else:
                stack.append(m.start())
        def in_fence(pos: int) -> bool:
            return any(a <= pos < b for a,b in fences)

        sections: List[Tuple[Optional[str], str, Tuple[int,int]]] = []
        last_pos = 0
        last_heading: Optional[str] = None
        for m in _HEADING_RE.finditer(text):
            if in_fence(m.start()):
                continue
            # close previous
            if last_heading is not None:
                body = text[last_pos:m.start()].strip("\n")
                sections.append((last_heading, body, (last_pos, m.start())))
            # start new
            last_heading = m.group(0)
            last_pos = m.end()
        # tail
        tail = text[last_pos:].strip("\n")
        sections.append((last_heading, tail, (last_pos, len(text))))
        # If the file starts with body (no heading), keep first tuple with None heading
        if sections and sections[0][0] is None and sections[0][1].strip() == "":
            sections = sections[1:]
        return sections

    # ===== Code =====
    def chunk_code(self, code: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        lang = _lang_from_ext(source) or "code"
        # try function/class splits for common langs
        if lang in {"python","javascript","typescript","tsx","jsx","java","go","rust","cpp","c"}:
            blocks = self._split_code_blocks(code, lang)
        else:
            blocks = [(None, code, (0, len(code)))]

        out: List[Dict[str, Any]] = []
        for bi, (name, body, span) in enumerate(blocks):
            # pack by logical lines
            lines = [ln for ln in body.splitlines() if ln.strip() != ""]
            windows = _window_tokens([ "\n".join(lines) ], self.max_tokens, self.overlap)
            for i, w in enumerate(windows):
                meta = {"symbol": name, "language": lang}
                out.append(self._make_chunk(w, source, "code", idx=len(out), start=span[0], end=span[1], extra_meta=meta))
        return out

    def _split_code_blocks(self, code: str, lang: str) -> List[Tuple[Optional[str], str, Tuple[int,int]]]:
        # very light heuristics; avoids heavy parsing deps
        patterns = []
        if lang == "python":
            patterns = [re.compile(r"^(class|def)\s+[\w_]+.*:$", re.MULTILINE)]
        elif lang in {"javascript","typescript","tsx","jsx"}:
            patterns = [re.compile(r"^(export\s+)?(function|class)\s+[\w$]+", re.MULTILINE)]
        elif lang in {"java","go","rust","cpp","c"}:
            patterns = [re.compile(r"^(class|struct|fn|void|int|float|double|public\s+class|package\s+[\w\.]+)", re.MULTILINE)]

        if not patterns:
            return [(None, code, (0, len(code)))]

        indices = []
        for pat in patterns:
            indices.extend([m.start() for m in pat.finditer(code)])
        indices = sorted(set([0] + indices + [len(code)]))

        blocks: List[Tuple[Optional[str], str, Tuple[int,int]]] = []
        for i in range(len(indices)-1):
            a, b = indices[i], indices[i+1]
            block = code[a:b]
            header = block.splitlines()[0] if block.strip() else None
            blocks.append((header, block, (a, b)))
        return blocks

    # ===== CSV/TSV Tables =====
    def chunk_table(self, data: str, source: Optional[str] = None, delimiter: Optional[str] = None,
                    rows_per_chunk: int = 200) -> List[Dict[str, Any]]:
        """
        Splits CSV/TSV into row batches, keeps header each chunk.
        """
        if delimiter is None:
            delimiter = "\t" if "\t" in data and "," not in data else ","
        reader = csv.reader(io.StringIO(data), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return []
        header, body = rows[0], rows[1:]

        out: List[Dict[str, Any]] = []
        for i in range(0, len(body), rows_per_chunk):
            batch = body[i:i+rows_per_chunk]
            text = "\n".join([delimiter.join(header)] + [delimiter.join(map(str, r)) for r in batch])
            if _count_tokens(text) > self.max_tokens:
                # if too large, fall back to half window
                text = _clip_tokens(text, self.max_tokens)
            out.append(self._make_chunk(text, source, "table", idx=len(out), start=i, end=min(i+rows_per_chunk, len(body))))
        return out

    # ===== Generic builder =====
    def _make_chunk(self, text: str, source: Optional[str], kind: str,
                    idx: int, start: int, end: int,
                    extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "source": source,
            "kind": kind,
            "idx": idx,
            "start_char": start,
            "end_char": end,
            "n_tokens": _count_tokens(text),
        }
        if extra_meta:
            meta.update(extra_meta)
        return {
            "id": _hash_id(source or "", kind, str(idx), str(start), str(end), text[:64]),
            "text": text,
            "meta": meta,
        }


# ---------- Convenience functions ----------

def chunk_any(text: str, source: Optional[str] = None, kind_hint: Optional[str] = None,
              max_tokens: int = 800, overlap: int = 120) -> List[Dict[str, Any]]:
    """
    One-shot helper that routes by hint or filename:
      - '.md' -> markdown
      - '.py', '.js', '.ts', etc. -> code
      - '.csv', '.tsv' -> table
      - else -> plain text
    """
    ch = Chunker(max_tokens=max_tokens, overlap=overlap)
    ext = os.path.splitext(source or "")[1].lower()
    if kind_hint == "markdown" or ext == ".md":
        return ch.chunk_markdown(text, source)
    if kind_hint == "code" or ext in {".py",".js",".ts",".tsx",".jsx",".java",".go",".rs",".cpp",".cc",".c",".sql",".sh"}:
        return ch.chunk_code(text, source)
    if kind_hint == "table" or ext in {".csv",".tsv"}:
        return ch.chunk_table(text, source, delimiter=("\t" if ext == ".tsv" else ","))
    return ch.chunk_text(text, source, kind="text")


def print_chunks(chunks: Iterable[Dict[str, Any]]) -> None:
    for c in chunks:
        print(f"--- Chunk {c['meta']['idx']} ({c['meta']['n_tokens']} tokens) ---")
        print(c['text'])
        print()             