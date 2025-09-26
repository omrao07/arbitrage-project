#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb/asr/utils.py
---------------
Shared utilities for ASR pipeline:
  - File I/O (read/write audio safely)
  - Path helpers
  - Normalization (text, loudness)
  - Timer context manager
  - Logging helper

All functions are safe to import across kb/asr modules.
"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf

# Optional
try:
    import librosa
except Exception:
    librosa = None


# ---------------- Logging ----------------

def get_logger(name: str = "asr", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    return logger


# ---------------- Timer ----------------

class Timer:
    """Context manager to measure elapsed wall time (seconds)."""
    def __init__(self, label: str = ""):
        self.label = label
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self.start #type:ignore
        if self.label:
            print(f"{self.label}: {self.elapsed:.3f}s", file=sys.stderr)


# ---------------- Path helpers ----------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def base_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ---------------- Audio I/O ----------------

def read_audio(path: str, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Read audio to float32 numpy array, resample if needed."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1 and mono:
        y = y.mean(axis=1)
    if sr != target_sr:
        if librosa is None:
            raise RuntimeError("Resampling requires librosa. Install with `pip install librosa`.")
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    sf.write(path, audio, sr, subtype="PCM_16")


# ---------------- Text normalization ----------------

def normalize_text_basic(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    import re
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def tokenize_words(text: str) -> List[str]:
    return normalize_text_basic(text).split()


# ---------------- Loudness ----------------

def rms_dbfs(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(audio)) + 1e-12)
    return 20 * math.log10(rms + 1e-12)

def normalize_loudness(audio: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    cur = rms_dbfs(audio)
    gain_db = target_dbfs - cur
    gain = 10 ** (gain_db / 20)
    out = (audio * gain).astype(np.float32)
    # prevent clipping
    max_abs = np.max(np.abs(out)) + 1e-9
    if max_abs > 1.0:
        out = (out / max_abs).astype(np.float32)
    return out


# ---------------- JSON helpers ----------------

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows