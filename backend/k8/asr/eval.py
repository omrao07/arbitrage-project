#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr/eval.py
-----------
Evaluate ASR hypotheses vs reference transcripts.

Supports:
  - References: .txt or .jsonl ({"text": "..."} at top level)
  - Hypotheses: .txt or .jsonl (your whisper_loader/wav2vec_loader outputs)
  - Per-file metrics: WER, CER, MER, WIL, token/char counts
  - Optional per-segment metrics if "chunks":[{"text":..., "timestamp":[s,e]}]
  - Summaries (micro/macro averages) + CSV/JSON exports

Usage:
  python asr/eval.py --refs data/asr/refs --hyps outputs/transcripts/whisper --ref-ext txt --hyp-ext jsonl --out-csv asr_eval.csv

  # Segment-level scoring (if hyps JSONL has chunks):
  python asr/eval.py --refs refs/ --hyps outputs/transcripts/wav2vec --by-segment --out-json seg_eval.json

Dependencies:
  pip install jiwer pandas numpy python-levenshtein tabulate
"""

from __future__ import annotations
import os
import re
import json
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional pretty table
try:
    from tabulate import tabulate  # type: ignore
    _HAS_TAB = True
except Exception:
    _HAS_TAB = False

# jiwer for WER/CER/MER/WIL
try:
    import jiwer  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: jiwer. Install with `pip install jiwer`.") from e


# ============================
# I/O helpers
# ============================

def _read_txt(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return {"text": text, "chunks": [], "latency_s": None, "duration_s": None}

def _read_jsonl(path: str) -> Dict[str, Any]:
    # we expect a single-line JSON as written by loaders; if multiple lines, read first non-empty
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                obj = json.loads(s)
                break
        else:
            obj = {}
    text = obj.get("text") or ""
    chunks = obj.get("chunks") or []
    latency = obj.get("latency_s")
    duration = obj.get("duration_s")
    return {"text": str(text), "chunks": chunks, "latency_s": latency, "duration_s": duration}

def _load_item(path: str, ext_hint: Optional[str]) -> Dict[str, Any]:
    ext = ext_hint or os.path.splitext(path)[1].lstrip(".").lower()
    if ext == "txt":
        return _read_txt(path)
    return _read_jsonl(path)

def _gather_files(dirpath: str, ext: str) -> Dict[str, str]:
    patt = os.path.join(dirpath, f"**/*.{ext}")
    files = [p for p in glob.glob(patt, recursive=True) if os.path.isfile(p)]
    # key by basename without extension
    out = {}
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        out[base] = p
    return out


# ============================
# Normalization pipeline
# ============================

def _default_text_normalizer() -> jiwer.Compose:
    """
    Reasonable normalization for ASR:
      - lowercase
      - strip punctuation
      - remove extra spaces
      - normalize numbers (optional; left off by default)
    """
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),  # tokenization
    ])

def _char_normalizer() -> jiwer.Compose:
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])

def normalize_words(text: str) -> List[str]:
    return _default_text_normalizer()(text)[0] if text else []

def normalize_chars(text: str) -> str:
    return _char_normalizer()(text)


# ============================
# Metrics
# ============================

@dataclass
class ASRMetrics:
    wer: float
    cer: float
    mer: float
    wil: float
    n_words_ref: int
    n_words_hyp: int
    n_chars_ref: int
    n_chars_hyp: int
    latency_s: Optional[float]
    duration_s: Optional[float]
    rtf: Optional[float]  # real-time factor = latency/duration

def compute_metrics(ref_text: str, hyp_text: str,
                    latency_s: Optional[float] = None,
                    duration_s: Optional[float] = None) -> ASRMetrics:
    # Word-level
    wer = jiwer.wer(ref_text, hyp_text, truth_transform=_default_text_normalizer(), hypothesis_transform=_default_text_normalizer())
    mer = jiwer.mer(ref_text, hyp_text, truth_transform=_default_text_normalizer(), hypothesis_transform=_default_text_normalizer())
    wil = jiwer.wil(ref_text, hyp_text, truth_transform=_default_text_normalizer(), hypothesis_transform=_default_text_normalizer())
    # Char-level CER (use normalized chars)
    ref_c = normalize_chars(ref_text)
    hyp_c = normalize_chars(hyp_text)
    cer = jiwer.cer(ref_c, hyp_c, truth_transform=jiwer.IdentityTransformation(), hypothesis_transform=jiwer.IdentityTransformation())

    nwr = len(normalize_words(ref_text))
    nwh = len(normalize_words(hyp_text))
    ncr = len(ref_c)
    nch = len(hyp_c)

    rtf = (latency_s / duration_s) if (latency_s and duration_s and duration_s > 0) else None

    return ASRMetrics(
        wer=float(wer), cer=float(cer), mer=float(mer), wil=float(wil),
        n_words_ref=int(nwr), n_words_hyp=int(nwh),
        n_chars_ref=int(ncr), n_chars_hyp=int(nch),
        latency_s=float(latency_s) if latency_s is not None else None,
        duration_s=float(duration_s) if duration_s is not None else None,
        rtf=float(rtf) if rtf is not None else None
    )


# ============================
# Segment alignment (optional)
# ============================

def segments_from_jsonl(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Expect list of {"text":..., "timestamp":[s,e]}
    out = []
    for ch in obj.get("chunks", []):
        txt = (ch.get("text") or "").strip()
        ts = ch.get("timestamp") or ch.get("timestamps")
        if not txt:
            continue
        out.append({"text": txt, "timestamp": ts})
    return out

def score_by_segments(ref_text: str, hyp_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    If we only have reference as a single text but hypothesis has timestamped chunks,
    we split the reference to K segments by greedy alignment on running length.
    Heuristic but useful for sanity checks.
    """
    chunks = segments_from_jsonl(hyp_obj)
    if not chunks:
        return []

    ref_tokens = normalize_words(ref_text)
    ref_len = len(ref_tokens)
    if ref_len == 0:
        return []

    # target tokens per chunk proportional to chunk durations if timestamps exist
    total_dur = 0.0
    durs = []
    for ch in chunks:
        ts = ch.get("timestamp")
        if isinstance(ts, (list, tuple)) and ts and ts[0] is not None and len(ts) == 2:
            dur = max(0.0, float(ts[1] or 0) - float(ts[0] or 0))
        else:
            dur = 1.0
        durs.append(dur)
        total_dur += (dur or 0.0)
    # convert to token quotas
    quotas = [max(1, int(round(ref_len * (d / total_dur)))) if total_dur > 0 else max(1, ref_len // max(1,len(chunks)))
              for d in durs]
    # adjust last quota to hit total exactly
    diff = ref_len - sum(quotas)
    quotas[-1] += diff

    # slice ref tokens to segments
    segs = []
    pos = 0
    for i, ch in enumerate(chunks):
        take = quotas[i] if i < len(quotas) else max(1, ref_len // len(chunks))
        ref_slice = " ".join(ref_tokens[pos:pos+take])
        pos += take
        hyp_text = ch["text"]
        m = compute_metrics(ref_slice, hyp_text)
        segs.append({
            "index": i,
            "timestamp": ch.get("timestamp"),
            "ref_text": ref_slice,
            "hyp_text": hyp_text,
            "wer": m.wer, "cer": m.cer, "mer": m.mer, "wil": m.wil,
            "n_words_ref": m.n_words_ref, "n_words_hyp": m.n_words_hyp
        })
    return segs


# ============================
# Main evaluation
# ============================

def evaluate(ref_dir: str, hyp_dir: str, ref_ext: str, hyp_ext: str,
             by_segment: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    refs = _gather_files(ref_dir, ref_ext)
    hyps = _gather_files(hyp_dir, hyp_ext)

    common = sorted(set(refs.keys()) & set(hyps.keys()))
    missing_ref = sorted(set(hyps.keys()) - set(refs.keys()))
    missing_hyp = sorted(set(refs.keys()) - set(hyps.keys()))
    if missing_ref:
        print(f"⚠️  {len(missing_ref)} hyps without refs (ignored), e.g. {missing_ref[:3]}")
    if missing_hyp:
        print(f"⚠️  {len(missing_hyp)} refs without hyps (ignored), e.g. {missing_hyp[:3]}")

    rows = []
    seg_rows = []

    for base in common:
        ref_obj = _load_item(refs[base], ref_ext)
        hyp_obj = _load_item(hyps[base], hyp_ext)

        m = compute_metrics(ref_obj["text"], hyp_obj["text"],
                            latency_s=hyp_obj.get("latency_s"),
                            duration_s=hyp_obj.get("duration_s"))

        rows.append({
            "file": base,
            "wer": m.wer, "cer": m.cer, "mer": m.mer, "wil": m.wil,
            "n_words_ref": m.n_words_ref, "n_words_hyp": m.n_words_hyp,
            "n_chars_ref": m.n_chars_ref, "n_chars_hyp": m.n_chars_hyp,
            "latency_s": m.latency_s, "duration_s": m.duration_s, "rtf": m.rtf
        })

        if by_segment:
            segs = score_by_segments(ref_obj["text"], hyp_obj)
            for s in segs:
                s["file"] = base
            seg_rows.extend(segs)

    df = pd.DataFrame(rows).sort_values("file")
    seg_df = pd.DataFrame(seg_rows).sort_values(["file","index"]) if by_segment and seg_rows else None
    return df, seg_df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    # micro-averages weight by reference length (words/chars)
    w_words = df["n_words_ref"].replace(0, np.nan)
    w_chars = df["n_chars_ref"].replace(0, np.nan)

    def wavg(col: str, w):
        x = df[col].astype(float)
        return np.nansum(x * w) / np.nansum(w) if np.nansum(w) > 0 else np.nan

    out = pd.DataFrame([{
        "files": len(df),
        "WER_micro": wavg("wer", w_words),
        "CER_micro": wavg("cer", w_chars),
        "MER_micro": wavg("mer", w_words),
        "WIL_micro": wavg("wil", w_words),
        "WER_macro": float(df["wer"].mean()),
        "CER_macro": float(df["cer"].mean()),
        "MER_macro": float(df["mer"].mean()),
        "WIL_macro": float(df["wil"].mean()),
        "RTF_median": float(df["rtf"].median()) if "rtf" in df.columns else np.nan,
        "RTF_mean": float(df["rtf"].mean()) if "rtf" in df.columns else np.nan,
    }])
    return out


# ============================
# CLI
# ============================

def main():
    ap = argparse.ArgumentParser(description="Evaluate ASR outputs vs references")
    ap.add_argument("--refs", required=True, help="Directory of reference transcripts")
    ap.add_argument("--hyps", required=True, help="Directory of hypothesis transcripts")
    ap.add_argument("--ref-ext", default="txt", choices=["txt","jsonl"], help="Reference file extension")
    ap.add_argument("--hyp-ext", default="jsonl", choices=["txt","jsonl"], help="Hypothesis file extension")
    ap.add_argument("--by-segment", action="store_true", help="Score per-segment when chunks w/ timestamps exist (JSONL)")
    ap.add_argument("--out-csv", default=None, help="Write per-file metrics CSV")
    ap.add_argument("--out-json", default=None, help="Write per-file metrics JSON")
    ap.add_argument("--out-seg-csv", default=None, help="Write per-segment CSV (when --by-segment)")
    ap.add_argument("--out-seg-json", default=None, help="Write per-segment JSON (when --by-segment)")
    args = ap.parse_args()

    df, seg_df = evaluate(args.refs, args.hyps, args.ref_ext, args.hyp_ext, by_segment=args.by_segment)
    summ = summarize(df)

    # Print summary
    if not summ.empty:
        print("\n=== Summary ===")
        if _HAS_TAB:
            print(tabulate(summ, headers="keys", floatfmt=".4f", tablefmt="github"))#type:ignore
        else:
            print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Print head of file-level table
    if not df.empty:
        print("\n=== Per-file (head) ===")
        if _HAS_TAB:
            print(tabulate(df.head(10), headers="keys", floatfmt=".4f", tablefmt="github", showindex=False))#type:ignore
        else:
            print(df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    # Save outputs
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"✅ wrote {args.out_csv}")
    if args.out_json:
        df.to_json(args.out_json, orient="records", force_ascii=False, indent=2)
        print(f"✅ wrote {args.out_json}")
    if args.by_segment and seg_df is not None:
        if args.out_seg_csv:
            seg_df.to_csv(args.out_seg_csv, index=False)
            print(f"✅ wrote {args.out_seg_csv}")
        if args.out_seg_json:
            seg_df.to_json(args.out_seg_json, orient="records", force_ascii=False, indent=2)
            print(f"✅ wrote {args.out_seg_json}")

if __name__ == "__main__":
    main()