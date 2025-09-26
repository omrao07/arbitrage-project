#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr/preprocess.py
-----------------
Audio preprocessing pipeline for ASR:
  - Load (wav/flac/mp3/m4a) → mono float32
  - Resample to target sample rate
  - (optional) Loudness normalization
  - (optional) Denoise (spectral gating)
  - (optional) VAD trimming (WebRTC VAD) or VAD-based segmentation
  - (optional) Fixed-length chunking with overlap
  - Save cleaned .wav + manifest (CSV/JSONL)

Examples:
  # simple standardize to 16k mono wav
  python asr/preprocess.py --in data/raw_calls --out data/clean --sr 16000

  # with VAD segmentation + denoise + normalize
  python asr/preprocess.py --in data/raw_calls --out data/clean --sr 16000 \
      --vad --vad-aggressiveness 2 --denoise --normalize \
      --max-chunk-s 30 --overlap-s 5 --manifest out/clean_manifest.csv

Dependencies:
  pip install soundfile librosa numpy pandas pydub
  # optional:
  pip install webrtcvad noisereduce
"""

from __future__ import annotations
import os
import io
import re
import sys
import json
import math
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# I/O
import soundfile as sf

# Optional deps
try:
    import librosa  # resample + effects
except Exception:
    librosa = None

try:
    import webrtcvad  # fast VAD
except Exception:
    webrtcvad = None

try:
    import noisereduce as nr  # type: ignore # spectral noise gating
except Exception:
    nr = None


# =========================
# Helpers
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_audio(path: str) -> Tuple[np.ndarray, int]:
    """Read audio file to float32 array and sample rate."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # mix to mono
    return data, sr

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if orig_sr == target_sr:
        return audio, orig_sr
    if librosa is None:
        raise RuntimeError("Resampling requires librosa. Install with `pip install librosa`.")
    y = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return y.astype(np.float32), target_sr

def normalize_loudness(audio: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    """Simple RMS loudness normalize to target dBFS."""
    rms = np.sqrt(np.mean(np.square(audio)) + 1e-12)
    dbfs = 20 * math.log10(rms + 1e-12)
    gain_db = target_dbfs - dbfs
    gain = 10 ** (gain_db / 20)
    out = (audio * gain).astype(np.float32)
    # prevent clipping
    max_abs = np.max(np.abs(out)) + 1e-9
    if max_abs > 1.0:
        out = (out / max_abs).astype(np.float32)
    return out

def spectral_denoise(audio: np.ndarray, sr: int, prop_decrease: float = 0.9) -> np.ndarray:
    if nr is None:
        raise RuntimeError("Denoise requires noisereduce. Install with `pip install noisereduce`.")
    # Estimate noise from first 0.5s (heuristic)
    n = int(sr * 0.5)
    noise_clip = audio[:min(n, len(audio))]
    out = nr.reduce_noise(y=audio, y_noise=noise_clip, prop_decrease=prop_decrease, sr=sr)
    return out.astype(np.float32)

def frame_generator(audio: np.ndarray, sr: int, frame_ms: int = 30, hop_ms: int = 10) -> List[np.ndarray]:
    """Yield overlapping frames."""
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    out = []
    for start in range(0, max(1, len(audio) - frame_len + 1), hop):
        out.append(audio[start:start + frame_len])
    return out

def vad_segments(audio: np.ndarray, sr: int, aggressiveness: int = 2,
                 frame_ms: int = 30, hop_ms: int = 10, min_speech_ms: int = 250,
                 max_silence_ms: int = 300) -> List[Tuple[int, int]]:
    """
    Return speech segments as list of (start_sample, end_sample).
    """
    if webrtcvad is None:
        raise RuntimeError("VAD requires webrtcvad. Install with `pip install webrtcvad`.")
    vad = webrtcvad.Vad(aggressiveness)

    # webrtcvad expects 16-bit PCM 8/16/32k mono with frame sizes 10/20/30ms
    # Ensure 16k for best results; resample if needed (non-destructive to caller)
    work = audio
    work_sr = sr
    if sr not in (8000, 16000, 32000):
        if librosa is None:
            raise RuntimeError("VAD resampling requires librosa. Install with `pip install librosa`.")
        work, work_sr = resample(audio, sr, 16000)

    # build frames and raw bytes
    frame_len = int(work_sr * frame_ms / 1000)
    hop = int(work_sr * hop_ms / 1000)
    speech_flags: List[bool] = []

    for start in range(0, max(1, len(work) - frame_len + 1), hop):
        frame = work[start:start+frame_len]
        pcm16 = np.clip(frame * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        is_speech = vad.is_speech(pcm16, work_sr)
        speech_flags.append(is_speech)

    # merge flags into segments on original SR scale
    min_speech_frames = max(1, int(min_speech_ms / hop_ms))
    max_silence_frames = max(1, int(max_silence_ms / hop_ms))

    segments = []
    start_idx = None
    silence_run = 0
    for i, flag in enumerate(speech_flags):
        if flag:
            silence_run = 0
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                silence_run += 1
                if silence_run >= max_silence_frames:
                    # close segment
                    end_idx = i - silence_run + 1
                    if (end_idx - start_idx) >= min_speech_frames:
                        a = int((start_idx * hop) * (sr / work_sr))
                        b = int((end_idx * hop + frame_len) * (sr / work_sr))
                        segments.append((max(0, a), min(len(audio), b)))
                    start_idx, silence_run = None, 0
    # tail
    if start_idx is not None:
        a = int((start_idx * hop) * (sr / work_sr))
        b = len(audio)
        if b - a > int(sr * (min_speech_ms / 1000.0)):
            segments.append((max(0, a), min(len(audio), b)))

    # merge small gaps
    merged = []
    gap_thresh = int(sr * 0.2)
    for s in segments:
        if not merged:
            merged.append(s); continue
        last_s, last_e = merged[-1]
        if s[0] - last_e <= gap_thresh:
            merged[-1] = (last_s, s[1])
        else:
            merged.append(s)
    return merged

def fixed_chunks(total_len: int, sr: int, max_chunk_s: float, overlap_s: float) -> List[Tuple[int,int]]:
    if max_chunk_s <= 0:
        return [(0, total_len)]
    step = int((max_chunk_s - overlap_s) * sr) if max_chunk_s > overlap_s else int(max_chunk_s * sr)
    win = int(max_chunk_s * sr)
    out = []
    pos = 0
    while pos < total_len:
        a = pos
        b = min(total_len, pos + win)
        out.append((a, b))
        if b == total_len:
            break
        pos += step
    return out

def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio, sr, subtype="PCM_16")

def relpath_no_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

# =========================
# Core pipeline
# =========================

@dataclass
class PreprocessConfig:
    sr: int = 16000
    normalize: bool = False
    target_dbfs: float = -23.0
    denoise: bool = False
    denoise_strength: float = 0.9
    vad: bool = False
    vad_aggressiveness: int = 2  # 0..3
    vad_frame_ms: int = 30
    vad_hop_ms: int = 10
    vad_min_speech_ms: int = 250
    vad_max_silence_ms: int = 300
    max_chunk_s: float = 0.0
    overlap_s: float = 0.0

def process_file(path: str, out_dir: str, cfg: PreprocessConfig) -> List[Dict[str, Any]]:
    # load
    y, sr = read_audio(path)
    # resample
    y, sr = resample(y, sr, cfg.sr)
    # denoise
    if cfg.denoise:
        y = spectral_denoise(y, sr, prop_decrease=cfg.denoise_strength)
    # normalize loudness
    if cfg.normalize:
        y = normalize_loudness(y, target_dbfs=cfg.target_dbfs)

    # segmenting
    segments: List[Tuple[int,int]]
    if cfg.vad:
        segments = vad_segments(
            y, sr,
            aggressiveness=cfg.vad_aggressiveness,
            frame_ms=cfg.vad_frame_ms,
            hop_ms=cfg.vad_hop_ms,
            min_speech_ms=cfg.vad_min_speech_ms,
            max_silence_ms=cfg.vad_max_silence_ms
        )
        # if VAD found nothing, fallback to whole file
        if not segments:
            segments = [(0, len(y))]
    else:
        segments = [(0, len(y))]

    # optional fixed chunking inside each segment
    final_spans: List[Tuple[int,int]] = []
    for (a, b) in segments:
        if cfg.max_chunk_s and cfg.max_chunk_s > 0:
            final_spans.extend([(a+sa, a+sb) for (sa, sb) in fixed_chunks(b - a, sr, cfg.max_chunk_s, cfg.overlap_s)])
        else:
            final_spans.append((a, b))

    # write
    rel = relpath_no_ext(path)
    file_rows: List[Dict[str, Any]] = []
    for i, (a, b) in enumerate(final_spans):
        chunk = y[a:b]
        out_name = f"{rel}_seg{i:03d}.wav" if len(final_spans) > 1 else f"{rel}.wav"
        out_path = os.path.join(out_dir, out_name)
        save_wav(out_path, chunk, sr)
        file_rows.append({
            "source": path,
            "output": out_path,
            "sample_rate": sr,
            "start_s": round(a / sr, 3),
            "end_s": round(b / sr, 3),
            "duration_s": round((b - a) / sr, 3),
            "vad": cfg.vad,
            "denoise": cfg.denoise,
            "normalize": cfg.normalize
        })
    return file_rows


def gather_inputs(inp: str, pattern: Optional[str]) -> List[str]:
    if os.path.isfile(inp):
        return [inp]
    if os.path.isdir(inp):
        pat = pattern or "**/*"
        exts = (".wav",".flac",".mp3",".m4a",".ogg")
        files = [p for p in glob.glob(os.path.join(inp, pat), recursive=True) if os.path.splitext(p)[1].lower() in exts]
        return files
    raise FileNotFoundError(inp)


def write_manifest(rows: List[Dict[str, Any]], path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Preprocess audio for ASR (resample/mono/denoise/VAD/chunk)")
    ap.add_argument("--in", dest="inp", required=True, help="Input file or directory")
    ap.add_argument("--glob", default=None, help="Glob pattern within directory (default=**/*)")
    ap.add_argument("--out", required=True, help="Output directory for cleaned wavs")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate (default 16000)")
    ap.add_argument("--normalize", action="store_true", help="Enable loudness normalization (to --target-dbfs)")
    ap.add_argument("--target-dbfs", type=float, default=-23.0, help="Target loudness dBFS for normalization")
    ap.add_argument("--denoise", action="store_true", help="Enable spectral denoise (noisereduce)")
    ap.add_argument("--denoise-strength", type=float, default=0.9, help="Denoise prop_decrease (0..1)")
    ap.add_argument("--vad", action="store_true", help="Enable WebRTC VAD trimming/segmentation")
    ap.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0,1,2,3], help="VAD aggressiveness (0..3)")
    ap.add_argument("--vad-frame-ms", type=int, default=30, help="VAD frame size ms (10/20/30)")
    ap.add_argument("--vad-hop-ms", type=int, default=10, help="VAD hop size ms")
    ap.add_argument("--vad-min-speech-ms", type=int, default=250, help="Min speech length to keep (ms)")
    ap.add_argument("--vad-max-silence-ms", type=int, default=300, help="Silence threshold to close a segment (ms)")
    ap.add_argument("--max-chunk-s", type=float, default=0.0, help="Fixed chunk length seconds (0 = off)")
    ap.add_argument("--overlap-s", type=float, default=0.0, help="Overlap between fixed chunks seconds")
    ap.add_argument("--manifest", default=None, help="Write CSV or JSONL manifest mapping source→output(s)")
    args = ap.parse_args()

    cfg = PreprocessConfig(
        sr=args.sr,
        normalize=args.normalize,
        target_dbfs=args.target_dbfs,
        denoise=args.denoise,
        denoise_strength=args.denoise_strength,
        vad=args.vad,
        vad_aggressiveness=args.vad_aggressiveness,
        vad_frame_ms=args.vad_frame_ms,
        vad_hop_ms=args.vad_hop_ms,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_max_silence_ms=args.vad_max_silence_ms,
        max_chunk_s=args.max_chunk_s,
        overlap_s=args.overlap_s,
    )

    ensure_dir(args.out)
    files = gather_inputs(args.inp, args.glob)
    all_rows: List[Dict[str, Any]] = []

    for p in files:
        try:
            rows = process_file(p, args.out, cfg)
            all_rows.extend(rows)
            print(f"✓ {p} → {len(rows)} file(s)")
        except Exception as e:
            print(f"✗ {p}: {e}", file=sys.stderr)

    if all_rows:
        write_manifest(all_rows, args.manifest)
        if args.manifest:
            print(f"✅ wrote manifest: {args.manifest}")
        print(f"✅ finished: {len(all_rows)} output file(s)")
    else:
        print("No outputs produced.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()