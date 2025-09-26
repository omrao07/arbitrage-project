#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wav2vec_loader.py
-----------------
Loader + thin wrapper around Hugging Face Wav2Vec2 ASR.

- Reads kb/asr/configs/wav2vec.yaml
- Builds a pipeline("automatic-speech-recognition")
- Provides: transcribe_file(), transcribe_batch(), transcribe_dir()
- Writes JSONL outputs compatible with your kb/asr outputs/

Dependencies:
  pip install transformers datasets soundfile librosa pyyaml torch --upgrade
"""

from __future__ import annotations
import os
import sys
import json
import time
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Optional heavy deps (fail fast with a clear message)
try:
    from transformers import pipeline  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: transformers. pip install transformers") from e

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import soundfile as sf  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: soundfile. pip install soundfile") from e

# -------------------------------------------------------

DEFAULT_CFG_PATH = os.path.join("kb", "asr", "configs", "wav2vec.yaml")


@dataclass
class Wav2VecConfig:
    model_name: str = "facebook/wav2vec2-large-960h"
    tokenizer_name: Optional[str] = None
    language: str = "en-US"
    sample_rate: int = 16000
    chunk_size_s: float = 20.0
    stride_s: float = 5.0
    padding: bool = True
    normalize: bool = True
    device: str = "auto"
    transcripts_dir: str = os.path.join("outputs", "transcripts", "wav2vec")
    save_format: str = "jsonl"
    include_timestamps: bool = True
    save_confidence: bool = True
    hotwords: Optional[List[Dict[str, Any]]] = None
    log_level: str = "INFO"


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str = DEFAULT_CFG_PATH) -> Wav2VecConfig:
    cfg = _read_yaml(path)
    asr = cfg.get("asr", {})
    infer = cfg.get("inference", {})
    runtime = cfg.get("runtime", {})
    out = cfg.get("output", {})

    return Wav2VecConfig(
        model_name=asr.get("model_name", "facebook/wav2vec2-large-960h"),
        tokenizer_name=asr.get("tokenizer_name") or asr.get("model_name"),
        language=asr.get("language", "en-US"),
        sample_rate=asr.get("sample_rate", 16000),
        chunk_size_s=float(infer.get("chunk_size_s", 20.0)),
        stride_s=float(infer.get("stride_s", 5.0)),
        padding=bool(infer.get("padding", True)),
        normalize=bool(infer.get("normalize", True)),
        device=runtime.get("device", "auto"),
        transcripts_dir=out.get("transcripts_dir", os.path.join("outputs", "transcripts", "wav2vec")),
        save_format=out.get("save_format", "jsonl"),
        include_timestamps=bool(out.get("include_timestamps", True)),
        save_confidence=bool(out.get("save_confidence", True)),
        hotwords=infer.get("hotwords"),
        log_level=runtime.get("log_level", "INFO"),
    )


def _pick_device(device: str) -> int:
    """
    Returns HF pipeline device index:
      -1 = CPU, 0..N = CUDA device
    """
    if device == "cpu":
        return -1
    if device == "cuda" or device == "auto":
        if torch is not None and torch.cuda.is_available():
            return 0
        return -1
    try:
        # numeric string like "1"
        return int(device)
    except Exception:
        return -1


class Wav2VecLoader:
    def __init__(self, cfg_path: str = DEFAULT_CFG_PATH):
        self.cfg = load_config(cfg_path)
        os.makedirs(self.cfg.transcripts_dir, exist_ok=True)
        self.device = _pick_device(self.cfg.device)

        # HF ASR pipeline with chunking/stride to handle long files
        kwargs = dict(
            model=self.cfg.model_name,
            tokenizer=self.cfg.tokenizer_name or self.cfg.model_name,
            chunk_length_s=self.cfg.chunk_size_s,
            stride_length_s=self.cfg.stride_s,
            return_timestamps=self.cfg.include_timestamps,  # per-chunk timestamps
        )
        # Some models support "language" argument (esp. Whisper), wav2vec2 usually ignores
        try:
            self.asr = pipeline(
                task="automatic-speech-recognition",#type:ignore
                model=self.cfg.model_name,
                device=self.device,
              
            )
        except TypeError:
            # Older transformers versions may not support some kwargs
            kwargs.pop("return_timestamps", None)
            self.asr = pipeline(
                task="automatic-speech-recognition",#type:ignore
                model=self.cfg.model_name,
                device=self.device,
                
            )

    # ---------- I/O ----------
    @staticmethod
    def _read_audio(path: str, target_sr: int) -> Tuple[List[float], int]:
        """
        Returns (mono_float32, sample_rate). Resamples if needed using soundfile’s data.
        """
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            # mix down to mono
            audio = audio.mean(axis=1)
        if sr != target_sr:
            # light resample via librosa (optional dep)
            try:
                import librosa  # type: ignore
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception as e:
                raise RuntimeError(
                    f"Audio sample rate {sr} != {target_sr}. Install librosa for resampling: pip install librosa"
                ) from e
        return audio, sr#type:ignore

    def _to_output_path(self, audio_path: str) -> str:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        ext = "jsonl" if self.cfg.save_format.lower() == "jsonl" else "txt"
        return os.path.join(self.cfg.transcripts_dir, f"{base}.{ext}")

    # ---------- Core ----------
    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Returns dict:
          {
            "audio_path": ...,
            "model": ...,
            "language": ...,
            "duration_s": ...,
            "text": "...",
            "chunks": [ { "timestamp": [start,end], "text": "...", "confidence": 0.xx? } ... ]
          }
        """
        t0 = time.time()
        audio, sr = self._read_audio(audio_path, self.cfg.sample_rate)

        # Hugging Face pipeline can accept raw arrays with sampling rate
        result = self.asr({"array": audio, "sampling_rate": sr})

        payload: Dict[str, Any] = {
            "audio_path": audio_path,
            "model": self.cfg.model_name,
            "language": self.cfg.language,
            "duration_s": len(audio) / float(sr),
            "text": result["text"] if isinstance(result, dict) and "text" in result else str(result),#type:ignore
            "chunks": [],
        }

        # Attempt to capture timestamps and confidence if provided
        if isinstance(result, dict):
            # Some models return "chunks": [{"text":..., "timestamp": (s,e)}]
            chunks = result.get("chunks") or []
            out_chunks: List[Dict[str, Any]] = []
            for ch in chunks:
                item = {
                    "text": ch.get("text", ""),
                    "timestamp": ch.get("timestamp") or ch.get("timestamps"),
                }
                if self.cfg.save_confidence and "score" in ch:
                    item["confidence"] = ch["score"]
                out_chunks.append(item)
            payload["chunks"] = out_chunks

        payload["latency_s"] = round(time.time() - t0, 3)
        return payload

    def save_transcript(self, transcript: Dict[str, Any]) -> str:
        out_path = self._to_output_path(transcript["audio_path"])
        if self.cfg.save_format.lower() == "jsonl":
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(transcript, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(transcript.get("text", ""))
        return out_path

    # ---------- Batch helpers ----------
    def transcribe_batch(self, audio_paths: List[str], save: bool = True) -> List[Dict[str, Any]]:
        outs = []
        for p in audio_paths:
            try:
                tr = self.transcribe_file(p)
                if save:
                    self.save_transcript(tr)
                outs.append(tr)
                print(f"✓ {p}  →  {len(tr.get('text',''))} chars")
            except Exception as e:
                print(f"✗ {p}: {e}", file=sys.stderr)
        return outs

    def transcribe_dir(self, audio_dir: str, pattern: str = "**/*.wav", save: bool = True) -> List[Dict[str, Any]]:
        paths = [p for p in glob.glob(os.path.join(audio_dir, pattern), recursive=True) if os.path.isfile(p)]
        return self.transcribe_batch(paths, save=save)


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Wav2Vec2 ASR loader/transcriber")
    ap.add_argument("--config", default=DEFAULT_CFG_PATH, help="Path to kb/asr/configs/wav2vec.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("file", help="Transcribe a single audio file")
    s1.add_argument("audio", help="Path to audio (wav/flac/mp3)")
    s1.add_argument("--no-save", action="store_true", help="Do not write transcript file")

    s2 = sub.add_parser("dir", help="Transcribe a directory of audio files")
    s2.add_argument("audio_dir", help="Directory path")
    s2.add_argument("--glob", default="**/*.wav", help="Glob pattern (default **/*.wav)")
    s2.add_argument("--no-save", action="store_true")

    args = ap.parse_args()

    loader = Wav2VecLoader(args.config)

    if args.cmd == "file":
        tr = loader.transcribe_file(args.audio)
        if not args.no_save:
            out = loader.save_transcript(tr)
            print(f"✅ wrote {out}")
        else:
            print(json.dumps(tr, ensure_ascii=False, indent=2))
        return

    if args.cmd == "dir":
        trs = loader.transcribe_dir(args.audio_dir, pattern=args.glob, save=not args.no_save)
        print(f"✅ processed {len(trs)} files")

if __name__ == "__main__":
    main()