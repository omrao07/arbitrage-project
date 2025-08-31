# backend/ai/agents/connectors/llm/local_gguf.py
from __future__ import annotations

import os
import time
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# ============================================================
# Environment configuration
# ============================================================
GGUF_MODEL       = os.getenv("GGUF_MODEL", "./models/llama-3.1-8b-instruct.Q4_K_M.gguf")
GGUF_CTX         = int(os.getenv("GGUF_CTX", "4096"))
GGUF_THREADS     = int(os.getenv("GGUF_THREADS", "4"))
GGUF_GPU_LAYERS  = int(os.getenv("GGUF_GPU_LAYERS", "0"))  # >0 uses GPU offload (if compiled)
GGUF_BATCH       = int(os.getenv("GGUF_BATCH", "512"))
GGUF_TEMP        = float(os.getenv("GGUF_TEMP", "0.7"))
GGUF_TOP_P       = float(os.getenv("GGUF_TOP_P", "0.9"))
GGUF_TOP_K       = int(os.getenv("GGUF_TOP_K", "40"))
GGUF_REP_PEN     = float(os.getenv("GGUF_REP_PEN", "1.1"))

# Embeddings (use the same model if it supports it; set EMBED_MODEL to override)
EMBED_MODEL      = os.getenv("EMBED_MODEL", GGUF_MODEL)

# Optional speech models (offline)
WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "base")         # faster-whisper model id or path
PIPER_VOICE      = os.getenv("PIPER_VOICE", "")               # e.g., "en_US-amy-low.onnx"
PIPER_NOISE_W    = float(os.getenv("PIPER_NOISE_W", "0.667"))
PIPER_LENGTH_W   = float(os.getenv("PIPER_LENGTH_W", "1.0"))

# ============================================================
# Optional backends
# ============================================================
_llama: Any = None
_backend: str = "stub"
_embed_support: bool = False

def _try_init_llama() -> None:
    """Initialize a local GGUF runtime if possible (llama-cpp preferred, then ctransformers)."""
    global _llama, _backend, _embed_support
    if _llama is not None:
        return
    # Try llama-cpp-python
    try:
        from llama_cpp import Llama  # type: ignore
        _llama = Llama(
            model_path=GGUF_MODEL,
            n_ctx=GGUF_CTX,
            n_threads=GGUF_THREADS,
            n_gpu_layers=GGUF_GPU_LAYERS,
            n_batch=GGUF_BATCH,
            logits_all=False,
            embedding=True,  # allow embeddings if model supports
            verbose=False,
        )
        _backend = "llama_cpp"
        _embed_support = True
        return
    except Exception:
        _llama = None

    # Try ctransformers (fallback)
    try:
        from ctransformers import AutoModelForCausalLM  # type: ignore
        _llama = AutoModelForCausalLM.from_pretrained(
            GGUF_MODEL,
            model_type="llama",
            context_length=GGUF_CTX,
            gpu_layers=GGUF_GPU_LAYERS,
            threads=GGUF_THREADS,
        )
        _backend = "ctransformers"
        _embed_support = False  # ctransformers embedding support varies; default off
        return
    except Exception:
        _llama = None
        _backend = "stub"
        _embed_support = False

# ============================================================
# Public: Text generation
# ============================================================
def generate(prompt: str, *, temperature: float = GGUF_TEMP, top_p: float = GGUF_TOP_P,
             top_k: int = GGUF_TOP_K, max_tokens: int = 512, stop: Optional[List[str]] = None) -> str:
    """
    Simple single-turn generation.
    """
    _try_init_llama()
    if _backend == "llama_cpp":
        out = _llama(  # type: ignore
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
            repeat_penalty=GGUF_REP_PEN,
        )
        return (out.get("choices", [{}])[0].get("text") or "").strip()
    elif _backend == "ctransformers":
        # ctransformers uses a generator interface
        txt = _llama(  # type: ignore
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            repetition_penalty=GGUF_REP_PEN,
        )
        return str(txt).strip()
    # stub
    return "[local-gguf-stub] (install llama-cpp-python or ctransformers)"

# ============================================================
# Public: Chat completion (messages = [{role, content}])
# ============================================================
def chat_complete(messages: List[Dict[str, str]], *,
                  system: Optional[str] = None,
                  temperature: float = GGUF_TEMP,
                  top_p: float = GGUF_TOP_P,
                  top_k: int = GGUF_TOP_K,
                  max_tokens: int = 512,
                  stop: Optional[List[str]] = None) -> str:
    """
    Minimal chat wrapper that builds a prompt in OpenAI-ish format.
    """
    _try_init_llama()
    sys_prompt = system or "You are a helpful trading assistant."
    # Build a simple chat template
    chat_txt = f"<s>[SYSTEM]\n{sys_prompt}\n"
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        chat_txt += f"[{role}]\n{content}\n"
    chat_txt += "[ASSISTANT]\n"

    if _backend == "llama_cpp":
        out = _llama(  # type: ignore
            prompt=chat_txt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=(stop or []) + ["</s>", "[USER]", "[SYSTEM]"],
            repeat_penalty=GGUF_REP_PEN,
        )
        return (out.get("choices", [{}])[0].get("text") or "").strip()

    elif _backend == "ctransformers":
        txt = _llama(  # type: ignore
            chat_txt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=(stop or []) + ["</s>", "[USER]", "[SYSTEM]"],
            repetition_penalty=GGUF_REP_PEN,
        )
        return str(txt).strip()

    return "[local-gguf-stub] (no backend available)"

# ============================================================
# Public: Embeddings
# ============================================================
def embed(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Returns a list of embeddings (one per text). If backend/model doesn’t
    support embeddings, returns small deterministic stubs.
    """
    _try_init_llama()
    if isinstance(texts, str):
        texts = [texts]

    if _backend == "llama_cpp" and _embed_support:
        vecs: List[List[float]] = []
        for t in texts:
            r = _llama.create_embedding(t)  # type: ignore
            v = r.get("data", [{}])[0].get("embedding") or []
            vecs.append([float(x) for x in v])
        return vecs

    # fallback deterministic 16-dim pseudo-embeddings
    def _stub_vec(s: str) -> List[float]:
        h = abs(hash(s))
        return [((h >> (i * 3)) & 0xFF) / 255.0 for i in range(16)]
    return [_stub_vec(t) for t in texts]

# ============================================================
# Public: Speech-to-Text (offline optional)
# ============================================================
def stt_transcribe(audio_bytes: bytes, language: Optional[str] = None) -> str:
    """
    Transcribe audio with faster-whisper if installed; otherwise a stub.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(audio_bytes)
            f.flush()
            segments, _ = model.transcribe(f.name, language=language, vad_filter=True)
            txt = " ".join([s.text for s in segments]).strip()
            return txt or "[stt-empty]"
    except Exception:
        return "[no-stt] faster-whisper not installed or audio invalid"

# ============================================================
# Public: Text-to-Speech (offline optional)
# ============================================================
def tts_speak(text: str, voice: str = "neutral") -> bytes:
    """
    Synthesize speech with piper-tts if installed and voice provided; otherwise a stub.
    Returns WAV bytes.
    """
    try:
        if not PIPER_VOICE:
            raise RuntimeError("no piper voice configured")
        import subprocess
        import tempfile
        # call piper CLI: echo "text" | piper --model VOICE --output_file out.wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_path = tmp_out.name
        cmd = [
            "piper",
            "--model", PIPER_VOICE,
            "--output_file", out_path,
            "--noise_w", str(PIPER_NOISE_W),
            "--length_scale", str(PIPER_LENGTH_W),
        ]
        # Pipe the text to stdin
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate(input=text.encode("utf-8"), timeout=30)
        # Read WAV bytes
        with open(out_path, "rb") as f:
            data = f.read()
        try:
            os.remove(out_path)
        except Exception:
            pass
        return data
    except Exception:
        # Return a tiny “fake wav” header + text payload (for UIs that expect bytes)
        fake = f"FAKE-TTS(local_gguf::{voice})::{text}".encode("utf-8")
        return fake

# ============================================================
# Convenience: base64 helpers (useful for web sockets / UI)
# ============================================================
def tts_speak_b64(text: str, voice: str = "neutral") -> str:
    return base64.b64encode(tts_speak(text, voice)).decode("ascii")

# ============================================================
# Health
# ============================================================
def health() -> Dict[str, Any]:
    _try_init_llama()
    return {
        "backend": _backend,
        "model": GGUF_MODEL,
        "ctx": GGUF_CTX,
        "threads": GGUF_THREADS,
        "gpu_layers": GGUF_GPU_LAYERS,
        "embed_support": _embed_support,
    }

# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":  # pragma: no cover
    print("health:", health())
    t0 = time.time()
    out = chat_complete([
        {"role": "user", "content": "In one line, what is VWAP?"}
    ], max_tokens=64)
    print("chat:", out)
    print("took:", round(time.time() - t0, 2), "s")
    print("embed:", len(embed("hello")[0]), "dims")
    print("tts_b64_len:", len(tts_speak_b64("Hello from local GGUF!") or ""))