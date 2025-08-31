# backend/ai/agents/connectors/llm/openai.py
from __future__ import annotations

import os
import base64
import time
from typing import List, Dict, Any, Optional, Union

# ============================================================
# Environment
# ============================================================
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED = os.getenv("OPENAI_EMBED", "text-embedding-3-small")
OPENAI_TTS   = os.getenv("OPENAI_TTS", "gpt-4o-mini-tts")
OPENAI_STT   = os.getenv("OPENAI_STT", "gpt-4o-mini-transcribe")

# ============================================================
# Client
# ============================================================
_client: Any = None
_backend_ok: bool = False

def _init_client() -> None:
    """Lazy-init OpenAI client if possible."""
    global _client, _backend_ok
    if _client is not None:
        return
    try:
        from openai import OpenAI # type: ignore
        _client = OpenAI(api_key=OPENAI_KEY)
        _backend_ok = True
    except Exception:
        _client = None
        _backend_ok = False

# ============================================================
# Chat completion
# ============================================================
def chat_complete(messages: List[Dict[str,str]], *,
                  model: Optional[str] = None,
                  max_tokens: int = 512,
                  temperature: float = 0.7,
                  stop: Optional[List[str]] = None,
                  system: Optional[str] = None) -> str:
    _init_client()
    if not _backend_ok:
        return "[openai-stub: no client]"
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)

    resp = _client.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    return resp.choices[0].message.content.strip()

# ============================================================
# Simple text generation
# ============================================================
def generate(prompt: str, *,
             model: Optional[str] = None,
             max_tokens: int = 512,
             temperature: float = 0.7,
             stop: Optional[List[str]] = None) -> str:
    return chat_complete([{"role":"user","content":prompt}],
                         model=model,
                         max_tokens=max_tokens,
                         temperature=temperature,
                         stop=stop)

# ============================================================
# Embeddings
# ============================================================
def embed(texts: Union[str,List[str]], *, model: Optional[str] = None) -> List[List[float]]:
    _init_client()
    if not _backend_ok:
        return [[0.0]*16] if isinstance(texts,str) else [[0.0]*16 for _ in texts]
    if isinstance(texts,str):
        texts=[texts]
    resp = _client.embeddings.create(
        model=model or OPENAI_EMBED,
        input=texts
    )
    return [d.embedding for d in resp.data]

# ============================================================
# Speech-to-Text
# ============================================================
def stt_transcribe(audio_bytes: bytes, *, model: Optional[str] = None) -> str:
    _init_client()
    if not _backend_ok:
        return "[openai-stub: no STT]"
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(audio_bytes)
        f.flush()
        resp = _client.audio.transcriptions.create(
            model=model or OPENAI_STT,
            file=open(f.name, "rb"),
        )
        return resp.text.strip()

# ============================================================
# Text-to-Speech
# ============================================================
def tts_speak(text: str, *, model: Optional[str] = None, voice: str = "alloy") -> bytes:
    _init_client()
    if not _backend_ok:
        return f"[openai-stub: no TTS]::{text}".encode()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    resp = _client.audio.speech.create(
        model=model or OPENAI_TTS,
        voice=voice,
        input=text
    )
    data = resp.read()
    return data

def tts_speak_b64(text: str, *, model: Optional[str] = None, voice: str = "alloy") -> str:
    return base64.b64encode(tts_speak(text, model=model, voice=voice)).decode("ascii")

# ============================================================
# Health
# ============================================================
def health() -> Dict[str,Any]:
    _init_client()
    return {
        "backend": "openai" if _backend_ok else "stub",
        "model": OPENAI_MODEL,
        "embed": OPENAI_EMBED,
        "tts": OPENAI_TTS,
        "stt": OPENAI_STT,
        "key_present": bool(OPENAI_KEY),
    }

# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":  # pragma: no cover
    print("health:", health())
    t0=time.time()
    print("chat:", chat_complete([{"role":"user","content":"Say hi in one word"}]))
    print("took:", round(time.time()-t0,2),"s")
    print("embed dims:", len(embed("hello")[0]))