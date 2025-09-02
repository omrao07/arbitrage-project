# workers/sentiment_worker.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from platform import bootstrap # type: ignore
from platform import envelope as env # type: ignore

SERVICE = "sentiment-worker"

# Streams & consumer group (override via env)
IN_STREAM = os.getenv("IN_STREAM", "STREAM_SENTIMENT_REQUESTS")
OUT_STREAM = os.getenv("OUT_STREAM", "STREAM_SENTIMENT_RESULTS")
DLQ_STREAM = os.getenv("DLQ_STREAM", "STREAM_DLQ")
GROUP = os.getenv("GROUP", "sentiment_v1")
CONSUMER = os.getenv("CONSUMER", f"sentiment-{int(time.time())}")

# Idempotency TTL (seconds)
IDEMP_TTL = int(os.getenv("IDEMP_TTL", "86400"))  # 1 day

# Language handling
DEFAULT_LANG = os.getenv("SENTIMENT_LANG", "en")


# --------------------------------------------------------------------------------------
# Domain: Sentiment (NLTK VADER)
# --------------------------------------------------------------------------------------
def _ensure_vader():
    # Lazy download to avoid startup dependency if not present in image
    import nltk

    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # noqa: F401
    except Exception:
        pass  # import triggers no download

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def _score_vader(texts: List[str]) -> List[Dict[str, float]]:
    from nltk.sentiment import SentimentIntensityAnalyzer

    _ensure_vader()
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t or "") for t in texts]


def analyze_texts(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected payload shape:
    {
      "texts": ["headline 1", "headline 2", ...],   # required
      "language": "en",                              # optional (default: env SENTIMENT_LANG)
      "meta": {"source":"news|transcript|tweet", ...}
    }
    Returns:
    {
      "scores": [{"neg":..,"neu":..,"pos":..,"compound":..}, ...],
      "summary": {"avg_compound": ..., "n": N}
    }
    """
    texts = payload.get("texts") or []
    if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
        raise ValueError("invalid_payload: 'texts' must be list[str]")

    language = str(payload.get("language") or DEFAULT_LANG).lower()
    # For now, VADER is English-oriented; for other langs we just return neutral scores.
    if language != "en":
        scores = [{"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0} for _ in texts]
    else:
        scores = _score_vader(texts)

    n = len(scores)
    avg_comp = (sum(s["compound"] for s in scores) / n) if n else 0.0
    return {"scores": scores, "summary": {"avg_compound": avg_comp, "n": n, "language": language}}


# --------------------------------------------------------------------------------------
# Worker plumbing
# --------------------------------------------------------------------------------------
def _ensure_group(r, stream: str, group: str) -> None:
    try:
        r.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception:
        pass  # already exists


def _publish(r, result_env: env.Envelope) -> None:
    r.xadd(OUT_STREAM, result_env.flatten_for_stream())


def main() -> None:
    ctx = bootstrap.init(SERVICE)
    tracer = ctx["tracer"]
    METRICS = ctx["metrics"]
    r = ctx["redis"]
    ent = ctx["ent"]
    audit = ctx["audit"]
    dlq = ctx["dlq"]

    _ensure_group(r, IN_STREAM, GROUP)

    while True:
        resp = r.xreadgroup(GROUP, CONSUMER, {IN_STREAM: ">"}, count=100, block=5000) or []
        if not resp:
            continue

        for stream, messages in resp:
            for msg_id, fields in messages:
                try:
                    raw = fields.get("payload") or "{}"
                    e = env.parse(raw)
                    topic, corr = env.correlation_tuple(e)

                    # Idempotency
                    if env.is_duplicate(r, e.id, ttl_seconds=IDEMP_TTL):
                        r.xack(stream, GROUP, msg_id)
                        continue

                    # Required fields
                    e.require(["texts"])
                    language = (e.payload.get("language") or DEFAULT_LANG).lower()

                    # Entitlements: require 'sentiment/*' read
                    resource = "sentiment/analyze"
                    roles = e.producer.get("roles", []) if isinstance(e.producer, dict) else []
                    if not ent.allow(roles, resource, action="read", region=(e.payload.get("region") or "*")):
                        raise PermissionError("entitlement_denied sentiment/analyze")

                    with tracer.start_as_current_span(
                        "sentiment.analyze", attributes={"corr_id": corr, "topic": topic, "language": language}
                    ):
                        with METRICS.latency_timer("sentiment_analyze"):
                            result = analyze_texts(e.payload)

                    # Audit (store only summary, not full texts)
                    audit.record(
                        action="sentiment_analyze",
                        resource=resource,
                        user=(e.producer or {}).get("user"),
                        corr_id=corr,
                        region=str(e.payload.get("region") or ""),
                        policy_hash=os.getenv("POLICY_HASH"),
                        details={"summary": result["summary"]},
                        input_for_hash={"n": len(e.payload.get("texts") or []), "language": language},
                    )

                    # Publish
                    out = env.new(
                        schema_name="sentiment.result",
                        payload={
                            "request_id": e.id,
                            "corr_id": e.corr_id,
                            "summary": result["summary"],
                            "scores": result["scores"],
                            "meta": e.payload.get("meta", {}),
                        },
                        corr_id=e.corr_id,
                        producer={"svc": SERVICE},
                    )
                    _publish(r, out)

                    r.xack(stream, GROUP, msg_id)

                except PermissionError as pe:
                    METRICS.inc_task("sentiment_analyze", error=True)
                    dlq.push(payload=fields, reason=str(pe), corr_id=fields.get("corr_id"), retries=int(fields.get("retries", "0")))
                    r.xack(stream, GROUP, msg_id)

                except Exception as ex:
                    METRICS.inc_task("sentiment_analyze", error=True)
                    dlq.push(payload=fields, reason=str(ex), corr_id=fields.get("corr_id"), retries=int(fields.get("retries", "0")))
                    r.xack(stream, GROUP, msg_id)


if __name__ == "__main__":
    main()