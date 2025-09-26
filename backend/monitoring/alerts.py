# orchestrator/alerts.py
from __future__ import annotations

import os
import smtplib
import ssl
import time
import json
import logging
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "alerts.log", mode="a"), logging.StreamHandler()]
)
log = logging.getLogger("alerts")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    slack_webhook: Optional[str] = None         # https://hooks.slack.com/services/...
    discord_webhook: Optional[str] = None       # https://discord.com/api/webhooks/...
    telegram_bot_token: Optional[str] = None    # e.g. '123456:ABC...'
    telegram_chat_id: Optional[str] = None      # numeric or @username
    smtp_host: Optional[str] = None             # e.g. 'smtp.gmail.com'
    smtp_port: Optional[int] = 587
    smtp_user: Optional[str] = None
    smtp_pass: Optional[str] = None
    email_to: List[str] = field(default_factory=list)  # recipient list
    file_sink: Optional[Path] = LOG_DIR / "alerts_events.jsonl"  # persistent JSONL sink
    # Defaults: pick up from env if not provided
    def hydrate_from_env(self):
        self.slack_webhook = self.slack_webhook or os.getenv("SLACK_WEBHOOK_URL")
        self.discord_webhook = self.discord_webhook or os.getenv("DISCORD_WEBHOOK_URL")
        self.telegram_bot_token = self.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = self.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.smtp_host = self.smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", str(self.smtp_port or 587)))
        self.smtp_user = self.smtp_user or os.getenv("SMTP_USER")
        self.smtp_pass = self.smtp_pass or os.getenv("SMTP_PASS")
        if not self.email_to:
            env_to = os.getenv("ALERT_EMAIL_TO", "")
            if env_to:
                self.email_to = [x.strip() for x in env_to.split(",") if x.strip()]


@dataclass
class PolicyConfig:
    throttle_secs: int = 30                    # minimum seconds between duplicate alerts
    max_per_minute: int = 30                   # circuit-breaker on spam
    dedup_window_secs: int = 300               # dedup identical messages
    min_level: str = "INFO"                    # INFO | WARN | ERROR | CRITICAL
    # routing overrides by event kind
    route_overrides: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        # kind -> channel flags
        "fills":     {"slack": True, "discord": False, "email": False, "telegram": False},
        "risk":      {"slack": True, "discord": True,  "email": True,  "telegram": False},
        "error":     {"slack": True, "discord": True,  "email": True,  "telegram": True},
        "pnl":       {"slack": False,"discord": False, "email": False, "telegram": False},
        "heartbeat": {"slack": False,"discord": False, "email": False, "telegram": False},
    })


# --------------------------------------------------------------------------------------
# Alert Client
# --------------------------------------------------------------------------------------

class Alerts:
    """
    Usage:
      alerts = Alerts()
      alerts.info("heartbeat", "Live job running", meta={"sid": "CIT-BA01"})
      alerts.risk("limit_breach", f"Per-name cap hit: {issuer}", meta={...})
      alerts.error("router", "Order rejected", meta=err_payload)

    Kinds used in templates: 'fills', 'risk', 'error', 'pnl', 'heartbeat', 'custom'
    """

    def __init__(self, channels: Optional[ChannelConfig] = None, policy: Optional[PolicyConfig] = None, app_name: str = "HF-Orchestrator"):
        self.channels = channels or ChannelConfig()
        self.channels.hydrate_from_env()
        self.policy = policy or PolicyConfig()
        self.app_name = app_name

        self._last_sent_by_key: Dict[str, float] = {}
        self._minute_bucket_ts = 0
        self._minute_count = 0

        # Create persistent sink file
        if self.channels.file_sink:
            self.channels.file_sink.parent.mkdir(parents=True, exist_ok=True)
            self.channels.file_sink.touch(exist_ok=True)

    # -------------- public API (severity helpers) -----------------

    def info(self, kind: str, title: str, meta: Optional[Dict[str, Any]] = None):
        self._emit("INFO", kind, title, meta)

    def warn(self, kind: str, title: str, meta: Optional[Dict[str, Any]] = None):
        self._emit("WARN", kind, title, meta)

    def error(self, kind: str, title: str, meta: Optional[Dict[str, Any]] = None):
        self._emit("ERROR", kind, title, meta)

    def critical(self, kind: str, title: str, meta: Optional[Dict[str, Any]] = None):
        self._emit("CRITICAL", kind, title, meta)

    # Convenience wrappers for common event kinds
    def fills(self, title: str, fills_df_like: Any):
        meta = _coerce_meta(fills_df_like)
        self._emit("INFO", "fills", title, meta)

    def risk(self, title: str, meta: Optional[Dict[str, Any]] = None):
        self._emit("WARN", "risk", title, meta)

    def pnl(self, title: str, meta: Optional[Dict[str, Any]] = None):
        self._emit("INFO", "pnl", title, meta)

    def heartbeat(self, title: str = "alive", meta: Optional[Dict[str, Any]] = None):
        self._emit("INFO", "heartbeat", title, meta)

    # -------------- core emit -----------------

    def _emit(self, level: str, kind: str, title: str, meta: Optional[Dict[str, Any]]):
        now = time.time()
        lvl_order = ["INFO", "WARN", "ERROR", "CRITICAL"]
        if lvl_order.index(level) < lvl_order.index(self.policy.min_level):
            return

        payload = {
            "ts": int(now),
            "app": self.app_name,
            "level": level,
            "kind": kind,
            "title": title,
            "meta": meta or {},
        }

        # dedup/throttle
        key = self._dedup_key(payload)
        if self._should_throttle(key, now):
            log.debug(f"throttled: {key}")
            return

        # minute-rate breaker
        if not self._under_rate_limit(now):
            log.warning("alerts rate limit reached; dropping message")
            return

        # persist to file sink first (for audit)
        self._write_sink(payload)

        # route to channels
        routes = self._resolve_routes(kind)
        text = self._format_text(payload)
        try:
            if routes.get("slack") and self.channels.slack_webhook:
                self._send_slack(text)
            if routes.get("discord") and self.channels.discord_webhook:
                self._send_discord(text)
            if routes.get("telegram") and self.channels.telegram_bot_token and self.channels.telegram_chat_id:
                self._send_telegram(text)
            if routes.get("email") and self.channels.smtp_host and self.channels.email_to:
                self._send_email(subject=f"[{self.app_name}] {level} / {kind}: {title}", html=_html_wrap(text))
        except Exception as e:
            log.exception(f"alert send failed: {e}")

        # mark last sent
        self._last_sent_by_key[key] = now

    # -------------- routing & formatting -----------------

    def _resolve_routes(self, kind: str) -> Dict[str, bool]:
        # default: everything to slack if configured
        default = {"slack": bool(self.channels.slack_webhook), "discord": False, "email": False, "telegram": False}
        return self.policy.route_overrides.get(kind, default)

    @staticmethod
    def _format_text(payload: Dict[str, Any]) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(payload["ts"]))
        level = payload["level"]
        kind = payload["kind"]
        title = payload["title"]
        meta = payload.get("meta") or {}

        # Pretty known kinds
        if kind == "fills" and isinstance(meta, dict) and "rows" in meta:
            rows = meta["rows"]
            lines = [f"*Fills* ({len(rows)} orders) — {title} [{ts} UTC]"]
            for r in rows[:10]:
                lines.append(f"- `{r.get('ticker','?')}` {r.get('side','?')} ${r.get('filled_usd',0):,.0f} @ {r.get('avg_price_bps','?')}bps ({r.get('status','?')})")
            if len(rows) > 10:
                lines.append(f"... and {len(rows)-10} more")
            return "\n".join(lines)

        if kind == "risk":
            return f"*RISK* — {title}\n```{json.dumps(meta, indent=2, default=str)}```\n[{ts} UTC]"

        if kind == "pnl":
            return f"*PnL* — {title}\n```{json.dumps(meta, indent=2, default=str)}```\n[{ts} UTC]"

        if kind == "error":
            return f"*ERROR* — {title}\n```{json.dumps(meta, indent=2, default=str)}```\n[{ts} UTC]"

        # generic
        return f"*{level}/{kind}* — {title}\n```{json.dumps(meta, indent=2, default=str)}```\n[{ts} UTC]"

    # -------------- channels -----------------

    def _send_slack(self, text: str):
        data = json.dumps({"text": text}).encode("utf-8")
        req = Request(self.channels.slack_webhook, data=data, headers={"Content-Type": "application/json"}) # type: ignore
        with urlopen(req, timeout=5) as _:
            pass

    def _send_discord(self, text: str):
        data = json.dumps({"content": text}).encode("utf-8")
        req = Request(self.channels.discord_webhook, data=data, headers={"Content-Type": "application/json"}) # type: ignore
        with urlopen(req, timeout=5) as _:
            pass

    def _send_telegram(self, text: str):
        token = self.channels.telegram_bot_token
        chat_id = self.channels.telegram_chat_id
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as _:
            pass

    def _send_email(self, subject: str, html: str):
        msg = MIMEText(html, "html", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self.channels.smtp_user or "alerts@localhost"
        msg["To"] = ", ".join(self.channels.email_to)
        context = ssl.create_default_context()
        with smtplib.SMTP(self.channels.smtp_host, self.channels.smtp_port) as server: # type: ignore
            server.starttls(context=context)
            if self.channels.smtp_user and self.channels.smtp_pass:
                server.login(self.channels.smtp_user, self.channels.smtp_pass)
            server.sendmail(msg["From"], self.channels.email_to, msg.as_string())

    # -------------- dedup / throttle / rate -----------------------------------

    def _dedup_key(self, payload: Dict[str, Any]) -> str:
        # identical event type + title + coarse meta hash
        meta = payload.get("meta") or {}
        # keep only lightweight fields to hash
        key_meta = {k: meta[k] for k in sorted(meta.keys()) if isinstance(meta[k], (str, int, float))}
        return f"{payload['level']}|{payload['kind']}|{payload['title']}|{json.dumps(key_meta, sort_keys=True)}"

    def _should_throttle(self, key: str, now: float) -> bool:
        last = self._last_sent_by_key.get(key)
        if last is None:
            return False
        return (now - last) < max(0, int(self.policy.throttle_secs))

    def _under_rate_limit(self, now: float) -> bool:
        bucket = int(now // 60)
        if bucket != self._minute_bucket_ts:
            self._minute_bucket_ts = bucket
            self._minute_count = 0
        self._minute_count += 1
        return self._minute_count <= int(self.policy.max_per_minute)

    # -------------- persistence sink -----------------------------------------

    def _write_sink(self, payload: Dict[str, Any]):
        if not self.channels.file_sink:
            return
        try:
            with self.channels.file_sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            log.warning(f"sink write failed: {e}")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _coerce_meta(fills_df_like: Any) -> Dict[str, Any]:
    """
    Accept a DataFrame or list[dict] of fills and convert to a compact meta dict.
    Expected columns: ticker, side, filled_usd, avg_price_bps, status, ts
    """
    try:
        import pandas as pd
        if hasattr(fills_df_like, "to_dict"):
            df = fills_df_like
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            cols = [c for c in ["ticker","side","filled_usd","avg_price_bps","status","ts"] if c in df.columns]
            rows = df[cols].to_dict(orient="records")
            return {"rows": rows, "n": len(rows)}
        if isinstance(fills_df_like, list):
            return {"rows": fills_df_like, "n": len(fills_df_like)}
    except Exception:
        pass
    return {"rows": [], "n": 0}


def _html_wrap(text_md: str) -> str:
    # very simple Markdown-ish → HTML wrapper for email
    safe = text_md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n", "<br>")
    safe = safe.replace("```", "<pre>").replace("*", "<b>")
    return f"<html><body style='font-family: Inter, Segoe UI, Arial; font-size: 14px;'>{safe}</body></html>"


# --------------------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Set env vars to try quickly:
    # export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/xxx/yyy/zzz"
    # export ALERT_EMAIL_TO="ops@example.com,trader@example.com"
    # export SMTP_HOST="smtp.gmail.com"; export SMTP_USER="user"; export SMTP_PASS="pass"

    alerts = Alerts(app_name="Quant-Stack")
    alerts.heartbeat(meta={"sid": "DEMO-0001", "mode": "paper"})
    alerts.fills("Executed weekly rebalance", [
        {"ticker":"IG_A_5Y","side":"BUY_PROTECTION","filled_usd":1000000,"avg_price_bps":120,"status":"FILLED","ts":int(time.time())},
        {"ticker":"HY_B_5Y","side":"SELL_PROTECTION","filled_usd":800000,"avg_price_bps":420,"status":"FILLED","ts":int(time.time())},
    ])
    alerts.risk("Per-name limit breached", {"name":"IG_A","limit_usd":5_000_000,"attempted":6_200_000})
    alerts.error("router", "Order reject: price outside band", {"ticker":"IG_A_5Y","side":"BUY_PROTECTION","quote":"n/a"})