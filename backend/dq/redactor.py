# backend/safety/redactor.py
from __future__ import annotations

import os, re, sys, json, hmac, hashlib, time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

# ---------- Optional YAML (graceful) -----------------------------------------
HAVE_YAML = True
try:
    import yaml  # type: ignore
except Exception:
    HAVE_YAML = False
    yaml = None  # type: ignore

# ---------- Helpers ----------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)
def _hmac(data: str, key: str) -> str:
    return hmac.new(key.encode("utf-8"), data.encode("utf-8"), hashlib.sha256).hexdigest()

def _mask_middle(s: str, visible_prefix: int = 2, visible_suffix: int = 2, char: str = "•") -> str:
    if len(s) <= visible_prefix + visible_suffix:
        return char * len(s)
    return s[:visible_prefix] + char * (len(s) - visible_prefix - visible_suffix) + s[-visible_suffix:]

def _last4(s: str, char: str = "•") -> str:
    s = re.sub(r"\s+", "", s)
    return (char * max(0, len(s) - 4)) + s[-4:]

# ---------- Strategy ---------------------------------------------------------
RedactFn = Callable[[str, Dict[str, Any]], str]

def strategy_mask(value: str, opts: Dict[str, Any]) -> str:
    mode = opts.get("mode", "middle")
    if mode == "last4": return _last4(value)
    if mode == "all": return "•" * max(4, len(value))
    return _mask_middle(value, int(opts.get("prefix", 2)), int(opts.get("suffix", 2)))

def strategy_hash(value: str, opts: Dict[str, Any]) -> str:
    salt = os.getenv("REDACT_SALT", "change-me")
    return "hash_" + _hmac(value, salt)[:32]

class TokenVault:
    """Simple reversible tokenization (local JSON map)."""
    def __init__(self, path: str = "artifacts/redactor/vault.json"):
        self.path = path
        self.map: Dict[str, str] = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.map = json.load(f) or {}
            except Exception:
                self.map = {}

    def tokenize(self, value: str) -> str:
        if value in self.map.values():
            # reuse existing token
            for tok, raw in self.map.items():
                if raw == value:
                    return tok
        tok = f"tok_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"
        self.map[tok] = value
        self._save()
        return tok

    def detokenize(self, token: str) -> Optional[str]:
        return self.map.get(token)

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.map, f, indent=2)
        except Exception:
            pass

def strategy_tokenize(value: str, opts: Dict[str, Any]) -> str:
    vault = TokenVault(opts.get("vault_path", "artifacts/redactor/vault.json"))
    return vault.tokenize(value)

STRATEGIES: Dict[str, RedactFn] = {
    "mask": strategy_mask,
    "hash": strategy_hash,
    "tokenize": strategy_tokenize,
}

# ---------- Built-in patterns (names → regex + default strategy) ------------
# NOTE: patterns aim to be high-coverage and safe; tune to your environment.
PATTERNS: Dict[str, Dict[str, Any]] = {
    # Secrets / tokens / keys
    "aws_access_key":   {"re": r"\bAKIA[0-9A-Z]{16}\b", "strategy": "mask", "opts": {"mode":"last4"}},
    "aws_secret_key":   {"re": r"\b(?<![A-Za-z0-9])[A-Za-z0-9/\+=]{40}(?![A-Za-z0-9])\b", "strategy": "hash", "opts": {}},
    "gcp_service_key":  {"re": r'"private_key"\s*:\s*"-----BEGIN PRIVATE KEY-----[^"]+END PRIVATE KEY-----"', "strategy": "tokenize", "opts": {}},
    "azure_conn":       {"re": r"(Endpoint=sb://[^;]+;SharedAccessKeyName=[^;]+;SharedAccessKey=)[A-Za-z0-9+/=]+", "strategy": "hash", "opts": {}},
    "openai_key":       {"re": r"\b(sk|rk)-[A-Za-z0-9]{20,}\b", "strategy": "mask", "opts": {"mode":"last4"}},
    "jwt":              {"re": r"\beyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b", "strategy": "hash", "opts": {}},
    "bearer_token":     {"re": r"(?i)\b(bearer)\s+[A-Za-z0-9._~-]{12,}\b", "strategy": "hash", "opts": {}},
    "password_param":   {"re": r"(?i)(password|passwd|pwd)\s*[:=]\s*([^\s,'\"]+)", "strategy": "mask", "opts": {"mode":"all"}},
    "generic_secret":   {"re": r"(?i)(api[-_ ]?key|secret|token)\s*[:=]\s*([^\s,'\"]+)", "strategy": "hash", "opts": {}},

    # PII / Compliance
    "email":            {"re": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "flags": re.I, "strategy": "mask", "opts": {"prefix":2,"suffix":6}},
    "phone":            {"re": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b", "strategy": "mask", "opts": {"mode":"last4"}},
    "credit_card":      {"re": r"\b(?:\d[ -]*?){13,19}\b", "strategy": "mask", "opts": {"mode":"last4"}},
    "iban":             {"re": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", "strategy": "mask", "opts": {"mode":"last4"}},
    "ipv4":             {"re": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b", "strategy": "mask", "opts": {"prefix":0,"suffix":0}},
    "ipv6":             {"re": r"\b([0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}\b", "flags": re.I, "strategy": "mask", "opts": {"prefix":0,"suffix":0}},
    "url_creds":        {"re": r"(?i)\b[a-z]+://[^/\s:@]+:([^@\s]+)@[^ \n]+", "strategy": "mask", "opts": {"mode":"all"}},

    # India-specific (tune with Compliance)
    "aadhaar":          {"re": r"\b\d{4}\s?\d{4}\s?\d{4}\b", "strategy": "mask", "opts": {"mode":"last4"}},
    "pan":              {"re": r"\b[A-Z]{5}\d{4}[A-Z]\b", "strategy": "mask", "opts": {"prefix":3,"suffix":1}},

    # Trading identifiers
    "account_id":       {"re": r"\bACC[-_]?[A-Z0-9]{6,}\b", "strategy": "hash", "opts": {}},
    "client_id":        {"re": r"\bCL(?:IENT)?[-_]?[A-Z0-9]{6,}\b", "strategy": "hash", "opts": {}},
    "iban_like":        {"re": r"\b[0-9]{2}[A-Z]{4}\d{10,}\b", "strategy": "mask", "opts": {"mode":"last4"}},
}

# ---------- Policy -----------------------------------------------------------
@dataclass
class Rule:
    name: str
    regex: re.Pattern
    strategy: str = "mask"
    opts: Dict[str, Any] = field(default_factory=dict)
    group: Optional[int] = None  # if only a capture group should be replaced

@dataclass
class RedactorPolicy:
    rules: List[Rule] = field(default_factory=list)
    default_strategy: str = "mask"
    respect_allowlist_keys: bool = True
    allowlist_keys: List[str] = field(default_factory=lambda: ["note", "description", "comment"])  # won't redact entire value unless a rule hits
    redact_keys_like: List[str] = field(default_factory=lambda: ["password","passwd","secret","token","api_key","authorization","auth","cookie"])

    @staticmethod
    def from_yaml(path: str) -> "RedactorPolicy":
        if not HAVE_YAML:
            raise RuntimeError("pyyaml not installed; cannot load YAML policy")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {} # type: ignore
        rules: List[Rule] = []
        for r in cfg.get("rules", []):
            flags = 0
            if r.get("flags"):
                for fl in r["flags"]:
                    flags |= getattr(re, fl, 0)
            rules.append(Rule(
                name=r["name"],
                regex=re.compile(r["re"], flags),
                strategy=r.get("strategy","mask"),
                opts=r.get("opts",{}) or {},
                group=r.get("group")
            ))
        pol = RedactorPolicy(
            rules=rules,
            default_strategy=cfg.get("default_strategy","mask"),
            respect_allowlist_keys=bool(cfg.get("respect_allowlist_keys", True)),
            allowlist_keys=cfg.get("allowlist_keys", []) or [],
            redact_keys_like=cfg.get("redact_keys_like", []) or [],
        )
        return pol

def build_default_policy() -> RedactorPolicy:
    rules: List[Rule] = []
    for name, spec in PATTERNS.items():
        flags = spec.get("flags", 0)
        rules.append(Rule(
            name=name,
            regex=re.compile(spec["re"], flags),
            strategy=spec.get("strategy", "mask"),
            opts=spec.get("opts", {}) or {}
        ))
    return RedactorPolicy(rules=rules)

# ---------- Redactor ---------------------------------------------------------
class Redactor:
    def __init__(self, policy: Optional[RedactorPolicy] = None):
        self.policy = policy or build_default_policy()

    def _apply_strategy(self, value: str, strategy: str, opts: Dict[str, Any]) -> str:
        fn = STRATEGIES.get(strategy, strategy_mask)
        try:
            return fn(value, opts or {})
        except Exception:
            return "[REDACTED]"

    def redact_text(self, text: str) -> str:
        out = text
        for rule in self.policy.rules:
            def _repl(m: re.Match) -> str:
                if rule.group:
                    g = m.group(rule.group)
                    repl = self._apply_strategy(g, rule.strategy, rule.opts)
                    # replace only the group inside the whole match
                    s, e = m.span(rule.group)
                    return out[m.start():s] + repl + out[e:m.end()]  # will be recomputed below safely
                else:
                    return self._apply_strategy(m.group(0), rule.strategy, rule.opts)
            # apply safely by rebuilding on each pattern
            out = re.sub(rule.regex, _repl, out)
        return out

    def redact_value(self, key: Optional[str], value: Any) -> Any:
        # Key-based blanket redaction
        lk = (key or "").lower()
        if lk and any(k in lk for k in self.policy.redact_keys_like):
            if isinstance(value, str):
                return self._apply_strategy(value, "mask", {"mode":"all"})
            return "[REDACTED]"

        # Structured types
        if isinstance(value, dict):
            return {k: self.redact_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact_value(None, v) for v in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        # Strings
        try:
            return self.redact_text(str(value))
        except Exception:
            return "[REDACTED]"

    def redact_event(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self.redact_value(k, v) for k, v in obj.items()}

    # -------- File / stream helpers -----------------------------------------
    def redact_jsonl_file(self, in_path: str, out_path: str) -> int:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        n = 0
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj2 = self.redact_event(obj) if isinstance(obj, dict) else self.redact_value(None, obj)
                    fout.write(json.dumps(obj2, ensure_ascii=False) + "\n")
                except Exception:
                    fout.write(self.redact_text(line) + "\n")
                n += 1
        return n

    def redact_text_file(self, in_path: str, out_path: str) -> int:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(in_path, "r", encoding="utf-8") as fin:
            data = fin.read()
        red = self.redact_text(data)
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write(red)
        return len(red)

# ---------- CLI --------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("redactor")
    ap.add_argument("--policy", type=str, default=None, help="YAML policy path (optional)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text", help="Redact a text file")
    p_text.add_argument("--in", dest="inp", required=True)
    p_text.add_argument("--out", dest="outp", required=True)

    p_jsonl = sub.add_parser("jsonl", help="Redact a JSONL file (one object per line)")
    p_jsonl.add_argument("--in", dest="inp", required=True)
    p_jsonl.add_argument("--out", dest="outp", required=True)

    p_pipe = sub.add_parser("pipe", help="Read stdin, write redacted stdout")
    args = ap.parse_args()

    policy = None
    if args.policy:
        if HAVE_YAML:
            policy = RedactorPolicy.from_yaml(args.policy)
        else:
            print("WARNING: pyyaml not installed; ignoring --policy", file=sys.stderr)

    r = Redactor(policy)

    if args.cmd == "text":
        r.redact_text_file(args.inp, args.outp)
    elif args.cmd == "jsonl":
        r.redact_jsonl_file(args.inp, args.outp)
    elif args.cmd == "pipe":
        for line in sys.stdin:
            sys.stdout.write(r.redact_text(line))
    else:
        ap.print_help()

if __name__ == "__main__":
    _cli()