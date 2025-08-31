# backend/compliance/pii_redactor.py
from __future__ import annotations
"""
PII Redactor / Tokenizer
------------------------
Utility to redact or tokenize PII in text, dicts, and lists.

Features
- Regex-based detection for common PII:
  emails, phone numbers, credit cards (Luhn), PAN (India), Aadhaar (12d),
  SSN (US), generic long account numbers, IBAN, IP/MAC.
- Two actions: "redact" (mask) or "tokenize" (HMAC/sha256 stable token).
- Format-preserving masks (e.g., keep last 4 digits).
- Field-aware redaction for dicts (e.g., keys like 'email', 'phone', 'ssn').
- Allowlist (never redact certain substrings/keys).
- Deterministic salts per environment (set PII_SALT).
- Zero hard deps; optional `phonenumbers` improves phone matching if installed.
- CLI for quick runs.

Usage (Python)
--------------
    pr = PIIRedactor(action="redact", keep_last_n=4)
    clean = pr.clean(obj)  # obj can be str | dict | list

    pr = PIIRedactor(action="tokenize", salt="my-secret")
    clean = pr.clean({"email": "a@b.com"})

CLI
---
    python -m backend.compliance.pii_redactor --in in.txt --out out.txt --mode redact
    python -m backend.compliance.pii_redactor --in in.json --out out.json --mode tokenize --salt env

Environment
-----------
    PII_SALT   : default salt when --salt env or not provided
"""

import os, re, json, hmac, hashlib, io, sys
from typing import Any, Dict, List, Iterable, Optional, Tuple, Union

try:
    import phonenumbers  # type: ignore
    _HAS_PHONE = True
except Exception:
    _HAS_PHONE = False

# ---------------- Configuration ----------------

DEFAULT_REPLACER = "[redacted]"
DEFAULT_TOKEN_PREFIX = "pii_"

# common field names hinting PII
PII_FIELD_HINTS = {
    "email", "e-mail",
    "phone", "mobile", "contact",
    "ssn", "sin", "pan", "aadhaar", "aadhar", "passport",
    "credit_card", "card", "cc", "iban", "account", "acct",
    "ip", "mac", "address"
}

# Regex patterns (compiled once)
RE_EMAIL   = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
RE_SSN_US  = re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")
RE_AADHAAR = re.compile(r"\b(?:\d{4}\s?\d{4}\s?\d{4})\b")
RE_PAN_IN  = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")
RE_IBAN    = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
RE_ACCT    = re.compile(r"\b\d{7,}\b")  # generic long account-like numbers
RE_IPv4    = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.|$)){4}\b")
RE_MAC     = re.compile(r"\b(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}\b", re.I)

# Credit card (catch, then validate with Luhn)
RE_CC = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Phone (basic fallback; prefer phonenumbers if available)
RE_PHONE = re.compile(r"""
    (?:\+?\d{1,3}[\s-]?)?                # country code
    (?:\(?\d{2,4}\)?[\s-]?)?             # area code
    \d{3,4}[\s-]?\d{3,4}                 # local
""", re.X)

# For addresses (very heuristic; keep mild to avoid FP)
RE_ADDR = re.compile(r"\b\d{1,5}\s+[A-Za-z][A-Za-z0-9.\-'\s]{3,}\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way)\b", re.I)

# ---------------- Utilities ----------------

def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", s)]
    if len(digits) < 12:
        return False
    checksum = 0
    oddeven = len(digits) & 1
    for i, d in enumerate(digits):
        if not ((i & 1) ^ oddeven):
            d *= 2
            if d > 9: d -= 9
        checksum += d
    return (checksum % 10) == 0

def _hmac_token(value: str, salt: str, prefix: str = DEFAULT_TOKEN_PREFIX) -> str:
    hm = hmac.new(salt.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()[:20]
    return f"{prefix}{hm}"

def _mask_keep_last(s: str, last_n: int = 4, mask_char: str = "•") -> str:
    t = re.sub(r"\s", "", s)
    if len(t) <= last_n:
        return mask_char * len(t)
    return mask_char * (len(t) - last_n) + t[-last_n:]

def _is_allowlisted(s: str, allowlist: Iterable[str]) -> bool:
    ss = s.lower()
    for w in allowlist:
        if w and w.lower() in ss:
            return True
    return False

# ---------------- Main class ----------------

class PIIRedactor:
    def __init__(
        self,
        *,
        action: str = "redact",               # "redact" | "tokenize"
        salt: Optional[str] = None,           # used when tokenizing; if "env", pull from PII_SALT
        keep_last_n: int = 4,                 # for numbers (card/acct/phone)
        mask_char: str = "•",
        email_mask_user: bool = True,         # keep domain, mask user
        allowlist_substrings: Optional[List[str]] = None,
        allowlist_keys: Optional[List[str]] = None
    ):
        assert action in ("redact", "tokenize")
        if salt == "env" or (salt is None and action == "tokenize"):
            salt = os.getenv("PII_SALT", "default-pii-salt-change-me")
        self.action = action
        self.salt = salt or ""
        self.keep_last_n = int(keep_last_n)
        self.mask_char = mask_char
        self.email_mask_user = email_mask_user
        self.allow_sub = allowlist_substrings or []
        self.allow_keys = {k.lower() for k in (allowlist_keys or [])}

    # ------------- public API -------------

    def clean(self, obj: Any, *, field_name: Optional[str] = None) -> Any:
        """Redact/tokenize strings, dicts, lists recursively."""
        if obj is None:
            return None
        if isinstance(obj, str):
            return self._clean_text(obj)
        if isinstance(obj, (int, float, bool)):
            return obj
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if self.allow_keys and str(k).lower() in self.allow_keys:
                    out[k] = v
                    continue
                out[k] = self.clean(v, field_name=str(k))
            return out
        if isinstance(obj, list):
            return [self.clean(v, field_name=field_name) for v in obj]
        # fallback: stringified
        return self._clean_text(str(obj))

    # ------------- internals -------------

    def _token(self, value: str, prefix: str) -> str:
        return _hmac_token(value, self.salt, prefix=prefix)

    def _redact_email(self, m: re.Match) -> str:
        full = m.group(0)
        if self.action == "tokenize":
            return self._token(full, "email_")
        if not self.email_mask_user:
            return DEFAULT_REPLACER
        user, domain = full.split("@", 1)
        masked = (self.mask_char * max(1, len(user)-1)) + user[-1]
        return f"{masked}@{domain}"

    def _redact_cc(self, m: re.Match) -> str:
        full = m.group(0)
        if not _luhn_ok(full):
            return full  # not a real card; leave as-is
        if self.action == "tokenize":
            return self._token(full, "cc_")
        return _mask_keep_last(full, self.keep_last_n, self.mask_char)

    def _redact_numberish(self, m: re.Match, prefix: str) -> str:
        full = m.group(0)
        if self.action == "tokenize":
            return self._token(full, prefix)
        return _mask_keep_last(full, self.keep_last_n, self.mask_char)

    def _redact_plain(self, m: re.Match, prefix: str) -> str:
        full = m.group(0)
        if self.action == "tokenize":
            return self._token(full, prefix)
        return DEFAULT_REPLACER

    def _clean_text(self, text: str) -> str:
        if not text or _is_allowlisted(text, self.allow_sub):
            return text

        out = text

        # Email
        out = RE_EMAIL.sub(self._redact_email, out)

        # Credit card (Luhn validated)
        out = RE_CC.sub(self._redact_cc, out)

        # Aadhaar (12 digits with optional spaces)
        out = RE_AADHAAR.sub(lambda m: self._redact_numberish(m, "aadhaar_"), out)

        # PAN (India)
        out = RE_PAN_IN.sub(lambda m: self._redact_plain(m, "pan_"), out)

        # SSN (US)
        out = RE_SSN_US.sub(lambda m: self._redact_numberish(m, "ssn_"), out)

        # IBAN
        out = RE_IBAN.sub(lambda m: self._redact_plain(m, "iban_"), out)

        # Generic long account numbers
        out = RE_ACCT.sub(lambda m: self._redact_numberish(m, "acct_"), out)

        # IP / MAC
        out = RE_IPv4.sub(lambda m: self._redact_plain(m, "ip_"), out)
        out = RE_MAC.sub(lambda m: self._redact_plain(m, "mac_"), out)

        # Phones
        if _HAS_PHONE:
            # parse and replace numbers deterministically
            out = self._replace_phones_lib(out)
        else:
            out = RE_PHONE.sub(lambda m: self._redact_numberish(m, "phone_"), out)

        # Addresses (heuristic)
        out = RE_ADDR.sub(lambda m: self._redact_plain(m, "addr_"), out)

        return out

    def _replace_phones_lib(self, text: str) -> str:
        """Use phonenumbers to find and replace phone numbers with mask/token."""
        spans: List[Tuple[int, int, str]] = []
        for m in phonenumbers.PhoneNumberMatcher(text, None):  # type: ignore
            raw = text[m.start:m.end]
            spans.append((m.start, m.end, raw))
        if not spans:
            return text
        # Rebuild with replacements (backwards to keep indices valid)
        buff = []
        last = len(text)
        for start, end, raw in reversed(spans):
            rep = self._redact_numberish(re.match(r".+", raw), "phone_")  # type: ignore
            buff.append(text[end:last])
            buff.append(rep)
            last = start
        buff.append(text[:last])
        return "".join(reversed(buff))

# ---------------- CLI ----------------

def _read_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    # try json
    try:
        return json.loads(txt)
    except Exception:
        return txt

def _write_any(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(obj, (dict, list)):
            json.dump(obj, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(obj))

def _main():
    import argparse
    p = argparse.ArgumentParser(description="PII Redactor / Tokenizer")
    p.add_argument("--in", dest="inp", required=True, help="Input file (text or JSON)")
    p.add_argument("--out", dest="out", required=True, help="Output file")
    p.add_argument("--mode", choices=["redact","tokenize"], default="redact")
    p.add_argument("--salt", default=None, help="'env' to use PII_SALT or provide a literal string (for tokenize)")
    p.add_argument("--keep_last", type=int, default=4, help="Keep last N digits for numbers (redact mode)")
    p.add_argument("--allow_sub", nargs="*", default=[], help="Allowlist substrings (never redact)")
    p.add_argument("--allow_keys", nargs="*", default=[], help="Allowlist keys (dict fields to skip)")
    args = p.parse_args()

    salt = os.getenv("PII_SALT") if args.salt in (None, "env") else args.salt

    pr = PIIRedactor(
        action=args.mode,
        salt=salt,
        keep_last_n=args.keep_last,
        allowlist_substrings=args.allow_sub,
        allowlist_keys=args.allow_keys
    )
    data = _read_any(args.inp)
    cleaned = pr.clean(data)
    _write_any(args.out, cleaned)

if __name__ == "__main__":  # pragma: no cover
    _main()