# backend/compliance/cftc_part43.py
"""
CFTC Part 43 real-time public reporting helper
----------------------------------------------
Features
- Build Part 43 "swap transaction and pricing data" payloads (Appendix A, minimized set)
- Round & cap notionals per §43.4(f)-(h); configurable post-initial cap sizes
- Compute dissemination deferrals per §43.5 (blocks/large notionals)
- Anonymize parties; optional geographic masking for Other Commodity per App. E
- Emit NEW / CORRECT / CANCEL; simple validator
- Optional event loop: consume internal swap executions -> publish to reg.part43 (after delay)

Dependencies
- stdlib only; auto-uses PyYAML if present for config; auto-uses backend.bus.streams if available

CLI
    python -m backend.compliance.cftc_part43 --probe
    python -m backend.compliance.cftc_part43 --run        # needs backend.bus.streams
    python -m backend.compliance.cftc_part43 --check msg.json

Notes
- Part 43 requires reporting ASATP (as soon as technologically practicable) after execution (§43.3),
  SDRs publicly disseminate Appendix A fields with anonymity (§43.4), subject to time delays (§43.5).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

# Optional YAML config for caps/blocks/masks
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# Optional bus (graceful if missing)
try:
    from backend.bus.streams import consume_stream, publish_stream
except Exception:  # pragma: no cover
    consume_stream = publish_stream = None  # type: ignore


# ----------------------------- config -----------------------------

DEFAULT_CAPS = {
    # Initial cap sizes from §43.4(g); can be overridden by post-initial caps via config
    "IR": { "0_2Y": 250_000_000, "2_10Y": 100_000_000, "10Y_+": 75_000_000 },
    "CR": 100_000_000,
    "EQ": 250_000_000,
    "FX": 250_000_000,
    "OC": 25_000_000,
}

@dataclass
class Part43Config:
    post_initial_caps: Dict[str, Any]  # e.g., {"IR":{"0_2Y":250e6,...},"CR":100e6,...} based on annual CFTC tables
    block_thresholds: Dict[str, Any]   # your asset-class buckets/tenors -> threshold notionals
    geo_mask: Dict[str, Any]           # for Other Commodity masking (App. E), optional
    sdr: str = "mock-sdr"              # placeholder SDR name/topic
    topic_in: str = "oms.swap.executions"
    topic_out: str = "reg.part43"

def load_config(
    caps_path: str = "config/part43_caps.yaml",
    blocks_path: str = "config/part43_block_thresholds.yaml",
    geo_mask_path: str = "config/part43_geo_mask.yaml",
) -> Part43Config:
    def _load(path, default):
        if yaml and os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or default
        return default
    caps = _load(caps_path, DEFAULT_CAPS)
    blocks = _load(blocks_path, {})
    geo = _load(geo_mask_path, {})
    return Part43Config(post_initial_caps=caps, block_thresholds=blocks, geo_mask=geo)


# ----------------------------- helpers -----------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _round_notional(n: float) -> float:
    """
    Part 43 §43.4(f) rounding ladder.
    """
    x = float(n)
    if x < 1_000:         step = 5
    elif x < 10_000:      step = 100
    elif x < 100_000:     step = 1_000
    elif x < 1_000_000:   step = 10_000
    elif x < 100_000_000: step = 1_000_000
    elif x < 500_000_000: step = 10_000_000
    elif x < 1_000_000_000: step = 50_000_000
    elif x < 100_000_000_000: step = 100_000_000
    else:                 step = 10_000_000_000
    # "in no case less than five" for very small numbers
    rounded = max(5.0, round(x / step) * step)
    return float(rounded)

def _cap_notional(rounded: float, cap: Optional[float]) -> Tuple[float, bool]:
    if cap is None or cap <= 0:
        return rounded, False
    if rounded > cap:
        return float(cap), True
    return float(rounded), False

def _tenor_bucket(years: float) -> str:
    if years <= 2.0: return "0_2Y"
    if years <= 10.0: return "2_10Y"
    return "10Y_+"

def _asset_key(asset_class: str) -> str:
    ac = (asset_class or "").upper()
    if ac.startswith("IR"): return "IR"
    if ac.startswith("CRED") or ac == "CR": return "CR"
    if ac.startswith("EQ"): return "EQ"
    if ac.startswith("FX"): return "FX"
    return "OC"  # Other Commodity default

def _delay_ms_for_swap(*, is_block: bool, is_off_facility_large: bool, asset_key: str,
                       mandatory_clearing: bool, has_sd_or_msp: bool, on_sef_or_dcm: bool) -> int:
    """
    Compute §43.5 delays (simplified to fixed values; see regulation for full matrix).
    """
    MIN = 60_000
    if on_sef_or_dcm and is_block:
        return 15 * MIN  # §43.5(d)(2)
    if is_off_facility_large:
        if mandatory_clearing and has_sd_or_msp:
            return 15 * MIN  # §43.5(e)(2)(ii)
        if (asset_key in {"IR","CR","FX","EQ"}) and has_sd_or_msp and not mandatory_clearing:
            return 30 * MIN  # §43.5(f)(3)
        if asset_key == "OC" and has_sd_or_msp and not mandatory_clearing:
            return 120 * MIN  # §43.5(g)(3) (2 hours)
        if not has_sd_or_msp and not mandatory_clearing:
            # All asset classes: 24 business hours; we approximate 24h wall-clock here
            return 24 * 60 * MIN  # §43.5(h)(3)
    # Non-block / non-large: SDR disseminates ASATP once received (§43.3(b)(1))
    return 0

def _mask_underlying_if_needed(asset_key: str, underlying: str, cfg: Part43Config) -> str:
    if asset_key != "OC":
        return underlying
    # basic geographic masking example: replace precise loc with region bucket per cfg
    if not cfg.geo_mask:
        return underlying
    for needle, repl in cfg.geo_mask.items():
        if needle and needle.lower() in (underlying or "").lower():
            return repl
    return underlying

def _pick_cap(cfg: Part43Config, asset_key: str, tenor_years: Optional[float]) -> Optional[float]:
    caps = cfg.post_initial_caps or {}
    if asset_key == "IR":
        bucket = _tenor_bucket(tenor_years or 2.0)
        return float(caps.get("IR", {}).get(bucket, DEFAULT_CAPS["IR"][bucket]))
    val = caps.get(asset_key)
    if isinstance(val, dict):  # unusual
        # try "default" key else first value
        return float(val.get("default", next(iter(val.values()))))
    if isinstance(val, (int, float)):
        return float(val)
    # fallback to initial caps
    if asset_key in DEFAULT_CAPS:
        v = DEFAULT_CAPS[asset_key]
        return float(v if isinstance(v, (int, float)) else v.get("0_2Y", 250_000_000))
    return None

def _is_block(cfg: Part43Config, asset_key: str, tenor_years: Optional[float], notional: float) -> bool:
    """
    Compare to your configured block thresholds (post-initial appropriate minimum block size).
    """
    bt = cfg.block_thresholds or {}
    if asset_key == "IR":
        bucket = _tenor_bucket(tenor_years or 2.0)
        thr = float(bt.get("IR", {}).get(bucket, float("inf")))
        return notional >= thr
    thr = bt.get(asset_key)
    if isinstance(thr, dict):
        thr = thr.get("default")
    return (float(notional) >= float(thr)) if thr else False


# ----------------------------- public message model -----------------------------

@dataclass
class Part43Message:
    """
    Simplified public payload (Appendix A-aligned, minimal fields).
    """
    action: str                    # NEW | CORRECT | CANCEL
    public_id: str                 # non-identifying, internal->public map
    asof_ms: int                   # time built
    exec_ts_ms: int                # execution timestamp (to the second)
    asset_class: str               # IR | CR | EQ | FX | OC
    product: str                   # e.g., IRS, CDS, NDF, OIS
    underlying: str                # actual description per §43.4(c); masked for OC if needed
    tenor_years: Optional[float]   # for IR buckets, etc.
    side: Optional[str]            # e.g., pay_fix/rec_fix, buy/sell
    price: float                   # rate/price
    price_unit: Optional[str]      # e.g., "rate", "USD/BRL", "$/mmBtu"
    currency: str                  # reporting ccy
    notional_public: float         # rounded & capped for dissemination
    is_capped: bool
    on_sef_or_dcm: bool
    off_facility: bool
    mandatory_clearing: bool
    has_sd_or_msp: bool
    is_block: bool
    is_large_off_facility: bool
    dissemination_delay_ms: int    # computed per §43.5
    ready_at_ms: int               # exec_ts_ms + delay
    # optional extras for traceability (not publicly disseminated)
    internal_trade_id: Optional[str] = None
    notes: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        """
        Dict you would hand to an SDR per its tech spec (minimal representative fields).
        Counterparty identifiers intentionally excluded per §43.4(c).
        """
        return {
            "action": self.action,
            "public_id": self.public_id,
            "exec_ts_ms": int(self.exec_ts_ms),
            "asset_class": self.asset_class,
            "product": self.product,
            "underlying": self.underlying,
            "price": float(self.price),
            "price_unit": self.price_unit,
            "currency": self.currency,
            "notional": float(self.notional_public),
            "is_capped": bool(self.is_capped),
        }


# ----------------------------- builder / validator -----------------------------

def build_part43_message(
    trade: Dict[str, Any],
    cfg: Optional[Part43Config] = None,
    *,
    action: str = "NEW",
    public_id_prefix: str = "P43"
) -> Part43Message:
    """
    Convert internal trade dict -> Part43Message.

    Expected trade keys (superset; unused keys ignored):
      id, exec_ts_ms, asset_class, product, underlying, tenor_years, side,
      price, price_unit, currency, notional, venue: {"sef":bool,"dcm":bool,"off_facility":bool},
      clearing: {"mandatory":bool}, counterparties: {"has_sd_or_msp":bool}

    """
    cfg = cfg or load_config()

    asset_key = _asset_key(trade.get("asset_class", ""))
    tenor_years = trade.get("tenor_years")
    underlying = _mask_underlying_if_needed(asset_key, trade.get("underlying",""), cfg)

    raw_notional = float(trade.get("notional") or 0.0)
    rounded = _round_notional(raw_notional)
    cap = _pick_cap(cfg, asset_key, tenor_years)
    notional_pub, is_capped = _cap_notional(rounded, cap)

    on_sef_or_dcm = bool(trade.get("venue", {}).get("sef") or trade.get("venue", {}).get("dcm"))
    off_facility = bool(trade.get("venue", {}).get("off_facility"))
    has_sd_or_msp = bool(trade.get("counterparties", {}).get("has_sd_or_msp", True))
    mandatory_clearing = bool(trade.get("clearing", {}).get("mandatory", False))

    block = _is_block(cfg, asset_key, tenor_years, raw_notional)
    large = (off_facility and block)  # modeling "large notional off-facility" via same thresholds
    delay_ms = _delay_ms_for_swap(
        is_block=block, is_off_facility_large=large, asset_key=asset_key,
        mandatory_clearing=mandatory_clearing, has_sd_or_msp=has_sd_or_msp, on_sef_or_dcm=on_sef_or_dcm
    )

    exec_ts = int(trade.get("exec_ts_ms") or _utc_ms())
    pid = f"{public_id_prefix}-{int(exec_ts)}-{str(trade.get('id','X'))[-6:]}"

    msg = Part43Message(
        action=action,
        public_id=pid,
        asof_ms=_utc_ms(),
        exec_ts_ms=exec_ts,
        asset_class=asset_key,
        product=str(trade.get("product") or ""),
        underlying=underlying,
        tenor_years=float(tenor_years) if tenor_years is not None else None,
        side=trade.get("side"),
        price=float(trade.get("price") or 0.0),
        price_unit=trade.get("price_unit"),
        currency=str(trade.get("currency") or "USD"),
        notional_public=notional_pub,
        is_capped=is_capped,
        on_sef_or_dcm=on_sef_or_dcm,
        off_facility=off_facility,
        mandatory_clearing=mandatory_clearing,
        has_sd_or_msp=has_sd_or_msp,
        is_block=block,
        is_large_off_facility=large,
        dissemination_delay_ms=delay_ms,
        ready_at_ms=exec_ts + delay_ms,
        internal_trade_id=str(trade.get("id") or None),
        notes=("capped" if is_capped else None),
    )
    validate_part43(msg)
    return msg

def validate_part43(msg: Part43Message) -> None:
    """
    Minimal schema/lawfulness checks (engineering guards; not legal advice).
    """
    if msg.action not in {"NEW","CORRECT","CANCEL"}:
        raise ValueError("action must be NEW|CORRECT|CANCEL")
    if msg.price <= 0:
        raise ValueError("price must be > 0")
    if msg.notional_public <= 0:
        raise ValueError("notional must be > 0 after rounding/capping")
    if not msg.asset_class or msg.asset_class not in {"IR","CR","EQ","FX","OC"}:
        raise ValueError("asset_class must be one of IR|CR|EQ|FX|OC")
    if not msg.product:
        raise ValueError("product required")
    if not msg.currency:
        raise ValueError("currency required")
    # anonymity: ensure we didn't leak IDs
    forbidden_keys = ("party", "counterparty", "lei", "name")
    pub_json = json.dumps(msg.to_public_dict()).lower()
    if any(k in pub_json for k in forbidden_keys):
        raise ValueError("public payload appears to contain counterparty identifiers")
    # time ordering: ready_at >= exec_ts
    if msg.ready_at_ms < msg.exec_ts_ms:
        raise ValueError("ready_at_ms < exec_ts_ms (delay computation bug)")

def make_correction(original: Part43Message, *, new_price: Optional[float]=None,
                    new_notional: Optional[float]=None, note: str = "price correction") -> Part43Message:
    """
    Build a CORRECT message for SDR, as per §43.3(e).
    """
    data = asdict(original)
    data["action"] = "CORRECT"
    if new_price is not None:
        data["price"] = float(new_price)
    if new_notional is not None:
        # re-apply rounding/cap to the new notional with a quick pass
        cfg = load_config()
        asset_key = data["asset_class"]
        tenor_years = data.get("tenor_years")
        rounded = _round_notional(float(new_notional))
        cap = _pick_cap(cfg, asset_key, tenor_years)
        notional_pub, is_capped = _cap_notional(rounded, cap)
        data["notional_public"] = notional_pub
        data["is_capped"] = is_capped
    data["asof_ms"] = _utc_ms()
    data["notes"] = (note or "correction")
    msg = Part43Message(**data)
    validate_part43(msg)
    return msg

def make_cancel(original: Part43Message, note: str = "cancel") -> Part43Message:
    data = asdict(original)
    data["action"] = "CANCEL"
    data["asof_ms"] = _utc_ms()
    data["notes"] = note
    msg = Part43Message(**data)
    validate_part43(msg)
    return msg


# ----------------------------- bus loop (optional) -----------------------------

def run_loop(cfg: Optional[Part43Config] = None):
    """
    Listen for internal swap executions, publish Part 43 messages when ready.
    Internal message shape is the same as expected by build_part43_message().
    """
    assert consume_stream and publish_stream, "bus streams not available"
    cfg = cfg or load_config()
    cursor = "$"
    while True:
        for _, raw in consume_stream(cfg.topic_in, start_id=cursor, block_ms=500, count=200):
            cursor = "$"
            try:
                exec_msg = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            try:
                p43 = build_part43_message(exec_msg, cfg=cfg)
                now = _utc_ms()
                if now >= p43.ready_at_ms:
                    publish_stream(cfg.topic_out, p43.to_public_dict())
                else:
                    # schedule: simple sleep-until (coarse); in practice use your scheduler
                    time.sleep(max(0.0, (p43.ready_at_ms - now) / 1000.0))
                    publish_stream(cfg.topic_out, p43.to_public_dict())
            except Exception as e:
                # emit a minimal error envelope
                publish_stream("reg.errors", {
                    "ts_ms": _utc_ms(), "kind": "part43_build_error", "err": str(e), "raw": exec_msg
                })


# ----------------------------- CLI -----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CFTC Part 43 reporting helper")
    ap.add_argument("--run", action="store_true", help="Run bus bridge from internal executions to reg.part43")
    ap.add_argument("--probe", action="store_true", help="Emit an example public payload to stdout")
    ap.add_argument("--check", type=str, help="Validate a JSON message file (public form)")
    args = ap.parse_args()

    if args.check:
        with open(args.check, "r") as f:
            data = json.load(f)
        # load into model for validation
        msg = Part43Message(
            action=data["action"], public_id=data["public_id"], asof_ms=_utc_ms(),
            exec_ts_ms=int(data["exec_ts_ms"]), asset_class=data["asset_class"],
            product=data["product"], underlying=data.get("underlying",""),
            tenor_years=data.get("tenor_years"), side=data.get("side"),
            price=float(data["price"]), price_unit=data.get("price_unit"),
            currency=data.get("currency","USD"), notional_public=float(data["notional"]),
            is_capped=bool(data.get("is_capped", False)),
            on_sef_or_dcm=False, off_facility=True, mandatory_clearing=False,
            has_sd_or_msp=True, is_block=False, is_large_off_facility=False,
            dissemination_delay_ms=0, ready_at_ms=int(data["exec_ts_ms"]),
        )
        validate_part43(msg)
        print("OK")
        return

    if args.probe:
        cfg = load_config()
        demo = {
            "id": "SWAP123456",
            "exec_ts_ms": _utc_ms() - 5_000,
            "asset_class": "IR",
            "product": "IRS",
            "underlying": "USD-LIBOR-3M",   # or SOFR, etc.
            "tenor_years": 5.0,
            "side": "pay_fix",
            "price": 0.04125,
            "price_unit": "rate",
            "currency": "USD",
            "notional": 375_000_000,
            "venue": {"sef": True, "dcm": False, "off_facility": False},
            "clearing": {"mandatory": True},
            "counterparties": {"has_sd_or_msp": True},
        }
        p = build_part43_message(demo, cfg=cfg)
        print(json.dumps(p.to_public_dict(), indent=2))
        return

    if args.run:
        try:
            run_loop(load_config())
        except KeyboardInterrupt:
            pass
        return

    ap.print_help()

if __name__ == "__main__":
    main()