# backend/regulatory/mifid_reporter.py
from __future__ import annotations

import os, csv, json, time, uuid, asyncio, datetime, re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# -------- Optional deps (graceful) -------------------------------------------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

HAVE_YAML = True
try:
    import yaml  # type: ignore
except Exception:
    HAVE_YAML = False

# -------- Env / Streams / Paths ---------------------------------------------
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
S_FILLS         = os.getenv("ORDERS_FILLED", "orders.filled")
OUT_DIR         = os.getenv("MIFID_OUT_DIR", "artifacts/mifid")
REG_PATH        = os.getenv("INSTRUMENT_REGISTRY", "configs/instruments.yaml")  # symbol -> {isin, cfi, mic}

os.makedirs(OUT_DIR, exist_ok=True)

# -------- Helpers ------------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

def iso_utc(ts_ms: int) -> str:
    return datetime.datetime.utcfromtimestamp(ts_ms/1000.0).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

LEI_RE  = re.compile(r"^[A-Z0-9]{20}$")
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")   # basic check
MIC_RE  = re.compile(r"^[A-Z0-9]{4}$")

def _num(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

def _str(x: Any, default: str = "") -> str:
    return default if x is None else str(x)

# -------- Config -------------------------------------------------------------
@dataclass
class MifidConfig:
    firm_lei: str                          # your firm LEI (20 chars)
    submitting_entity_lei: Optional[str] = None  # usually same as firm_lei
    branch_country: Optional[str] = None   # e.g., "GB", "IE" if applicable
    default_mic: str = "XOFF"              # XOFF for OTC, else venue MIC (e.g., XNSE)
    currency: str = "USD"                  # price currency
    short_sale_default: str = "3"          # 1=short,2=short-exempt,3=not-short, per ESMA code list
    algo_indicator: str = "Y"              # Y/N whether an algo was used
    algo_id: Optional[str] = None          # optional algorithm identifier
    buyer_decision_id: Optional[str] = None
    seller_decision_id: Optional[str] = None
    arm_mode: str = "file_drop"            # 'file_drop' placeholder
    batch_rows: int = 5000                 # rotate CSV after N rows
    filename_prefix: str = "RTS22"
    include_headers: bool = True

# -------- Minimal RTS 22 fields (subset) ------------------------------------
# Column order chosen for readability + common ARM templates.
RTS22_ORDER = [
    "reportingFirmId",        # LEI of reporting firm
    "submittingEntityId",     # LEI if submitting on behalf
    "transactionId",          # internal unique ID
    "executionDateTime",      # UTC ISO 8601 with ms
    "instrumentIdType",       # ISIN
    "instrumentId",
    "price", "priceCurrency",
    "quantity",
    "tradingVenue",           # MIC or XOFF/XOFP
    "buyerId", "buyerIdType", # LEI|NATN|PRSN (we'll use LEI or 'UNK')
    "sellerId", "sellerIdType",
    "shortSaleIndicator",     # 1/2/3 (ESMA code)
    "algorithmicIndicator",   # Y/N
    "algorithmId",
    "buyerDecisionMaker",     # trader ID / algo ID (optional)
    "sellerDecisionMaker",
    "branchCountry",          # optional
    "strategyTag",            # your strategy / book
]

# -------- Instrument registry -----------------------------------------------
class InstrumentRegistry:
    """
    symbol -> {isin, cfi, mic?}
    """
    def __init__(self, path: str = REG_PATH):
        self.path = path
        self.map: Dict[str, Dict[str, str]] = {}
        if os.path.exists(path):
            try:
                if HAVE_YAML and (path.endswith(".yml") or path.endswith(".yaml")):
                    with open(path, "r") as f:
                        self.map = yaml.safe_load(f) or {}
                else:
                    with open(path, "r") as f:
                        self.map = json.load(f)
            except Exception:
                self.map = {}

    def resolve(self, symbol: str) -> Dict[str, str]:
        sym = symbol.upper()
        d = self.map.get(sym, {})
        return {
            "isin": d.get("isin", ""),
            "cfi": d.get("cfi", ""),
            "mic": d.get("mic", ""),
        }

# -------- Builder / Validator -----------------------------------------------
class MifidBuilder:
    def __init__(self, cfg: MifidConfig, registry: Optional[InstrumentRegistry] = None):
        self.cfg = cfg
        self.reg = registry or InstrumentRegistry()

    def from_fill(self, fill: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform your internal fill to a minimal RTS22 dict using the field names in RTS22_ORDER.
        Expected fill keys (from your stack): ts_ms, order_id, symbol, side, qty, price, fee, venue, strategy, account?
        """
        ts_ms   = int(fill.get("ts_ms") or now_ms())
        symbol  = _str(fill.get("symbol")).upper()
        qty     = abs(_num(fill.get("qty") or fill.get("quantity")))
        price   = _num(fill.get("price"))
        venue   = _str(fill.get("venue") or self.cfg.default_mic).upper()
        side    = _str(fill.get("side")).lower()  # 'buy'/'sell'
        strategy= _str(fill.get("strategy") or "")
        account = _str(fill.get("account") or self.cfg.firm_lei)

        inst = self.reg.resolve(symbol)
        isin = inst.get("isin") or _str(fill.get("isin") or "")
        mic  = inst.get("mic") or venue or self.cfg.default_mic

        # buyer/seller attribution (very simplified):
        if side == "buy":
            buyer_id, seller_id = account, _str(fill.get("counterparty_lei") or "UNK")
        else:
            buyer_id, seller_id = _str(fill.get("counterparty_lei") or "UNK"), account

        row = {
            "reportingFirmId": self.cfg.firm_lei,
            "submittingEntityId": self.cfg.submitting_entity_lei or self.cfg.firm_lei,
            "transactionId": _str(fill.get("fill_id") or fill.get("order_id") or f"tx-{uuid.uuid4().hex[:16]}"),
            "executionDateTime": iso_utc(ts_ms),
            "instrumentIdType": "ISIN",
            "instrumentId": isin,
            "price": f"{price:.10f}",
            "priceCurrency": self.cfg.currency,
            "quantity": f"{qty:.8f}",
            "tradingVenue": mic if mic else self.cfg.default_mic,
            "buyerId": buyer_id,
            "buyerIdType": "LEI" if LEI_RE.match(buyer_id) else "UNK",
            "sellerId": seller_id,
            "sellerIdType": "LEI" if LEI_RE.match(seller_id) else "UNK",
            "shortSaleIndicator": _str(fill.get("short_sale_indicator") or self.cfg.short_sale_default),
            "algorithmicIndicator": _str(fill.get("algo_indicator") or self.cfg.algo_indicator),
            "algorithmId": _str(fill.get("algo_id") or self.cfg.algo_id or ""),
            "buyerDecisionMaker": _str(fill.get("buyer_trader_id") or self.cfg.buyer_decision_id or ""),
            "sellerDecisionMaker": _str(fill.get("seller_trader_id") or self.cfg.seller_decision_id or ""),
            "branchCountry": _str(self.cfg.branch_country or ""),
            "strategyTag": strategy,
        }
        return row

    def validate(self, row: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        # Required: firm LEI
        if not LEI_RE.match(row.get("reportingFirmId","")):
            errs.append("reportingFirmId: invalid LEI")
        # Timestamps
        if not _str(row.get("executionDateTime")).endswith("Z"):
            errs.append("executionDateTime: must be UTC ISO with 'Z'")
        # Instrument
        if not ISIN_RE.match(row.get("instrumentId","")):
            errs.append("instrumentId: invalid/missing ISIN")
        # MIC or XOFF/XOFP
        mic = _str(row.get("tradingVenue")).upper()
        if mic not in ("XOFF","XOFP") and not MIC_RE.match(mic):
            errs.append("tradingVenue: must be MIC or XOFF/XOFP")
        # Price & qty positive
        try:
            if float(row.get("price", 0)) <= 0: errs.append("price must be > 0")
        except Exception:
            errs.append("price not numeric")
        try:
            if float(row.get("quantity", 0)) <= 0: errs.append("quantity must be > 0")
        except Exception:
            errs.append("quantity not numeric")
        # Short sale indicator in {1,2,3}
        if _str(row.get("shortSaleIndicator")) not in ("1","2","3"):
            errs.append("shortSaleIndicator must be 1,2,or 3")
        # Algo Y/N
        if _str(row.get("algorithmicIndicator")).upper() not in ("Y","N"):
            errs.append("algorithmicIndicator must be Y/N")
        return errs

# -------- CSV Writer with rotation ------------------------------------------
class CsvBatchWriter:
    def __init__(self, base_dir: str, prefix: str, include_headers: bool, batch_rows: int):
        self.base_dir = base_dir
        self.prefix = prefix
        self.include_headers = include_headers
        self.batch_rows = int(batch_rows)
        self._rows_in_file = 0
        self._f = None
        self._writer: Optional[csv.DictWriter] = None
        self._seq = 0

    def _open_new(self):
        if self._f:
            try: self._f.close()
            except Exception: pass
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self._seq += 1
        path = os.path.join(self.base_dir, f"{self.prefix}_{ts}_{self._seq:03d}.csv")
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=RTS22_ORDER, extrasaction="ignore")
        if self.include_headers:
            self._writer.writeheader()
        self._rows_in_file = 0
        return path

    def write(self, row: Dict[str, Any]) -> str:
        if not self._f or self._rows_in_file >= self.batch_rows:
            path = self._open_new()
        assert self._writer is not None
        self._writer.writerow({k: row.get(k,"") for k in RTS22_ORDER})
        self._rows_in_file += 1
        return self._f.name  # type: ignore

    def close(self):
        if self._f:
            try: self._f.close()
            except Exception: pass
        self._f, self._writer = None, None

# -------- Reporter (Redis live or file replay) -------------------------------
class MifidReporter:
    def __init__(self, cfg: MifidConfig, registry: Optional[InstrumentRegistry] = None):
        self.cfg = cfg
        self.builder = MifidBuilder(cfg, registry)
        self.writer = CsvBatchWriter(OUT_DIR, cfg.filename_prefix, cfg.include_headers, cfg.batch_rows)
        self.r: Optional[AsyncRedis] = None # type: ignore

    async def connect(self):
        if not HAVE_REDIS:
            return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def run_live(self):
        """
        Listen to Redis `orders.filled`, convert to RTS22 rows, validate, and write CSV batches.
        """
        await self.connect()
        last_id = "$"
        if not self.r:
            print("[mifid] Redis unavailable; idle. Use --from-jsonl to convert offline.")
            while True:
                await asyncio.sleep(1)
        try:
            while True:
                resp = await self.r.xread({S_FILLS: last_id}, count=500, block=1000)  # type: ignore
                if not resp:
                    continue
                for stream, entries in resp:
                    last_id = entries[-1][0]
                    for _id, fields in entries:
                        row = self._row_from_fields(fields)
                        if not row:
                            continue
                        errs = self.builder.validate(row)
                        if errs:
                            self._write_error(row, errs)
                        else:
                            path = self.writer.write(row)
                            # You can kick an SFTP uploader to pick files from OUT_DIR
        finally:
            self.writer.close()

    async def from_jsonl(self, path: str):
        """
        Convert a JSONL of fills (one JSON per line) into batched RTS22 CSV files.
        """
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    fill = json.loads(line)
                except Exception:
                    continue
                row = self.builder.from_fill(fill)
                errs = self.builder.validate(row)
                if errs:
                    self._write_error(row, errs)
                else:
                    self.writer.write(row)
        self.writer.close()

    def _row_from_fields(self, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = fields.get("json")
        if raw:
            try:
                fill = json.loads(raw)
            except Exception:
                return None
        else:
            # flat map
            fill = {k: fields.get(k) for k in ("ts_ms","order_id","symbol","side","qty","price","fee","venue","strategy","account","counterparty_lei")}
        return self.builder.from_fill(fill)

    def _write_error(self, row: Dict[str, Any], errs: List[str]):
        # write a sidecar .bad file with the offending row + reasons
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        name = f"{self.cfg.filename_prefix}_errors_{ts}.jsonl"
        with open(os.path.join(OUT_DIR, name), "a", encoding="utf-8") as f:
            f.write(json.dumps({"row": row, "errors": errs}, ensure_ascii=False) + "\n")

# -------- CLI ----------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("mifid_reporter (RTS 22 minimal)")
    ap.add_argument("--firm-lei", required=True, help="Your firm LEI (20 chars)")
    ap.add_argument("--submitting-lei", default=None)
    ap.add_argument("--branch", default=None, help="Branch country code (e.g., GB, IE)")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--default-mic", default="XOFF")
    ap.add_argument("--algo-ind", default="Y", choices=["Y","N"])
    ap.add_argument("--short", default="3", choices=["1","2","3"])
    ap.add_argument("--algo-id", default=None)
    ap.add_argument("--buyer-id", default=None)
    ap.add_argument("--seller-id", default=None)
    ap.add_argument("--from-jsonl", default=None, help="Convert fills JSONL -> RTS22 CSV")
    ap.add_argument("--live", action="store_true", help="Read from Redis orders.filled")
    ap.add_argument("--batch-rows", type=int, default=5000)
    ap.add_argument("--prefix", default="RTS22")
    args = ap.parse_args()

    cfg = MifidConfig(
        firm_lei=args.firm_lei,
        submitting_entity_lei=args.submitting_lei or args.firm_lei,
        branch_country=args.branch,
        default_mic=args.default_mic,
        currency=args.currency,
        short_sale_default=args.short,
        algo_indicator=args.algo_ind,
        algo_id=args.algo_id,
        buyer_decision_id=args.buyer_id,
        seller_decision_id=args.seller_id,
        batch_rows=args.batch_rows,
        filename_prefix=args.prefix,
    )
    rep = MifidReporter(cfg)

    async def _run():
        if args.from_jsonl:
            await rep.from_jsonl(args.from_jsonl)
        elif args.live:
            await rep.run_live()
        else:
            print("Nothing to do. Use --live or --from-jsonl path.")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()