# backend/treasury/bank_adapters.py
from __future__ import annotations

import abc
import csv
import hashlib
import hmac
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional helpers from your codebase (kept optional to avoid hard deps)
try:
    from backend.utils.rate_linits import RateGate, SlidingWindowLimiter, WindowRule # type: ignore
except Exception:
    class RateGate:
        def __init__(self, rps: float): pass
        def wait(self): pass
    class SlidingWindowLimiter:
        def __init__(self, *_a, **_k): pass
        def wait(self, *_a, **_k): pass
    class WindowRule:
        def __init__(self, *_a, **_k): pass

try:
    from backend.utils.secrets import secrets # type: ignore
except Exception:
    class _DummySecrets:
        def get(self, k: str, default: Optional[str] = None, required: bool = False):
            v = os.getenv(k, default)
            if required and v is None:
                raise KeyError(f"Missing secret {k}")
            return v
    secrets = _DummySecrets()  # type: ignore


# ============================== Models ==============================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class BankAccount:
    bank_name: str
    account_id: str
    account_number_masked: str
    currency: str = "USD"
    meta: Dict[str, Any] = None # type: ignore

@dataclass
class Balance:
    available: float
    ledger: float
    currency: str
    ts: str = now_iso()

@dataclass
class Transaction:
    tx_id: str
    account_id: str
    amount: float                 # positive = credit to account, negative = debit
    currency: str
    description: str
    booked_at: str                # ISO 8601
    value_date: Optional[str] = None
    counterparty: Optional[str] = None
    meta: Dict[str, Any] = None # type: ignore

@dataclass
class TransferRequest:
    from_account_id: str
    to_beneficiary: str           # bank routing reference / vpa / iban / etc.
    amount: float
    currency: str
    reference: str                # human ref
    idempotency_key: Optional[str] = None
    meta: Dict[str, Any] = None # type: ignore

@dataclass
class TransferResult:
    ok: bool
    transfer_id: Optional[str]
    status: str                   # 'queued' | 'processing' | 'settled' | 'failed'
    reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


# ============================== Base Adapter ==============================

class BankAdapterBase(abc.ABC):
    """
    Abstract adapter for bank / payout rails.
    Concrete implementations: Plaid-like aggregator, bank CSV, mock, Razorpay/Stripe-like payout, etc.
    """

    name: str = "base"

    def __init__(self, *, rps: float = 2.0, per_min_limit: int = 60):
        self._gate = RateGate(rps=rps)
        self._lim = SlidingWindowLimiter(WindowRule(per_min_limit, 60))

    # ---- required APIs ----

    @abc.abstractmethod
    def list_accounts(self) -> List[BankAccount]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_balance(self, account_id: str) -> Balance:
        raise NotImplementedError

    @abc.abstractmethod
    def list_transactions(
        self, account_id: str, *, since: Optional[str] = None, until: Optional[str] = None, limit: int = 200
    ) -> List[Transaction]:
        raise NotImplementedError

    @abc.abstractmethod
    def initiate_transfer(self, req: TransferRequest) -> TransferResult:
        raise NotImplementedError

    @abc.abstractmethod
    def transfer_status(self, transfer_id: str) -> TransferResult:
        raise NotImplementedError

    # ---- optional helpers ----

    def verify_webhook(self, payload: bytes, header_sig: str, secret_env_key: str) -> bool:
        """
        HMAC-SHA256 verification (typical scheme). Configure bank webhook secret via env/Secrets.
        """
        secret = secrets.get(secret_env_key, required=True)
        digest = hmac.new(str(secret).encode(), payload, hashlib.sha256).hexdigest()
        # Many providers prefix with "sha256="; normalize both sides
        header_sig = header_sig.lower().replace("sha256=", "")
        return hmac.compare_digest(digest, header_sig)

    # ---- rate limits wrappers ----

    def _guard(self, key: str = "global") -> None:
        self._gate.wait()
        try:
            self._lim.wait(key)
        except Exception:
            # best-effort if limiter not available
            pass


# ============================== Mock Adapter ==============================

class MockBankAdapter(BankAdapterBase):
    """
    In-memory mock ledger: perfect for local dev / tests.
    - deterministic account ids
    - idempotent transfers via idempotency key
    """

    name = "mock"

    def __init__(self, *, currency: str = "USD", start_cash: float = 1_000_000.0):
        super().__init__(rps=50.0, per_min_limit=10_000)
        self._lock = threading.RLock()
        self._currency = currency
        self._accounts: Dict[str, Dict[str, Any]] = {}
        self._tx: Dict[str, List[Transaction]] = {}
        self._transfers: Dict[str, TransferResult] = {}       # by transfer_id
        self._idem: Dict[str, TransferResult] = {}            # by idempotency key

        # create a default "operating" account
        acct_id = "acct-operating"
        self._accounts[acct_id] = dict(
            bank_name="MockBank",
            account_number_masked="****1234",
            currency=currency,
            ledger=start_cash,
            available=start_cash,
        )
        self._tx[acct_id] = []

    def list_accounts(self) -> List[BankAccount]:
        self._guard("list_accounts")
        with self._lock:
            out = []
            for aid, m in self._accounts.items():
                out.append(BankAccount(
                    bank_name=m["bank_name"], account_id=aid,
                    account_number_masked=m["account_number_masked"], currency=m["currency"], meta={}
                ))
            return out

    def get_balance(self, account_id: str) -> Balance:
        self._guard(f"bal:{account_id}")
        with self._lock:
            m = self._accounts[account_id]
            return Balance(available=float(m["available"]), ledger=float(m["ledger"]), currency=m["currency"], ts=now_iso())

    def list_transactions(self, account_id: str, *, since: Optional[str] = None, until: Optional[str] = None, limit: int = 200) -> List[Transaction]:
        self._guard(f"tx:{account_id}")
        with self._lock:
            txs = list(self._tx.get(account_id, []))
        # naive time filter on ISO strings (UTC)
        def _ok(t: Transaction) -> bool:
            if since and t.booked_at < since:  return False
            if until and t.booked_at > until:  return False
            return True
        out = [t for t in txs if _ok(t)]
        return out[: int(limit)]

    def initiate_transfer(self, req: TransferRequest) -> TransferResult:
        self._guard("transfer")
        # idempotency
        idem = req.idempotency_key or f"{req.from_account_id}:{req.to_beneficiary}:{req.amount}:{req.currency}:{req.reference}"
        with self._lock:
            if idem in self._idem:
                return self._idem[idem]

            # balance check
            acct = self._accounts[req.from_account_id]
            if req.amount <= 0:
                res = TransferResult(ok=False, transfer_id=None, status="failed", reason="invalid_amount")
                self._idem[idem] = res
                return res
            if acct["available"] < req.amount:
                res = TransferResult(ok=False, transfer_id=None, status="failed", reason="insufficient_funds")
                self._idem[idem] = res
                return res

            # debit immediately; settle after a small delay to mimic rails
            acct["available"] -= req.amount
            acct["ledger"] -= req.amount
            tid = "trf_" + uuid.uuid4().hex[:20]
            booked_at = now_iso()
            t = Transaction(
                tx_id="tx_" + uuid.uuid4().hex[:12],
                account_id=req.from_account_id,
                amount=-req.amount,
                currency=req.currency,
                description=f"Transfer to {req.to_beneficiary} ({req.reference})",
                booked_at=booked_at,
                counterparty=req.to_beneficiary,
                meta=dict(kind="transfer_out", ref=req.reference),
            )
            self._tx[req.from_account_id].insert(0, t)
            res = TransferResult(ok=True, transfer_id=tid, status="processing", raw={"booked_at": booked_at})
            self._transfers[tid] = res
            self._idem[idem] = res
            # simulate settlement
            threading.Timer(0.5, self._settle, args=(tid,)).start()
            return res

    def _settle(self, transfer_id: str) -> None:
        with self._lock:
            r = self._transfers.get(transfer_id)
            if r:
                r.status = "settled"
                r.ok = True

    def transfer_status(self, transfer_id: str) -> TransferResult:
        self._guard("status")
        with self._lock:
            r = self._transfers.get(transfer_id)
            if not r:
                return TransferResult(ok=False, transfer_id=transfer_id, status="failed", reason="unknown_transfer")
            return r


# ============================== CSV Adapter ==============================

class CSVBankAdapter(BankAdapterBase):
    """
    Read-only adapter that exposes balances/transactions from a CSV export
    (useful for reconciliation tests). CSV columns expected:

    transactions.csv:
        account_id,booked_at_iso,amount,currency,description,counterparty

    balances.csv:
        account_id,available,ledger,currency,ts_iso
    """

    name = "csv"

    def __init__(self, *, tx_csv_path: str, bal_csv_path: str):
        super().__init__(rps=10.0, per_min_limit=600)
        self._tx_path = tx_csv_path
        self._bal_path = bal_csv_path
        self._acc_index: Dict[str, BankAccount] = {}
        self._load_accounts()

    def _load_accounts(self):
        # infer accounts from balances.csv
        try:
            with open(self._bal_path, newline="") as f:
                for row in csv.DictReader(f):
                    aid = row["account_id"]
                    self._acc_index[aid] = BankAccount(
                        bank_name="CSV/Offline",
                        account_id=aid,
                        account_number_masked="****CSV",
                        currency=row.get("currency") or "USD",
                        meta={}
                    )
        except FileNotFoundError:
            pass

    def list_accounts(self) -> List[BankAccount]:
        self._guard("list_accounts")
        return list(self._acc_index.values())

    def get_balance(self, account_id: str) -> Balance:
        self._guard(f"bal:{account_id}")
        with open(self._bal_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["account_id"] == account_id:
                    return Balance(
                        available=float(row["available"]),
                        ledger=float(row["ledger"]),
                        currency=row.get("currency") or "USD",
                        ts=row.get("ts_iso") or now_iso(),
                    )
        raise KeyError(f"balance not found for {account_id}")

    def list_transactions(self, account_id: str, *, since: Optional[str] = None, until: Optional[str] = None, limit: int = 200) -> List[Transaction]:
        self._guard(f"tx:{account_id}")
        out: List[Transaction] = []
        with open(self._tx_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["account_id"] != account_id:
                    continue
                t = Transaction(
                    tx_id=row.get("tx_id") or uuid.uuid4().hex[:16],
                    account_id=account_id,
                    amount=float(row["amount"]),
                    currency=row.get("currency") or "USD",
                    description=row.get("description") or "",
                    booked_at=row.get("booked_at_iso") or now_iso(),
                    counterparty=row.get("counterparty"),
                    meta={k: v for k, v in row.items() if k not in {"account_id","amount","currency","description","booked_at_iso","counterparty"}},
                )
                out.append(t)
        # filter & cap
        def _ok(t: Transaction) -> bool:
            if since and t.booked_at < since:  return False
            if until and t.booked_at > until:  return False
            return True
        out = [t for t in out if _ok(t)]
        return out[: int(limit)]

    def initiate_transfer(self, req: TransferRequest) -> TransferResult:
        # read-only rails
        return TransferResult(ok=False, transfer_id=None, status="failed", reason="csv_adapter_read_only")

    def transfer_status(self, transfer_id: str) -> TransferResult:
        return TransferResult(ok=False, transfer_id=transfer_id, status="failed", reason="csv_adapter_read_only")


# ============================== (Skeleton) Real Adapter ==============================

class PayoutRailsAdapter(BankAdapterBase):
    """
    Skeleton for a real payout/collections adapter (e.g., RazorpayX / Stripe Treasury / Wise).
    Fill in _http_get/_http_post with your client; use secrets for keys.

    Env keys expected (example):
        PAYOUT_BASE_URL, PAYOUT_API_KEY, PAYOUT_API_SECRET
    """

   