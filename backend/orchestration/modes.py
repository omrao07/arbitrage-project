# orchestrator/modes.py
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any

import pandas as pd


# =============================================================================
# Enums
# =============================================================================

class RunMode(Enum):
    BACKTEST = auto()   # offline backtest/simulation
    PAPER    = auto()   # live prices, simulated fills
    LIVE     = auto()   # live prices, real routing/fills


class ControlMode(Enum):
    MANUAL    = auto()  # human approves every trade
    SEMI_AUTO = auto()  # auto if small/low-risk, else hold for approval
    AUTO      = auto()  # fully automated within limits


# =============================================================================
# Policies & Limits
# =============================================================================

@dataclass
class RiskLimits:
    max_gross_usd: float = 25_000_000.0
    max_name_usd: float = 5_000_000.0
    max_daily_turnover_usd: float = 10_000_000.0
    allow_short: bool = True
    min_order_usd: float = 5_000.0          # ignore dust
    max_order_usd: float = 5_000_000.0      # single ticket cap
    cooldown_secs: int = 20                 # per-symbol cooldown between executions


@dataclass
class ApprovalsState:
    """Holds orders awaiting manual approval."""
    queue: List[Dict[str, Any]] = field(default_factory=list)  # proposed orders
    decisions: Dict[str, str] = field(default_factory=dict)    # client_order_id -> "APPROVE"/"REJECT"
    last_action_ts: Dict[str, int] = field(default_factory=dict)  # symbol base -> last send ts


# =============================================================================
# Controller
# =============================================================================

@dataclass
class ModeController:
    run_mode: RunMode = RunMode.PAPER
    control_mode: ControlMode = ControlMode.SEMI_AUTO
    limits: RiskLimits = field(default_factory=RiskLimits)
    # Providers from the host system (injected):
    get_position: Callable[[str], float] = lambda symbol: 0.0
    get_traded_today: Callable[[], float] = lambda: 0.0
    logger: Callable[[str], None] = print

    # Internal
    approvals: ApprovalsState = field(default_factory=ApprovalsState)

    # Optional custom hook to score riskiness of each order (0..1)
    risk_score_fn: Optional[Callable[[Dict[str, Any]], float]] = None
    # Threshold for SEMI_AUTO auto-approve (e.g., <=0.35 is small/benign)
    semi_auto_threshold: float = 0.35

    def describe(self) -> Dict[str, Any]:
        return {
            "run_mode": self.run_mode.name,
            "control_mode": self.control_mode.name,
            "limits": asdict(self.limits),
            "queue_len": len(self.approvals.queue),
        }

    # -------------------------------------------------------------------------
    # Public API: gate orders according to mode & limits
    # -------------------------------------------------------------------------

    def process_orders(self, orders: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Input DataFrame columns (required):
          ['ticker','trade_notional','side'] and optionally
          ['tenor_years','currency','client_order_id','px_hint_bps','meta']
        Returns dict:
          {
            "approved": DataFrame (ready to send),
            "queued":   DataFrame (waiting for approval),
            "rejected": DataFrame (dropped due to limits/invalid)
          }
        """
        required = {"ticker", "trade_notional", "side"}
        if not required.issubset(orders.columns):
            raise ValueError(f"orders must include {required}")

        orders = orders.copy()
        # Normalize types
        orders["ticker"] = orders["ticker"].astype(str)
        orders["trade_notional"] = orders["trade_notional"].astype(float)
        orders["side"] = orders["side"].astype(str)

        # 1) Hard risk filters
        ok_mask, rej_rows = self._hard_filters(orders)

        filtered = orders[ok_mask].copy()
        rejected = orders.loc[~ok_mask].copy()
        if not rejected.empty:
            self.logger(f"[modes] rejected {len(rejected)} orders by hard limits")

        # 2) Cooldown per symbol base
        cd_mask, cd_rej = self._cooldown_filter(filtered)
        filtered = filtered[cd_mask]
        if not cd_rej.empty:
            rejected = pd.concat([rejected, cd_rej], ignore_index=True)
            self.logger(f"[modes] cooldown blocked {len(cd_rej)} orders")

        # 3) Control mode branching
        if self.control_mode == ControlMode.AUTO:
            approved = filtered
            queued = filtered.iloc[0:0]
        elif self.control_mode == ControlMode.MANUAL:
            approved = filtered.iloc[0:0]
            queued = filtered
            self._enqueue_for_approval(queued)
        else:  # SEMI_AUTO
            auto_mask, to_queue = self._semi_auto_split(filtered)
            approved = filtered[auto_mask]
            queued = filtered[~auto_mask]
            if not queued.empty:
                self._enqueue_for_approval(queued)

        # 4) Soft caps on daily turnover (scale down proportionally)
        approved = self._apply_turnover_cap(approved)

        return {"approved": approved.reset_index(drop=True),
                "queued": queued.reset_index(drop=True),
                "rejected": rejected.reset_index(drop=True)}

    # -------------------------------------------------------------------------
    # Manual approval surface
    # -------------------------------------------------------------------------

    def pending(self) -> pd.DataFrame:
        return pd.DataFrame(self.approvals.queue)

    def approve(self, client_order_id: str, decision: str) -> None:
        decision = decision.strip().upper()
        if decision not in ("APPROVE", "REJECT"):
            raise ValueError("decision must be APPROVE or REJECT")
        self.approvals.decisions[client_order_id] = decision

    def drain_approved(self) -> pd.DataFrame:
        """
        Move approved items from queue to an 'approved' batch and remove them from queue.
        """
        if not self.approvals.queue:
            return pd.DataFrame(columns=["ticker","trade_notional","side","client_order_id"])
        rows, remain = [], []
        for row in self.approvals.queue:
            cid = str(row.get("client_order_id", ""))
            dec = self.approvals.decisions.get(cid)
            if dec == "APPROVE":
                rows.append(row)
                # update cooldown timer on approval
                base = base_symbol(row["ticker"])
                self.approvals.last_action_ts[base] = int(time.time())
            elif dec == "REJECT" or dec is None:
                remain.append(row) if dec is None else None
        # Keep only items still pending (None)
        self.approvals.queue = remain
        # Clean consumed decisions
        for r in rows:
            self.approvals.decisions.pop(r.get("client_order_id", ""), None)
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _hard_filters(self, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        mask = pd.Series(True, index=df.index)

        # size bounds
        mask &= df["trade_notional"].abs() >= self.limits.min_order_usd
        mask &= df["trade_notional"].abs() <= self.limits.max_order_usd

        # per-name limit
        def within_name_cap(row) -> bool:
            name = base_symbol(row["ticker"])
            cur = float(self.get_position(name))
            signed = float(row["trade_notional"]) if "BUY" in row["side"].upper() else -float(row["trade_notional"])
            return abs(cur + signed) <= (self.limits.max_name_usd + 1e-6)
        mask &= df.apply(within_name_cap, axis=1)

        # shorting allowed?
        if not self.limits.allow_short:
            def no_short(row) -> bool:
                return "BUY" in row["side"].upper()
            mask &= df.apply(no_short, axis=1)

        rejected = df.loc[~mask].copy()
        return mask, rejected

    def _cooldown_filter(self, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        if df.empty or self.limits.cooldown_secs <= 0:
            return pd.Series(True, index=df.index), df.iloc[0:0]
        now = int(time.time())
        def ok(row) -> bool:
            base = base_symbol(row["ticker"])
            last = int(self.approvals.last_action_ts.get(base, 0))
            return (now - last) >= self.limits.cooldown_secs
        mask = df.apply(ok, axis=1)
        return mask, df.loc[~mask].copy()

    def _semi_auto_split(self, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        if df.empty:
            return pd.Series(dtype=bool), df
        if self.risk_score_fn is None:
            # default heuristic: smaller orders auto, larger queued
            score = (df["trade_notional"].abs() / max(1.0, self.limits.max_order_usd)).clip(0, 1.0)
        else:
            score = df.apply(lambda r: float(self.risk_score_fn(r.to_dict())), axis=1).clip(0, 1.0) # type: ignore
        auto_mask = score <= float(self.semi_auto_threshold)
        return auto_mask, df.loc[~auto_mask].copy()

    def _apply_turnover_cap(self, approved: pd.DataFrame) -> pd.DataFrame:
        if approved.empty:
            return approved
        already = float(self.get_traded_today())
        budget = max(0.0, self.limits.max_daily_turnover_usd - already)
        want = float(approved["trade_notional"].abs().sum())
        if want <= budget or budget <= 0:
            if budget <= 0 and want > 0:
                self.logger("[modes] daily turnover limit reached; zeroing approved orders")
                return approved.iloc[0:0]
            return approved
        scale = budget / max(want, 1e-9)
        self.logger(f"[modes] scaling approved orders by {scale:.2%} to fit daily turnover cap")
        out = approved.copy()
        out["trade_notional"] = out["trade_notional"] * scale
        return out

    def _enqueue_for_approval(self, df: pd.DataFrame) -> None:
        for _, r in df.iterrows():
            item = r.to_dict()
            cid = item.get("client_order_id")
            if not cid:
                item["client_order_id"] = f"pending-{int(time.time()*1000)}"
            self.approvals.queue.append(item)


# =============================================================================
# Helpers
# =============================================================================

def base_symbol(ticker: str) -> str:
    """
    Reduce instrument to a base name for risk/cooldown (e.g., 'IG_A_5Y' -> 'IG_A').
    Customize if you need more precise grouping (e.g., by issuer).
    """
    parts = str(ticker).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Toy demo
    ctrl = ModeController(
        run_mode=RunMode.PAPER,
        control_mode=ControlMode.SEMI_AUTO,
        limits=RiskLimits(max_name_usd=2_000_000, max_order_usd=1_000_000, cooldown_secs=5),
        get_position=lambda name: {"IG_A": 500_000.0}.get(name, 0.0),
        get_traded_today=lambda: 1_000_000.0,
    )

    batch = pd.DataFrame([
        {"ticker": "IG_A_5Y", "trade_notional": 800_000, "side": "BUY_PROTECTION", "client_order_id": "x1"},
        {"ticker": "HY_B_5Y", "trade_notional": 3_000_000, "side": "SELL_PROTECTION", "client_order_id": "x2"},  # above max_order → reject
        {"ticker": "IG_A_5Y", "trade_notional": 300_000, "side": "SELL_PROTECTION", "client_order_id": "x3"},    # maybe cooldown reject
    ])

    out = ctrl.process_orders(batch)
    print("APPROVED\n", out["approved"])
    print("QUEUED\n", out["queued"])
    print("REJECTED\n", out["rejected"])

    # Manually approve queued orders
    for _, row in out["queued"].iterrows():
        ctrl.approve(row["client_order_id"], "APPROVE")

    to_send = ctrl.drain_approved()
    print("DRAINED (now approved)\n", to_send)