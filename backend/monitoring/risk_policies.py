# risk/risk_policies.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Optional: your existing stack (soft imports so this module is standalone)
try:
    from risk.risk_limits import RiskLimits, LimitsConfig # type: ignore
except Exception:  # pragma: no cover
    RiskLimits = object             # type: ignore
    LimitsConfig = object           # type: ignore

try:
    from orchestrator.alerts import Alerts # type: ignore
except Exception:  # pragma: no cover
    class Alerts:                      # minimal shim
        def __init__(self, *_, **__): pass
        def risk(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass


# ======================================================================
# Data models
# ======================================================================

@dataclass
class EscalationTarget:
    name: str
    channel: str = "slack"          # slack|email|discord|telegram
    address: Optional[str] = None   # webhook/email/chat id


@dataclass
class PolicyContext:
    sid: str                              # strategy id
    book: str = "default"                 # book/portfolio name
    user: str = "system"                  # actor (trader/service)
    tz: str = "UTC"
    regime: Dict[str, Any] = field(default_factory=dict)   # e.g., {"risk_off": True, "fed_day": True}
    now_ts: int = field(default_factory=lambda: int(time.time()))


@dataclass
class Decision:
    action: str                           # APPROVE | QUEUE | REJECT | HALT
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """Declarative rule using a tiny predicate DSL evaluated row-wise or batch-wise."""
    name: str
    when: Dict[str, Any]                  # predicate spec (see _eval_predicate)
    then: str                             # APPROVE|QUEUE|REJECT|HALT|TAG:<name>|SET:<k>=<v>
    priority: int = 100                   # lower runs first
    scope: str = "row"                    # row | batch


@dataclass
class Policy:
    name: str
    version: str = "1.0"
    enabled: bool = True
    families: List[str] = field(default_factory=list)        # strategy families it applies to
    tags_any: List[str] = field(default_factory=list)         # registry tags the policy targets
    rules: List[Rule] = field(default_factory=list)
    exemptions: Dict[str, Any] = field(default_factory=dict)  # {"users":["alice"], "sids":["BW-0001"], "until":"2025-12-31"}
    escalation: List[EscalationTarget] = field(default_factory=list)
    circuit_breaker: Dict[str, Any] = field(default_factory=dict)  # {"pnl_day$": -750000, "var$": 500000}
    hard_stop: bool = False


@dataclass
class AuditEvent:
    ts: int
    sid: str
    policy: str
    rule: Optional[str]
    action: str
    reason: str
    meta: Dict[str, Any]


# ======================================================================
# Policy Engine
# ======================================================================

class RiskPolicyEngine:
    """
    Evaluate orders against higher-level policies (macro/news/regime governance),
    then pass the approved batch to `RiskLimits.pretrade_check()` if you wire it.

    Input orders DataFrame requires columns:
      - 'ticker', 'trade_notional', 'side'
    Optional (recommended) columns:
      - 'asset', 'family', 'tags', 'adv_usd', 'latency_ms', 'cost_bps', 'country', 'sector', 'issuer',
        'greeks' (gamma/vega/theta as separate cols), 'strategy_id', 'client_order_id'
    """

    def __init__(
        self,
        *,
        alerts: Optional[Alerts] = None,
        risk_limits: Optional[RiskLimits] = None, # type: ignore
        audit_path: Path = Path("runtime/logs/policy_audit.jsonl"),
        registry_view: Optional[pd.DataFrame] = None,  # strategy registry for metadata matchers
    ):
        self.alerts = alerts or Alerts()
        self.risk_limits = risk_limits
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = registry_view

        self._policies: List[Policy] = []

    # ------------------------------------------------------------------
    # Load / manage policies
    # ------------------------------------------------------------------

    def load_yaml(self, path: Path | str) -> None:
        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        # file may define a single policy or a list under 'policies'
        entries = data.get("policies", data if isinstance(data, list) else [data])
        for item in entries:
            pol = _policy_from_dict(item)
            self._policies.append(pol)

    def add_policy(self, policy: Policy) -> None:
        self._policies.append(policy)

    def list_policies(self) -> List[Dict[str, Any]]:
        return [asdict(p) for p in self._policies]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        orders: pd.DataFrame,
        ctx: PolicyContext,
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns dict of DataFrames:
          - approved  → pass into RiskLimits / router
          - queued    → requires manual approval
          - rejected  → blocked by policies
          - halted    → if any policy HALT, everything pauses
        """
        req = {"ticker", "trade_notional", "side"}
        if not req.issubset(orders.columns):
            raise ValueError(f"orders must include {req}")

        df = orders.copy()
        df["__action__"] = "APPROVE"
        df["__reason__"] = "DEFAULT_ALLOW"
        df["__tags__"] = df.get("tags", "")

        # 0) Circuit breakers (book-level)
        if self._should_halt(ctx):
            self._audit(AuditEvent(ts=ctx.now_ts, sid=ctx.sid, policy="circuit_breaker",
                                   rule=None, action="HALT", reason="BREACH", meta={"regime": ctx.regime}))
            self.alerts.risk("Circuit breaker HALT", {"sid": ctx.sid, "regime": ctx.regime})
            return {"approved": df.iloc[0:0], "queued": df.iloc[0:0], "rejected": df.iloc[0:0], "halted": df}

        # 1) Apply policies
        applicable = [p for p in self._policies if p.enabled and _policy_matches(p, ctx, self.registry)] # type: ignore
        applicable.sort(key=lambda p: p.name)

        for pol in applicable:
            for rule in sorted(pol.rules, key=lambda r: r.priority):
                try:
                    if rule.scope == "batch":
                        action, mask, reason = _apply_rule_batch(rule, df, ctx)
                        if action:
                            df.loc[:, "__action__"] = df["__action__"].where(~mask, action)
                            df.loc[mask, "__reason__"] = reason
                            self._audit_many(pol, rule, action, reason, df.loc[mask])
                            if action == "HALT":
                                self.alerts.risk(f"HALT by policy {pol.name}", {"rule": rule.name, "sid": ctx.sid})
                                return {"approved": df.iloc[0:0], "queued": df.iloc[0:0], "rejected": df.iloc[0:0], "halted": df}
                    else:
                        action, mask, reason = _apply_rule_row(rule, df, ctx)
                        df.loc[mask, "__action__"] = action
                        df.loc[mask, "__reason__"] = reason
                        if mask.any():
                            self._audit_many(pol, rule, action, reason, df.loc[mask])
                except Exception as e:  # robust against bad policy
                    self.alerts.error("policy", f"Rule '{rule.name}' failed", {"err": str(e), "policy": pol.name})
                    continue

            # Handle policy-level hard stop
            if pol.hard_stop and (df["__action__"] == "REJECT").any():
                self.alerts.risk(f"Hard-stop REJECT by {pol.name}", {"sid": ctx.sid})
                break

        # 2) Exemptions (last — to allow known whitelists)
        df = self._apply_exemptions(df, applicable, ctx)

        # 3) Split by action and (optional) pass approved into RiskLimits
        approved = df[df["__action__"] == "APPROVE"].drop(columns=["__action__","__reason__","__tags__"], errors="ignore")
        queued   = df[df["__action__"] == "QUEUE"].drop(columns=["__action__","__reason__","__tags__"], errors="ignore")
        rejected = df[df["__action__"] == "REJECT"].drop(columns=["__action__","__reason__","__tags__"], errors="ignore")

        # Optional downstream hard limits
        if isinstance(self.risk_limits, RiskLimits) and len(approved):
            checks = self.risk_limits.pretrade_check(approved) # type: ignore
            approved = checks["approved"]
            # carry scaled rejects into "queued" with reason
            scaled = checks["scaled"]
            if len(scaled):
                scaled["__policy_note__"] = scaled.get("reason", "SCALED")
                queued = pd.concat([queued, scaled.drop(columns=["reason"])], ignore_index=True)
            rej = checks["rejected"]
            if len(rej):
                rej["__policy_note__"] = rej.get("reason", "RISK_LIMITS")
                rejected = pd.concat([rejected, rej.drop(columns=["reason"])], ignore_index=True)

        return {"approved": approved.reset_index(drop=True),
                "queued": queued.reset_index(drop=True),
                "rejected": rejected.reset_index(drop=True),
                "halted": df[df["__action__"] == "HALT"].reset_index(drop=True)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_halt(self, ctx: PolicyContext) -> bool:
        # If any policy defines circuit_breaker with breached metrics (requires risk_limits for some)
        for pol in self._policies:
            if not pol.enabled or not pol.circuit_breaker:
                continue
            br = pol.circuit_breaker
            if "pnl_day$" in br:
                # If you wire daily pnl via risk_limits (or external), read from there; else allow ctx.regime feed-in
                pnl = float(ctx.regime.get("pnl_day$", 0.0))
                if hasattr(self.risk_limits, "state"):
                    pnl = getattr(self.risk_limits.state, "daily_pnl_usd", pnl) # type: ignore
                if pnl <= -abs(float(br["pnl_day$"])):
                    return True
            if "var$" in br and hasattr(self.risk_limits, "var_parametric"):
                var_now = float(self.risk_limits.var_parametric() or 0.0) # type: ignore
                if var_now > float(br["var$"]):
                    return True
            if "flag" in br and ctx.regime.get(br["flag"], False):
                return True
        return False

    def _apply_exemptions(self, df: pd.DataFrame, policies: List[Policy], ctx: PolicyContext) -> pd.DataFrame:
        if df.empty:
            return df
        exempt_users = set()
        exempt_sids  = set()
        until_ts = None
        for p in policies:
            ex = p.exemptions or {}
            exempt_users |= set(ex.get("users", []))
            exempt_sids  |= set(ex.get("sids", []))
            if ex.get("until"):
                try:
                    until_ts = max(until_ts or 0, int(pd.Timestamp(ex["until"]).timestamp()))
                except Exception:
                    pass

        if (ctx.user in exempt_users or ctx.sid in exempt_sids) and (until_ts is None or ctx.now_ts <= until_ts):
            # Downgrade REJECT->QUEUE for audit, keep APPROVE as-is
            mask = df["__action__"] == "REJECT"
            if mask.any():
                df.loc[mask, "__action__"] = "QUEUE"
                df.loc[mask, "__reason__"] = "EXEMPTION_DOWNGRADE"
        return df

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _audit_many(self, pol: Policy, rule: Rule, action: str, reason: str, rows: pd.DataFrame):
        if rows is None or rows.empty:
            return
        for _, r in rows.iterrows():
            evt = AuditEvent(
                ts=int(time.time()),
                sid=str(r.get("strategy_id", "")) or pol.name,
                policy=pol.name,
                rule=rule.name,
                action=action,
                reason=reason,
                meta={"ticker": r.get("ticker"), "notional": r.get("trade_notional"), "side": r.get("side")}
            )
            self._audit(evt)

    def _audit(self, evt: AuditEvent):
        try:
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(evt)) + "\n")
        except Exception:
            pass


# ======================================================================
# Rule evaluation helpers (tiny DSL)
# ======================================================================

def _apply_rule_row(rule: Rule, df: pd.DataFrame, ctx: PolicyContext) -> Tuple[str, pd.Series, str]:
    mask = _eval_predicate(rule.when, df, ctx, row_scope=True)
    action = rule.then.strip().upper()
    reason = f"{rule.name}"
    return action, mask, reason # type: ignore

def _apply_rule_batch(rule: Rule, df: pd.DataFrame, ctx: PolicyContext) -> Tuple[Optional[str], pd.Series, str]:
    # batch predicate returns a boolean; if True → action applies to ALL rows by default,
    # but you can combine with a 'where' clause inside the predicate to narrow it.
    mask = _eval_predicate(rule.when, df, ctx, row_scope=False)
    if mask is None:
        return None, pd.Series(False, index=df.index), ""
    m = mask if isinstance(mask, pd.Series) else pd.Series(bool(mask), index=df.index)
    action = rule.then.strip().upper()
    reason = f"{rule.name}"
    return action, m, reason

def _eval_predicate(spec: Dict[str, Any], df: pd.DataFrame, ctx: PolicyContext, *, row_scope: bool) -> pd.Series | bool | None:
    """
    Supported operators:
      - eq/ne/gt/gte/lt/lte: {"col": "country", "op": "eq", "value": "RU"}
      - in/not_in: {"col": "sector", "op":"in", "value":["Energy","Defense"]}
      - contains: {"col":"tags","op":"contains","value":"event:CPI"}
      - regex: {"col":"ticker","op":"regex","value":"^IG_.*_5Y$"}
      - between: {"col":"hhmm","op":"between","value":[935, 1555]}  (requires precomputed hhmm col)
      - any/all (nested): {"any":[{...},{...}]}, {"all":[{...},{...}]}
      - regime flag: {"regime":"risk_off"} (True/False)
      - constant true/false: {"true":1} / {"false":1}
      - where (narrow batch mask): {"where": {"col":"country","op":"eq","value":"RU"}}
    """
    if not spec:
        return True if not row_scope else pd.Series(True, index=df.index)

    if "any" in spec:
        parts = [_eval_predicate(s, df, ctx, row_scope=row_scope) for s in spec["any"]]
        if row_scope:
            out = pd.Series(False, index=df.index)
            for p in parts:
                out = out | (p if isinstance(p, pd.Series) else bool(p))
            return out
        return any(bool(p.iloc[-1] if isinstance(p, pd.Series) else p) for p in parts)

    if "all" in spec:
        parts = [_eval_predicate(s, df, ctx, row_scope=row_scope) for s in spec["all"]]
        if row_scope:
            out = pd.Series(True, index=df.index)
            for p in parts:
                out = out & (p if isinstance(p, pd.Series) else bool(p))
            return out
        return all(bool(p.iloc[-1] if isinstance(p, pd.Series) else p) for p in parts)

    if "regime" in spec:
        key = str(spec["regime"])
        return bool(ctx.regime.get(key, False))

    if "true" in spec:
        return True if not row_scope else pd.Series(True, index=df.index)
    if "false" in spec:
        return False if not row_scope else pd.Series(False, index=df.index)

    # column operator
    col = spec.get("col")
    op = str(spec.get("op", "eq")).lower()
    val = spec.get("value")

    if col is None and "where" in spec:
        # batch narrowing
        mask = _eval_predicate(spec["where"], df, ctx, row_scope=True)
        return mask

    series = df[col] if (col in df.columns) else pd.Series([None]*len(df), index=df.index)

    if op == "eq":   return series == val
    if op == "ne":   return series != val
    if op == "gt":   return series.astype(float) > float(val) # type: ignore
    if op == "gte":  return series.astype(float) >= float(val) # type: ignore
    if op == "lt":   return series.astype(float) < float(val) # type: ignore
    if op == "lte":  return series.astype(float) <= float(val) # type: ignore
    if op == "in":   return series.isin(val if isinstance(val, list) else [val])
    if op == "not_in": return ~series.isin(val if isinstance(val, list) else [val])
    if op == "contains":
        return series.astype(str).str.contains(str(val), case=False, na=False)
    if op == "regex":
        return series.astype(str).str.contains(val, regex=True, na=False) # type: ignore
    if op == "between":
        lo, hi = val # type: ignore
        return (series.astype(float) >= float(lo)) & (series.astype(float) <= float(hi))

    # default: unknown predicate → no-op
    return pd.Series(False, index=df.index)


# ======================================================================
# YAML helpers
# ======================================================================

def _policy_from_dict(d: Dict[str, Any]) -> Policy:
    rules = [Rule(**r) for r in d.get("rules", [])]
    esc = [EscalationTarget(**e) for e in d.get("escalation", [])]
    return Policy(
        name=d["name"],
        version=str(d.get("version","1.0")),
        enabled=bool(d.get("enabled", True)),
        families=list(d.get("families", [])),
        tags_any=list(d.get("tags_any", [])),
        rules=rules,
        exemptions=d.get("exemptions", {}) or {},
        escalation=esc,
        circuit_breaker=d.get("circuit_breaker", {}) or {},
        hard_stop=bool(d.get("hard_stop", False)),
    )


# ======================================================================
# Example policy YAML (save as risk/policies/example.yaml)
# ======================================================================

EXAMPLE_YAML = """\
policies:
  - name: "Geopolitical_Sanctions_Block"
    version: "1.2"
    families: ["equity_ls","futures_macro","credit_cds"]
    tags_any: ["em", "event:war"]
    rules:
      - name: "Block_Russia_Assets"
        priority: 1
        scope: "row"
        when: { col: "country", op: "eq", value: "RU" }
        then: "REJECT"
      - name: "Queue_Sanctioned_Issuers"
        priority: 5
        scope: "row"
        when: { col: "issuer", op: "in", value: ["GAZPROM","SBERBANK"] }
        then: "QUEUE"
  - name: "Macro_Regime_RiskOff"
    version: "1.0"
    rules:
      - name: "RiskOff_Cut_Procyc"
        priority: 10
        scope: "row"
        when:
          all:
            - { regime: "risk_off" }
            - { col: "tags", op: "contains", value: "style:procyc" }
        then: "QUEUE"
  - name: "CPI_FedDay_CircuitBreaker"
    version: "1.0"
    circuit_breaker:
      flag: "major_event"     # if ctx.regime.major_event True → HALT
    rules:
      - name: "CPI_Window"
        priority: 1
        scope: "batch"
        when:
          all:
            - { regime: "major_event" }
            - { where: { col: "hhmm", op: "between", value: [830, 845] } }
        then: "HALT"
"""

# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    # Build engine with example policies
    eng = RiskPolicyEngine()
    tmp = Path("risk/policies/example.yaml")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(EXAMPLE_YAML)
    eng.load_yaml(tmp)

    # Orders sample
    import numpy as np
    df = pd.DataFrame([
        {"ticker":"GAZP_RX", "trade_notional":250_000, "side":"BUY", "country":"RU", "issuer":"GAZPROM", "tags":"em,style:procyc", "hhmm": 931, "strategy_id":"BW-0001"},
        {"ticker":"SPY",     "trade_notional":500_000, "side":"BUY", "country":"US", "issuer":"-", "tags":"style:defens", "hhmm": 932, "strategy_id":"BW-0002"},
    ])

    ctx = PolicyContext(sid="BW-0001", regime={"risk_off": True, "major_event": True})
    out = eng.evaluate(df, ctx)
    print("\nAPPROVED\n", out["approved"])
    print("\nQUEUED\n", out["queued"])
    print("\nREJECTED\n", out["rejected"])
    print("\nHALTED\n", out["halted"])