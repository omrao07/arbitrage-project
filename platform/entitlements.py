# platform/entitlements.py
from __future__ import annotations

import fnmatch
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class Decision:
    allowed: bool
    rule: Optional[Dict[str, Any]] = None
    reason: str = ""


class Entitlements:
    """
    Role-based entitlements with wildcard resources, optional action/region scoping,
    explicit deny support, and hot-reload on policy file mtime change.

    YAML shape (configs/policies/entitlements.yml):
      version: 1
      rules:
        - roles: ["analyst", "risk"]
          resources: ["analytics/*", "dashboards/*"]
          actions: ["read"]            # optional, default: any
          region: ["US","EU"]          # optional, default: any
          effect: "allow"              # "allow" (default) or "deny"
        - roles: ["admin"]
          resources: ["*"]
          effect: "allow"

      # Optional role inheritance
      role_hierarchy:
        admin: ["trader", "risk", "analyst"]
        trader: ["research"]           # trader implicitly has research

    Usage:
      ent = Entitlements("configs/policies/entitlements.yml")
      ent.allow(["analyst"], "analytics/gnn_correlations", action="read", region="EU")
    """

    def __init__(self, policy_path: str, hot_reload: bool = True, cache_ttl_s: int = 30) -> None:
        self._path = policy_path
        self._hot_reload = hot_reload
        self._lock = threading.RLock()
        self._mtime = 0.0
        self._rules: List[Dict[str, Any]] = []
        self._role_graph: Dict[str, List[str]] = {}
        # tiny cache: (frozenset(roles), resource, action, region) -> (ts, Decision)
        self._cache_ttl = max(cache_ttl_s, 1)
        self._cache: Dict[Tuple[frozenset, str, str, str], Tuple[float, Decision]] = {}
        self._load()

    # ---------------------------- Public API -----------------------------

    def allow(
        self,
        user_roles: List[str],
        resource: str,
        *,
        action: Optional[str] = None,
        region: Optional[str] = None,
    ) -> bool:
        """Convenience: return True/False only."""
        return self.decide(user_roles, resource, action=action, region=region).allowed

    def decide(
        self,
        user_roles: List[str],
        resource: str,
        *,
        action: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Decision:
        """
        Returns a Decision with {allowed: bool, rule: matched_rule, reason: str}.
        Order of precedence:
          1) Explicit DENY matching user/ctx/resource
          2) ALLOW matching user/ctx/resource
          3) Default DENY
        """
        self._maybe_reload()

        roles_norm = frozenset(self._expand_roles(set(r.lower() for r in user_roles)))
        resource_norm = resource.strip()
        action_norm = (action or "*").lower()
        region_norm = (region or "*").upper()

        key = (roles_norm, resource_norm, action_norm, region_norm)
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] <= self._cache_ttl:
                return hit[1]

        # 1) Collect candidate rules matching context (roles x action x region x resource)
        allow_match: Optional[Dict[str, Any]] = None
        deny_match: Optional[Dict[str, Any]] = None

        for rule in self._rules:
            if not self._roles_intersect(roles_norm, rule.get("_roles_norm", [])):
                continue
            if not _match_any(action_norm, rule.get("_actions_norm", ["*"])):
                continue
            if not _match_any(region_norm, rule.get("_regions_norm", ["*"])):
                continue
            if not _match_any_path(resource_norm, rule.get("_resources_norm", ["*"])):
                continue

            eff = (rule.get("effect") or "allow").lower()
            if eff == "deny" and deny_match is None:
                deny_match = rule
                # keep scanning to locate more specific denies, but first deny is enough to reject
            elif eff == "allow" and allow_match is None:
                allow_match = rule

        if deny_match:
            dec = Decision(allowed=False, rule=_public_rule(deny_match), reason="explicit_deny")
        elif allow_match:
            dec = Decision(allowed=True, rule=_public_rule(allow_match), reason="matched_allow")
        else:
            dec = Decision(allowed=False, rule=None, reason="no_matching_rule")

        with self._lock:
            self._cache[key] = (now, dec)
        return dec

    # ---------------------------- Internals ------------------------------

    def _maybe_reload(self) -> None:
        if not self._hot_reload:
            return
        try:
            mtime = os.path.getmtime(self._path)
        except FileNotFoundError:
            return
        if mtime != self._mtime:
            with self._lock:
                # Double-checked locking
                if mtime != self._mtime:
                    self._load()

    def _load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        rules = cfg.get("rules") or []
        role_h = cfg.get("role_hierarchy") or {}

        # Normalize & compile rules
        compiled: List[Dict[str, Any]] = []
        for r in rules:
            roles = [str(x).lower() for x in r.get("roles", [])]
            resources = [str(x) for x in r.get("resources", [])]
            actions = [str(x).lower() for x in r.get("actions", ["*"])]
            regions = [str(x).upper() for x in r.get("region", ["*"])]  # may be str or list

            compiled.append(
                {
                    **r,
                    "_roles_norm": roles,
                    "_resources_norm": resources,
                    "_actions_norm": actions,
                    "_regions_norm": regions,
                    "effect": (r.get("effect") or "allow").lower(),
                }
            )

        with self._lock:
            self._rules = compiled
            self._role_graph = {str(k).lower(): [str(v).lower() for v in (vals or [])] for k, vals in role_h.items()}
            self._mtime = os.path.getmtime(self._path) if os.path.exists(self._path) else time.time()
            self._cache.clear()

    def _expand_roles(self, roles: set[str]) -> set[str]:
        """Return roles plus inherited roles via role_hierarchy (transitively)."""
        expanded = set(roles)
        stack = list(roles)
        # DFS through role graph
        while stack:
            r = stack.pop()
            for child in self._role_graph.get(r, []):
                if child not in expanded:
                    expanded.add(child)
                    stack.append(child)
        return expanded

    @staticmethod
    def _roles_intersect(user_roles: frozenset[str], rule_roles: List[str]) -> bool:
        if not rule_roles:
            return False
        # Direct match or via inheritance handled by caller (expand_roles).
        return bool(set(rule_roles) & set(user_roles))


def _public_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal normalized keys before returning rule in decisions."""
    return {
        k: v
        for k, v in rule.items()
        if not k.startswith("_")
    }


def _match_any(value: str, patterns: List[str]) -> bool:
    """Case-insensitive wildcard match (supports '*' and '?')."""
    v = value.lower()
    for p in patterns:
        if p == "*":
            return True
        if fnmatch.fnmatch(v, str(p).lower()):
            return True
    return False


def _match_any_path(resource: str, patterns: List[str]) -> bool:
    """
    Resource matcher for path-like strings: supports wildcards at any segment.
    Examples:
      resource='analytics/gnn' matches ['analytics/*']
      resource='exec/orders/place' matches ['exec/*', 'exec/orders/*']
    """
    r = resource.strip()
    for p in patterns:
        p_str = str(p).strip()
        if p_str == "*" or fnmatch.fnmatch(r, p_str):
            return True
    return False