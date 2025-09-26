# backend/compliance/entitlements.py
"""
Entitlements Manager
--------------------
Purpose:
  • Centralize user/role/permission mapping.
  • Enforce least privilege: only authorized users can run allocators,
    view risk dashboards, or trigger hedges.
  • Provide machine-checkable proofs for audit logs.
  • Optionally back by YAML/JSON file.

Entities
--------
- User: identified by user_id / email.
- Role: "pm", "quant", "ops", "admin".
- Permissions: strings like "alloc.write", "risk.view", "oms.trade".

Typical usage
-------------
from backend.compliance.entitlements import Entitlements, EntitlementError

ents = Entitlements.from_file("config/entitlements.yml")
if ents.check("alice@example.com", "alloc.write"):
    ...
else:
    raise EntitlementError("not authorized")

ents.audit_proof("alice@example.com", "alloc.write", verdict=True)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml  # pip install pyyaml


# -------------------- Errors --------------------

class EntitlementError(Exception):
    pass


# -------------------- Data structures --------------------

@dataclass
class UserEnt:
    user_id: str
    roles: List[str]

@dataclass
class RoleDef:
    role: str
    permissions: List[str]


# -------------------- Core class --------------------

class Entitlements:
    def __init__(self, users: Dict[str, UserEnt], roles: Dict[str, RoleDef]) -> None:
        self._users = users
        self._roles = roles

    # ---- Factory loaders ----

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Entitlements":
        users = {u["user_id"]: UserEnt(user_id=u["user_id"], roles=u.get("roles", []))
                 for u in d.get("users", [])}
        roles = {r["role"]: RoleDef(role=r["role"], permissions=r.get("permissions", []))
                 for r in d.get("roles", [])}
        return Entitlements(users, roles)

    @staticmethod
    def from_file(path: str) -> "Entitlements":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        with open(p, "r") as f:
            if path.endswith(".json"):
                d = json.load(f)
            else:
                d = yaml.safe_load(f)
        return Entitlements.from_dict(d)

    # ---- API ----

    def user_roles(self, user_id: str) -> List[str]:
        return self._users.get(user_id, UserEnt(user_id, [])).roles

    def role_permissions(self, role: str) -> List[str]:
        return self._roles.get(role, RoleDef(role, [])).permissions

    def check(self, user_id: str, perm: str) -> bool:
        """Return True if user has permission perm."""
        roles = self.user_roles(user_id)
        for r in roles:
            if perm in self.role_permissions(r):
                return True
        return False

    def require(self, user_id: str, perm: str) -> None:
        if not self.check(user_id, perm):
            raise EntitlementError(f"{user_id} not entitled to {perm}")

    def audit_proof(self, user_id: str, perm: str, verdict: bool) -> Dict[str, Any]:
        """Produce an audit-friendly proof dict for Merkle ledger."""
        return {
            "ts": int(time.time() * 1000),
            "proof_type": "entitlement_check",
            "user_id": user_id,
            "perm": perm,
            "verdict": "PASS" if verdict else "FAIL",
            "roles": self.user_roles(user_id),
        }


# -------------------- Example config --------------------
"""
YAML example (config/entitlements.yml):

roles:
  - role: admin
    permissions: ["*"]
  - role: pm
    permissions: ["alloc.write", "risk.view"]
  - role: quant
    permissions: ["alloc.sim", "risk.view"]
  - role: ops
    permissions: ["oms.trade", "alloc.read"]

users:
  - user_id: alice@example.com
    roles: ["pm"]
  - user_id: bob@example.com
    roles: ["ops","quant"]
"""