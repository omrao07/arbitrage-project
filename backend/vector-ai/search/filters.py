# search/filters.py
"""
Unified filter builder for vector + lexical backends.

Supports ops:
- eq, ne, gt, gte, lt, lte
- in, nin
- contains (substring for strings)
- prefix (startsWith)
- between (inclusive)
- exists (bool)

Emits:
- to_predicate(): Python callable for client-side filtering (dict/pandas rows)
- to_pandas_query(): pandas .query() string (best-effort)
- to_pinecone(): Pinecone filter JSON
- to_weaviate(): Weaviate "where" JSON
- to_whoosh(): whoosh query string (basic)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import re
import datetime as dt

JSON = Dict[str, Any]

# ----------------------------- Public API -----------------------------

@dataclass
class Clause:
    field: str
    op: str
    value: Any

@dataclass
class Filter:
    """CNF: AND of OR groups. Example:
       (sector IN ["Tech","Health"]) AND (region="US" OR region="EU") AND (date BETWEEN [..])
    """
    should: List[List[Clause]]  # list of OR lists; each inner list is OR-ed, outer list AND-ed

    # ---------- Constructors ----------

    @staticmethod
    def from_dict(d: JSON) -> "Filter":
        """
        Accepts either:
          { "field": {"op": value, ...}, ... }  OR a compact dict {"field": value} (eq)
        Also accepts {"$and":[...], "$or":[...]} recursively.
        """
        if not d:
            return Filter(should=[])

        if "$and" in d or "$or" in d:
            return _from_bool_tree(d)

        # flat map: join via AND; each field may map to scalar (eq) or op-map
        groups: List[List[Clause]] = []
        for field, spec in d.items():
            if isinstance(spec, dict):
                ors: List[Clause] = []
                for op, val in spec.items():
                    ors.append(Clause(field=field, op=_normalize_op(op), value=val))
                # each {field: {op1,op2}} means AND across ops; model that as separate groups
                # e.g., price {gte: 10, lte: 20} -> AND [price>=10] AND [price<=20]
                for c in ors:
                    groups.append([c])
            else:
                groups.append([Clause(field=field, op="eq", value=spec)])
        return Filter(should=groups)

    # ---------- Backends ----------

    def to_predicate(self) -> Callable[[Dict[str, Any]], bool]:
        """Return a Python predicate(row_dict)->bool for client-side filtering."""
        compiled = []
        for or_group in self.should:
            compiled.append([_compile_clause(c) for c in or_group])

        def _pred(row: Dict[str, Any]) -> bool:
            for ors in compiled:
                # each group must have at least one True
                ok = any(fn(row) for fn in ors) if ors else True
                if not ok:
                    return False
            return True
        return _pred

    def to_pandas_query(self) -> str:
        """Best-effort pandas .query string. For complex ops (contains/prefix), falls back to .str methods."""
        parts: List[str] = []
        for ors in self.should:
            sub = []
            for c in ors:
                sub.append(_pandas_expr(c)) # type: ignore
            if sub:
                parts.append("(" + " or ".join(sub) + ")")
        return " and ".join(parts) if parts else ""

    def to_pinecone(self) -> JSON:
        """
        Pinecone filter grammar:
          { "$and": [ {field: {"$in":[...]}}, {field: {"$gte": 1}} ] }
        """
        if not self.should:
            return {}

        def one(c: Clause) -> JSON:
            field = c.field
            op = c.op
            v = c.value
            if op == "eq":   return {field: {"$eq": v}}
            if op == "ne":   return {field: {"$ne": v}}
            if op == "gt":   return {field: {"$gt": v}}
            if op == "gte":  return {field: {"$gte": v}}
            if op == "lt":   return {field: {"$lt": v}}
            if op == "lte":  return {field: {"$lte": v}}
            if op == "in":   return {field: {"$in": _as_list(v)}}
            if op == "nin":  return {field: {"$nin": _as_list(v)}}
            if op == "between":
                lo, hi = _as_range(v)
                return {field: {"$gte": lo, "$lte": hi}}
            if op == "exists":
                return {field: {"$exists": bool(v)}}
            if op == "contains":
                # Pinecone doesn't support substring; store ngrams or filter client-side.
                # Emit a hint key for your app to post-filter.
                return {"$expr_contains": {field: str(v)}}
            if op == "prefix":
                return {"$expr_prefix": {field: str(v)}}
            raise ValueError(f"Unsupported op for Pinecone: {op}")

        # Convert CNF (AND of ORs) into Pinecone AND of ORs
        and_terms: List[JSON] = []
        for ors in self.should:
            if len(ors) == 1:
                and_terms.append(one(ors[0]))
            else:
                and_terms.append({"$or": [one(c) for c in ors]})
        return {"$and": and_terms} if len(and_terms) > 1 else (and_terms[0] if and_terms else {})

    def to_weaviate(self) -> JSON:
        """
        Weaviate "where" filter (text props):
          { operator: "And", operands: [ ... ] }
        Supports Equal, NotEqual, GreaterThan, GreaterThanEqual, LessThan, LessThanEqual,
                 ContainsAny (for IN), Like (prefix), IsNull (exists:false).
        """
        if not self.should:
            return {}

        def one(c: Clause) -> JSON:
            f = c.field
            op = c.op
            v = c.value
            if op == "eq":   return {"path": [f], "operator": "Equal", "valueText": _as_text(v)}
            if op == "ne":   return {"path": [f], "operator": "NotEqual", "valueText": _as_text(v)}
            if op == "gt":   return {"path": [f], "operator": "GreaterThan", "valueText": _as_text(v)}
            if op == "gte":  return {"path": [f], "operator": "GreaterThanEqual", "valueText": _as_text(v)}
            if op == "lt":   return {"path": [f], "operator": "LessThan", "valueText": _as_text(v)}
            if op == "lte":  return {"path": [f], "operator": "LessThanEqual", "valueText": _as_text(v)}
            if op == "in":   return {"path": [f], "operator": "ContainsAny", "valueTextArray": [_as_text(x) for x in _as_list(v)]}
            if op == "nin":
                # emulate NOT IN: NOT (ContainsAny)
                return {"operator": "Not", "operands": [
                    {"path": [f], "operator": "ContainsAny", "valueTextArray": [_as_text(x) for x in _as_list(v)]}
                ]}
            if op == "between":
                lo, hi = _as_range(v)
                return {"operator": "And", "operands": [
                    {"path": [f], "operator": "GreaterThanEqual", "valueText": _as_text(lo)},
                    {"path": [f], "operator": "LessThanEqual", "valueText": _as_text(hi)},
                ]}
            if op == "exists":
                return {"path": [f], "operator": "IsNull", "valueBoolean": (not bool(v))}
            if op == "prefix":
                return {"path": [f], "operator": "Like", "valueText": f"{_escape_like(str(v))}*"}
            if op == "contains":
                return {"path": [f], "operator": "Like", "valueText": f"*{_escape_like(str(v))}*"}
            raise ValueError(f"Unsupported op for Weaviate: {op}")

        and_ops: List[JSON] = []
        for ors in self.should:
            if len(ors) == 1:
                and_ops.append(one(ors[0]))
            else:
                and_ops.append({"operator": "Or", "operands": [one(c) for c in ors]})
        return {"operator": "And", "operands": and_ops} if len(and_ops) > 1 else (and_ops[0] if and_ops else {})

    def to_whoosh(self) -> str:
        """
        Basic Whoosh query string for TEXT fields; numeric comparisons should be
        handled post-query or by storing numeric fields as TEXT with range codec.
        """
        parts: List[str] = []
        for ors in self.should:
            sub = []
            for c in ors:
                f, op, v = c.field, c.op, c.value
                if op == "eq":
                    sub.append(f'{f}:"{_escape_q(str(v))}"')
                elif op == "prefix":
                    sub.append(f'{f}:{_escape_q(str(v))}*')
                elif op == "contains":
                    # Whoosh doesn't have substring; approximate by wildcards
                    sub.append(f'{f}:*{_escape_q(str(v))}*')
                elif op == "in":
                    sub.append("(" + " OR ".join(f'{f}:"{_escape_q(str(x))}"' for x in _as_list(v)) + ")")
                # other ops left for client-side post-filter
            if sub:
                parts.append("(" + " OR ".join(sub) + ")")
        return " AND ".join(parts) if parts else ""


# ----------------------------- Internals ------------------------------

def _from_bool_tree(d: JSON) -> Filter:
    """
    Build CNF from a boolean JSON like:
      {"$and":[ {"field":{"gte":1}}, {"$or":[{"a":1},{"b":2}]} ]}
    """
    if not d:
        return Filter(should=[])

    def norm(node: Any) -> List[List[Clause]]:
        # returns CNF (list of OR-groups)
        if isinstance(node, dict):
            if "$and" in node:
                groups: List[List[Clause]] = []
                for child in node["$and"]:
                    groups.extend(norm(child))
                return groups
            if "$or" in node:
                or_group: List[Clause] = []
                for child in node["$or"]:
                    # child can be {field: spec}
                    sub = Filter.from_dict(child).should
                    # flatten: each from_dict(child) yields AND groups; for OR we only take singletons
                    for g in sub:
                        if len(g) == 1:
                            or_group.append(g[0])
                        else:
                            # if child produced multiple ANDed clauses, wrap them by splitting
                            for c in g:
                                or_group.append(c)
                return [or_group]
            # leaf {field: value/spec}
            return Filter.from_dict(node).should
        else:
            return []

    return Filter(should=norm(d))


def _normalize_op(op: str) -> str:
    op = op.lower().strip()
    aliases = {
        "equals": "eq", "==": "eq", "=": "eq",
        "!=": "ne", "<>": "ne",
        ">": "gt", ">=": "gte", "<": "lt", "<=": "lte",
        "startswith": "prefix",
        "icontains": "contains", "contains": "contains",
        "between": "between", "range": "between",
        "exists": "exists",
        "in": "in", "nin": "nin", "notin": "nin"
    }
    return aliases.get(op, op)


def _as_list(v: Any) -> List[Any]:
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _as_range(v: Any) -> Tuple[Any, Any]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return v[0], v[1]
    if isinstance(v, dict):
        return v.get("from"), v.get("to")
    raise ValueError("between expects [lo, hi] or {'from':..., 'to':...}")

def _as_text(v: Any) -> str:
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    return str(v)

def _escape_like(s: str) -> str:
    return s.replace("%", r"\%").replace("_", r"\_").replace("*", r"\*")

def _escape_q(s: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r'\\\1', s)

# -------------------------- Clause Evaluation -------------------------

def _compile_clause(c: Clause) -> Callable[[Dict[str, Any]], bool]:
    f, op, v = c.field, c.op, c.value
    vlist = _as_list(v)

    def get(row: Dict[str, Any]):
        return row.get(f, None)

    if op == "eq":
        return lambda row: get(row) == v
    if op == "ne":
        return lambda row: get(row) != v
    if op == "gt":
        return lambda row: _cmp(get(row), v, ">")
    if op == "gte":
        return lambda row: _cmp(get(row), v, ">=")
    if op == "lt":
        return lambda row: _cmp(get(row), v, "<")
    if op == "lte":
        return lambda row: _cmp(get(row), v, "<=")
    if op == "in":
        return lambda row: get(row) in set(vlist)
    if op == "nin":
        return lambda row: get(row) not in set(vlist)
    if op == "between":
        lo, hi = _as_range(v)
        return lambda row: _between(get(row), lo, hi)
    if op == "exists":
        want = bool(v)
        return lambda row: (get(row) is not None) if want else (get(row) is None)
    if op == "contains":
        val = str(v).lower()
        return lambda row: (str(get(row)).lower().find(val) >= 0) if get(row) is not None else False
    if op == "prefix":
        val = str(v).lower()
        return lambda row: str(get(row)).lower().startswith(val) if get(row) is not None else False

    # default: reject unknown op
    return lambda row: False

def _cmp(a: Any, b: Any, op: str) -> bool:
    try:
        if isinstance(a, str) and isinstance(b, str):
            pass
        elif _is_date_like(a) or _is_date_like(b):
            a = _to_dt(a); b = _to_dt(b)
        else:
            a = float(a) if a is not None else None
            b = float(b) if b is not None else None
        if a is None or b is None:
            return False
        if op == ">":  return a > b
        if op == ">=": return a >= b
        if op == "<":  return a < b
        if op == "<=": return a <= b
    except Exception:
        return False
    return False

def _between(a: Any, lo: Any, hi: Any) -> bool:
    try:
        if _is_date_like(a) or _is_date_like(lo) or _is_date_like(hi):
            a = _to_dt(a); lo = _to_dt(lo); hi = _to_dt(hi)
            if a is None or lo is None or hi is None:
                return False
            return lo <= a <= hi
        a = float(a); lo = float(lo); hi = float(hi)
        return lo <= a <= hi
    except Exception:
        return False

def _is_date_like(x: Any) -> bool:
    if isinstance(x, (dt.date, dt.datetime)):
        return True
    if isinstance(x, str) and re.match(r"^\d{4}-\d{2}-\d{2}", x):
        return True
    return False

def _to_dt(x: Any) -> Optional[dt.datetime]:
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.date):
        return dt.datetime.combine(x, dt.time())
    if isinstance(x, str):
        try:
            # ISO8601 best-effort
            return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


# ------------------------------- Examples ----------------------------

if __name__ == "__main__":
    # Example usage
    raw = {
        "sector": {"in": ["Tech", "Health"]},
        "region": {"eq": "US"},
        "date": {"between": ["2025-01-01", "2025-12-31"]},
        "headline": {"contains": "carry trade"}
    }
    f = Filter.from_dict(raw)

    # Client-side predicate
    pred = f.to_predicate()
    row = {"sector": "Tech", "region": "US", "date": "2025-06-10", "headline": "Yen carry trade unwinds"}
    print("match:", pred(row))

    # Pinecone
    print("pinecone:", f.to_pinecone())

    # Weaviate
    print("weaviate:", f.to_weaviate())

    # Whoosh
    print("whoosh:", f.to_whoosh())

    # Pandas
    print("pandas:", f.to_pandas_query())