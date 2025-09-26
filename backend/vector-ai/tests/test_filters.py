# tests/test_filters.py
"""
Unit tests for search/filters.py

Run:
  pytest -q tests/test_filters.py
"""

import datetime as dt
import pytest

from search.filters import Filter, Clause


# ----------------------------- Constructors -----------------------------

def test_from_dict_simple_eq():
    f = Filter.from_dict({"region": "US", "assetClass": {"in": ["FX", "Rates"]}})
    # should be AND of two groups
    assert isinstance(f.should, list) and len(f.should) == 2
    # group 1: region == US
    assert f.should[0][0].field == "region"
    assert f.should[0][0].op == "eq"
    assert f.should[0][0].value == "US"
    # group 2: assetClass IN [...]
    assert f.should[1][0].field == "assetClass"
    assert f.should[1][0].op == "in"
    assert f.should[1][0].value == ["FX", "Rates"]


def test_from_bool_tree_and_or():
    raw = {
        "$and": [
            {"sector": {"in": ["Tech", "Health"]}},
            {"$or": [
                {"region": "US"},
                {"region": "EU"}
            ]},
            {"date": {"between": ["2025-01-01", "2025-12-31"]}}
        ]
    }
    f = Filter.from_dict(raw)
    # Expect 3 groups: sector IN [...], (region=US OR region=EU), date between
    assert len(f.should) == 3
    assert f.should[0][0].field == "sector" and f.should[0][0].op == "in"
    assert {c.value for c in f.should[1]} == {"US", "EU"}
    assert f.should[2][0].op == "between"


# ----------------------------- Predicates -------------------------------

def test_predicate_basic_eq_in_between():
    f = Filter.from_dict({
        "sector": {"in": ["Tech", "Health"]},
        "region": {"eq": "US"},
        "date":   {"between": ["2025-01-01", "2025-12-31"]},
    })
    pred = f.to_predicate()
    row_ok = {"sector": "Tech", "region": "US", "date": "2025-06-10"}
    row_bad_sector = {"sector": "Energy", "region": "US", "date": "2025-06-10"}
    row_bad_region = {"sector": "Tech", "region": "EU", "date": "2025-06-10"}
    row_bad_date = {"sector": "Tech", "region": "US", "date": "2024-12-31"}
    assert pred(row_ok) is True
    assert pred(row_bad_sector) is False
    assert pred(row_bad_region) is False
    assert pred(row_bad_date) is False


def test_predicate_numeric_ops():
    f = Filter.from_dict({
        "score": {"gte": 0.7, "lte": 0.9},
        "count": {"gt": 10}
    })
    pred = f.to_predicate()
    assert pred({"score": 0.85, "count": 11}) is True
    assert pred({"score": 0.65, "count": 11}) is False
    assert pred({"score": 0.95, "count": 11}) is False
    assert pred({"score": 0.85, "count": 9}) is False


def test_predicate_exists_contains_prefix():
    f = Filter.from_dict({
        "headline": {"contains": "carry trade"},
        "ticker": {"prefix": "JP"},
        "notes": {"exists": False},
    })
    pred = f.to_predicate()
    row_ok = {"headline": "Yen carry trade unwind", "ticker": "JPY", "notes": None}
    row_bad_a = {"headline": "Other story", "ticker": "JPY", "notes": None}
    row_bad_b = {"headline": "Yen carry trade unwind", "ticker": "USD", "notes": None}
    row_bad_c = {"headline": "Yen carry trade unwind", "ticker": "JPY", "notes": "has text"}
    assert pred(row_ok) is True
    assert pred(row_bad_a) is False
    assert pred(row_bad_b) is False
    assert pred(row_bad_c) is False


def test_predicate_dates_dt_objects():
    f = Filter.from_dict({
        "ts": {"between": [dt.date(2025, 1, 1), dt.date(2025, 1, 31)]}
    })
    pred = f.to_predicate()
    assert pred({"ts": dt.datetime(2025, 1, 15, 12, 0, 0)}) is True
    assert pred({"ts": dt.datetime(2024, 12, 31, 23, 59, 59)}) is False


def test_predicate_nin():
    f = Filter.from_dict({"region": {"nin": ["US", "EU"]}})
    pred = f.to_predicate()
    assert pred({"region": "Asia"}) is True
    assert pred({"region": "US"}) is False


# ------------------------ Backend Translations --------------------------

def test_to_pandas_query_best_effort():
    f = Filter.from_dict({
        "sector": {"in": ["Tech", "Health"]},
        "region": "US",
        "score": {"gte": 0.5}
    })
    q = f.to_pandas_query()
    # we don't execute .query() here; just sanity-check structure
    assert "sector" in q and "in" in q or "==" in q
    assert "region" in q
    assert "score" in q


def test_to_pinecone_basic():
    f = Filter.from_dict({
        "region": "US",
        "assetClass": {"in": ["FX", "Rates"]},
        "date": {"between": ["2025-01-01", "2025-12-31"]},
        "haveNotes": {"exists": True}
    })
    pf = f.to_pinecone()
    assert "$and" in pf or "region" in pf
    # spot-check a couple clauses
    s = str(pf)
    assert "$eq" in s and "$in" in s and "$gte" in s and "$lte" in s


def test_to_weaviate_basic():
    f = Filter.from_dict({
        "ticker": {"prefix": "JP"},
        "headline": {"contains": "carry"},
        "region": {"eq": "Asia"},
        "haveNotes": {"exists": True}
    })
    wf = f.to_weaviate()
    # outer operator And or direct single object
    assert isinstance(wf, dict)
    s = str(wf)
    assert "Like" in s  # prefix/contains map to Like
    assert "Equal" in s or "valueText" in s
    assert "IsNull" in s  # exists -> IsNull false/true


def test_to_whoosh_query_string():
    f = Filter.from_dict({
        "region": "US",
        "ticker": {"prefix": "JP"},
        "headline": {"contains": "carry"},
        "sector": {"in": ["Tech", "Health"]}
    })
    wq = f.to_whoosh()
    # Escaped content with wildcards where expected
    assert 'region:"US"' in wq
    assert "ticker:JP*" in wq
    assert "headline:*carry*" in wq
    assert "(" in wq and "OR" in wq and ")" in wq


# ------------------------------ Edge Cases ------------------------------

def test_empty_filter_behaviour():
    f = Filter.from_dict({})
    assert f.to_pinecone() == {}
    assert f.to_weaviate() == {} or "operator" in f.to_weaviate() or f.to_weaviate() == {}
    assert f.to_whoosh() == ""
    pred = f.to_predicate()
    # empty filter should accept any row
    assert pred({"anything": 123}) is True


def test_between_input_formats():
    # list format
    f1 = Filter.from_dict({"date": {"between": ["2025-01-01", "2025-01-31"]}})
    pred1 = f1.to_predicate()
    assert pred1({"date": "2025-01-15"})
    assert not pred1({"date": "2024-12-31"})
    # dict format
    f2 = Filter.from_dict({"score": {"between": {"from": 0.2, "to": 0.3}}})
    pred2 = f2.to_predicate()
    assert pred2({"score": 0.25})
    assert not pred2({"score": 0.35})


def test_or_group_semantics():
    # (region=US OR region=EU) AND sector=Tech
    raw = {
        "$and": [
            {"$or": [{"region": "US"}, {"region": "EU"}]},
            {"sector": "Tech"}
        ]
    }
    f = Filter.from_dict(raw)
    pred = f.to_predicate()
    assert pred({"region": "US", "sector": "Tech"}) is True
    assert pred({"region": "EU", "sector": "Tech"}) is True
    assert pred({"region": "ASIA", "sector": "Tech"}) is False
    assert pred({"region": "US", "sector": "Health"}) is False