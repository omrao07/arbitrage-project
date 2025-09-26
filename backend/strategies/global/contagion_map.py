#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contagion_map.py — Network contagion simulator & systemic risk mapping

What this does
--------------
Given a directed exposure network (i -> j is exposure of i to j, i.e. asset of i; liability of j),
and node fundamentals (equity, sector, etc.), this tool:

1) Cleans & aligns nodes/exposures into an NxN matrix with id→index mapping
2) Computes network centralities (in/out strength, eigenvector, Katz, PageRank)
3) Runs contagion under three engines:
   • Furfine-style default cascade (iterative loss given default)
   • DebtRank distress propagation (Battiston et al.)
   • (Optional) Eisenberg–Noe clearing if external-assets are provided
4) Supports scenarios to shock node equities and/or rescale edges
5) Writes tidy CSVs + a compact JSON summary

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--nodes nodes.csv   REQUIRED
  Columns (any subset; best-effort):
    node_id, name, sector, country,
    equity_usd, assets_usd, liabilities_usd, capital_ratio_pct,
    external_assets_usd (for Eisenberg–Noe; optional)

--exposures exposures.csv   REQUIRED
  Directed exposures: from_id, to_id, exposure_usd  (i -> j is asset of i; liability of j)

--scenarios scenarios.csv   OPTIONAL
  Columns: scenario, key, value
  Keys (examples):
    shock.node.<ID>.equity_pct = -30          (hit to equity%)
    shock.sector.<SECTOR>.equity_pct = -15
    shock.edge.<ID1>-><ID2>.pct = -20         (scale that edge by 1 + value/100)
    scale.edges.sector_to.<SECTOR>.pct = -10  (scale all edges to nodes in sector)
    scale.edges.sector_from.<SECTOR>.pct = -10
    param.lgd = 60                            (LGD in %, default 55)
    param.alpha = 0.9                         (DebtRank damping)
    param.rounds = 20                         (max rounds for cascades)
    method = furfine | debtrank | en          (choose engine for this scenario; default = all)

CLI
---
--nodes NODES.csv --exposures EXP.csv [--scenarios SCN.csv]
--lgd 55 --alpha 0.9 --rounds 20 --pagerank_damp 0.85
--outdir out_contagion

Outputs
-------
- network_edges.csv        Cleaned directed edges (i→j) with exposure_usd (post-scenario, per scenario)
- network_nodes.csv        Node table (equity before/after shock, sector, etc.) (per scenario)
- centrality.csv           In/out strength, eigenvector, Katz, PageRank (baseline)
- cascade_path.csv         Furfine engine: per round node losses/equity/default state (per scenario)
- edge_loss_flows.csv      Furfine: per (round, defaulter→creditor) loss flows (per scenario)
- debtrank_scores.csv      DebtRank: h(t) paths & final DebtRank scores (per scenario)
- en_clearing.csv          Eisenberg–Noe payments, if external assets available (per scenario)
- summary.json             High-level metrics (defaults, top systemic nodes, etc.)
- config.json              Run configuration (for reproducibility)

DISCLAIMER
----------
Research tool only; simplifications abound. Interpret prudently.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None

def ensure_dir(p: str) -> Path:
    pp = Path(p); pp.mkdir(parents=True, exist_ok=True); return pp

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pct_to_mult(x: float) -> float:
    return 1.0 + (x / 100.0)

def df_rename(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    m2 = {}
    for k, v in mapping.items():
        col = ncol(df, k) or k
        if col in df.columns:
            m2[col] = v
    return df.rename(columns=m2)

def spectral_radius(M: np.ndarray) -> float:
    if M.size == 0:
        return 0.0
    vals = np.linalg.eigvals(M)
    return float(np.max(np.abs(vals)))

# ----------------------------- loaders -----------------------------

def load_nodes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df_rename(df, {
        "node_id": "node_id",
        "name": "name",
        "sector": "sector",
        "country": "country",
        "equity_usd": "equity",
        "assets_usd": "assets",
        "liabilities_usd": "liabilities",
        "capital_ratio_pct": "cap_ratio",
        "external_assets_usd": "external_assets",
    })
    # Minimal schema
    if "node_id" not in df.columns:
        # fall back to first column
        df = df.rename(columns={df.columns[0]: "node_id"})
    for c in ["name","sector","country"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["equity","assets","liabilities","cap_ratio","external_assets"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    # derive equity if missing
    if "equity" not in df.columns or df["equity"].isna().all():
        if "assets" in df.columns and "liabilities" in df.columns:
            df["equity"] = safe_num(df["assets"]) - safe_num(df["liabilities"])
        else:
            df["equity"] = np.nan
    # defaults
    if "name" not in df.columns: df["name"] = df["node_id"]
    if "sector" not in df.columns: df["sector"] = "UNKNOWN"
    if "country" not in df.columns: df["country"] = "UNK"
    return df[["node_id","name","sector","country","equity","assets","liabilities","external_assets"]].copy()

def load_exposures(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df_rename(df, {
        "from_id": "from_id",
        "to_id": "to_id",
        "exposure_usd": "exposure",
        "exposure": "exposure",
        "amount": "exposure",
        "value": "exposure",
    })
    # Minimal schema
    if "from_id" not in df.columns or "to_id" not in df.columns:
        # try first two columns
        cols = df.columns.tolist()
        assert len(cols) >= 3, "exposures.csv needs at least 3 columns (from,to,exposure)"
        df = df.rename(columns={cols[0]: "from_id", cols[1]: "to_id", cols[2]: "exposure"})
    df["from_id"] = df["from_id"].astype(str)
    df["to_id"] = df["to_id"].astype(str)
    df["exposure"] = safe_num(df["exposure"]).fillna(0.0).clip(lower=0.0)
    df = df[df["from_id"] != df["to_id"]]  # drop self-loops
    df = df.groupby(["from_id","to_id"], as_index=False)["exposure"].sum()
    return df

# ----------------------------- network build -----------------------------

@dataclass
class Net:
    nodes: pd.DataFrame          # index aligned to positions
    edges: pd.DataFrame          # from_id,to_id,exposure
    id2idx: Dict[str, int]
    A: np.ndarray                # exposure matrix A[i, j] = exposure i->j
    eq: np.ndarray               # equity vector
    sectors: List[str]

def build_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> Net:
    ids = nodes["node_id"].astype(str).unique().tolist()
    id2idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)
    A = np.zeros((n, n), dtype=float)
    for _, r in edges.iterrows():
        i = id2idx.get(str(r["from_id"]))
        j = id2idx.get(str(r["to_id"]))
        if i is None or j is None:  # exposure to unknown node — drop
            continue
        A[i, j] += float(r["exposure"])
    eq = nodes.set_index("node_id").reindex(ids)["equity"].fillna(0.0).values.astype(float)
    sectors = nodes.set_index("node_id").reindex(ids)["sector"].astype(str).tolist()
    return Net(nodes=nodes.copy(), edges=edges.copy(), id2idx=id2idx, A=A, eq=eq, sectors=sectors)

# ----------------------------- centralities -----------------------------

def centralities(A: np.ndarray, ids: List[str], damping: float=0.85) -> pd.DataFrame:
    n = len(ids)
    out_strength = A.sum(axis=1)   # assets to others
    in_strength  = A.sum(axis=0)   # liabilities to system
    # Eigenvector on symmetrized
    S = (A + A.T) * 0.5
    if n > 0 and np.any(S):
        vals, vecs = np.linalg.eig(S)
        k = int(np.argmax(np.real(vals)))
        eig = np.real(vecs[:, k])
        eig = np.abs(eig) / (np.abs(eig).sum() + 1e-12)
    else:
        eig = np.zeros(n)
    # Katz (I - β A^T)^-1 1
    rho = spectral_radius(A.T)
    beta = 0.9 / (rho + 1e-12) if rho > 0 else 0.1
    try:
        Katz = np.linalg.solve(np.eye(n) - beta * A.T, np.ones(n))
        Katz = Katz / (Katz.sum() + 1e-12)
    except np.linalg.LinAlgError:
        Katz = np.ones(n) / max(1, n)
    # PageRank-like on column-stochastic matrix (liability-side)
    G = A.copy()
    colsum = G.sum(axis=0)
    P = np.zeros_like(G, dtype=float)
    for j in range(n):
        if colsum[j] > 0:
            P[:, j] = G[:, j] / colsum[j]
        else:
            P[:, j] = 1.0 / n
    pr = np.ones(n) / n
    for _ in range(100):
        pr_new = damping * (P @ pr) + (1 - damping) * (np.ones(n) / n)
        if np.linalg.norm(pr_new - pr, 1) < 1e-10:
            break
        pr = pr_new
    df = pd.DataFrame({
        "node_id": ids,
        "out_strength_usd": out_strength,
        "in_strength_usd": in_strength,
        "eigen_centrality": eig,
        "katz_centrality": Katz,
        "pagerank": pr,
    })
    return df.sort_values("pagerank", ascending=False)

# ----------------------------- scenarios -----------------------------

@dataclass
class Params:
    lgd: float = 0.55
    alpha: float = 0.9
    rounds: int = 20
    method: str = "all"

def apply_scenario(net: Net, scen_rows: pd.DataFrame, params: Params) -> Tuple[np.ndarray, np.ndarray, Params]:
    A = net.A.copy()
    eq = net.eq.copy()
    id_map = net.id2idx
    # defaults
    lgd = params.lgd
    alpha = params.alpha
    rounds = params.rounds
    method = params.method

    for _, r in scen_rows.iterrows():
        key = str(r["key"]).strip()
        val = float(r["value"]) if pd.notna(r["value"]) else np.nan
        if key.lower().startswith("shock.node."):
            nid = key.split(".", 2)[2]
            if nid in id_map and pd.notna(val):
                i = id_map[nid]
                eq[i] *= pct_to_mult(val)  # e.g., -30 → *0.7
        elif key.lower().startswith("shock.sector."):
            sec = key.split(".", 2)[2].upper()
            idx = [i for i, s in enumerate(net.sectors) if s.upper()==sec]
            for i in idx:
                if pd.notna(val):
                    eq[i] *= pct_to_mult(val)
        elif key.lower().startswith("shock.edge."):
            # format shock.edge.ID1->ID2.pct
            body = key.split(".", 2)[2]
            if "->" in body:
                pair = body.split(".pct")[0]
                u, v = pair.split("->")
                if u in id_map and v in id_map and pd.notna(val):
                    A[id_map[u], id_map[v]] *= pct_to_mult(val)
        elif key.lower().startswith("scale.edges.sector_to."):
            sec = key.split(".", 2)[2].upper().replace(".pct","")
            if pd.notna(val):
                cols = [j for j, s in enumerate(net.sectors) if s.upper()==sec]
                A[:, cols] *= pct_to_mult(val)
        elif key.lower().startswith("scale.edges.sector_from."):
            sec = key.split(".", 2)[2].upper().replace(".pct","")
            if pd.notna(val):
                rows = [i for i, s in enumerate(net.sectors) if s.upper()==sec]
                A[rows, :] *= pct_to_mult(val)
        elif key.lower() == "param.lgd":
            lgd = max(0.0, min(1.0, (val/100.0 if val>1.0 else val)))
        elif key.lower() == "param.alpha":
            alpha = float(val)
        elif key.lower() == "param.rounds":
            rounds = int(val)
        elif key.lower() == "method":
            method = str(val).lower()

    return A, eq, Params(lgd=lgd, alpha=alpha, rounds=rounds, method=method)

# ----------------------------- Furfine cascade -----------------------------

def furfine_cascade(A: np.ndarray, equity: np.ndarray, lgd: float, max_rounds: int=20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    A[i,j] = exposure of i to j (asset of i). If j defaults, i loses lgd*A[i,j].
    """
    n = len(equity)
    state = np.zeros(n, dtype=int)  # 0=alive, 1=defaulted
    eq = equity.copy().astype(float)
    path_rows = []
    flow_rows = []
    # Initial defaults (negative equity)
    newly_default = np.where(eq <= 0.0)[0].tolist()
    for i in newly_default:
        state[i] = 1
        path_rows.append({"round": 0, "node_idx": i, "equity_before": float(eq[i]), "loss_in_round": 0.0,
                          "equity_after": float(eq[i]), "defaulted": 1})
    r = 1
    while r <= max_rounds and newly_default:
        # Losses to creditors from newly defaulted borrowers
        loss_to_i = np.zeros(n, dtype=float)
        for j in newly_default:
            # creditors i have asset A[i,j]
            loss_vec = lgd * A[:, j]
            loss_to_i += loss_vec
            for i in np.where(loss_vec > 0)[0]:
                flow_rows.append({"round": r, "from_defaulter_idx": int(j), "to_creditor_idx": int(i),
                                  "loss": float(loss_vec[i])})
        # apply losses to equity of alive nodes
        for i in range(n):
            if state[i] == 0 and loss_to_i[i] > 0:
                eb = float(eq[i])
                eq[i] -= loss_to_i[i]
                path_rows.append({"round": r, "node_idx": int(i), "equity_before": eb,
                                  "loss_in_round": float(loss_to_i[i]), "equity_after": float(eq[i]),
                                  "defaulted": 0})
        # new defaults this round
        newly_default = [i for i in range(n) if state[i]==0 and eq[i] <= 0.0]
        for i in newly_default:
            state[i] = 1
            path_rows.append({"round": r, "node_idx": int(i), "equity_before": float(eq[i]),
                              "loss_in_round": 0.0, "equity_after": float(eq[i]), "defaulted": 1})
        r += 1

    # Final state
    final = pd.DataFrame({
        "node_idx": np.arange(n),
        "equity_final": eq,
        "defaulted": state
    })
    path = pd.DataFrame(path_rows) if path_rows else pd.DataFrame(columns=["round","node_idx","equity_before","loss_in_round","equity_after","defaulted"])
    flows = pd.DataFrame(flow_rows) if flow_rows else pd.DataFrame(columns=["round","from_defaulter_idx","to_creditor_idx","loss"])
    return final, path, flows

# ----------------------------- DebtRank -----------------------------

def debtrank(A: np.ndarray, equity: np.ndarray, alpha: float=0.9, T: int=20, initial_equity_shock: Optional[np.ndarray]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    DebtRank with impact matrix W_ij = min(1, A_ij / E_i). Distress flows from borrower j to lender i.
    h_i in [0,1] is distress; Δh propagates once.
    """
    n = len(equity)
    E = equity.copy().astype(float)
    E[E <= 0] = 1e-6
    W = np.minimum(1.0, (A / E.reshape(-1, 1)))  # divide each row i by E_i

    # Initial shock as relative equity hit: s_i ∈ [0,1]
    if initial_equity_shock is None:
        s = np.zeros(n)
    else:
        s = np.clip(np.maximum(0.0, -initial_equity_shock) / (np.abs(E) + 1e-12), 0.0, 1.0)

    # States: U=0 (undistressed), A=1 (active), I=2 (inactive)
    S = np.zeros(n, dtype=int)
    h = s.copy()
    active = np.where(s > 0)[0].tolist()
    S[active] = 1

    path = [{"t": 0, "node_idx": int(i), "h": float(h[i]), "state": int(S[i])} for i in range(n)]

    for t in range(1, T+1):
        if not active:
            break
        delta = np.zeros(n)
        for j in active:
            # impact from j to its lenders i via W_ij (row i, col j)
            delta += alpha * W[:, j] * (1.0 - h) * (h[j] - 0.0)  # Δh_j is full h_j since we propagate once
        # update h; new active are those with delta>0 and not already inactive
        h = np.clip(h + delta, 0.0, 1.0)
        new_active = [i for i in range(n) if S[i]==0 and delta[i] > 1e-12]
        # Set previous active to inactive
        for j in active:
            S[j] = 2
        # New active
        for i in new_active:
            S[i] = 1
        active = new_active
        for i in range(n):
            path.append({"t": t, "node_idx": int(i), "h": float(h[i]), "state": int(S[i])})

    # DebtRank score R = sum_i (h_i(T) - h_i(0)) * v_i, v_i = economic value weight (use out_strength)
    v = A.sum(axis=1)
    v_sum = v.sum() if v.sum() > 0 else 1.0
    R = float(((h - s) * v).sum() / v_sum)
    scores = pd.DataFrame({"node_idx": np.arange(n), "h0": s, "hT": h, "value_weight": v/v_sum})
    scores["debt_rank"] = (scores["hT"] - scores["h0"]) * scores["value_weight"]
    path_df = pd.DataFrame(path)
    return scores, path_df

# ----------------------------- Eisenberg–Noe (optional) -----------------------------

def eisenberg_noe(A: np.ndarray, external_assets: Optional[np.ndarray], tol: float=1e-8, max_iter: int=10_000) -> Tuple[pd.DataFrame, Dict]:
    """
    Simple clearing for interbank liabilities. We treat L = A^T as liabilities matrix (j owes to i).
    Requires external_assets >= 0. If external_assets is None, returns empty result with a note.

    Notation:
      bar_p_j = total interbank liabilities of j = sum_i L_{ji} = sum_i A_{ij}
      Π_{ji} = L_{ji} / bar_p_j  (relative liabilities; if bar_p_j=0, Π row = 0)
      Clearing: p = min(bar_p, Π^T p + x), solved by fixed-point iteration on payments p.
    """
    n = A.shape[0]
    if external_assets is None:
        return pd.DataFrame(), {"note": "external_assets missing; skipping EN."}
    x = external_assets.astype(float).clip(min=0.0)
    L = A.T.copy()
    bar_p = L.sum(axis=1)  # liabilities per node j
    Pi = np.zeros_like(L)
    for j in range(n):
        if bar_p[j] > 0:
            Pi[j, :] = L[j, :] / bar_p[j]
    # Iterate
    p = bar_p.copy()
    for _ in range(max_iter):
        p_next = np.minimum(bar_p, Pi.T @ p + x)
        if np.linalg.norm(p_next - p, ord=1) < tol:
            p = p_next
            break
        p = p_next
    # Default set: those paying less than bar_p
    default = (p < bar_p - 1e-10).astype(int)
    paid_to_creditor_i = p @ Pi  # amount received by each creditor i from all j
    out = pd.DataFrame({
        "node_idx": np.arange(n),
        "bar_p_liabilities": bar_p,
        "payments": p,
        "shortfall": bar_p - p,
        "received_from_interbank": paid_to_creditor_i,
        "defaulted": default
    })
    meta = {"converged": True, "iterations": None}
    return out, meta

# ----------------------------- orchestrator -----------------------------

@dataclass
class Config:
    nodes: str
    exposures: str
    scenarios: Optional[str]
    lgd: float
    alpha: float
    rounds: int
    pagerank_damp: float
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Contagion map — cascades, DebtRank, EN clearing")
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--exposures", required=True)
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--lgd", type=float, default=55.0, help="LGD in percent (default 55)")
    ap.add_argument("--alpha", type=float, default=0.9, help="DebtRank damping (0..1)")
    ap.add_argument("--rounds", type=int, default=20, help="Max rounds for cascades")
    ap.add_argument("--pagerank_damp", type=float, default=0.85)
    ap.add_argument("--outdir", default="out_contagion")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load
    NODES = load_nodes(args.nodes)
    EDGES = load_exposures(args.exposures)
    net = build_network(NODES, EDGES)

    # Baseline centralities
    cent = centralities(net.A, list(net.id2idx.keys()), damping=args.pagerank_damp)
    # Attach labels
    cent = cent.merge(NODES[["node_id","name","sector","country"]], on="node_id", how="left")
    cent.to_csv(outdir / "centrality.csv", index=False)

    # Prepare scenarios
    scenarios = []
    if args.scenarios:
        SCN = pd.read_csv(args.scenarios)
        SCN = df_rename(SCN, {"scenario": "scenario", "key": "key", "value": "value"})
        if "scenario" not in SCN.columns: SCN["scenario"] = "SCENARIO"
        for scen, g in SCN.groupby("scenario"):
            scenarios.append((str(scen), g))
    else:
        # One baseline "BASELINE" with no changes
        scenarios.append(("BASELINE", pd.DataFrame(columns=["key","value"])))

    summary_rows = []
    # Cache reverse id map
    idx2id = {i: nid for nid, i in net.id2idx.items()}

    for scen_name, rows in scenarios:
        # Apply scenario to A and equity
        A_s, eq_s, p = apply_scenario(net, rows, Params(lgd=args.lgd/100.0, alpha=args.alpha, rounds=args.rounds, method="all"))

        # Write scenario network snapshot
        edges_out = []
        n = A_s.shape[0]
        for i in range(n):
            for j in np.where(A_s[i, :] > 0)[0]:
                edges_out.append({"scenario": scen_name, "from_id": idx2id[i], "to_id": idx2id[j], "exposure_usd": float(A_s[i, j])})
        edges_df = pd.DataFrame(edges_out)
        edges_df.to_csv(outdir / f"network_edges_{scen_name}.csv", index=False)

        nodes_df = NODES.copy()
        nodes_df["scenario"] = scen_name
        nodes_df["equity_post_shock"] = nodes_df["node_id"].map(lambda nid: eq_s[net.id2idx[nid]])
        nodes_df.to_csv(outdir / f"network_nodes_{scen_name}.csv", index=False)

        defaults_furfine = None
        R_score = None
        en_meta_note = None

        # Select engines
        engines = ["furfine","debtrank","en"] if p.method in ["all","",None] else [p.method]

        if "furfine" in engines:
            final, path, flows = furfine_cascade(A_s, eq_s, p.lgd, max_rounds=p.rounds)
            # map idx to id
            if not final.empty:
                final["node_id"] = final["node_idx"].map(idx2id)
                defaults_furfine = int(final["defaulted"].sum())
                final.to_csv(outdir / f"furfine_final_{scen_name}.csv", index=False)
            if not path.empty:
                path["node_id"] = path["node_idx"].map(idx2id)
                path["scenario"] = scen_name
                path.to_csv(outdir / f"cascade_path_{scen_name}.csv", index=False)
            if not flows.empty:
                flows["from_id"] = flows["from_defaulter_idx"].map(idx2id)
                flows["to_id"] = flows["to_creditor_idx"].map(idx2id)
                flows["scenario"] = scen_name
                flows.to_csv(outdir / f"edge_loss_flows_{scen_name}.csv", index=False)

        if "debtrank" in engines:
            # Use initial shock = equity_post - equity_pre (negative values propagate)
            init_shock = eq_s - net.eq
            scores, path = debtrank(A_s, np.maximum(eq_s, 1e-6), alpha=p.alpha, T=p.rounds, initial_equity_shock=init_shock)
            scores["node_id"] = scores["node_idx"].map(idx2id)
            scores["scenario"] = scen_name
            path["node_id"] = path["node_idx"].map(idx2id)
            path["scenario"] = scen_name
            scores.to_csv(outdir / f"debtrank_scores_{scen_name}.csv", index=False)
            path.to_csv(outdir / f"debtrank_path_{scen_name}.csv", index=False)
            R_score = float(scores["debt_rank"].sum())

        if "en" in engines:
            # Need external assets; if missing, skip gracefully
            if "external_assets" in NODES.columns and NODES["external_assets"].notna().any():
                ext = NODES.set_index("node_id").reindex(list(idx2id.values()))["external_assets"].fillna(0.0).values
                en_out, meta = eisenberg_noe(A_s, ext)
                if not en_out.empty:
                    en_out["node_id"] = en_out["node_idx"].map(idx2id)
                    en_out["scenario"] = scen_name
                    en_out.to_csv(outdir / f"en_clearing_{scen_name}.csv", index=False)
            else:
                en_meta_note = "external_assets missing; EN skipped."

        # Scenario summary
        tot_exposure = float(A_s.sum())
        tot_equity = float(eq_s[eq_s>0].sum())
        summary_rows.append({
            "scenario": scen_name,
            "engine": ",".join(engines),
            "total_system_exposure_usd": tot_exposure,
            "total_equity_usd": tot_equity,
            "furfine_defaults": defaults_furfine,
            "debtrank_total": R_score,
            "note": en_meta_note or ""
        })

    # Write summary
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary_table.csv", index=False)

    # JSON summary with top systemic nodes (baseline centrality)
    top_sys = (cent.sort_values("pagerank", ascending=False)
                   .head(10)[["node_id","name","sector","pagerank","in_strength_usd","out_strength_usd"]]
                   .to_dict(orient="records"))
    (outdir / "summary.json").write_text(json.dumps({
        "scenarios": summary_rows,
        "top_systemic_nodes_baseline": top_sys
    }, indent=2))

    # Config dump
    cfg = asdict(Config(
        nodes=args.nodes, exposures=args.exposures, scenarios=(args.scenarios or None),
        lgd=args.lgd, alpha=args.alpha, rounds=args.rounds, pagerank_damp=args.pagerank_damp, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Contagion Map ==")
    print(f"Nodes: {len(net.id2idx)} | Edges: {int((net.A>0).sum())} | Total exposure: {net.A.sum():,.0f}")
    print("Top systemic (baseline PageRank):")
    for r in top_sys[:5]:
        print(f"  {r['node_id']:<12} {r['name']:<20} PR={r['pagerank']:.4f} In={r['in_strength_usd']:,.0f} Out={r['out_strength_usd']:,.0f}")
    print("Scenarios summary written to:", (outdir / "summary_table.csv").resolve())

if __name__ == "__main__":
    main()
