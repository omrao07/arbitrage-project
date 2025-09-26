#!/usr/bin/env python3
"""
contagion_network.py — Interbank contagion simulator (default cascades, Eisenberg–Noe clearing, and DebtRank).

INPUTS
------
You can either:
1) Load a real network:
   - --exposures exposures.csv
       Square matrix with row/col labels = bank IDs.
       By default we assume ASSET orientation: exposures[i,j] = loan from i to j
       (i.e., asset of lender i on borrower j). Override with --matrix-type liabilities
       if your matrix is L[i,j] = liability of i to j (then assets = L.T).
   - --institutions banks.csv (optional)
       Columns (any casing): bank,id,name  (identifier)
                              equity|capital (float)
                              external_assets|ext_assets|EA (float, optional)
       If equity not provided, it will be derived from (ext_assets + interbank_assets - interbank_liabs)
       If ext_assets missing, assumed 0.

2) Generate a synthetic network (if --exposures not given):
   - --gen-n 50 --gen-p 0.06 --gen-total-assets 1e9 --seed 42
   Creates an Erdős–Rényi directed graph with random positive exposures, random equity ratios.

MODES
-----
--mode cascade        : Threshold default cascade with loss given default (LGD=1-recovery).
--mode en             : Eisenberg–Noe clearing payments with external assets and liabilities.
--mode debrank        : DebtRank systemic impact measure.

SHOCKS
------
Exactly one of:
  - --shock-bank AAPL,MS,jpm      (comma list of bank IDs present in exposures) with
    --shock-frac 0.3              (reduce their external assets by 30%)
  - --macro-frac 0.15             (reduce everyone’s external assets by 15%)

PARAMETERS
----------
--recovery 0.4         : Recovery rate on interbank assets when a borrower defaults (cascade mode).
--max-iter 1000        : Max iterations for clearing / propagation.
--tol 1e-10            : Convergence tolerance for Eisenberg–Noe.
--outdir out           : Directory to save CSV/JSON outputs.

OUTPUTS (written to --outdir)
-----------------------------
- nodes.csv             : Bank-level summary (equity0, equityT, default flag, degree, etc.)
- edges.csv             : Long-form edge list of exposures (lender -> borrower, amount)
- rounds.csv            : Per-round newly defaulted banks (cascade) OR EN convergence trace OR DebtRank states
- summary.json          : Topline metrics (defaults, equity loss, unpaid liabilities, DebtRank, etc.)

USAGE EXAMPLES
--------------
# Cascade with 40% recovery and idiosyncratic shock to 2 banks
python contagion_network.py --exposures exposures.csv --institutions banks.csv \
  --mode cascade --recovery 0.4 --shock-bank B1,B7 --shock-frac 0.5

# Eisenberg–Noe systemic clearing under macro shock 10%
python contagion_network.py --exposures liabilities.csv --matrix-type liabilities \
  --mode en --macro-frac 0.10

# DebtRank for a synthetic network of 100 banks, shock top-3 by asset
python contagion_network.py --gen-n 100 --gen-p 0.05 --mode debrank --seed 7 --shock-bank B0,B1,B2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------
# I/O and preprocessing
# ---------------------------
def load_exposures(path: str, matrix_type: str = "assets") -> pd.DataFrame:
    """Load square exposures matrix. Rows/cols are bank IDs. Returns ASSET-oriented matrix A[i,j]=asset of i on j."""
    A = pd.read_csv(path, index_col=0)
    if set(A.index) != set(A.columns):
        raise ValueError("Exposures CSV must be a square matrix with identical row/column labels.")
    A = A.astype(float).reindex(index=sorted(A.index), columns=sorted(A.columns))
    if matrix_type.lower().startswith("liab"):
        A = A.T  # convert liabilities L (i owes j) -> assets A (j has asset on i)
    # Zero out diagonals
    np.fill_diagonal(A.values, 0.0)
    return A


def load_banks(path: Optional[str], A: pd.DataFrame) -> pd.DataFrame:
    """
    Load banks info; derive equity if missing. Returns DataFrame indexed by bank id with columns:
    equity0, ext_assets, interbank_assets, interbank_liabs, total_assets0
    """
    idx = A.index
    if path:
        df = pd.read_csv(path)
        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        # Identify id/name column
        id_col = None
        for k in ("bank", "id", "name"):
            if k in cols:
                id_col = cols[k]
                break
        if id_col is None:
            raise ValueError("banks.csv must have a 'bank' or 'id' or 'name' column.")
        df = df.set_index(id_col)
        df.index = df.index.astype(str)
        df = df.reindex(idx)
        if df.isna().any().any():
            missing = df.index[df.isna().any(axis=1)].tolist()
            raise ValueError(f"banks.csv missing rows for banks: {missing}")

        # Extract numeric fields
        def pick(*cands):
            for c in cands:
                if c in {x.lower(): x for x in df.columns}:
                    return {x.lower(): x for x in df.columns}[c]
            return None

        eq_c = pick("equity", "capital")
        ea_c = pick("external_assets", "ext_assets", "ea")
        df_num = pd.DataFrame(index=df.index)
        df_num["ext_assets"] = df[ea_c] if ea_c else 0.0
        if eq_c:
            df_num["equity0"] = df[eq_c].astype(float)
        else:
            df_num["equity0"] = np.nan
    else:
        # If not provided, synthesize zero external assets; equity to be derived from balance sheet consistency later.
        df_num = pd.DataFrame(index=idx, data={"ext_assets": 0.0, "equity0": np.nan})

    # Interbank assets/liabs from A
    ib_assets = A.sum(axis=1)           # loans lent out
    ib_liabs = A.sum(axis=0)            # funding borrowed
    df_num["interbank_assets"] = ib_assets
    df_num["interbank_liabs"] = ib_liabs

    # Derive equity if missing: E = EA + IB_A - IB_L - other_liabs (assume other_liabs=0 if unknown)
    # We assume total assets = EA + IB_A; total liabs = IB_L + Equity
    # => Equity = total assets - IB_L - other_liabs (here 0)
    df_num["equity0"] = df_num["equity0"].astype(float)
    mask = df_num["equity0"].isna()
    df_num.loc[mask, "equity0"] = df_num.loc[mask, "ext_assets"] + df_num.loc[mask, "interbank_assets"] - df_num.loc[mask, "interbank_liabs"]
    df_num["total_assets0"] = df_num["ext_assets"] + df_num["interbank_assets"]
    # Ensure nonnegative equity baseline
    return df_num


def gen_random_network(n: int, p: float, total_assets: float, seed: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic asset-oriented exposures matrix A[i,j] with random weights on existing edges,
    plus derived banks DF with ext_assets and equity ratios.
    """
    rng = np.random.default_rng(seed)
    labels = [f"B{i}" for i in range(n)]

    # Directed Erdős–Rényi edges
    edges = rng.random((n, n)) < p
    np.fill_diagonal(edges, 0)

    # Random positive weights for edges, scaled so each lender’s interbank assets ~ U(0.2, 0.6) * total_assets/n
    raw = rng.gamma(shape=2.0, scale=1.0, size=(n, n)) * edges
    row_sum = raw.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    norm = raw / row_sum

    target_ib_assets = rng.uniform(0.2, 0.6, size=n) * (total_assets / n)
    A = norm * target_ib_assets[:, None]
    A = pd.DataFrame(A, index=labels, columns=labels)

    # External assets so that total assets per bank ~ total_assets/n
    ext_assets = (total_assets / n) - target_ib_assets
    ext_assets = np.maximum(ext_assets, 0.1 * (total_assets / n) * rng.random(n))

    # Equity ratio ~ U(5%, 12%)
    eq_ratio = rng.uniform(0.05, 0.12, size=n)
    equity0 = eq_ratio * (ext_assets + target_ib_assets)

    banks = pd.DataFrame(
        {
            "ext_assets": ext_assets,
            "interbank_assets": target_ib_assets,
            "interbank_liabs": A.sum(axis=0).values,
            "equity0": equity0,
        },
        index=labels,
    )
    banks["total_assets0"] = banks["ext_assets"] + banks["interbank_assets"]
    return A, banks


def save_edges(A: pd.DataFrame, out: Path) -> None:
    edges = A.stack().reset_index()
    edges.columns = ["lender", "borrower", "exposure"]
    edges = edges[edges["exposure"] > 0]
    edges.to_csv(out, index=False)


# ---------------------------
# Shocks
# ---------------------------
def apply_shock(banks: pd.DataFrame, shock_banks: List[str], shock_frac: float, macro_frac: float) -> pd.Series:
    ea = banks["ext_assets"].copy()
    if macro_frac > 0:
        ea *= (1 - macro_frac)
    if shock_banks:
        sel = [b for b in shock_banks if b in ea.index]
        ea.loc[sel] = ea.loc[sel] * (1 - shock_frac)
    return ea


# ---------------------------
# Cascade model (LGD-based)
# ---------------------------
def default_cascade(
    A: pd.DataFrame,
    banks: pd.DataFrame,
    ext_assets_after: pd.Series,
    recovery: float = 0.4,
    max_rounds: int = 10_000,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    A[i,j] = asset of i on j. If j defaults, i loses (1-recovery)*A[i,j].
    Start with equity0 and shocked external assets. Iterate loss propagation until fixed point.
    Returns (nodes DF with equity_t and default flag, rounds list with newly defaulted per round).
    """
    idx = A.index
    E0 = banks["equity0"].copy().astype(float)
    EA0 = banks["ext_assets"].astype(float)
    IB_A = banks["interbank_assets"].astype(float)
    # Initial equity after external shock (before counterparty losses)
    E = E0 + (ext_assets_after - EA0)

    defaulted = pd.Series(False, index=idx)
    rounds: List[List[str]] = []
    for r in range(max_rounds):
        # Loss to each lender from borrowers that newly default this round
        newly_default = (~defaulted) & (E < 0)
        if not newly_default.any():
            break
        D = newly_default[newly_default].index.tolist()
        rounds.append(D)

        # Distribute write-downs to lenders of these defaulters
        # Loss_i = sum_j (LGD * A[i,j] * 1_{j defaulted now} * remaining fraction not already written)
        LGD = 1 - recovery
        loss_vec = (A.loc[:, D].sum(axis=1)) * LGD
        # Reduce assets (already priced-in via equity update): equity absorbs losses
        E = E - loss_vec

        defaulted.loc[D] = True

    nodes = pd.DataFrame(
        {
            "equity0": E0,
            "equityT": E,
            "defaulted": defaulted.astype(int),
            "ext_assets0": EA0,
            "ext_assetsT": ext_assets_after,
            "interbank_assets0": IB_A,
            "interbank_liabs0": A.sum(axis=0).values,
        },
        index=idx,
    )
    return nodes, rounds


# ---------------------------
# Eisenberg–Noe clearing
# ---------------------------
def eisenberg_noe(
    A_assets: pd.DataFrame,
    ext_assets_after: pd.Series,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Convert assets matrix A to liabilities L = A.T. Let \bar p = total liabilities per bank (sum of row in L).
    Let Π be relative liabilities matrix (rows sum to 1 where \bar p_i>0).
    Fixed point: p = min(\bar p, a + Π^T p)
    where a = external assets.
    Returns: clearing payments p, unpaid u=(\bar p - p), and equity (post-clearing): a + Π^T p - \bar p.
    """
    L = A_assets.T.copy()
    banks = L.index
    pbar = L.sum(axis=1).astype(float)
    # Relative liabilities Π (row-stochastic on liabilities side)
    Pi = pd.DataFrame(0.0, index=banks, columns=banks)
    mask = pbar > 0
    Pi.loc[mask, :] = L.loc[mask, :].div(pbar[mask], axis=0).fillna(0.0)

    # iterate
    p = pbar.copy()
    a = ext_assets_after[banks].astype(float).copy()
    for _ in range(max_iter):
        p_next = np.minimum(pbar.values, (a.values + (Pi.T @ p.values)))#type:ignore
        if np.max(np.abs(p_next - p.values)) < tol:
            p = pd.Series(p_next, index=banks)
            break
        p = pd.Series(p_next, index=banks)
    else:
        # did not converge within max_iter
        pass

    unpaid = (pbar - p).clip(lower=0.0)
    equity = a + (Pi.T @ p.values) - pbar.values#type:ignore
    equity = pd.Series(equity, index=banks)#type:ignore
    return p, unpaid, equity


# ---------------------------
# DebtRank
# ---------------------------
def debt_rank(
    A: pd.DataFrame,
    equity0: pd.Series,
    shocked: List[str],
    shock_level: float = 1.0,  # 1.0 means initial banks are fully distressed (h=1)
    max_iter: int = 1_000,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Battiston et al. DebtRank.
    A[i,j] = asset of i on j. Impact matrix W_{i->j} = min(1, A[i,j] / equity0[j]).
    h_i in [0,1] distress; s_i in {U(unaffected), D(distressed), I(inactive)}.
    Start with h=shock_level on shocked nodes (state D), else 0 (U).
    Repeat: Δh_j(t) = sum_i W_{i->j} * Δh_i(t-1) for nodes with s_i=D. Then set s_i=I after spreading once.
    Stops when no new distress.
    Returns (DebtRank value per node R_i = h_i(T) * v_i, and a time-series DataFrame of h).
    """
    banks = A.index
    # Economic value v_i proportional to total interbank assets
    v = A.sum(axis=1)
    total_v = v.sum()
    v = v / (total_v if total_v > 0 else 1.0)

    # Impact matrix on borrowers j from lenders i (distress propagates via borrowers’ equity shortfall)
    W = A.div(equity0.replace(0, np.nan), axis=1).clip(upper=1.0).fillna(0.0)

    h = pd.Series(0.0, index=banks)
    s = pd.Series("U", index=banks)
    h.loc[shocked] = shock_level
    s.loc[shocked] = "D"

    H_hist = [h.copy().rename(0)]
    for t in range(1, max_iter + 1):
        dh_prev = (H_hist[-1] - (H_hist[-2] if len(H_hist) > 1 else 0.0))
        # Only nodes that were D at t-1 can transmit
        active = (s == "D").astype(float)
        # Δh_j = sum_i W_{i->j} * Δh_i(t-1) * 1_{i active}
        delta = (W.T @ (dh_prev.values * active.values))#type:ignore
        h_new = (h + delta).clip(upper=1.0)#type:ignore
        H_hist.append(pd.Series(h_new, index=banks).rename(t))#type:ignore

        # Update states: those D become I; any node with h increased becomes D (if was U)
        grew = (h_new > h + 1e-15)
        s[s == "D"] = "I"
        s[(s == "U") & grew] = "D"
        h = pd.Series(h_new, index=banks)#type:ignore
        if not grew.any():  # convergence#type:ignore
            break

    H = pd.concat(H_hist, axis=1)
    R = (H.iloc[:, -1] * v)  # contribution-weighted distress
    return R, H


# ---------------------------
# Orchestration / CLI
# ---------------------------
@dataclass
class Summary:
    mode: str
    n_banks: int
    total_exposure: float
    defaults: int = 0
    total_equity0: float = 0.0
    total_equityT: float = 0.0
    unpaid_fraction: float = 0.0
    debtrank_total: float = 0.0


def run(
    mode: str,
    exposures: Optional[str],
    matrix_type: str,
    institutions: Optional[str],
    gen_n: int,
    gen_p: float,
    gen_total_assets: float,
    seed: Optional[int],
    shock_bank: List[str],
    shock_frac: float,
    macro_frac: float,
    recovery: float,
    tol: float,
    max_iter: int,
    outdir: Path,
) -> None:
    # Load or generate network
    if exposures:
        A = load_exposures(exposures, matrix_type=matrix_type)
        banks = load_banks(institutions, A)
    else:
        A, banks = gen_random_network(gen_n, gen_p, gen_total_assets, seed)

    banks.index = A.index  # ensure ordering
    # Apply shocks to external assets
    shocked_ea = apply_shock(banks, shock_bank, shock_frac, macro_frac)

    outdir.mkdir(parents=True, exist_ok=True)
    save_edges(A, outdir / "edges.csv")

    if mode == "cascade":
        nodes, rounds = default_cascade(A, banks, shocked_ea, recovery=recovery, max_rounds=max_iter)
        nodes.to_csv(outdir / "nodes.csv")
        # Save rounds
        flat = []
        for r, lst in enumerate(rounds, start=1):
            for b in lst:
                flat.append({"round": r, "bank": b})
        pd.DataFrame(flat).to_csv(outdir / "rounds.csv", index=False)

        summ = Summary(
            mode=mode,
            n_banks=len(A),
            total_exposure=float(A.values.sum()),
            defaults=int(nodes["defaulted"].sum()),
            total_equity0=float(nodes["equity0"].sum()),
            total_equityT=float(nodes["equityT"].sum()),
        )
        out = asdict(summ)
        (outdir / "summary.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

    elif mode == "en":
        p, unpaid, equity = eisenberg_noe(A, shocked_ea, tol=tol, max_iter=max_iter)
        # package nodes
        nodes = pd.DataFrame(
            {
                "pbar": A.T.sum(axis=1),
                "p_clearing": p,
                "unpaid": unpaid,
                "equityT": equity,
            }
        )
        # For reference, reconstruct equity0 consistent with banks:
        nodes["equity0"] = banks["equity0"]
        nodes["defaulted"] = (nodes["unpaid"] > 1e-12).astype(int)
        nodes.to_csv(outdir / "nodes.csv")

        # Convergence trace not tracked here; write placeholder rounds with unpaid >0
        df_rounds = nodes[nodes["unpaid"] > 0][["unpaid"]].reset_index().rename(columns={"index": "bank"})
        df_rounds["round"] = 1
        df_rounds[["round", "bank", "unpaid"]].to_csv(outdir / "rounds.csv", index=False)

        unpaid_fraction = float(unpaid.sum() / max(nodes["pbar"].sum(), 1e-12))
        summ = Summary(
            mode=mode,
            n_banks=len(A),
            total_exposure=float(A.values.sum()),
            defaults=int(nodes["defaulted"].sum()),
            total_equity0=float(nodes["equity0"].sum()),
            total_equityT=float(nodes["equityT"].sum()),
            unpaid_fraction=unpaid_fraction,
        )
        out = asdict(summ)
        (outdir / "summary.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

    elif mode == "debrank":
        # Determine initial shocked set:
        S = shock_bank if shock_bank else []  # empty -> no initial shock (you can pass macro via ext assets, but DR needs set)
        if not S:
            # pick the largest by interbank assets as a sensible default
            S = [A.sum(axis=1).idxmax()]
        R, H = debt_rank(A, banks["equity0"], S, shock_level=1.0, max_iter=max_iter)#type:ignore
        # Save nodes + DR contributions
        nodes = banks.copy()
        nodes["DebtRank_v"] = (A.sum(axis=1) / max(A.values.sum(), 1e-12))
        nodes["DebtRank_R"] = R
        nodes["DebtRank_contrib"] = nodes["DebtRank_v"] * R
        nodes.to_csv(outdir / "nodes.csv")

        H.to_csv(outdir / "rounds.csv")  # time series of distress h

        summ = Summary(
            mode=mode,
            n_banks=len(A),
            total_exposure=float(A.values.sum()),
            defaults=0,
            total_equity0=float(banks["equity0"].sum()),
            total_equityT=float(banks["equity0"].sum()),
            debtrank_total=float(nodes["DebtRank_contrib"].sum()),
        )
        out = asdict(summ)
        (outdir / "summary.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

    else:
        raise ValueError("Unknown --mode. Choose from: cascade, en, debrank")


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interbank contagion simulator (cascade, Eisenberg–Noe, DebtRank)")
    # Inputs
    p.add_argument("--exposures", default="", help="CSV square matrix (rows/cols are banks).")
    p.add_argument("--matrix-type", default="assets", choices=["assets", "liabilities"], help="Orientation of exposures CSV.")
    p.add_argument("--institutions", default="", help="Optional banks.csv with equity/ext_assets info.")
    # Synthetic generation
    p.add_argument("--gen-n", type=int, default=50, help="Synthetic: number of banks (if exposures missing).")
    p.add_argument("--gen-p", type=float, default=0.06, help="Synthetic: edge probability for ER directed graph.")
    p.add_argument("--gen-total-assets", type=float, default=1e9, help="Synthetic: target total assets across system.")
    p.add_argument("--seed", type=int, default=None)
    # Mode
    p.add_argument("--mode", default="cascade", choices=["cascade", "en", "debrank"])
    # Shocks
    p.add_argument("--shock-bank", default="", help="Comma-separated bank IDs to shock (idiosyncratic).")
    p.add_argument("--shock-frac", type=float, default=0.3, help="Fractional reduction of external assets for shocked banks.")
    p.add_argument("--macro-frac", type=float, default=0.0, help="Macro shock: fractional reduction of all external assets.")
    # Params
    p.add_argument("--recovery", type=float, default=0.4, help="Recovery on interbank assets when borrower defaults (cascade).")
    p.add_argument("--tol", type=float, default=1e-10, help="Tolerance for Eisenberg–Noe fixed point.")
    p.add_argument("--max-iter", type=int, default=10000, help="Max iterations for cascade/clearing/debt rank.")
    # Output
    p.add_argument("--outdir", default="out", help="Output directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outdir = Path(args.outdir)
    shock_bank = [s.strip() for s in args.shock_bank.split(",") if s.strip()]

    run(
        mode=args.mode,
        exposures=args.exposures or None,
        matrix_type=args.matrix_type,
        institutions=args.institutions or None,
        gen_n=args.gen_n,
        gen_p=args.gen_p,
        gen_total_assets=args.gen_total_assets,
        seed=args.seed,
        shock_bank=shock_bank,
        shock_frac=args.shock_frac,
        macro_frac=args.macro_frac,
        recovery=args.recovery,
        tol=args.tol,
        max_iter=args.max_iter,
        outdir=outdir,
    )
# End of contagion_network.py