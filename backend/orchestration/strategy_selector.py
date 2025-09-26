# orchestrator/strategy_selector.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterable

import numpy as np
import pandas as pd


# ======================================================================================
# Data models
# ======================================================================================

@dataclass
class RiskCaps:
    max_strategies_per_asset: int = 5      # diversification across algos
    max_weight_per_strategy: float = 0.4   # cap as fraction of asset notional budget
    min_score_threshold: float = 0.0       # drop negative/low-quality scores
    total_weight_per_asset: float = 1.0    # sum of selected strategy weights = this


@dataclass
class SelectorConfig:
    mode: str = "ensemble"                 # "rules" | "ensemble" | "bandit"
    # ensemble scoring weights (combine multiple columns into a meta-score)
    score_columns: List[str] = field(default_factory=lambda: ["alpha_score"])
    score_weights: List[float] = field(default_factory=lambda: [1.0])
    zclip: float = 4.0
    # bandit settings
    bandit_kind: str = "thompson"          # "thompson" | "ucb"
    bandit_prior_alpha: float = 3.0        # Beta prior for success
    bandit_prior_beta: float  = 3.0        # Beta prior for failure
    bandit_explore_bonus: float = 1.5      # UCB bonus multiplier
    decay: float = 0.995                   # optional reward decay per day
    # persistence
    state_path: Optional[Path] = None      # where to store bandit state (json)


@dataclass
class SelectionResult:
    # allocation table: index=(asset), columns=[strategy_id, weight, score, reason, meta]
    allocations: pd.DataFrame
    # diagnostics per asset (top K candidates with scores/details)
    leaderboard: Dict[str, pd.DataFrame]
    # raw selector state (e.g., bandit posteriors)
    state: Dict[str, Any]


# ======================================================================================
# Core selector
# ======================================================================================

class StrategySelector:
    """
    Inputs:
      - candidates: DataFrame with rows per (asset, strategy_id)
          required columns: ['asset','strategy_id']
          optional: 'alpha_score', 'carry', 'momentum', 'sharpe', 'drawdown', 'latency_ms', 'cost_bps', ...
      - features: dict or DataFrame with macro/news/regime flags (optional)
    Output:
      - SelectionResult with allocations & diagnostics

    Modes:
      - "rules": drop/keep via simple thresholds (e.g., cost <= X, latency <= Y, dd <= Z),
                 then rank by 'alpha_score'.
      - "ensemble": z-score selected columns and linearly combine with weights to a meta-score.
      - "bandit": per (asset, strategy_id) posterior; pick the arms with highest samples or UCB.
                  Call `update_rewards()` after PnL realized to learn over time.
    """

    def __init__(self, cfg: SelectorConfig = SelectorConfig(), caps: RiskCaps = RiskCaps()):
        self.cfg = cfg
        self.caps = caps
        self._state: Dict[str, Any] = {"bandit": {}}  # per-asset dicts

        # load prior state if provided
        if cfg.state_path and Path(cfg.state_path).exists():
            try:
                self._state = json.loads(Path(cfg.state_path).read_text())
            except Exception:
                self._state = {"bandit": {}}

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------

    def select(
        self,
        candidates: pd.DataFrame,
        *,
        features: Optional[pd.DataFrame | Dict[str, Any]] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> SelectionResult:
        self._validate_candidates(candidates)
        df = candidates.copy()

        # 1) Rule filter (light, always useful)
        df = self._apply_rules(df, features or {}) # type: ignore
        if df.empty:
            return SelectionResult(allocations=_empty_alloc(), leaderboard={}, state=self._state)

        # 2) Scoring
        if self.cfg.mode == "rules":
            scored = self._score_rules(df)
        elif self.cfg.mode == "ensemble":
            scored = self._score_ensemble(df)
        elif self.cfg.mode == "bandit":
            scored = self._score_bandit(df)
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")

        # 3) Per-asset selection & weight assignment
        allocations, boards = self._allocate(scored)

        # 4) Persist state if needed
        self._persist_state()

        return SelectionResult(allocations=allocations, leaderboard=boards, state=self._state)

    def update_rewards(
        self,
        realized: pd.DataFrame,
        *,
        reward_col: str = "pnl$",
        asset_col: str = "asset",
        strat_col: str = "strategy_id",
        timestamp_col: Optional[str] = None,
    ) -> None:
        """
        realized: rows per (asset, strategy_id) with realized reward (pnl, win=1/0, sharpe, etc.)
        For Thompson (Beta), convert reward to success prob in [0,1].
        """
        if self.cfg.mode != "bandit":
            return

        if realized is None or realized.empty:
            return

        # Normalize to [0,1] with logistic; you can customize mapping
        r = realized.copy()
        if reward_col not in r.columns:
            raise ValueError(f"realized must include '{reward_col}'")

        # decay old counts
        self._bandit_decay_all()

        # update per (asset, strategy)
        for _, row in r.iterrows():
            asset = str(row[asset_col])
            sid = str(row[strat_col])
            rew = float(row[reward_col])
            p = _sigmoid(rew)  # success prob proxy
            arm = self._bandit_arm(asset, sid)
            arm["alpha"] += p
            arm["beta"]  += (1.0 - p)
            arm["last_ts"] = int(time.time())

    # ----------------------------------------------------------------------------------
    # Internals — scoring
    # ----------------------------------------------------------------------------------

    def _apply_rules(self, df: pd.DataFrame, feats: Dict[str, Any]) -> pd.DataFrame:
        out = df.copy()

        # Drop bad/slow/costly algos if those hints exist
        if "cost_bps" in out.columns:
            out = out[out["cost_bps"] <= out["cost_bps"].quantile(0.9)]
        if "latency_ms" in out.columns:
            out = out[out["latency_ms"] <= out["latency_ms"].quantile(0.9)]
        if "drawdown" in out.columns:
            out = out[out["drawdown"] <= out["drawdown"].quantile(0.9)]

        # Optional: regime gating via features (example: if "risk_off" drop pro-cyc algos)
        if isinstance(feats, dict) and feats.get("risk_off", False) and "style" in out.columns:
            out = out[~out["style"].str.contains("procyc", case=False, na=False)]

        return out

    def _score_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df.copy()
        # fall back to alpha_score; if missing, create from momentum/carry
        if "alpha_score" not in s.columns:
            s["alpha_score"] = 0.0
            if "momentum" in s.columns: s["alpha_score"] += s["momentum"]
            if "carry"    in s.columns: s["alpha_score"] += 0.5 * s["carry"]
        s["meta_score"] = s["alpha_score"]
        return s

    def _score_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df.copy()
        cols = list(self.cfg.score_columns)
        wts  = np.array(self.cfg.score_weights, dtype=float)
        if len(cols) != len(wts):
            raise ValueError("score_columns and score_weights length mismatch")
        # z-score each column within asset, clip, then weighted sum
        parts = []
        for c in cols:
            if c not in s.columns:
                s[c] = 0.0
            z = _group_zscore(s, by="asset", col=c, clip=self.cfg.zclip)
            parts.append(z.values.reshape(-1, 1)) # type: ignore
        Z = np.hstack(parts) if parts else np.zeros((len(s), 1))
        meta = (Z * wts.reshape(1, -1)).sum(axis=1)
        s["meta_score"] = meta
        return s

    def _score_bandit(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df.copy()
        # For each row, draw a sample from arm posterior (Thompson) or compute UCB
        samples = []
        for _, r in s.iterrows():
            asset = str(r["asset"]); sid = str(r["strategy_id"])
            arm = self._bandit_arm(asset, sid)
            if self.cfg.bandit_kind == "thompson":
                a, b = max(arm["alpha"], 1e-3), max(arm["beta"], 1e-3)
                draw = np.random.beta(a, b)
                samples.append(draw)
            else:  # UCB (approx with Beta mean + bonus)
                a, b = arm["alpha"], arm["beta"]
                n = a + b
                mean = a / max(n, 1e-9)
                bonus = self.cfg.bandit_explore_bonus * math.sqrt(math.log(max(arm["t"], 2)) / max(n, 1e-9))
                samples.append(mean + bonus)
            arm["t"] += 1
        s["meta_score"] = np.array(samples, dtype=float)
        return s

    # ----------------------------------------------------------------------------------
    # Internals — allocation
    # ----------------------------------------------------------------------------------

    def _allocate(self, s: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        # drop below threshold
        s = s[s["meta_score"] >= float(self.caps.min_score_threshold)].copy()
        if s.empty:
            return _empty_alloc(), {}

        boards: Dict[str, pd.DataFrame] = {}
        rows: List[Dict[str, Any]] = []

        for asset, grp in s.groupby("asset"):
            g = grp.sort_values("meta_score", ascending=False).reset_index(drop=True)
            boards[asset] = g.head(max(10, self.caps.max_strategies_per_asset)).copy() # type: ignore

            top = g.head(self.caps.max_strategies_per_asset).copy()
            if top.empty:
                continue
            # Convert scores to positive weights (softmax over positive scores)
            sc = np.maximum(0.0, top["meta_score"].values) # type: ignore
            if sc.sum() == 0:
                # equal weights if all zeros
                w = np.ones(len(top)) / len(top)
            else:
                w = sc / sc.sum()

            # cap per-strategy and renormalize to target total weight
            w = np.minimum(w, self.caps.max_weight_per_strategy)
            if w.sum() == 0:
                continue
            w = w / w.sum() * self.caps.total_weight_per_asset

            for i, row in top.iterrows():
                rows.append({
                    "asset": asset,
                    "strategy_id": row["strategy_id"],
                    "weight": float(w[top.index.get_loc(i)]),
                    "score": float(row["meta_score"]),
                    "reason": self.cfg.mode,
                    "meta": _row_meta(row),
                })

        alloc = pd.DataFrame(rows, columns=["asset","strategy_id","weight","score","reason","meta"])
        return alloc, boards

    # ----------------------------------------------------------------------------------
    # Bandit state helpers
    # ----------------------------------------------------------------------------------

    def _bandit_arm(self, asset: str, sid: str) -> Dict[str, float]:
        broot: Dict[str, Any] = self._state.setdefault("bandit", {})
        aset: Dict[str, Any] = broot.setdefault(asset, {})
        arm: Dict[str, float] = aset.get(sid) # type: ignore
        if arm is None:
            arm = {
                "alpha": float(self.cfg.bandit_prior_alpha),
                "beta":  float(self.cfg.bandit_prior_beta),
                "t": 1.0,             # time index for UCB bonus
                "last_ts": int(time.time())
            }
            aset[sid] = arm
        return arm

    def _bandit_decay_all(self):
        if self.cfg.decay >= 0.9999:
            return
        band = self._state.get("bandit", {})
        for aset in band.values():
            for sid, arm in aset.items():
                arm["alpha"] *= float(self.cfg.decay)
                arm["beta"]  *= float(self.cfg.decay)

    def _persist_state(self):
        if not self.cfg.state_path:
            return
        try:
            p = Path(self.cfg.state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._state))
        except Exception:
            pass

    # ----------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------

    @staticmethod
    def _validate_candidates(df: pd.DataFrame) -> None:
        req = {"asset", "strategy_id"}
        if not req.issubset(df.columns):
            raise ValueError(f"candidates must contain columns {req}")


# ======================================================================================
# Utilities
# ======================================================================================

def _group_zscore(df: pd.DataFrame, by: str, col: str, clip: float) -> pd.Series:
    def z(s: pd.Series) -> pd.Series:
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if sd <= 1e-12:
            out = (s - mu)
        else:
            out = (s - mu) / sd
        return out.clip(-clip, clip)
    return df.groupby(by, observed=True)[col].transform(z)

def _sigmoid(x: float) -> float:
    # squashes real-valued reward to (0,1) for Beta updates
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _row_meta(row: pd.Series) -> Dict[str, Any]:
    # keep a few helpful diagnostics if present
    keep = []
    for k in ("alpha_score","carry","momentum","sharpe","cost_bps","latency_ms","drawdown","style"):
        if k in row.index:
            keep.append((k, row[k]))
    return {k: (None if (isinstance(v, float) and (math.isnan(v))) else v) for k, v in keep}

def _empty_alloc() -> pd.DataFrame:
    return pd.DataFrame(columns=["asset","strategy_id","weight","score","reason","meta"])


# ======================================================================================
# Example usage (synthetic)
# ======================================================================================

if __name__ == "__main__":
    np.random.seed(7)

    # Build a candidate table: each row is (asset, strategy)
    assets = ["AAPL", "MSFT", "NVDA"]
    strategies = [f"strat_{i:03d}" for i in range(20)]
    rows = []
    for a in assets:
        for s in strategies:
            rows.append({
                "asset": a,
                "strategy_id": s,
                "alpha_score": np.random.normal(0, 1),
                "carry": np.random.normal(0, 0.5),
                "momentum": np.random.normal(0, 1.2),
                "sharpe": np.random.normal(0.5, 0.4),
                "cost_bps": np.random.uniform(5, 35),
                "latency_ms": np.random.uniform(2, 25),
                "drawdown": np.random.uniform(0.05, 0.3),
                "style": np.random.choice(["procyc", "defens", "neutral"]),
            })
    cand = pd.DataFrame(rows)

    # 1) Ensemble selector
    cfg = SelectorConfig(
        mode="ensemble",
        score_columns=["alpha_score", "momentum", "sharpe", "-cost_bps"],  # prefix '-' means inverse
        score_weights=[0.6, 0.3, 0.3, 0.2],
        state_path=Path("runtime/state/selector_bandit.json"),
    )
    # Allow "negative" columns by inverting them
    # (We’ll preprocess: create a flipped column if name starts with '-')
    # Quick preproc:
    to_flip = [c for c in cfg.score_columns if c.startswith("-")]
    for c in to_flip:
        base = c[1:]
        if base in cand.columns:
            cand[c] = -cand[base]
        else:
            cand[c] = 0.0

    sel = StrategySelector(cfg=cfg, caps=RiskCaps(max_strategies_per_asset=4, max_weight_per_strategy=0.5))
    res = sel.select(cand, features={"risk_off": False})
    print("ALLOCATIONS\n", res.allocations.head(15))
    print("\nLEADERBOARD (AAPL)\n", res.leaderboard["AAPL"].head(8))

    # 2) Bandit selector example (learning)
    cfg2 = SelectorConfig(mode="bandit", state_path=Path("runtime/state/selector_bandit.json"))
    sel2 = StrategySelector(cfg=cfg2, caps=RiskCaps(max_strategies_per_asset=3))
    res2 = sel2.select(cand)
    print("\nBANDIT PICK\n", res2.allocations.head(9))

    # Simulate realized PnL and update
    realized = res2.allocations.copy()
    realized["pnl$"] = np.random.normal(0, 100, len(realized))  # plug your real PnL here
    sel2.update_rewards(realized)
    # re-select (post-update)
    res3 = sel2.select(cand)
    print("\nPOST-UPDATE BANDIT PICK\n", res3.allocations.head(9))