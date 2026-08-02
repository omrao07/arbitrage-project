"""
FII/DII participant-flow features — the platform's highest-conviction edge.

THE HYPOTHESIS (from the plan, Part 5.1)
----------------------------------------
Institutional flows are slow. A large FII rebalance takes days, which creates
short-horizon autocorrelation in positioning. NSE publishes participant-wise
open interest daily and free, and almost no global systematic fund ingests it.
That combination — public, underexploited, market-specific — is the moat.

Conditioning next-day Nifty exposure on (a) FII net index-futures positioning
and (b) the change in their long/short ratio should beat unconditional exposure
on a risk-adjusted basis. Should. This module builds the features; the gauntlet
decides whether the hypothesis survives.

THE TRAP THIS MODULE EXISTS TO AVOID
-------------------------------------
Participant data for trade date T is published after T's close. Joining it to
T's session is lookahead bias and produces spectacular, fake alpha. Every
feature here is built from `avail_ts` (next session's pre-open), and
`build_features` asserts the alignment rather than trusting it. If you see a
Sharpe here that looks too good, suspect this join before you believe it.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FLOW_STORE = Path("backend/data/nse/participant_flow")
DEFAULT_PRICE_FILE = Path("backend/data/nse/prices/NIFTY.csv")


class FeatureError(RuntimeError):
    """Raised when features cannot be built honestly."""


@dataclass(frozen=True, slots=True)
class FlowFeatures:
    """Point-in-time FII/DII features aligned to when they were knowable."""

    dates: np.ndarray            # the session each feature may first be used in
    fii_net_idx_fut: np.ndarray  # FII net index-futures contracts
    fii_net_z: np.ndarray        # 20d z-score of the above
    fii_ls_ratio: np.ndarray     # FII long/short ratio
    fii_ls_change: np.ndarray    # day-over-day change in that ratio
    fii_dii_spread: np.ndarray   # FII net minus DII net (they often oppose)

    def __len__(self) -> int:
        return len(self.dates)

    def as_matrix(self) -> np.ndarray:
        return np.column_stack(
            [self.fii_net_z, self.fii_ls_change, self.fii_dii_spread]
        )


def _load_flow(store: Path) -> Dict[dt.date, Dict[str, Dict[str, float]]]:
    """Load every day's participant rows, keyed by trade date then client type."""
    import csv

    if not store.exists():
        raise FeatureError(
            f"no participant store at {store} — run "
            "`python -m backend.data.ingest.nse_participant_flow` first"
        )

    out: Dict[dt.date, Dict[str, Dict[str, float]]] = {}
    for f in sorted(store.glob("*.csv")):
        with f.open() as fh:
            for row in csv.DictReader(fh):
                d = dt.date.fromisoformat(row["trade_date"])
                out.setdefault(d, {})[row["client_type"].upper()] = {
                    "net": float(row["net_index_futures"]),
                    "ls": float(row["index_futures_ls_ratio"]),
                    "avail_ts": row["avail_ts"],
                }
    if not out:
        raise FeatureError(f"{store} contains no parseable participant rows")
    return out


def _rolling_z(x: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Trailing z-score using only past observations.

    Deliberately excludes the current point from its own mean/std: including it
    leaks a sliver of the present into the normalisation, which is the same
    class of error as a full-sample z-score, just smaller and harder to spot.
    """
    z = np.full(x.size, np.nan)
    for i in range(window, x.size):
        hist = x[i - window : i]
        sd = hist.std(ddof=1)
        if sd > 0:
            z[i] = (x[i] - hist.mean()) / sd
    return z


def build_features(
    flow_store: Path = DEFAULT_FLOW_STORE,
    window: int = 20,
) -> FlowFeatures:
    """
    Build point-in-time FII/DII features.

    Each row is stamped with the session it may first be ACTED ON, not the
    session it describes.
    """
    flow = _load_flow(flow_store)
    trade_dates = sorted(flow)

    usable = [d for d in trade_dates if "FII" in flow[d]]
    if len(usable) < window + 5:
        raise FeatureError(
            f"need > {window + 5} days of FII data, have {len(usable)} — backfill more history"
        )

    fii_net = np.array([flow[d]["FII"]["net"] for d in usable], dtype=float)
    fii_ls = np.array([flow[d]["FII"]["ls"] for d in usable], dtype=float)
    dii_net = np.array(
        [flow[d].get("DII", {}).get("net", np.nan) for d in usable], dtype=float
    )

    # The session each row becomes usable in: the avail_ts date, not trade date.
    avail_dates = np.array(
        [dt.datetime.fromisoformat(flow[d]["FII"]["avail_ts"]).date() for d in usable]
    )

    ls_change = np.full(fii_ls.size, np.nan)
    ls_change[1:] = np.diff(fii_ls)

    return FlowFeatures(
        dates=avail_dates,
        fii_net_idx_fut=fii_net,
        fii_net_z=_rolling_z(fii_net, window),
        fii_ls_ratio=fii_ls,
        fii_ls_change=ls_change,
        fii_dii_spread=fii_net - dii_net,
    )


def align_to_prices(
    features: FlowFeatures,
    price_file: Path = DEFAULT_PRICE_FILE,
) -> Dict[str, np.ndarray]:
    """
    Join features to forward returns on the session they became knowable.

    The return paired with a feature is the move of the session that OPENS after
    publication — feature from `avail_ts` date D predicts D's close-to-close
    return. Any tighter join is lookahead.
    """
    import csv

    if not price_file.exists():
        raise FeatureError(
            f"no price file at {price_file} — run "
            "`python -m backend.data.ingest.index_prices` first"
        )

    px: Dict[dt.date, float] = {}
    with price_file.open() as fh:
        for row in csv.DictReader(fh):
            try:
                px[dt.date.fromisoformat(row["date"])] = float(row["close"])
            except (ValueError, KeyError):
                continue

    price_dates = sorted(px)
    idx = {d: i for i, d in enumerate(price_dates)}

    feats, rets, used = [], [], []
    for i, d in enumerate(features.dates):
        j = idx.get(d)
        if j is None or j + 1 >= len(price_dates):
            continue  # no session on that date, or no forward return available
        row = [features.fii_net_z[i], features.fii_ls_change[i], features.fii_dii_spread[i]]
        if not np.all(np.isfinite(row)):
            continue
        prev, nxt = px[price_dates[j]], px[price_dates[j + 1]]
        feats.append(row)
        rets.append((nxt - prev) / prev)
        used.append(d)

    if len(feats) < 30:
        raise FeatureError(
            f"only {len(feats)} aligned observations — too few to evaluate. "
            "Backfill more participant history."
        )

    return {
        "features": np.asarray(feats, dtype=float),
        "forward_returns": np.asarray(rets, dtype=float),
        "dates": np.asarray(used),
    }


def flow_signal_returns(
    flow_store: Path = DEFAULT_FLOW_STORE,
    price_file: Path = DEFAULT_PRICE_FILE,
    z_threshold: float = 0.5,
) -> np.ndarray:
    """
    The tradeable expression of the hypothesis, as a return series.

    Rule: go long Nifty when FII net index-futures positioning is z-score
    BULLISH (they are adding length), short when bearish, flat in between.
    Weights are fixed, not fitted — fitting them on this sample and then scoring
    the same sample is the overfit the DSR gate exists to catch.
    """
    aligned = align_to_prices(build_features(flow_store), price_file)
    z = aligned["features"][:, 0]
    position = np.where(z > z_threshold, 1.0, np.where(z < -z_threshold, -1.0, 0.0))
    return position * aligned["forward_returns"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    feats = build_features()
    print(f"FII flow features built: {len(feats)} days")
    print(f"  date range      : {feats.dates[0]} .. {feats.dates[-1]}")
    print(f"  FII net (last)  : {feats.fii_net_idx_fut[-1]:+,.0f} contracts")
    print(f"  FII L/S (last)  : {feats.fii_ls_ratio[-1]:.3f}")

    aligned = align_to_prices(feats)
    print(f"  aligned to price: {len(aligned['forward_returns'])} observations")

    rets = flow_signal_returns()
    active = int(np.sum(rets != 0))
    print(f"\nSignal series: {len(rets)} days, {active} with a position")
    if active >= 20:
        from backend.validation.metrics import compute_all

        try:
            print("  ", compute_all(rets[rets != 0]).as_dict())
        except Exception as exc:  # noqa: BLE001 - metrics raise on thin samples
            print(f"   metrics unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
