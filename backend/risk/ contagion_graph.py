# backend/risk/contagion_graph.py
"""
Contagion Graph Simulator
-------------------------
Directed weighted graph where edge i->j means j has exposure to i (i's failure hurts j).
Weights represent exposure as a fraction of j's equity (or absolute if abs_mode=True).

Features
- Build graph (API or CSV)
- Apply shocks to nodes (equity hit or default)
- Iterate contagion with capital buffers + recovery rate
- Stop when fixed point (no new defaults) or max rounds
- Metrics & per-round audit trail
- Optional Plotly visualization
- Optional Redis publish of round snapshots

CLI
  python -m backend.risk.contagion_graph --probe
  python -m backend.risk.contagion_graph --csv data/exposures.csv --shock "BANK_A:-0.3,BANK_B:default" --plot
CSV format
  src,dst,weight
  # src fails -> dst loses weight*equity_dst (if abs_mode=False)
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any

try:
    import networkx as nx  # pip install networkx
except Exception:
    nx = None  # type: ignore

try:
    import plotly.graph_objects as go  # pip install plotly
except Exception:
    go = None  # type: ignore

# optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore


@dataclass
class NodeState:
    name: str
    equity: float                 # starting equity (>= 0)
    capital_ratio: float = 0.08   # min capital adequacy (8% default)
    alive: bool = True
    defaulted: bool = False
    equity_t: float = 0.0         # equity after last round (for delta tracking)
    sector: Optional[str] = None  # bank|broker|fund|sovereign|...
    region: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "equity": self.equity, "equity_t": self.equity_t,
            "alive": self.alive, "defaulted": self.defaulted,
            "capital_ratio": self.capital_ratio, "sector": self.sector,
            "region": self.region, **self.meta
        }


class ContagionGraph:
    def __init__(self, abs_mode: bool = False, recovery: float = 0.4, max_rounds: int = 50, publish_topic: Optional[str] = "risk.contagion"):
        """
        abs_mode=False: edge weight w_ij means fraction of dst equity at risk from src default
        abs_mode=True : edge weight interpreted as absolute currency loss to dst if src defaults
        recovery: fraction recovered from failed claims (0.4 typical)
        """
        if nx is None:
            raise RuntimeError("networkx not installed. Run: pip install networkx")
        self.G = nx.DiGraph()
        self.abs_mode = abs_mode
        self.recovery = float(recovery)
        self.max_rounds = int(max_rounds)
        self.publish_topic = publish_topic

    # ------------------------- build -------------------------
    def add_node(self, name: str, equity: float, *, capital_ratio: float = 0.08, sector: str | None = None, region: str | None = None, **meta):
        self.G.add_node(name, state=NodeState(name=name, equity=float(equity), equity_t=float(equity), capital_ratio=float(capital_ratio),
                                              sector=sector, region=region, meta=meta))

    def add_edge(self, src: str, dst: str, weight: float, *, label: Optional[str] = None):
        """
        Edge src->dst: dst is exposed to src. If src defaults, dst loses:
          - abs_mode=False: loss = weight * equity_dst * (1 - recovery)
          - abs_mode=True : loss = weight * (1 - recovery)
        """
        self.G.add_edge(src, dst, weight=float(weight), label=label)

    def load_csv(self, path: str, *, default_equity: float = 1e9, default_capital_ratio: float = 0.08):
        """
        CSV columns: src,dst,weight
        Creates nodes as needed with default equity/capital ratio.
        """
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s, d, w = row["src"].strip(), row["dst"].strip(), float(row["weight"])
                if s not in self.G:
                    self.add_node(s, default_equity, capital_ratio=default_capital_ratio)
                if d not in self.G:
                    self.add_node(d, default_equity, capital_ratio=default_capital_ratio)
                self.add_edge(s, d, w)

    # ------------------------- shocks -------------------------
    def apply_shocks(self, shocks: Dict[str, Any]):
        """
        shocks: { node: 'default' | -0.25 (equity hit fraction) | {'equity_abs': -1e8} }
        """
        for n, spec in shocks.items():
            if n not in self.G:
                continue
            st: NodeState = self.G.nodes[n]["state"]
            if isinstance(spec, str) and spec.lower() == "default":
                st.defaulted = True
                st.alive = False
                st.equity_t = 0.0
            elif isinstance(spec, (int, float)):
                st.equity_t = max(0.0, st.equity * (1.0 + float(spec)))  # e.g., -0.3 => lose 30%
                st.alive = st.equity_t > 0.0
                st.defaulted = not st.alive
            elif isinstance(spec, dict) and "equity_abs" in spec:
                st.equity_t = max(0.0, st.equity + float(spec["equity_abs"]))
                st.alive = st.equity_t > 0.0
                st.defaulted = not st.alive
            else:
                # unknown spec; ignore
                pass
            # commit new equity as baseline for first round
            st.equity = st.equity_t

    # ------------------------- contagion iterate -------------------------
    def step(self, round_id: int) -> Dict[str, Any]:
        """
        One contagion round:
          - for each newly defaulted node, propagate losses to its out-neighbors
          - apply capital adequacy threshold: default if equity_t <= 0 or equity_t/Assets < capital_ratio
        """
        newly_defaulted = [n for n, d in self.G.nodes(data=True) if d["state"].defaulted and d["state"].meta.get("_propagated_round", -1) < 0]
        if not newly_defaulted:
            return {"round": round_id, "new_defaults": [], "equity_losses": 0.0}

        # mark as will propagate this round
        for n in newly_defaulted:
            self.G.nodes[n]["state"].meta["_propagated_round"] = round_id

        total_loss = 0.0
        for src in newly_defaulted:
            for _, dst, ed in self.G.out_edges(src, data=True):
                st_dst: NodeState = self.G.nodes[dst]["state"]
                if not st_dst.alive:
                    continue
                w = float(ed.get("weight", 0.0))
                loss = (w * st_dst.equity * (1.0 - self.recovery)) if not self.abs_mode else (w * (1.0 - self.recovery))
                old = st_dst.equity_t
                st_dst.equity_t = max(0.0, st_dst.equity_t - loss)
                total_loss += max(0.0, old - st_dst.equity_t)

        # assess defaults after applying losses
        new_defaults: List[str] = []
        for n, d in self.G.nodes(data=True):
            st: NodeState = d["state"]
            if st.alive and st.equity_t <= 0.0:
                st.alive = False
                st.defaulted = True
                new_defaults.append(n)
            else:
                # (Optional) capital adequacy breach heuristic
                assets = st.meta.get("assets", st.equity_t / max(1e-9, st.capital_ratio))
                if assets > 0 and (st.equity_t / assets) < st.capital_ratio * 0.5:  # breach threshold
                    st.alive = False
                    st.defaulted = True
                    new_defaults.append(n)

        return {"round": round_id, "new_defaults": new_defaults, "equity_losses": total_loss}

    def run(self, *, max_rounds: Optional[int] = None, publish: bool = False) -> Dict[str, Any]:
        """
        Iterate until no new defaults or max_rounds reached.
        Returns summary with per-round trail.
        """
        R = int(max_rounds or self.max_rounds)
        trail: List[Dict[str, Any]] = []
        init_equity = sum(self.G.nodes[n]["state"].equity for n in self.G.nodes())

        # ensure equity_t initialized
        for _, d in self.G.nodes(data=True):
            st: NodeState = d["state"]
            if st.equity_t == 0.0 and st.equity > 0.0 and st.alive and not st.defaulted:
                st.equity_t = st.equity

        for r in range(1, R + 1):
            res = self.step(r)
            trail.append(res)
            if publish and publish_stream:
                try:
                    snapshot = self.snapshot(extra={"round": r, **res})
                    publish_stream(self.publish_topic or "risk.contagion", snapshot)
                except Exception:
                    pass
            if not res["new_defaults"]:
                break
            # commit new equity_t as equity baseline for next round
            for _, d in self.G.nodes(data=True):
                st: NodeState = d["state"]
                st.equity = st.equity_t

        final_equity = sum(self.G.nodes[n]["state"].equity_t for n in self.G.nodes())
        defaults = [n for n, d in self.G.nodes(data=True) if d["state"].defaulted]
        largest_cc = 0
        try:
            und = self.G.to_undirected()
            largest_cc = len(max(nx.connected_components(und), key=len)) if und.number_of_nodes() > 0 else 0 # type: ignore
        except Exception:
            pass

        return {
            "rounds": len(trail),
            "trail": trail,
            "defaults": defaults,
            "loss_total": max(0.0, init_equity - final_equity),
            "loss_pct": (1.0 - final_equity / max(1e-9, init_equity)) if init_equity > 0 else 0.0,
            "largest_component": largest_cc,
            "nodes": [self.G.nodes[n]["state"].as_dict() for n in self.G.nodes()],
        }

    # ------------------------- utils -------------------------
    def snapshot(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        nodes = []
        for n, d in self.G.nodes(data=True):
            st: NodeState = d["state"]
            nodes.append(st.as_dict())
        edges = [{"src": s, "dst": d, "weight": float(ed.get("weight", 0.0))}
                 for s, d, ed in self.G.edges(data=True)]
        snap = {"nodes": nodes, "edges": edges}
        if extra:
            snap.update(extra)
        return snap

    def plot(self, title: str = "Contagion Graph") -> "go.Figure": # type: ignore
        if go is None:
            raise RuntimeError("plotly not installed. Run: pip install plotly")
        pos = nx.spring_layout(self.G, seed=7) # type: ignore
        x_nodes = [pos[n][0] for n in self.G.nodes()]
        y_nodes = [pos[n][1] for n in self.G.nodes()]
        colors = []
        sizes = []
        texts = []
        for n in self.G.nodes():
            st: NodeState = self.G.nodes[n]["state"]
            colors.append("#d62728" if st.defaulted else ("#1f77b4" if st.alive else "#7f7f7f"))
            sizes.append(max(10, 20 * math.log10(max(2.0, st.equity_t or st.equity or 1.0))))
            texts.append(f"{n}<br>Eq:{st.equity_t:,.0f}{' (D)' if st.defaulted else ''}")
        edge_x = []
        edge_y = []
        for s, d in self.G.edges():
            x0, y0 = pos[s]; x1, y1 = pos[d]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", opacity=0.4))
        fig.add_trace(go.Scatter(x=x_nodes, y=y_nodes, mode="markers+text",
                                 marker=dict(size=sizes, line=dict(width=1)),
                                 text=[n for n in self.G.nodes()], textposition="bottom center",
                                 hovertext=texts, hoverinfo="text",
                                 marker_color=colors))
        fig.update_layout(title=title, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        return fig


# ------------------------- CLI -------------------------

def _parse_shocks(s: str) -> Dict[str, Any]:
    """
    "A:-0.3,B:default,C:{equity_abs:-1e8}"
    """
    out: Dict[str, Any] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        v = v.strip()
        if v.lower() == "default":
            out[k] = "default"
        elif v.startswith("{") and v.endswith("}"):
            try:
                out[k] = eval(v, {}, {})  # trusted CLI only; avoid in services
            except Exception:
                pass
        else:
            try:
                out[k] = float(v)
            except Exception:
                pass
    return out


def _probe():
    cg = ContagionGraph(abs_mode=False, recovery=0.4)
    # toy 5-node system
    for n in ["BANK_A", "BANK_B", "BROKER_C", "FUND_D", "SOV_E"]:
        cg.add_node(n, equity=1e9, capital_ratio=0.08, sector="bank" if "BANK" in n else "other")
    # exposures: if A defaults, B and C lose
    cg.add_edge("BANK_A", "BANK_B", 0.25)
    cg.add_edge("BANK_A", "BROKER_C", 0.20)
    cg.add_edge("BANK_B", "FUND_D", 0.10)
    cg.add_edge("BROKER_C", "FUND_D", 0.15)
    cg.add_edge("FUND_D", "SOV_E", 0.05)
    # shock A default
    cg.apply_shocks({"BANK_A": "default"})
    summary = cg.run(publish=False)
    print("Defaults:", summary["defaults"])
    print("Loss %:", round(100 * summary["loss_pct"], 2), "%")
    try:
        fig = cg.plot("Probe: Contagion")
        fig.show()
    except Exception:
        pass


def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Contagion Graph Simulator")
    ap.add_argument("--csv", type=str, help="CSV exposures file (src,dst,weight)")
    ap.add_argument("--abs-mode", action="store_true", help="Interpret weights as absolute currency")
    ap.add_argument("--recovery", type=float, default=0.4, help="Recovery rate [0..1]")
    ap.add_argument("--shock", type=str, help='Comma list: "A:-0.3,B:default,C:{equity_abs:-1e8}"')
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--publish", action="store_true", help="Publish snapshots to Redis stream")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--json", action="store_true", help="Print final JSON summary")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if args.probe:
        _probe()
        return

    cg = ContagionGraph(abs_mode=args.abs_mode, recovery=args.recovery, max_rounds=args.rounds)
    if args.csv:
        cg.load_csv(args.csv)
    else:
        # minimal fall-back example
        cg.add_node("A", equity=1e9); cg.add_node("B", equity=1e9); cg.add_node("C", equity=1e9)
        cg.add_edge("A","B",0.2); cg.add_edge("B","C",0.2); cg.add_edge("C","A",0.1)

    if args.shock:
        cg.apply_shocks(_parse_shocks(args.shock))

    summary = cg.run(publish=args.publish)
    if args.json:
        print(json.dumps(summary, indent=2))
    if args.plot:
        try:
            fig = cg.plot("Contagion Graph")
            fig.show()
        except Exception as e:
            print("Plot skipped:", e)

if __name__ == "__main__":
    main()