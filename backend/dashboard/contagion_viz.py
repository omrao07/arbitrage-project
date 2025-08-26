# backend/risk/contagion_viz.py
"""
Contagion Graph Visualizer (static + animated)

- Renders your bank/exposure network with Plotly (interactive HTML).
- Colors banks by status (defaulted / stressed / ok), sizes by equity or assets.
- Edge widths scale with exposure amount.
- Optional animation over time from CrisisTheatre / engine snapshots.

Soft-deps:
  - plotly>=5 (required)
  - networkx (optional, for nicer layouts; otherwise falls back to circular)

Expected graph-like object (flexible):
  graph.banks: Dict[str, BankLike]
      BankLike has attributes/keys: name?, equity, liquid_assets, illiquid_assets, liabilities, defaulted?
      (capital_ratio() method or 'capital_ratio' field if available is used when present.)
  graph.exposures or graph.edges():
      Iterable of (lender_id, borrower_id, amount, recovery_rate?) or dicts with these keys.

Frames for animation (optional):
  Use frames from CrisisTheatre Outcome or similar:
     frame["banks"] = { "A": {"equity":..., "liq":..., "illq":..., "dd": bool, "cr": float}, ... }
     frame["minute"] = int
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- required plotting lib
try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception as e:
    raise ImportError("contagion_viz requires 'plotly' (pip install plotly)") from e

# --- optional nicer layouts
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None

# =============================================================================
# Public API
# =============================================================================

@dataclass
class Layout:
    """Holds 2D positions for nodes."""
    pos: Dict[str, Tuple[float, float]]  # id -> (x,y)


def plot_snapshot(
    graph_like: Any,
    *,
    layout: Optional[Layout] = None,
    title: str = "Contagion Network",
    size_mode: str = "equity",       # or "assets"
    size_min: int = 14,
    size_max: int = 42,
    stressed_cap_ratio: float = 0.06,
    show_labels: bool = True,
) -> go.Figure:
    """
    Render a single snapshot of the network.
    """
    nodes, edges = _extract_graph(graph_like)
    if not layout:
        layout = _make_layout(nodes, edges)

    fig = _render(nodes, edges, layout, title=title,
                  size_mode=size_mode, size_min=size_min, size_max=size_max,
                  stressed_cap_ratio=stressed_cap_ratio, show_labels=show_labels)
    return fig


def animate_frames(
    graph_like: Any,
    frames: Iterable[Dict[str, Any]],
    *,
    layout: Optional[Layout] = None,
    title: str = "Contagion Animation",
    size_mode: str = "equity",
    size_min: int = 14,
    size_max: int = 42,
    stressed_cap_ratio: float = 0.06,
    label_format: str = "{name}\nCR={cr:.1%}",
    fps: int = 2,
) -> go.Figure:
    """
    Build an animated figure. 'frames' is an iterable of snapshots with bank fields.
    Node geometry (positions, ids) is kept constant, only marker colors/sizes/text update.

    label_format fields available per bank per frame:
      name, id, equity, liq, illq, cr (capital ratio), dd (defaulted: bool)
    """
    nodes, edges = _extract_graph(graph_like)
    if not layout:
        layout = _make_layout(nodes, edges)

    # Base (first frame) state
    frames = list(frames)
    if not frames:
        raise ValueError("No frames supplied for animation")
    first = frames[0]

    # prepare static edges
    edge_traces = _edge_traces(edges, layout)

    # initial node trace from first frame
    node_trace, _ = _node_trace(
        nodes, layout, size_mode=size_mode, size_min=size_min, size_max=size_max,
        stressed_cap_ratio=stressed_cap_ratio, override_state=first.get("banks"), label_format=label_format
    )

    # Build animated frames (update node markers & text only)
    pl_frames: List[go.Frame] = []
    for fr in frames:
        node_trace_f, _ = _node_trace(
            nodes, layout, size_mode=size_mode, size_min=size_min, size_max=size_max,
            stressed_cap_ratio=stressed_cap_ratio, override_state=fr.get("banks"), label_format=label_format
        )
        name = f"t={fr.get('minute', 0)}"
        pl_frames.append(go.Frame(name=name, data=[*edge_traces, node_trace_f]))

    # Compose figure
    fig = go.Figure(
        data=[*edge_traces, node_trace],
        frames=pl_frames,
        layout=go.Layout(
            title=dict(text=title),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 0.0, "y": 1.12, "xanchor": "left", "yanchor": "top",
                "pad": {"r": 8, "t": 6},
                "buttons": [
                    {"label": "Play", "method": "animate",
                     "args": [None, {"frame": {"duration": int(1000/max(1,fps))}, "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
                ],
            }],
            sliders=[{
                "y": 1.05, "x": 0.0, "len": 1.0, "xanchor":"left",
                "steps": [{"args": [[fr.name], {"frame": {"duration": 0}, "mode": "immediate"}], "label": fr.name, "method": "animate"} for fr in pl_frames]
            }],
            margin=dict(l=10, r=10, t=60, b=10),
        ),
    )
    return fig


def export_html(fig: go.Figure, path: str, *, auto_open: bool = False) -> None:
    """
    Write the figure to a self-contained HTML file.
    """
    pio.write_html(fig, file=path, include_plotlyjs="cdn", full_html=True, auto_open=auto_open) # type: ignore


# =============================================================================
# Internals
# =============================================================================

@dataclass
class _Node:
    id: str
    name: str
    equity: float
    liq: float
    illq: float
    liabilities: float
    defaulted: bool
    cap_ratio: float   # capital / liabilities (or provided)
    meta: Dict[str, Any]


@dataclass
class _Edge:
    src: str
    dst: str
    amount: float
    recovery: float


def _extract_graph(graph_like: Any) -> Tuple[Dict[str, _Node], List[_Edge]]:
    """
    Convert a user-provided graph object into our {_Node}s and [_Edge]s.
    """
    # ---- Banks / nodes
    banks_obj: Dict[str, Any] = getattr(graph_like, "banks", {})
    nodes: Dict[str, _Node] = {}
    for bid, b in banks_obj.items():
        name = str(getattr(b, "name", getattr(b, "id", bid)) or bid)
        equity = _as_float(getattr(b, "equity", _get(b, "equity", 0.0)))
        liq = _as_float(getattr(b, "liquid_assets", _get(b, "liquid_assets", 0.0)))
        illq = _as_float(getattr(b, "illiquid_assets", _get(b, "illiquid_assets", 0.0)))
        liab = _as_float(getattr(b, "liabilities", _get(b, "liabilities", 0.0)))
        defaulted = bool(getattr(b, "defaulted", _get(b, "defaulted", False)))
        # capital ratio, if provided as method/attr; else rough equity/liab
        cr = None
        if hasattr(b, "capital_ratio"):
            try:
                cr = float(b.capital_ratio() if callable(b.capital_ratio) else b.capital_ratio) # type: ignore
            except Exception:
                cr = None
        if cr is None:
            cr = (equity / max(1e-9, liab)) if liab > 0 else 0.0
        nodes[bid] = _Node(
            id=str(bid), name=name, equity=equity, liq=liq, illq=illq, liabilities=liab,
            defaulted=defaulted, cap_ratio=cr, meta={}
        )

    # ---- Exposures / edges
    edges: List[_Edge] = []
    # Accept graph.exposures (list/dict) OR graph.edges()/iter_edges()
    exposures = getattr(graph_like, "exposures", None)
    if exposures is None and hasattr(graph_like, "edges"):
        try:
            exposures = list(graph_like.edges())  # type: ignore
        except Exception:
            exposures = None

    if exposures is not None:
        for e in exposures:
            if isinstance(e, dict):
                src = str(e.get("lender") or e.get("src") or e.get("u") or e.get("from"))
                dst = str(e.get("borrower") or e.get("dst") or e.get("v") or e.get("to"))
                amt = _as_float(e.get("amount") or e.get("exposure") or e.get("w") or 0.0)
                rr  = _as_float(e.get("recovery_rate") or e.get("rr") or 0.4)
            else:
                # tuple-like
                try:
                    src, dst, amt, rr = str(e[0]), str(e[1]), _as_float(e[2]), _as_float(e[3] if len(e) > 3 else 0.4)
                except Exception:
                    continue
            if src and dst:
                edges.append(_Edge(src=src, dst=dst, amount=amt, recovery=rr))

    return nodes, edges


def _make_layout(nodes: Dict[str, _Node], edges: List[_Edge]) -> Layout:
    ids = list(nodes.keys())
    if nx is not None:
        G = nx.DiGraph()
        for i in ids:
            G.add_node(i)
        for e in edges:
            if e.src in nodes and e.dst in nodes:
                G.add_edge(e.src, e.dst, weight=max(1.0, e.amount))
        pos = nx.spring_layout(G, seed=42, k=1 / math.sqrt(max(1, len(ids))), weight="weight", iterations=200) # type: ignore
        return Layout({i: (float(p[0]), float(p[1])) for i, p in pos.items()})
    # fallback: circle
    n = max(1, len(ids))
    pos: Dict[str, Tuple[float, float]] = {}
    R = 1.0
    for k, i in enumerate(ids):
        a = 2 * math.pi * (k / n)
        pos[i] = (R * math.cos(a), R * math.sin(a))
    return Layout(pos)


def _render(
    nodes: Dict[str, _Node],
    edges: List[_Edge],
    layout: Layout,
    *,
    title: str,
    size_mode: str,
    size_min: int,
    size_max: int,
    stressed_cap_ratio: float,
    show_labels: bool,
) -> go.Figure:
    edge_traces = _edge_traces(edges, layout)
    node_trace, _ = _node_trace(
        nodes, layout, size_mode=size_mode, size_min=size_min, size_max=size_max,
        stressed_cap_ratio=stressed_cap_ratio, label_format="{name}\nCR={cr:.1%}" if show_labels else None
    )

    fig = go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            title=dict(text=title),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=10, r=10, t=60, b=10),
        ),
    )
    return fig


def _edge_traces(edges: List[_Edge], layout: Layout) -> List[go.Scatter]:
    if not edges:
        return []
    # scale edge widths by exposure amount (log-ish)
    max_amt = max(1.0, max(e.amount for e in edges))
    traces: List[go.Scatter] = []
    for e in edges:
        if e.src not in layout.pos or e.dst not in layout.pos:
            continue
        x0, y0 = layout.pos[e.src]
        x1, y1 = layout.pos[e.dst]
        width = 0.5 + 4.0 * math.sqrt(max(0.0, e.amount) / max_amt)
        traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=width, color="rgba(120,120,120,0.5)"),
            hoverinfo="text",
            text=[f"{e.src}→{e.dst}: {e.amount:,.0f} (RR={e.recovery:.0%})", ""],
        ))
    return traces


def _node_trace(
    nodes: Dict[str, _Node],
    layout: Layout,
    *,
    size_mode: str,
    size_min: int,
    size_max: int,
    stressed_cap_ratio: float,
    override_state: Optional[Dict[str, Any]] = None,
    label_format: Optional[str] = "{name}\nCR={cr:.1%}",
) -> Tuple[go.Scatter, Dict[str, Any]]:
    xs: List[float] = []
    ys: List[float] = []
    sizes: List[float] = []
    colors: List[str] = []
    texts: List[str] = []
    hover: List[str] = []
    ids: List[str] = []

    # derive scaling
    vals = []
    for n in nodes.values():
        assets = n.liq + n.illq
        base = n.equity if size_mode == "equity" else (assets if assets > 0 else n.equity)
        vals.append(max(0.0, base))
    vmin, vmax = (min(vals) if vals else 0.0), (max(vals) if vals else 1.0)

    for bid, n in nodes.items():
        st = n
        # override per-frame state if provided (animation)
        if override_state and bid in override_state:
            o = override_state[bid] or {}
            st = _Node(
                id=n.id, name=n.name,
                equity=float(o.get("equity", n.equity)),
                liq=float(o.get("liq", n.liq)),
                illq=float(o.get("illq", n.illq)),
                liabilities=float(o.get("liab", o.get("liabilities", n.liabilities))),
                defaulted=bool(o.get("dd", o.get("defaulted", n.defaulted))),
                cap_ratio=float(o.get("cr", n.cap_ratio)),
                meta=n.meta.copy(),
            )

        x, y = layout.pos.get(bid, (0.0, 0.0))
        xs.append(x); ys.append(y); ids.append(bid)

        assets = st.liq + st.illq
        base = st.equity if size_mode == "equity" else (assets if assets > 0 else st.equity)
        s = _scale(base, vmin, vmax, size_min, size_max)
        sizes.append(s)

        # color by status
        if st.defaulted:
            col = "#d62728"          # red
        elif st.cap_ratio < stressed_cap_ratio:
            col = "#ff7f0e"          # orange (stressed)
        else:
            col = "#2ca02c"          # green (ok)
        colors.append(col)

        # labels
        label = (label_format or "").format(
            id=st.id, name=st.name, equity=st.equity, liq=st.liq, illq=st.illq, cr=st.cap_ratio, dd=st.defaulted
        ) if label_format else ""
        texts.append(label)

        hover.append(
            f"<b>{st.name}</b><br>"
            f"Equity: {st.equity:,.0f}<br>"
            f"Capital Ratio: {st.cap_ratio:.2%}<br>"
            f"Assets (liq/illiq): {st.liq:,.0f}/{st.illq:,.0f}<br>"
            f"Defaulted: {st.defaulted}"
        )

    trace = go.Scatter(
        x=xs, y=ys, mode="markers+text" if label_format else "markers",
        text=texts if label_format else None,
        textposition="bottom center",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="rgba(40,40,40,0.6)")),
        hoverinfo="text",
        hovertext=hover,
    )
    state = {"ids": ids, "sizes": sizes, "colors": colors, "texts": texts}
    return trace, state


# =============================================================================
# Helpers
# =============================================================================

def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _get(obj: Any, key: str, default: Any = None) -> Any:
    try:
        return obj.get(key, default)  # type: ignore
    except Exception:
        return default

def _scale(x: float, vmin: float, vmax: float, smin: float, smax: float) -> float:
    if vmax <= vmin:
        return (smin + smax) / 2.0
    t = (x - vmin) / (vmax - vmin)
    return smin + t * (smax - smin)


# =============================================================================
# Tiny CLI (optional)
# =============================================================================

if __name__ == "__main__":
    # Minimal demo with a synthetic graph (only runs if you call this file directly)
    class _Bank:
        def __init__(self, id, name, equity, liq, illq, liab, defaulted=False):
            self.id=id; self.name=name; self.equity=equity; self.liquid_assets=liq; self.illiquid_assets=illq; self.liabilities=liab; self.defaulted=defaulted
        def capital_ratio(self): return self.equity / max(1e-9, self.liabilities)

    class _G:
        def __init__(self):
            self.banks = {
                "A": _Bank("A","Alpha",100,300,700,800),
                "B": _Bank("B","Beta",80,200,500,620),
                "C": _Bank("C","Gamma",60,150,450,540),
            }
            self.exposures = 
                {"lender":"A","borrower":"B","amount":120.0,"recovery_rate":0.5}, # type: ignore
                {"lender":"B","borrower":"C","amount":100.0,"recovery_rate":0.4}, # type: ignore
                {"lender":"C","borrower":"A","amount":90.0, "recovery_rate":0.3}, # type: ignore
            

    G = _G()
    fig = plot_snapshot(G, title="Contagion Snapshot (demo)")
    export_html(fig, "contagion_snapshot.html", auto_open=False)

    # Fake animation frames (toggle a default)
    frames = [
        {"minute": 0, "banks": {}},
        {"minute": 1, "banks": {"B": {"dd": True, "cr": 0.0, "equity": 0.0}}},
        {"minute": 2, "banks": {"C": {"dd": True, "cr": 0.0, "equity": 0.0}}},
    ]
    fig2 = animate_frames(G, frames, title="Contagion Animation (demo)", fps=1)
    export_html(fig2, "contagion_animation.html", auto_open=False)