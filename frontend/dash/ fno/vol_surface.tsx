'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ---------- Types ---------- */
type VolNode = { strike: number; days: number; iv: number }; // iv in 0..1
type Props = {
  title?: string;
  symbol?: string;
  spot?: number;
  expiriesDays?: number[];     // e.g., [7, 30, 60, 90, 180, 365]
  strikes?: number[];          // custom strike grid
  nodes?: VolNode[];           // optional pre-fetched nodes
  fetchEndpoint?: string;      // optional: GET ?symbol=AAPL → VolNode[]
};

/* ---------- Helpers ---------- */
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const fmt = (n: number, d = 2) => (n == null ? '-' : n.toLocaleString(undefined, { maximumFractionDigits: d }));

/** Tiny SABR-ish mock smile/term structure generator */
function mockSurface(spot = 225, expiries: number[] = [7, 30, 60, 90, 180, 365]) {
  const atmBase = 0.32;      // base ATM vol
  const termDecay = (d: number) => 0.85 ** Math.log2(1 + d / 30);
  const strikes = Array.from({ length: 21 }, (_, i) => Math.round((spot * 0.8) + i * (spot * 0.02)));
  const nodes: VolNode[] = [];
  for (const T of expiries) {
    const term = termDecay(T);
    for (const K of strikes) {
      const m = (K - spot) / spot;             // moneyness
      const smile = 1 + 0.9 * Math.abs(m) + (m < 0 ? 0.25 * (-m) : 0); // smirk
      const iv = clamp(atmBase * term * smile, 0.05, 1.0);
      nodes.push({ strike: K, days: T, iv: +iv.toFixed(4) });
    }
  }
  return { strikes, expiries, nodes };
}

/** Build grid from sparse/flat nodes */
function gridFromNodes(nodes: VolNode[]) {
  const strikes = Array.from(new Set(nodes.map(n => n.strike))).sort((a, b) => a - b);
  const expiries = Array.from(new Set(nodes.map(n => n.days))).sort((a, b) => a - b);
  const map = new Map<string, number>();
  for (const n of nodes) map.set(`${n.strike}|${n.days}`, n.iv);
  const grid = expiries.map((d) => strikes.map((k) => map.get(`${k}|${d}`) ?? 0));
  return { strikes, expiries, grid };
}

/** Simple sequential colormap (blue → green → yellow → red) */
function colorFor(v: number, vmin: number, vmax: number) {
  const t = clamp((v - vmin) / Math.max(1e-9, vmax - vmin), 0, 1);
  // piecewise: 0..0.33 blue→green, 0.33..0.66 green→yellow, 0.66..1 yellow→red
  const seg = t < 0.33 ? 0 : t < 0.66 ? 1 : 2;
  const local = seg === 0 ? t / 0.33 : seg === 1 ? (t - 0.33) / 0.33 : (t - 0.66) / 0.34;
  let r = 0, g = 0, b = 0;
  if (seg === 0) { r = 0; g = Math.round(120 * local + 30); b = Math.round(255 * (1 - local)); }
  else if (seg === 1) { r = Math.round(255 * local * 0.8); g = 150 + Math.round(80 * (1 - local)); b = 40; }
  else { r = 200 + Math.round(55 * local); g = Math.round(150 * (1 - local)); b = 30; }
  return `rgb(${r},${g},${b})`;
}

/* ---------- Component ---------- */
export default function VolSurface({
  title = 'Vol Surface',
  symbol: symProp = 'AAPL',
  spot: spotProp = 225,
  expiriesDays = [7, 30, 60, 90, 180, 365],
  strikes,
  nodes,
  fetchEndpoint,
}: Props) {
  const [symbol, setSymbol] = useState(symProp);
  const [spot, setSpot] = useState(spotProp);
  const [hover, setHover] = useState<{ k?: number; d?: number; iv?: number; x?: number; y?: number } | null>(null);
  const [view, setView] = useState<'heatmap' | 'table'>('heatmap');

  // Data loading / mocking
  const [rows, setRows] = useState<VolNode[] | null>(nodes ?? null);
  useEffect(() => {
    let dead = false;
    if (!fetchEndpoint) {
      const m = mockSurface(spot, expiriesDays);
      const K = strikes ?? m.strikes;
      const filtered = m.nodes.filter(n => K.includes(n.strike));
      setRows(filtered);
      return;
    }
    (async () => {
      try {
        const q = new URLSearchParams({ symbol });
        const res = await fetch(`${fetchEndpoint}?${q.toString()}`);
        const json = await res.json();
        if (!dead && Array.isArray(json)) setRows(json as VolNode[]);
      } catch {
        if (!dead) {
          const m = mockSurface(spot, expiriesDays);
          setRows(m.nodes);
        }
      }
    })();
    return () => { dead = true; };
  }, [fetchEndpoint, symbol, spot, strikes?.join('|'), expiriesDays.join('|')]);

  const { strikes: K, expiries: T, grid, ivMin, ivMax } = useMemo(() => {
    const base = rows ?? mockSurface(spot, expiriesDays).nodes;
    const g = gridFromNodes(base);
    let mn = Infinity, mx = -Infinity;
    g.grid.forEach(row => row.forEach(v => { if (v > 0) { mn = Math.min(mn, v); mx = Math.max(mx, v); } }));
    if (!isFinite(mn)) { mn = 0.1; mx = 1.0; }
    return { ...g, ivMin: mn, ivMax: mx };
  }, [rows]);

  /* ---------- Canvas Heatmap ---------- */
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const W = 760, H = 420, padL = 60, padR = 16, padT = 24, padB = 44;

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    // clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // axes
    ctx.font = '12px system-ui';
    ctx.fillStyle = '#374151';
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;

    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // grid/ticks
    ctx.strokeStyle = '#f3f4f6';
    ctx.beginPath();
    for (let i = 0; i < K.length; i++) {
      const x = padL + (i / Math.max(1, K.length - 1)) * plotW;
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
    }
    for (let j = 0; j < T.length; j++) {
      const y = padT + (j / Math.max(1, T.length - 1)) * plotH;
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
    }
    ctx.stroke();

    // heatmap cells
    for (let j = 0; j < T.length; j++) {
      for (let i = 0; i < K.length; i++) {
        const iv = grid[j]?.[i] ?? 0;
        if (!iv) continue;
        const x = padL + (i / K.length) * plotW;
        const y = padT + (j / T.length) * plotH;
        const cw = plotW / K.length;
        const ch = plotH / T.length;
        ctx.fillStyle = colorFor(iv, ivMin, ivMax);
        ctx.fillRect(x, y, cw + 1, ch + 1);
      }
    }

    // axes labels
    ctx.fillStyle = '#6b7280';
    // X (strikes)
    for (let i = 0; i < K.length; i += Math.ceil(K.length / 8)) {
      const x = padL + (i / Math.max(1, K.length - 1)) * plotW;
      ctx.fillText(String(K[i]), x - 12, padT + plotH + 16);
    }
    // Y (days)
    for (let j = 0; j < T.length; j += Math.ceil(T.length / 6)) {
      const y = padT + (j / Math.max(1, T.length - 1)) * plotH;
      ctx.fillText(String(T[j]) + 'd', 8, y + 4);
    }

    // axes titles
    ctx.fillText('Strike', padL + plotW / 2 - 16, H - 8);
    ctx.save();
    ctx.translate(16, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Expiry (days)', -40, 0);
    ctx.restore();

    // legend
    const Lx = W - padR - 160, Ly = padT + 6, Lw = 120, Lh = 10;
    for (let i = 0; i < Lw; i++) {
      const v = ivMin + (i / (Lw - 1)) * (ivMax - ivMin);
      ctx.fillStyle = colorFor(v, ivMin, ivMax);
      ctx.fillRect(Lx + i, Ly, 1, Lh);
    }
    ctx.strokeStyle = '#9ca3af';
    ctx.strokeRect(Lx, Ly, Lw, Lh);
    ctx.fillStyle = '#6b7280';
    ctx.fillText(`${(ivMin * 100).toFixed(0)}%`, Lx, Ly + 22);
    ctx.fillText(`${(ivMax * 100).toFixed(0)}%`, Lx + Lw - 28, Ly + 22);
  }, [grid, K, T, ivMin, ivMax]);

  // mouse hover → find nearest cell
  function onMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    if (mx < padL || mx > padL + plotW || my < padT || my > padT + plotH) {
      setHover(null); return;
    }
    const i = clamp(Math.floor(((mx - padL) / plotW) * K.length), 0, K.length - 1);
    const j = clamp(Math.floor(((my - padT) / plotH) * T.length), 0, T.length - 1);
    const iv = grid[j]?.[i] ?? 0;
    setHover({ k: K[i], d: T[j], iv, x: mx, y: my });
  }

  /* ---------- Table View ---------- */
  const flat = useMemo(() => {
    const out: { strike: number; days: number; iv: number }[] = [];
    for (let j = 0; j < T.length; j++) for (let i = 0; i < K.length; i++) {
      const v = grid[j]?.[i]; if (!v) continue;
      out.push({ strike: K[i], days: T[j], iv: v });
    }
    return out.sort((a, b) => a.days - b.days || a.strike - b.strike);
  }, [grid, K, T]);

  function exportJson() {
    const json = JSON.stringify(flat, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${symbol}_vol_surface.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <div style={S.wrap}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>{symbol}</span>
          <span style={{ ...S.badge, background: '#eef2ff', color: '#3730a3' }}>{spot.toFixed(2)}</span>
        </div>
        <div style={S.controls}>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Symbol</span>
            <input value={symbol} onChange={(e)=>setSymbol(e.target.value.toUpperCase())} style={S.input} />
          </label>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Spot</span>
            <input type="number" value={spot} onChange={(e)=>setSpot(+e.target.value)} style={S.input} />
          </label>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>View</span>
            <select value={view} onChange={(e)=>setView(e.target.value as any)} style={S.select}>
              <option value="heatmap">Heatmap</option>
              <option value="table">Table</option>
            </select>
          </label>
          <button onClick={exportJson} style={S.btn}>Export JSON</button>
        </div>
      </div>

      {/* Body */}
      {view === 'heatmap' ? (
        <div style={{ position: 'relative', padding: 12 }}>
          <canvas
            ref={canvasRef}
            width={760}
            height={420}
            style={{ width: '100%', maxWidth: 980, border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff' }}
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
          />
          {hover && (
            <div
              style={{
                position: 'absolute',
                left: clamp((hover.x ?? 0) + 18, 0, 760),
                top: clamp((hover.y ?? 0) + 10, 0, 420),
                background: '#111827', color: '#fff', fontSize: 12, padding: '6px 8px',
                borderRadius: 8, pointerEvents: 'none', boxShadow: '0 2px 6px rgba(0,0,0,0.2)'
              }}
            >
              <div><b>Strike:</b> {hover.k}</div>
              <div><b>Expiry:</b> {hover.d}d</div>
              <div><b>IV:</b> {(hover.iv! * 100).toFixed(2)}%</div>
            </div>
          )}
        </div>
      ) : (
        <div style={{ padding: 12, overflow: 'auto' }}>
          <table style={S.table}>
            <thead>
              <tr>
                <th style={S.th}>Days</th>
                <th style={S.th}>Strike</th>
                <th style={S.th}>IV</th>
              </tr>
            </thead>
            <tbody>
              {flat.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={S.td}>{r.days}</td>
                  <td style={S.td}>{r.strike}</td>
                  <td style={S.td}>{(r.iv * 100).toFixed(2)}%</td>
                </tr>
              ))}
              {flat.length === 0 && (
                <tr><td colSpan={3} style={{ ...S.td, textAlign: 'center', color: '#9ca3af' }}>No surface data</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ---------- Styles ---------- */
const S: Record<string, React.CSSProperties> = {
  wrap: { border: '1px solid #e5e7eb', borderRadius: 16, background: '#fff', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', width: '100%', fontFamily: 'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding: 16, borderBottom: '1px solid #eee' },
  titleRow: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },

  controls: { display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' },
  ctrlItem: { display: 'flex', flexDirection: 'column', gap: 6 },
  lbl: { fontSize: 12, color: '#6b7280' },
  input: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 10px', fontSize: 14, outline: 'none', minWidth: 120 },
  select: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 8px', fontSize: 14, background: '#fff', minWidth: 120 },
  btn: { height: 36, padding: '0 12px', borderRadius: 10, border: '1px solid transparent', background: '#111', color: '#fff', cursor: 'pointer', fontSize: 13 },

  table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0 },
  th: { textAlign: 'right', padding: '8px 10px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #eee', position: 'sticky' as any, top: 0, background: '#fff' },
  td: { textAlign: 'right', padding: '8px 10px', fontSize: 13, color: '#111827' },
};