'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */
type SeriesDict = Record<string, number[]>;     // { AAPL:[…], MSFT:[…] } aligned by index
type CorrMatrix = number[][];                   // [row][col] in [-1,1]
type Props = {
  title?: string;
  data?: SeriesDict;            // pass timeseries -> we compute correlations
  labels?: string[];            // order of symbols for timeseries (optional; else Object.keys(data))
  matrix?: CorrMatrix;          // alternatively, pass a correlation matrix directly
  matrixLabels?: string[];      // labels for the provided matrix
  fetchEndpoint?: string;       // optional GET -> { labels: string[], data?: SeriesDict, matrix?: number[][] }
  clampMin?: number;            // default -1
  clampMax?: number;            // default +1
};

/* ================= Helpers ================= */
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const fmt = (n: number, d = 2) => (Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: d }) : '-');

/** Pearson correlation between two aligned arrays */
function corr(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let sa = 0, sb = 0, s2a = 0, s2b = 0, sab = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    sa += x; sb += y;
    s2a += x * x; s2b += y * y;
    sab += x * y;
  }
  const ma = sa / n, mb = sb / n;
  const va = s2a / n - ma * ma;
  const vb = s2b / n - mb * mb;
  const cov = sab / n - ma * mb;
  const den = Math.sqrt(Math.max(va, 1e-12) * Math.sqrt(Math.max(vb, 1e-12)));
  const d = Math.sqrt(Math.max(va * vb, 1e-24));
  return d === 0 ? 0 : clamp(cov / d, -1, 1);
}

/** Build corr matrix from dict */
function buildMatrix(d: SeriesDict, labels?: string[]): { labels: string[]; matrix: CorrMatrix } {
  const L = labels && labels.length ? labels : Object.keys(d);
  const n = L.length;
  const M: CorrMatrix = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    M[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const c = corr(d[L[i]] || [], d[L[j]] || []);
      M[i][j] = c; M[j][i] = c;
    }
  }
  return { labels: L, matrix: M };
}

/** Diverging colormap: blue (-1) → white (0) → red (+1) */
function colorFor(v: number) {
  const t = clamp((v + 1) / 2, 0, 1);
  // Interpolate blue (52,106,235) -> white (245,245,245) -> red (239,68,68)
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const u = t / 0.5; // 0..1
    r = 52 + (245 - 52) * u;
    g = 106 + (245 - 106) * u;
    b = 235 + (245 - 235) * u;
  } else {
    const u = (t - 0.5) / 0.5;
    r = 245 + (239 - 245) * u;
    g = 245 + (68 - 245) * u;
    b = 245 + (68 - 245) * u;
  }
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

/* ================= Mock ================= */
function mockTimeseries(): SeriesDict {
  const labels = ['AAPL','MSFT','NVDA','AMZN','META','TSLA','XOM','JPM'];
  const len = 180;
  const out: SeriesDict = {};
  let base1 = 0, base2 = 0, base3 = 0;
  for (const s of labels) {
    const arr: number[] = [];
    let drift = (Math.random() - 0.5) * 0.02;
    for (let i = 0; i < len; i++) {
      // three latent factors + noise
      base1 += (Math.random() - 0.5) * 0.2;
      base2 += (Math.random() - 0.5) * 0.15;
      base3 += (Math.random() - 0.5) * 0.1;
      const val = 0.6 * base1 + 0.3 * base2 + 0.2 * base3 + drift * i + (Math.random() - 0.5) * 0.6;
      arr.push(val);
    }
    out[s] = arr;
  }
  return out;
}

/* ================= Component ================= */
export default function CorrMatrixView({
  title = 'Correlation Matrix',
  data,
  labels,
  matrix,
  matrixLabels,
  fetchEndpoint,
  clampMin = -1,
  clampMax = 1,
}: Props) {
  // Load/fetch data
  const [series, setSeries] = useState<SeriesDict | null>(data ?? null);
  const [mProvided, setMProvided] = useState<CorrMatrix | null>(matrix ?? null);
  const [lblsProvided, setLblsProvided] = useState<string[] | null>(matrixLabels ?? null);

  useEffect(() => {
    let dead = false;
    if (!fetchEndpoint) {
      if (!series && !mProvided) setSeries(mockTimeseries());
      return;
    }
    (async () => {
      try {
        const res = await fetch(fetchEndpoint);
        const json = await res.json();
        if (dead) return;
        if (json?.matrix && Array.isArray(json.matrix) && json?.labels) {
          setMProvided(json.matrix as CorrMatrix);
          setLblsProvided(json.labels as string[]);
        } else if (json?.data) {
          setSeries(json.data as SeriesDict);
        }
      } catch {
        if (!dead) setSeries(mockTimeseries());
      }
    })();
    return () => { dead = true; };
  }, [fetchEndpoint]);

  // Build matrix if needed
  const { L, M } = useMemo(() => {
    if (mProvided && (lblsProvided?.length ?? 0) > 0) return { L: lblsProvided as string[], M: mProvided as CorrMatrix };
    const src = series ?? mockTimeseries();
    const { labels: L0, matrix: M0 } = buildMatrix(src, labels);
    return { L: L0, M: M0 };
  }, [series, mProvided, lblsProvided, labels]);

  // Sort mode
  const [sortMode, setSortMode] = useState<'none' | 'var' | 'absCorrSum'>('none');
  const order = useMemo(() => {
    const n = L.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    if (sortMode === 'var' && series) {
      const vars = L.map(s => {
        const a = series[s] || [];
        const m = a.reduce((p, c) => p + c, 0) / Math.max(1, a.length);
        return a.reduce((p, c) => p + (c - m) * (c - m), 0) / Math.max(1, a.length);
      });
      idx.sort((i, j) => vars[j] - vars[i]);
    } else if (sortMode === 'absCorrSum') {
      const sums = L.map((_, i) => M[i].reduce((s, v) => s + Math.abs(v), 0));
      idx.sort((i, j) => sums[j] - sums[i]);
    }
    return idx;
  }, [L, M, series, sortMode]);

  // Canvas rendering
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const W = 920, H = 640;
  const padL = 140, padT = 36, padR = 16, padB = 110;

  const [hover, setHover] = useState<{ r: number; c: number; x: number; y: number } | null>(null);

  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    const n = L.length;
    if (!n) {
      ctx.font = '14px system-ui';
      ctx.fillStyle = '#6b7280';
      ctx.fillText('No data', 20, 30);
      return;
    }

    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const cw = plotW / n;
    const ch = plotH / n;

    // cells
    for (let rr = 0; rr < n; rr++) {
      for (let cc = 0; cc < n; cc++) {
        const i = order[rr], j = order[cc];
        const v = clamp(M[i][j], clampMin, clampMax);
        const x = padL + cc * cw;
        const y = padT + rr * ch;
        ctx.fillStyle = colorFor(v);
        ctx.fillRect(x, y, cw + 1, ch + 1);
      }
    }

    // diagonal stroke
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // row labels
    ctx.font = '12px system-ui';
    ctx.fillStyle = '#374151';
    ctx.textBaseline = 'middle';
    for (let rr = 0; rr < n; rr++) {
      const y = padT + rr * ch + ch / 2;
      ctx.fillText(L[order[rr]], 10, y);
    }

    // column labels (angled)
    ctx.save();
    ctx.textBaseline = 'top';
    for (let cc = 0; cc < n; cc++) {
      const x = padL + cc * cw + cw / 2;
      ctx.save();
      ctx.translate(x, padT + plotH + 8);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = '#374151';
      ctx.fillText(L[order[cc]], -30, 0);
      ctx.restore();
    }
    ctx.restore();

    // legend -1→+1
    const Lx = W - 180, Ly = padT + 6, Lw = 140, Lh = 10;
    for (let i = 0; i < Lw; i++) {
      const v = -1 + (i / (Lw - 1)) * 2;
      ctx.fillStyle = colorFor(v);
      ctx.fillRect(Lx + i, Ly, 1, Lh);
    }
    ctx.strokeStyle = '#9ca3af';
    ctx.strokeRect(Lx, Ly, Lw, Lh);
    ctx.fillStyle = '#6b7280';
    ctx.font = '11px system-ui';
    ctx.fillText('-1.0', Lx, Ly + 18);
    ctx.fillText('0', Lx + Lw / 2 - 4, Ly + 18);
    ctx.fillText('+1.0', Lx + Lw - 28, Ly + 18);
  }, [L, M, order, clampMin, clampMax]);

  // mouse hover
  function onMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;

    const n = L.length;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    if (mx < padL || mx > padL + plotW || my < padT || my > padT + plotH) { setHover(null); return; }

    const cw = plotW / n, ch = plotH / n;
    const cc = clamp(Math.floor((mx - padL) / cw), 0, n - 1);
    const rr = clamp(Math.floor((my - padT) / ch), 0, n - 1);
    setHover({ r: rr, c: cc, x: mx, y: my });
  }
  function onLeave() { setHover(null); }

  // tooltip info
  const tip = useMemo(() => {
    if (!hover) return null;
    const i = order[hover.r], j = order[hover.c];
    const v = M[i]?.[j];
    if (!Number.isFinite(v)) return null;
    return { a: L[i], b: L[j], v, x: hover.x, y: hover.y };
  }, [hover, L, M, order]);

  // exports
  function exportCSV() {
    const n = L.length;
    const header = ['Symbol', ...order.map(i => L[i])].join(',');
    const lines = order.map((ri) => {
      const row = [L[ri], ...order.map((ci) => M[ri][ci])];
      return row.join(',');
    });
    const csv = [header, ...lines].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'correlation_matrix.csv'; a.click();
    URL.revokeObjectURL(a.href);
  }
  function exportPNG() {
    const canvas = canvasRef.current; if (!canvas) return;
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = 'correlation_matrix.png';
    a.click();
  }

  return (
    <div style={S.wrap}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>{L.length}×{L.length}</span>
        </div>
        <div style={S.controls}>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Sort</span>
            <select value={sortMode} onChange={(e)=>setSortMode(e.target.value as any)} style={S.select}>
              <option value="none">None</option>
              <option value="var">By variance (desc)</option>
              <option value="absCorrSum">By |corr| sum (desc)</option>
            </select>
          </label>
          <button onClick={exportCSV} style={S.btn}>Export CSV</button>
          <button onClick={exportPNG} style={S.btn}>Export PNG</button>
        </div>
      </div>

      {/* Canvas */}
      <div style={{ position: 'relative', padding: 12 }}>
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          style={{ width: '100%', maxWidth: 1120, border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff' }}
          onMouseMove={onMove}
          onMouseLeave={onLeave}
        />
        {tip && (
          <div
            style={{
              position: 'absolute',
              left: clamp(tip.x + 14, 0, W - 160),
              top: clamp(tip.y + 10, 0, H - 80),
              background: '#111827', color: '#fff',
              padding: '6px 8px', borderRadius: 8, fontSize: 12,
              pointerEvents: 'none', boxShadow: '0 2px 6px rgba(0,0,0,0.25)',
            }}
          >
            <div><b>{tip.a}</b> × <b>{tip.b}</b></div>
            <div>ρ = {fmt(tip.v, 3)}</div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ================= Styles ================= */
const S: Record<string, React.CSSProperties> = {
  wrap: { border: '1px solid #e5e7eb', borderRadius: 16, background: '#fff', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', width: '100%', fontFamily: 'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding: 16, borderBottom: '1px solid #eee' },
  titleRow: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, flexWrap: 'wrap' },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },

  controls: { display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' },
  ctrlItem: { display: 'flex', flexDirection: 'column', gap: 6 },
  lbl: { fontSize: 12, color: '#6b7280' },
  select: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 8px', fontSize: 14, background: '#fff', minWidth: 180 },
  btn: { height: 36, padding: '0 12px', borderRadius: 10, border: '1px solid #e5e7eb', background: '#fff', cursor: 'pointer', fontSize: 13 },
};