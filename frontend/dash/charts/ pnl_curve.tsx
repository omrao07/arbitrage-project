'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */
type Row = { t: number | string; pnl?: number; ret?: number }; // t = ms epoch or ISO string
type Props = {
  title?: string;
  currency?: string;
  data?: Row[];               // if empty, a mock will be used
  fetchEndpoint?: string;     // optional GET → Row[]
  height?: number;            // total canvas height (default 420)
  startingEquity?: number;    // starting capital for equity curve (default 0)
};

/* ================= Helpers ================= */
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const fmt = (n: number, d = 2) => n.toLocaleString(undefined, { maximumFractionDigits: d });
const fmtDate = (ms: number) => new Date(ms).toLocaleDateString();

/** Generate mock daily PnL (random walk with mild drift) */
function mockData(days = 260): Row[] {
  const today = new Date();
  const arr: Row[] = [];
  let t = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  const dayMs = 24 * 3600 * 1000;
  // go back `days`
  t -= days * dayMs;
  let mu = 220;   // daily mean PnL
  let sigma = 800; // daily std PnL
  for (let i = 0; i < days; i++) {
    // skip weekends to look a bit realistic
    const d = new Date(t);
    if (d.getDay() === 0 || d.getDay() === 6) { t += dayMs; i--; continue; }
    const pnl = mu + (Math.random() - 0.5) * 2 * sigma;
    arr.push({ t, pnl: Math.round(pnl) });
    t += dayMs;
  }
  return arr;
}

/** Build time-aligned arrays & analytics from rows */
function buildSeries(rows: Row[], startingEquity = 0) {
  // normalize timestamps and pick pnl (fallback to ret*1000 if provided)
  const pts = rows
    .map(r => {
      const ts = typeof r.t === 'string' ? Date.parse(r.t) : r.t;
      const pnl = r.pnl != null ? r.pnl : (r.ret != null ? r.ret * 1000 : 0);
      return Number.isFinite(ts) ? { ts, pnl: Number(pnl || 0) } : null;
    })
    .filter((x): x is { ts: number; pnl: number } => !!x)
    .sort((a, b) => a.ts - b.ts);

  // dedupe same-day by summing pnl
  const byDay = new Map<number, number>();
  for (const p of pts) {
    const d = new Date(p.ts); d.setHours(0,0,0,0);
    const k = d.getTime();
    byDay.set(k, (byDay.get(k) || 0) + p.pnl);
  }
  const ts: number[] = [];           // day timestamps
  const pnl: number[] = [];          // daily pnl
  byDay.forEach((v, k) => { ts.push(k); pnl.push(v); });

  // equity curve
  const eq: number[] = [];
  let cum = startingEquity;
  for (const v of pnl) { cum += v; eq.push(cum); }

  // drawdown (peak-to-date)
  const peak: number[] = [];
  const dd: number[] = [];
  let pmax = eq.length ? eq[0] : 0;
  for (const e of eq) {
    pmax = Math.max(pmax, e);
    peak.push(pmax);
    dd.push(e - pmax); // negative or 0
  }
  const minDD = Math.min(0, ...dd);
  const minDDIdx = dd.indexOf(minDD);

  // simple Sharpe ≈ mean(daily pnl)/std(daily pnl)*sqrt(252)
  const mean = pnl.reduce((a, b) => a + b, 0) / Math.max(1, pnl.length);
  const std = Math.sqrt(
    pnl.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(1, pnl.length)
  );
  const sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0;

  return { ts, pnl, eq, peak, dd, sharpe, minDD, minDDIdx };
}

/* ================= Component ================= */
export default function PnlCurve({
  title = 'PnL Curve',
  currency = 'USD',
  data,
  fetchEndpoint,
  height = 420,
  startingEquity = 0,
}: Props) {
  // load or mock
  const [rows, setRows] = useState<Row[] | null>(data ?? null);
  useEffect(() => {
    let dead = false;
    if (!fetchEndpoint) {
      if (!rows) setRows(mockData());
      return;
    }
    (async () => {
      try {
        const res = await fetch(fetchEndpoint);
        const json = await res.json();
        if (!dead && Array.isArray(json)) setRows(json as Row[]);
      } catch {
        if (!dead) setRows(mockData());
      }
    })();
    return () => { dead = true; };
  }, [fetchEndpoint]);

  // derived series
  const { ts, pnl, eq, peak, dd, sharpe, minDD, minDDIdx } = useMemo(
    () => buildSeries(rows ?? mockData(), startingEquity),
    [rows, startingEquity]
  );

  // range filtering
  const [range, setRange] = useState<'1M' | '3M' | '6M' | 'YTD' | 'ALL'>('ALL');
  const { i0, i1 } = useMemo(() => {
    if (ts.length === 0) return { i0: 0, i1: -1 };
    const last = ts[ts.length - 1];
    const yearStart = new Date(new Date(last).getFullYear(), 0, 1).getTime();
    const pick = (days: number) => {
      const cutoff = last - days * 24 * 3600 * 1000;
      let idx = ts.findIndex(x => x >= cutoff);
      if (idx < 0) idx = 0;
      return { i0: idx, i1: ts.length - 1 };
    };
    switch (range) {
      case '1M': return pick(31);
      case '3M': return pick(93);
      case '6M': return pick(186);
      case 'YTD': {
        let idx = ts.findIndex(x => x >= yearStart);
        if (idx < 0) idx = 0;
        return { i0: idx, i1: ts.length - 1 };
      }
      default: return { i0: 0, i1: ts.length - 1 };
    }
  }, [ts, range]);

  // view arrays
  const T = ts.slice(i0, i1 + 1);
  const EQ = eq.slice(i0, i1 + 1);
  const DD = dd.slice(i0, i1 + 1);
  const PNL = pnl.slice(i0, i1 + 1);

  // canvas dims
  const W = 920;
  const topH = Math.floor(height * 0.74);
  const bottomH = height - topH - 20; // 20px gap
  const padL = 64, padR = 16, padT = 24, padB = 30;
  const padL2 = 64, padR2 = 16, padT2 = 6, padB2 = 26;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // scales
  function xAt(i: number, plotW: number, N: number, padLeft: number) {
    if (N <= 1) return padLeft;
    return padLeft + (i / (N - 1)) * plotW;
  }
  function yAt(v: number, vmin: number, vmax: number, plotH: number, padTop: number) {
    if (vmax - vmin <= 1e-9) return padTop + plotH / 2;
    return padTop + (1 - (v - vmin) / (vmax - vmin)) * plotH;
  }

  // hover
  const [hover, setHover] = useState<{ i: number; x: number; y: number; panel: 'eq' | 'dd' } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // guards
    const N = T.length;
    if (N === 0) {
      ctx.font = '14px system-ui';
      ctx.fillStyle = '#6b7280';
      ctx.fillText('No data', 16, 24);
      return;
    }

    /* ===== Top panel: Equity curve ===== */
    const plotW = W - padL - padR;
    const plotH = topH - padT - padB;
    const vmin = Math.min(...EQ), vmax = Math.max(...EQ);

    // grid
    ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = 0; g <= 4; g++) {
      const y = padT + (g / 4) * plotH;
      ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
    }
    ctx.stroke();

    // equity line
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = xAt(i, plotW, N, padL);
      const y = yAt(EQ[i], vmin, vmax, plotH, padT);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#111827';
    ctx.stroke();

    // high-water line (peak)
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = xAt(i, plotW, N, padL);
      const y = yAt(Math.max(...EQ.slice(0, i + 1)), vmin, vmax, plotH, padT);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#9ca3af';
    ctx.stroke();
    ctx.setLineDash([]);

    // axes
    ctx.font = '11px system-ui'; ctx.fillStyle = '#6b7280';
    // y labels
    for (let g = 0; g <= 4; g++) {
      const y = padT + (g / 4) * plotH;
      const val = vmax - (g / 4) * (vmax - vmin);
      ctx.fillText(`${fmt(val, 0)} ${currency}`, 8, y + 4);
    }
    // x labels (dates)
    const tickN = Math.min(6, N);
    for (let t = 0; t < tickN; t++) {
      const i = Math.round((t / (tickN - 1)) * (N - 1));
      const x = xAt(i, plotW, N, padL);
      ctx.fillText(fmtDate(T[i]), x - 30, padT + plotH + 18);
    }

    // annotate max drawdown point on top panel
    if (minDDIdx >= i0 && minDDIdx <= i1) {
      const iLocal = minDDIdx - i0;
      const x = xAt(iLocal, plotW, N, padL);
      const y = yAt(EQ[iLocal], vmin, vmax, plotH, padT);
      ctx.fillStyle = '#ef4444';
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    }

    /* ===== Gap between panels ===== */
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, topH, W, 20);

    /* ===== Bottom panel: Drawdown (negative) ===== */
    const plotW2 = W - padL2 - padR2;
    const plotH2 = bottomH - padT2 - padB2;
    const vmin2 = Math.min(-1, ...DD); // ensure negative range
    const vmax2 = 0;

    // grid
    ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = 0; g <= 3; g++) {
      const y = topH + 20 + padT2 + (g / 3) * plotH2;
      ctx.moveTo(padL2, y); ctx.lineTo(padL2 + plotW2, y);
    }
    ctx.stroke();

    // drawdown area
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = xAt(i, plotW2, N, padL2);
      const y = yAt(DD[i], vmin2, vmax2, plotH2, topH + 20 + padT2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    // close to zero-line
    const yZero = yAt(0, vmin2, vmax2, plotH2, topH + 20 + padT2);
    ctx.lineTo(padL2 + plotW2, yZero);
    ctx.lineTo(padL2, yZero);
    ctx.closePath();
    ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
    ctx.fill();

    // dd axis labels
    ctx.font = '11px system-ui'; ctx.fillStyle = '#6b7280';
    for (let g = 0; g <= 3; g++) {
      const y = topH + 20 + padT2 + (g / 3) * plotH2;
      const val = vmax2 - (g / 3) * (vmax2 - vmin2);
      ctx.fillText(fmt(val, 0), 8, y + 4);
    }
    // x labels already drawn on top (avoid overlap)

    // hover vertical crosshair
    if (hover && hover.i >= 0 && hover.i < N) {
      const x1 = xAt(hover.i, plotW, N, padL);
      ctx.strokeStyle = '#d1d5db';
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(x1, padT);
      ctx.lineTo(x1, padT + plotH);
      const x2 = xAt(hover.i, plotW2, N, padL2);
      ctx.moveTo(x2, topH + 20 + padT2);
      ctx.lineTo(x2, topH + 20 + padT2 + plotH2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [T, EQ, DD, PNL, currency, i0, i1, minDDIdx, topH, bottomH]);

  // mouse events
  function onMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const N = T.length;
    if (N <= 1) { setHover(null); return; }

    // detect panel
    let panel: 'eq' | 'dd' | null = null;
    if (my >= 0 && my <= topH) panel = 'eq';
    else if (my >= topH + 20 && my <= topH + 20 + bottomH) panel = 'dd';

    if (!panel) { setHover(null); return; }

    const plotW = panel === 'eq' ? (W - padL - padR) : (W - padL2 - padR2);
    const padLeft = panel === 'eq' ? padL : padL2;
    if (mx < padLeft || mx > padLeft + plotW) { setHover(null); return; }
    const rel = (mx - padLeft) / plotW;
    const i = clamp(Math.round(rel * (N - 1)), 0, N - 1);
    setHover({ i, x: mx, y: my, panel });
  }
  function onLeave() { setHover(null); }

  // exports
  function exportCSV() {
    const out = ['date,pnl,equity,drawdown'];
    for (let i = 0; i < T.length; i++) {
      out.push(`${new Date(T[i]).toISOString().slice(0,10)},${PNL[i] ?? 0},${EQ[i] ?? 0},${DD[i] ?? 0}`);
    }
    const blob = new Blob([out.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'pnl_curve.csv'; a.click();
    URL.revokeObjectURL(a.href);
  }
  function exportPNG() {
    const c = canvasRef.current; if (!c) return;
    const a = document.createElement('a');
    a.href = c.toDataURL('image/png'); a.download = 'pnl_curve.png'; a.click();
  }

  // header KPIs
  const totalPnL = (EQ[EQ.length - 1] ?? 0) - (EQ[0] ?? 0);
  const maxDDAbs = Math.abs(Math.min(...DD, 0));
  const hoverInfo = useMemo(() => {
    if (!hover) return null;
    const i = hover.i;
    return {
      date: T[i] ? fmtDate(T[i]) : '',
      pnl: PNL[i] ?? 0,
      eq: EQ[i] ?? 0,
      dd: DD[i] ?? 0,
    };
  }, [hover, T, PNL, EQ, DD]);

  return (
    <div style={S.wrap}>
      {/* Header & controls */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>Sharpe ≈ {fmt(sharpe, 2)}</span>
          <span style={{ ...S.badge, background: '#eef2ff', color: '#3730a3' }}>
            Total PnL: {fmt(totalPnL, 0)} {currency}
          </span>
          <span style={{ ...S.badge, background: '#fef2f2', color: '#991b1b' }}>
            Max DD: {fmt(maxDDAbs, 0)} {currency}
          </span>
        </div>

        <div style={S.controls}>
          <div style={S.rangeBtns}>
            {(['1M','3M','6M','YTD','ALL'] as const).map(r => (
              <button
                key={r}
                onClick={()=>setRange(r)}
                style={{ ...S.btn, ...(range===r ? S.btnActive : {}) }}
              >{r}</button>
            ))}
          </div>
          <button onClick={exportCSV} style={S.btn}>Export CSV</button>
          <button onClick={exportPNG} style={S.btn}>Export PNG</button>
        </div>
      </div>

      {/* Canvas */}
      <div style={{ position: 'relative', padding: 12 }}>
        <canvas
          ref={canvasRef}
          width={W}
          height={height}
          style={{ width: '100%', maxWidth: 1120, border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff' }}
          onMouseMove={onMove}
          onMouseLeave={onLeave}
        />

        {/* Tooltip */}
        {hoverInfo && (
          <div
            style={{
              position: 'absolute',
              left: clamp((hover?.x ?? 0) + 16, 0, W - 160),
              top: clamp((hover?.y ?? 0) + 10, 0, height - 80),
              background: '#111827', color: '#fff',
              borderRadius: 10, padding: '8px 10px', fontSize: 12,
              pointerEvents: 'none', boxShadow: '0 2px 6px rgba(0,0,0,0.25)',
            }}
          >
            <div style={{ fontWeight: 700 }}>{hoverInfo.date}</div>
            <div>Equity: {fmt(hoverInfo.eq, 0)} {currency}</div>
            <div>PnL: {fmt(hoverInfo.pnl, 0)} {currency}</div>
            <div>Drawdown: {fmt(Math.abs(hoverInfo.dd), 0)} {currency}</div>
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
  titleRow: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 10 },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },

  controls: { display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' },
  rangeBtns: { display: 'flex', gap: 6, alignItems: 'center' },
  btn: { height: 32, padding: '0 10px', borderRadius: 10, border: '1px solid #e5e7eb', background: '#fff', cursor: 'pointer', fontSize: 13 },
  btnActive: { background: '#111', color: '#fff', borderColor: 'transparent' },
};