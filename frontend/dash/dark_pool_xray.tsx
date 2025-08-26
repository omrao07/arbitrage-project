'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* =============== Types =============== */
type Print = {
  t: number;                 // epoch ms
  venue: string;             // e.g., "ATS:SIGMA", "TRF:FINRA"
  qty: number;               // shares/contracts
  price: number;
  side?: 'buy' | 'sell';     // optional; if missing we'll infer randomly for mock
};

type Props = {
  title?: string;
  symbol?: string;
  fetchEndpoint?: string;                    // GET ?symbol=AAPL -> Print[]
  rowsPerPage?: number;
};

/* =============== Helpers =============== */
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const fmt0 = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 0 });
const fmt2 = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 2 });
const dayMS = 24 * 3600 * 1000;

/* Mock generator: one session of dark prints with ATS labels */
function mock(symbol = 'AAPL', px = 225): Print[] {
  const open = new Date(); open.setHours(9, 30, 0, 0);
  const base = +open - 3 * dayMS; // pretend 3 days ago session, but we only show intraday
  const prints: Print[] = [];
  const venues = ['ATS:SIGMA', 'ATS:UBS', 'ATS:CSGP', 'ATS:JANE', 'TRF:FINRA', 'ATS:CITADEL', 'ATS:G1X'];
  for (let i = 0; i < 900; i++) {
    const t = base + 9.5 * 3600 * 1000 + Math.floor(Math.random() * (6.5 * 3600 * 1000)); // 9:30 to 16:00
    const drift = (Math.sin(i / 60) + Math.cos(i / 47)) * 0.6;
    const price = +(px + drift + (Math.random() - 0.5) * 0.8).toFixed(2);
    const qty = Math.max(50, Math.round(50 + Math.random() * 5000));
    const venue = venues[Math.floor(Math.random() * venues.length)];
    const side = Math.random() < 0.5 ? 'buy' : 'sell';
    prints.push({ t, venue, qty, price, side });
  }
  return prints.sort((a, b) => a.t - b.t);
}

/* Group by venue */
function groupByVenue(prints: Print[]) {
  const map = new Map<string, { notional: number; volume: number; trades: number }>();
  for (const p of prints) {
    const n = p.price * p.qty;
    const cur = map.get(p.venue) ?? { notional: 0, volume: 0, trades: 0 };
    cur.notional += n; cur.volume += p.qty; cur.trades += 1;
    map.set(p.venue, cur);
  }
  const rows = Array.from(map.entries()).map(([venue, v]) => ({ venue, ...v }))
    .sort((a, b) => b.notional - a.notional);
  const totalNotional = rows.reduce((s, r) => s + r.notional, 0);
  rows.forEach(r => (r as any).pct = totalNotional ? r.notional / totalNotional : 0);
  return { rows, totalNotional };
}

/* 15-minute buckets */
function intradayBuckets(prints: Print[]) {
  if (!prints.length) return { xs: [] as number[], buy: [] as number[], sell: [] as number[] };
  // align to the session start (09:30)
  const d0 = new Date(prints[0].t); d0.setHours(9, 30, 0, 0);
  const start = d0.getTime();
  const bins = 26; // 6.5h * 4
  const buy = Array(bins).fill(0);
  const sell = Array(bins).fill(0);
  const xs = Array.from({ length: bins }, (_, i) => start + i * 15 * 60 * 1000);
  for (const p of prints) {
    const idx = clamp(Math.floor((p.t - start) / (15 * 60 * 1000)), 0, bins - 1);
    const notional = p.price * p.qty;
    if ((p.side ?? (Math.random() < 0.5 ? 'buy' : 'sell')) === 'buy') buy[idx] += notional;
    else sell[idx] += notional;
  }
  return { xs, buy, sell };
}

/* =============== Component =============== */
export default function DarkPoolXray({
  title = 'Dark Pool X-Ray',
  symbol: symProp = 'AAPL',
  fetchEndpoint,
  rowsPerPage = 12,
}: Props) {
  const [symbol, setSymbol] = useState(symProp);
  const [prints, setPrints] = useState<Print[] | null>(null);

  // Load
  useEffect(() => {
    let dead = false;
    (async () => {
      if (!fetchEndpoint) { setPrints(mock(symbol)); return; }
      try {
        const q = new URLSearchParams({ symbol });
        const res = await fetch(`${fetchEndpoint}?${q.toString()}`);
        const json = await res.json();
        if (!dead && Array.isArray(json)) setPrints(json as Print[]);
        else if (!dead) setPrints(mock(symbol));
      } catch {
        if (!dead) setPrints(mock(symbol));
      }
    })();
    return () => { dead = true; };
  }, [fetchEndpoint, symbol]);

  const data = prints ?? mock(symbol);

  /* Derived stats */
  const { rows: venueRows, totalNotional } = useMemo(() => groupByVenue(data), [data]);
  const totalVol = useMemo(() => data.reduce((s, p) => s + p.qty, 0), [data]);
  const vwap = useMemo(() => {
    const n = data.reduce((s, p) => s + p.price * p.qty, 0);
    return totalVol ? n / totalVol : 0;
  }, [data, totalVol]);

  const imbalance = useMemo(() => {
    let buyN = 0, sellN = 0;
    for (const p of data) ((p.side ?? 'buy') === 'buy') ? buyN += p.price * p.qty : sellN += p.price * p.qty;
    const tot = buyN + sellN;
    return { buyN, sellN, pctBuy: tot ? buyN / tot : 0 };
  }, [data]);

  const { xs, buy, sell } = useMemo(() => intradayBuckets(data), [data]);

  /* Charts */
  const barRef = useRef<HTMLCanvasElement | null>(null);
  const histRef = useRef<HTMLCanvasElement | null>(null);

  // Venue bars (top 10)
  useEffect(() => {
    const c = barRef.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);

    const top = venueRows.slice(0, 10);
    const W = 920, H = 280, padL = 120, padR = 16, padT = 24, padB = 40;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    const maxV = Math.max(1, ...top.map(r => r.notional));
    const bw = plotW / Math.max(1, top.length);

    // grid
    ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = 0; g <= 4; g++) {
      const y = padT + (g / 4) * plotH;
      ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
    }
    ctx.stroke();

    // bars
    for (let i = 0; i < top.length; i++) {
      const v = top[i].notional;
      const h = (v / maxV) * plotH;
      const x = padL + i * bw + 8;
      const y = padT + (plotH - h);
      ctx.fillStyle = 'rgba(37,99,235,0.75)';
      ctx.fillRect(x, y, bw - 16, h);
      // label
      ctx.save();
      ctx.translate(x + (bw - 16) / 2, padT + plotH + 14);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = '#374151';
      ctx.font = '12px system-ui';
      ctx.fillText(top[i].venue, -40, 0);
      ctx.restore();
    }

    // y labels
    ctx.fillStyle = '#6b7280'; ctx.font = '11px system-ui';
    for (let g = 0; g <= 4; g++) {
      const val = (g / 4) * maxV;
      const y = padT + (1 - g / 4) * plotH;
      ctx.fillText(fmt0(val), 8, y + 4);
    }
    // title
    ctx.fillStyle = '#111827'; ctx.font = '13px system-ui';
    ctx.fillText('Top venues by notional', padL, padT - 8);
  }, [venueRows]);

  // Intraday histogram (stacked buy/sell)
  useEffect(() => {
    const c = histRef.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);

    const W = 920, H = 260, padL = 60, padR = 16, padT = 20, padB = 44;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const n = xs.length; if (!n) return;
    const bw = plotW / n;
    const maxV = Math.max(1, ...xs.map((_, i) => buy[i] + sell[i]));

    // grid
    ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = 0; g <= 3; g++) {
      const y = padT + (g / 3) * plotH;
      ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
    }
    ctx.stroke();

    // stacked bars
    for (let i = 0; i < n; i++) {
      const x = padL + i * bw + 1;
      const y0 = padT + plotH;
      const hBuy = (buy[i] / maxV) * plotH;
      const hSell = (sell[i] / maxV) * plotH;
      // sell (red) at bottom
      ctx.fillStyle = 'rgba(239,68,68,0.6)';
      ctx.fillRect(x, y0 - hSell, bw - 2, hSell);
      // buy (green) on top
      ctx.fillStyle = 'rgba(16,185,129,0.75)';
      ctx.fillRect(x, y0 - hSell - hBuy, bw - 2, hBuy);
    }

    // x ticks (every ~hour)
    ctx.fillStyle = '#6b7280'; ctx.font = '11px system-ui';
    const every = Math.max(1, Math.floor(n / 6));
    for (let i = 0; i < n; i += every) {
      const d = new Date(xs[i]);
      const label = `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
      const x = padL + i * bw;
      ctx.fillText(label, x - 12, padT + plotH + 16);
    }
    // y labels
    for (let g = 0; g <= 3; g++) {
      const val = (g / 3) * maxV;
      const y = padT + (1 - g / 3) * plotH;
      ctx.fillText(fmt0(val), 8, y + 4);
    }
    // title
    ctx.fillStyle = '#111827'; ctx.font = '13px system-ui';
    ctx.fillText('Intraday dark notional (15-min, buy/sell)', padL, padT - 6);
  }, [xs, buy, sell]);

  /* Exports */
  function exportCSV() {
    const header = 'ts,venue,qty,price,side,notional';
    const lines = data.map(p => [
      new Date(p.t).toISOString(),
      p.venue,
      p.qty,
      p.price.toFixed(2),
      p.side ?? '',
      (p.price * p.qty).toFixed(2),
    ].join(','));
    const csv = [header, ...lines].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = `${symbol}_dark_pool.csv`; a.click();
    URL.revokeObjectURL(a.href);
  }
  function exportPNGs() {
    // combine both canvases onto a temporary canvas
    const b1 = barRef.current, b2 = histRef.current;
    if (!b1 || !b2) return;
    const W = 940, H = 560;
    const temp = document.createElement('canvas');
    temp.width = W; temp.height = H;
    const ctx = temp.getContext('2d'); if (!ctx) return;
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
    ctx.drawImage(b1, 10, 10);
    ctx.drawImage(b2, 10, 300);
    const a = document.createElement('a');
    a.href = temp.toDataURL('image/png'); a.download = `${symbol}_dark_xray.png`; a.click();
  }

  /* Table (paged) */
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil(data.length / rowsPerPage));
  const rows = data.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  return (
    <div style={S.wrap}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>{symbol}</span>
          <span style={{ ...S.badge, background: '#eef2ff', color: '#3730a3' }}>
            VWAP: {fmt2(vwap)}
          </span>
          <span style={{ ...S.badge, background: '#ecfeff', color: '#0369a1' }}>
            Notional: {fmt0(totalNotional)}
          </span>
          <span style={{ ...S.badge, background: '#fef2f2', color: '#991b1b' }}>
            Buy%: {(imbalance.pctBuy * 100).toFixed(1)}%
          </span>
        </div>
        <div style={S.controls}>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Symbol</span>
            <input value={symbol} onChange={(e)=>setSymbol(e.target.value.toUpperCase())} style={S.input}/>
          </label>
          <button onClick={exportCSV} style={S.btn}>Export CSV</button>
          <button onClick={exportPNGs} style={S.btn}>Export PNG</button>
        </div>
      </div>

      {/* Charts */}
      <div style={{ padding: 12, display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
        <div style={S.card}>
          <canvas ref={barRef} width={920} height={280} style={S.canvas}/>
        </div>
        <div style={S.card}>
          <canvas ref={histRef} width={920} height={260} style={S.canvas}/>
        </div>
      </div>

      {/* Venue table */}
      <div style={{ padding: 12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>Top venues</div>
          <div style={{ overflow: 'auto' }}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={{...S.th, textAlign:'left'}}>Venue</th>
                  <th style={S.th}>Trades</th>
                  <th style={S.th}>Volume</th>
                  <th style={S.th}>Notional</th>
                  <th style={S.th}>Share</th>
                </tr>
              </thead>
              <tbody>
                {venueRows.map((r, i) => (
                  <tr key={r.venue} style={{ background: i === 0 ? '#f8fafc' : '#fff' }}>
                    <td style={{...S.td, textAlign:'left', fontWeight:600}}>{r.venue}</td>
                    <td style={S.td}>{fmt0(r.trades)}</td>
                    <td style={S.td}>{fmt0(r.volume)}</td>
                    <td style={S.td}>{fmt0(r.notional)}</td>
                    <td style={S.td}>{(r as any).pct ? ((r as any).pct * 100).toFixed(1) + '%' : '-'}</td>
                  </tr>
                ))}
                {venueRows.length === 0 && <tr><td style={S.td} colSpan={5}>No prints</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Raw prints table (paged) */}
      <div style={{ padding: 12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>
            Raw dark prints
            <div style={{ marginLeft: 'auto', display: 'inline-flex', gap: 8 }}>
              <button onClick={()=>setPage(p => Math.max(0, p-1))} style={S.btnSmall}>Prev</button>
              <span style={S.pageLabel}>{page+1} / {pageCount}</span>
              <button onClick={()=>setPage(p => Math.min(pageCount-1, p+1))} style={S.btnSmall}>Next</button>
            </div>
          </div>
          <div style={{ overflow: 'auto' }}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>Time</th>
                  <th style={{...S.th, textAlign:'left'}}>Venue</th>
                  <th style={S.th}>Side</th>
                  <th style={S.th}>Qty</th>
                  <th style={S.th}>Price</th>
                  <th style={S.th}>Notional</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((p, i) => {
                  const d = new Date(p.t);
                  const ts = `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
                  const notional = p.price * p.qty;
                  const sideColor = (p.side ?? 'buy') === 'buy' ? '#065f46' : '#991b1b';
                  const sideBg = (p.side ?? 'buy') === 'buy' ? '#ecfdf5' : '#fef2f2';
                  return (
                    <tr key={i}>
                      <td style={S.td}>{ts}</td>
                      <td style={{...S.td, textAlign:'left'}}>{p.venue}</td>
                      <td style={S.td}>
                        <span style={{ padding:'2px 8px', borderRadius:999, border:'1px solid #e5e7eb', background:sideBg, color:sideColor, fontSize:11, fontWeight:700 }}>
                          {(p.side ?? 'buy').toUpperCase()}
                        </span>
                      </td>
                      <td style={S.td}>{fmt0(p.qty)}</td>
                      <td style={S.td}>{fmt2(p.price)}</td>
                      <td style={S.td}>{fmt0(notional)}</td>
                    </tr>
                  );
                })}
                {rows.length === 0 && <tr><td style={S.td} colSpan={6}>No data</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

/* =============== Styles =============== */
const S: Record<string, React.CSSProperties> = {
  wrap: { border: '1px solid #e5e7eb', borderRadius: 16, background: '#fff', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', width: '100%', fontFamily: 'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding: 16, borderBottom: '1px solid #eee' },
  titleRow: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 10 },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },

  controls: { display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' },
  ctrlItem: { display: 'flex', flexDirection: 'column', gap: 6 },
  lbl: { fontSize: 12, color: '#6b7280' },
  input: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 10px', fontSize: 14, outline: 'none', minWidth: 140 },
  btn: { height: 36, padding: '0 12px', borderRadius: 10, border: '1px solid #e5e7eb', background: '#fff', cursor: 'pointer', fontSize: 13 },
  btnSmall: { height: 28, padding: '0 10px', borderRadius: 8, border: '1px solid #e5e7eb', background: '#fff', cursor: 'pointer', fontSize: 12 },
  pageLabel: { display: 'inline-flex', alignItems: 'center', height: 28, padding: '0 8px', borderRadius: 8, background: '#f3f4f6', fontSize: 12, fontWeight: 700, color: '#111827' },

  card: { border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff', overflow: 'hidden', padding: 8 },
  cardHeader: { padding: '10px 12px', borderBottom: '1px solid #eee', fontWeight: 700, fontSize: 14, color: '#111827', display: 'flex', alignItems: 'center' },
  canvas: { width: '100%', maxWidth: 1120, borderRadius: 8, background: '#fff' },

  table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0 },
  th: { textAlign: 'right', padding: '8px 10px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #eee', position: 'sticky' as any, top: 0, background: '#fff' },
  td: { textAlign: 'right', padding: '8px 10px', fontSize: 13, color: '#111827', borderBottom: '1px solid #f3f4f6' },
};