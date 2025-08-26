'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ================= Types ================= */
type Side = 'buy' | 'sell';
type Trade = {
  ts: number;         // epoch ms
  symbol: string;
  side: Side;
  qty: number;        // positive numbers
  price: number;      // trade price
  fee?: number;       // optional fees/commissions in base ccy
  venue?: string;
  strategy?: string;
  orderId?: string;
};

type Props = {
  title?: string;
  fetchEndpoint?: string;          // GET with optional query -> Trade[]
  initialSymbol?: string;
  rowsPerPage?: number;
};

/* ================= Mock data ================= */
function mockTrades(symbol = 'AAPL'): Trade[] {
  const start = new Date(); start.setHours(9, 30, 0, 0);
  const base = +start - 2 * 24 * 3600 * 1000; // two days ago 09:30
  const out: Trade[] = [];
  let px = 180;
  for (let i = 0; i < 240; i++) {
    // one trade ~ every 2 minutes
    const ts = base + i * (2 * 60 * 1000);
    const drift = Math.sin(i / 20) * 0.4;
    const noise = (Math.random() - 0.5) * 1.2;
    px = Math.max(1, +(px + drift + noise).toFixed(2));
    const side: Side = Math.random() < 0.53 ? 'buy' : 'sell';
    const qty = Math.max(1, Math.round(10 + Math.random() * 90));
    out.push({ ts, symbol, side, qty, price: px, fee: 0.01 * qty, venue: 'SIM', strategy: 'demo' });
  }
  // shuffle a second symbol to test filters
  return out.concat(
    mockTradesAlt('MSFT', base + 24 * 3600 * 1000) // next day
  );
}
function mockTradesAlt(symbol: string, base: number): Trade[] {
  const out: Trade[] = [];
  let px = 310;
  for (let i = 0; i < 180; i++) {
    const ts = base + i * (3 * 60 * 1000);
    px = Math.max(1, +(px + (Math.random() - 0.5) * 1.1).toFixed(2));
    const side: Side = Math.random() < 0.5 ? 'buy' : 'sell';
    const qty = Math.max(1, Math.round(5 + Math.random() * 40));
    out.push({ ts, symbol, side, qty, price: px, fee: 0.01 * qty, venue: 'SIM', strategy: 'demo' });
  }
  return out;
}

/* ================= Helpers ================= */
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const fmt0 = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 0 });
const fmt2 = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 2 });

/** Aggregate to equity & positions per symbol (simple FIFO realized PnL + mark to last trade) */
function computeEquity(trades: Trade[]) {
  // separate per symbol
  const bySym = new Map<string, Trade[]>();
  for (const t of trades) {
    const arr = bySym.get(t.symbol) ?? [];
    arr.push(t);
    bySym.set(t.symbol, arr);
  }

  type EquityPoint = { ts: number; total: number; realized: number; unrealized: number; pos: number; avg: number; last: number };
  const curves = new Map<string, EquityPoint[]>();

  for (const [sym, arr0] of bySym.entries()) {
    const arr = [...arr0].sort((a, b) => a.ts - b.ts);
    // FIFO inventory
    type Lot = { qty: number; px: number };
    const lots: Lot[] = [];
    let realized = 0;
    let pos = 0;
    let last = arr.length ? arr[0].price : 0;
    const pts: EquityPoint[] = [];

    for (const t of arr) {
      last = t.price;

      if (t.side === 'buy') {
        lots.push({ qty: t.qty, px: t.price });
        pos += t.qty;
        realized -= t.fee ?? 0;
      } else {
        // sell: close from FIFO lots
        let qty = t.qty;
        let proceeds = t.price * qty;
        let cost = 0;
        while (qty > 0 && lots.length) {
          const lot = lots[0];
          const take = Math.min(qty, lot.qty);
          cost += take * lot.px;
          lot.qty -= take;
          qty -= take;
          if (lot.qty <= 0) lots.shift();
        }
        // if we sold more than we had (short) — treat extra as short-open lot at sale px
        if (qty > 0) {
          // opening short lot at t.price
          lots.unshift({ qty: -qty, px: t.price }); // negative qty lot to represent short
          pos -= qty;
          cost += 0; // no cost on opening short; realized comes when covering
        } else {
          pos -= t.qty;
        }
        realized += (proceeds - cost) - (t.fee ?? 0);
      }

      // compute weighted avg price of current pos (positive long; if net short, avg based on negative)
      let sumQty = 0, sumNotional = 0;
      for (const l of lots) { sumQty += l.qty; sumNotional += l.qty * l.px; }
      const avg = sumQty !== 0 ? sumNotional / sumQty : 0;

      // unrealized: position * (last - avg)
      const unrealized = sumQty * (last - avg);
      const total = realized + unrealized;

      pts.push({ ts: t.ts, total, realized, unrealized, pos: sumQty, avg, last });
    }
    curves.set(sym, pts);
  }

  return curves; // per symbol: array of points over time
}

/* ================= Component ================= */
export default function VisualTradeHistory({
  title = 'Visual Trade History',
  fetchEndpoint,
  initialSymbol,
  rowsPerPage = 20,
}: Props) {
  const [trades, setTrades] = useState<Trade[] | null>(null);
  const [symbol, setSymbol] = useState(initialSymbol ?? 'AAPL');
  const [from, setFrom] = useState<string>(''); // ISO date local
  const [to, setTo] = useState<string>('');

  // Load or mock
  useEffect(() => {
    let dead = false;
    (async () => {
      if (!fetchEndpoint) {
        setTrades(mockTrades());
        return;
      }
      try {
        const q = new URLSearchParams();
        if (symbol) q.set('symbol', symbol);
        if (from) q.set('from', String(new Date(from).getTime()));
        if (to) q.set('to', String(new Date(to).getTime()));
        const res = await fetch(`${fetchEndpoint}?${q.toString()}`);
        const json = await res.json();
        if (!dead && Array.isArray(json)) setTrades(json as Trade[]);
        else if (!dead) setTrades(mockTrades());
      } catch {
        if (!dead) setTrades(mockTrades());
      }
    })();
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchEndpoint, symbol, from, to]);

  const allTrades = trades ?? mockTrades();
  const symbols = useMemo(() => Array.from(new Set(allTrades.map(t => t.symbol))), [allTrades]);

  // filter
  const filtered = useMemo(() => {
    let arr = allTrades;
    if (symbol) arr = arr.filter(t => t.symbol === symbol);
    const fromMs = from ? new Date(from).getTime() : -Infinity;
    const toMs = to ? new Date(to).getTime() : Infinity;
    arr = arr.filter(t => t.ts >= fromMs && t.ts <= toMs);
    return arr.sort((a, b) => a.ts - b.ts);
  }, [allTrades, symbol, from, to]);

  // equity
  const curves = useMemo(() => computeEquity(filtered), [filtered]);
  const pts = curves.get(symbol) ?? [];
  const realized = pts.length ? pts[pts.length - 1].realized : 0;
  const unrealized = pts.length ? pts[pts.length - 1].unrealized : 0;
  const total = pts.length ? pts[pts.length - 1].total : 0;
  const pos = pts.length ? pts[pts.length - 1].pos : 0;
  const avg = pts.length ? pts[pts.length - 1].avg : 0;
  const last = pts.length ? pts[pts.length - 1].last : 0;

  /* ================= Canvases ================= */
  const curveRef = useRef<HTMLCanvasElement | null>(null);
  const scatterRef = useRef<HTMLCanvasElement | null>(null);
  const curveW = 920, curveH = 320;
  const padL = 60, padR = 16, padT = 20, padB = 44;

  // Equity curve with markers
  useEffect(() => {
    const c = curveRef.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);

    const plotW = curveW - padL - padR;
    const plotH = curveH - padT - padB;

    if (!pts.length) {
      ctx.fillStyle = '#6b7280'; ctx.font = '13px system-ui';
      ctx.fillText('No trades in selected range.', 20, 28);
      return;
    }

    const xs = pts.map(p => p.ts);
    const ys = pts.map(p => p.total);
    const minX = xs[0], maxX = xs[xs.length - 1];
    const minY = Math.min(...ys, 0), maxY = Math.max(...ys, 1);

    const xAt = (t: number) => padL + ((t - minX) / Math.max(1, maxX - minX)) * plotW;
    const yAt = (v: number) => padT + (1 - (v - minY) / Math.max(1e-9, (maxY - minY))) * plotH;

    // grid
    ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = 0; g <= 4; g++) {
      const y = padT + (g / 4) * plotH;
      ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
    }
    ctx.stroke();

    // zero line
    const y0 = yAt(0);
    ctx.setLineDash([4,4]); ctx.strokeStyle = '#9ca3af';
    ctx.beginPath(); ctx.moveTo(padL, y0); ctx.lineTo(padL + plotW, y0); ctx.stroke();
    ctx.setLineDash([]);

    // equity line
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const x = xAt(pts[i].ts), y = yAt(pts[i].total);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#111827'; ctx.lineWidth = 2; ctx.stroke();

    // trades markers
    const tri = (x: number, y: number, side: Side) => {
      ctx.beginPath();
      if (side === 'buy') { // up triangle
        ctx.moveTo(x, y - 8); ctx.lineTo(x - 7, y + 6); ctx.lineTo(x + 7, y + 6);
      } else { // down triangle
        ctx.moveTo(x, y + 8); ctx.lineTo(x - 7, y - 6); ctx.lineTo(x + 7, y - 6);
      }
      ctx.closePath();
      ctx.fillStyle = side === 'buy' ? '#10b981' : '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();
    };
    for (const t of filtered) {
      const pi = nearestIndex(xs, t.ts);
      const x = xAt(t.ts);
      const y = yAt(pts[pi]?.total ?? 0);
      tri(x, y, t.side);
    }

    // axes labels
    ctx.fillStyle = '#6b7280'; ctx.font = '11px system-ui';
    const ticks = 6;
    for (let t = 0; t < ticks; t++) {
      const tt = minX + (t / (ticks - 1)) * (maxX - minX);
      const d = new Date(tt);
      const label = `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
      const x = xAt(tt);
      ctx.fillText(label, x - 12, padT + plotH + 16);
    }
    for (let g = 0; g <= 4; g++) {
      const val = minY + (g / 4) * (maxY - minY);
      const y = yAt(val);
      ctx.fillText(fmt0(val), 8, y + 4);
    }
  }, [pts, filtered]);

  // Scatter: price vs time with colored buys/sells (secondary visual)
  useEffect(() => {
    const c = scatterRef.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);

    const plotW = curveW - padL - padR;
    const plotH = 220 - padT - padB + 20;

    if (!filtered.length) {
      ctx.fillStyle = '#6b7280'; ctx.font = '13px system-ui';
      ctx.fillText('No trades in selected range.', 20, 28);
      return;
    }

    const xs = filtered.map(t => t.ts);
    const ps = filtered.map(t => t.price);
    const minX = xs[0], maxX = xs[xs.length - 1];
    const minP = Math.min(...ps), maxP = Math.max(...ps);

    const xAt = (t: number) => padL + ((t - minX) / Math.max(1, maxX - minX)) * plotW;
    const yAt = (v: number) => padT + (1 - (v - minP) / Math.max(1e-9, (maxP - minP))) * plotH;

    // grid
    ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = 0; g <= 3; g++) {
      const y = padT + (g / 3) * plotH;
      ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
    }
    ctx.stroke();

    // dots
    for (const t of filtered) {
      const x = xAt(t.ts), y = yAt(t.price);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = (t.side === 'buy') ? 'rgba(16,185,129,0.85)' : 'rgba(239,68,68,0.85)';
      ctx.fill();
    }

    // axes
    ctx.fillStyle = '#6b7280'; ctx.font = '11px system-ui';
    const ticks = 6;
    for (let t = 0; t < ticks; t++) {
      const tt = minX + (t / (ticks - 1)) * (maxX - minX);
      const d = new Date(tt);
      const label = `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
      const x = xAt(tt);
      ctx.fillText(label, x - 12, padT + plotH + 16);
    }
    for (let g = 0; g <= 3; g++) {
      const val = minP + (g / 3) * (maxP - minP);
      const y = yAt(val);
      ctx.fillText(fmt2(val), 8, y + 4);
    }
  }, [filtered]);

  /* ============== Table (paged) ============== */
  const [page, setPage] = useState(0);
  const rowsPer = rowsPerPage;
  const pageCount = Math.max(1, Math.ceil(filtered.length / rowsPer));
  const rows = filtered.slice(page * rowsPer, page * rowsPer + rowsPer);

  /* ============== Exports ============== */
  function exportCSV() {
    const head = 'ts,symbol,side,qty,price,fee,venue,strategy,notional';
    const lines = filtered.map(t =>
      [new Date(t.ts).toISOString(), t.symbol, t.side, t.qty, t.price.toFixed(2), t.fee ?? '', t.venue ?? '', t.strategy ?? '', (t.qty * t.price).toFixed(2)].join(',')
    );
    const csv = [head, ...lines].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = `trades_${symbol}.csv`; a.click();
    URL.revokeObjectURL(a.href);
  }
  function exportPNG() {
    const c1 = curveRef.current, c2 = scatterRef.current;
    if (!c1 || !c2) return;
    const W = 940, H = 600;
    const tmp = document.createElement('canvas'); tmp.width = W; tmp.height = H;
    const ctx = tmp.getContext('2d'); if (!ctx) return;
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
    ctx.drawImage(c1, 10, 10);
    ctx.drawImage(c2, 10, 340);
    const a = document.createElement('a');
    a.href = tmp.toDataURL('image/png'); a.download = `trade_history_${symbol}.png`; a.click();
  }

  /* ============== UI ============== */
  return (
    <div style={S.wrap}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>{symbol}</span>
          <span style={{...S.badge, background:'#ecfeff', color:'#0369a1'}}>Total: {fmt0(total)}</span>
          <span style={{...S.badge, background:'#eef2ff', color:'#3730a3'}}>Realized: {fmt0(realized)}</span>
          <span style={{...S.badge, background:'#fef3c7', color:'#92400e'}}>Unrealized: {fmt0(unrealized)}</span>
          <span style={{...S.badge, background:'#fef2f2', color:'#991b1b'}}>Pos: {fmt0(pos)}</span>
          <span style={S.badgeSmall}>Avg: {fmt2(avg)}</span>
          <span style={S.badgeSmall}>Last: {fmt2(last)}</span>
        </div>
        <div style={S.controls}>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Symbol</span>
            <select value={symbol} onChange={(e)=>{ setSymbol(e.target.value); setPage(0); }} style={S.select}>
              {[...new Set([symbol, ...symbols])].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>From</span>
            <input type="datetime-local" value={from} onChange={(e)=>{ setFrom(e.target.value); setPage(0); }} style={S.input}/>
          </label>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>To</span>
            <input type="datetime-local" value={to} onChange={(e)=>{ setTo(e.target.value); setPage(0); }} style={S.input}/>
          </label>
          <button onClick={exportCSV} style={S.btn}>Export CSV</button>
          <button onClick={exportPNG} style={S.btn}>Export PNG</button>
        </div>
      </div>

      {/* Charts */}
      <div style={{ padding: 12, display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>Equity curve (total PnL) & trade markers</div>
          <canvas ref={curveRef} width={curveW} height={curveH} style={S.canvas}/>
        </div>
        <div style={S.card}>
          <div style={S.cardHeader}>Trade prices (buy/sell)</div>
          <canvas ref={scatterRef} width={curveW} height={260} style={S.canvas}/>
        </div>
      </div>

      {/* Trades Table */}
      <div style={{ padding: 12 }}>
        <div style={S.card}>
          <div style={S.cardHeader}>
            Trades
            <div style={{ marginLeft: 'auto', display: 'inline-flex', gap: 8 }}>
              <button onClick={()=>setPage(p=>Math.max(0,p-1))} style={S.btnSmall}>Prev</button>
              <span style={S.pageLabel}>{page+1} / {pageCount}</span>
              <button onClick={()=>setPage(p=>Math.min(pageCount-1,p+1))} style={S.btnSmall}>Next</button>
            </div>
          </div>
          <div style={{ overflow: 'auto' }}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>Time</th>
                  <th style={{...S.th, textAlign:'left'}}>Symbol</th>
                  <th style={S.th}>Side</th>
                  <th style={S.th}>Qty</th>
                  <th style={S.th}>Price</th>
                  <th style={S.th}>Notional</th>
                  <th style={{...S.th, textAlign:'left'}}>Venue</th>
                  <th style={{...S.th, textAlign:'left'}}>Strategy</th>
                  <th style={S.th}>Fee</th>
                  <th style={{...S.th, textAlign:'left'}}>Order ID</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((t, i) => {
                  const d = new Date(t.ts);
                  const ts = `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
                  const notional = t.qty * t.price;
                  const sideColor = t.side === 'buy' ? '#065f46' : '#991b1b';
                  const sideBg = t.side === 'buy' ? '#ecfdf5' : '#fef2f2';
                  return (
                    <tr key={i}>
                      <td style={S.td}>{ts}</td>
                      <td style={{...S.td, textAlign:'left', fontWeight:600}}>{t.symbol}</td>
                      <td style={S.td}>
                        <span style={{ padding:'2px 8px', borderRadius:999, border:'1px solid #e5e7eb', background:sideBg, color:sideColor, fontSize:11, fontWeight:700 }}>
                          {t.side.toUpperCase()}
                        </span>
                      </td>
                      <td style={S.td}>{fmt0(t.qty)}</td>
                      <td style={S.td}>{fmt2(t.price)}</td>
                      <td style={S.td}>{fmt0(notional)}</td>
                      <td style={{...S.td, textAlign:'left'}}>{t.venue ?? '-'}</td>
                      <td style={{...S.td, textAlign:'left'}}>{t.strategy ?? '-'}</td>
                      <td style={S.td}>{t.fee != null ? fmt2(t.fee) : '-'}</td>
                      <td style={{...S.td, textAlign:'left'}}>{t.orderId ?? '-'}</td>
                    </tr>
                  );
                })}
                {rows.length === 0 && <tr><td style={S.td} colSpan={10}>No trades</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================= Utils ================= */
function nearestIndex(xs: number[], x: number): number {
  if (xs.length === 0) return 0;
  let best = 0, bestD = Math.abs(xs[0] - x);
  for (let i = 1; i < xs.length; i++) {
    const d = Math.abs(xs[i] - x);
    if (d < bestD) { best = i; bestD = d; }
  }
  return best;
}

/* ================= Styles ================= */
const S: Record<string, React.CSSProperties> = {
  wrap: { border: '1px solid #e5e7eb', borderRadius: 16, background: '#fff', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', width: '100%', fontFamily: 'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding: 16, borderBottom: '1px solid #eee' },
  titleRow: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 10 },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },
  badgeSmall: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '2px 8px', fontSize: 12, fontWeight: 600, background: '#eef2ff', color: '#3730a3' },

  controls: { display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' },
  ctrlItem: { display: 'flex', flexDirection: 'column', gap: 6 },
  lbl: { fontSize: 12, color: '#6b7280' },
  input: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 10px', fontSize: 14, outline: 'none', minWidth: 200 },
  select: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 8px', fontSize: 14, background: '#fff', minWidth: 140 },
  btn: { height: 36, padding: '0 12px', borderRadius: 10, border: '1px solid #e5e7eb', background: '#fff', cursor: 'pointer', fontSize: 13 },

  card: { border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff', overflow: 'hidden', padding: 8 },
  cardHeader: { padding: '10px 12px', borderBottom: '1px solid #eee', fontWeight: 700, fontSize: 14, color: '#111827', display: 'flex', alignItems: 'center' },
  canvas: { width: '100%', maxWidth: 1120, borderRadius: 8, background: '#fff' },

  table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0 },
  th: { textAlign: 'right', padding: '8px 10px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #eee', position: 'sticky' as any, top: 0, background: '#fff' },
  td: { textAlign: 'right', padding: '8px 10px', fontSize: 13, color: '#111827', borderBottom: '1px solid #f3f4f6' },

  btnSmall: { height: 28, padding: '0 10px', borderRadius: 8, border: '1px solid #e5e7eb', background: '#fff', cursor: 'pointer', fontSize: 12 },
  pageLabel: { display: 'inline-flex', alignItems: 'center', height: 28, padding: '0 8px', borderRadius: 8, background: '#f3f4f6', fontSize: 12, fontWeight: 700, color: '#111827' },
};