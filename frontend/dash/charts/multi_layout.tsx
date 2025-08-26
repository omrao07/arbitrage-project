'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* =============== Types =============== */
type Level = { price: number; size: number };
type Book = { bids: Level[]; asks: Level[] };

type Props = {
  title?: string;
  symbol?: string;
  decimals?: number;          // price formatting
  rows?: number;              // max levels per side to show & plot
  data?: Book;                // pass a book directly
  fetchEndpoint?: string;     // optional: GET ?symbol=AAPL -> { bids:[{price,size}], asks:[...] }
  pollMs?: number;            // if fetchEndpoint set, poll interval
};

/* =============== Helpers =============== */
const fmt = (n: number, d = 2) => n.toLocaleString(undefined, { maximumFractionDigits: d });
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));

function mockBook(spot = 225, levels = 16): Book {
  const step = Math.max(0.1, spot * 0.0015);
  const bids: Level[] = [];
  const asks: Level[] = [];
  for (let i = 0; i < levels; i++) {
    const pb = +(spot - (i + 1) * step).toFixed(2);
    const pa = +(spot + (i + 1) * step).toFixed(2);
    const s = Math.max(1, Math.round((levels - i) * (1 + Math.random() * 0.6)));
    bids.push({ price: pb, size: s });
    asks.push({ price: pa, size: s * (0.8 + Math.random() * 0.6) });
  }
  return { bids, asks };
}

function cumulative(levels: Level[]) {
  const out: (Level & { cum: number })[] = [];
  let c = 0;
  for (const l of levels) { c += l.size; out.push({ ...l, cum: c }); }
  return out;
}

/* =============== Component =============== */
export default function Depth({
  title = 'Order Book Depth',
  symbol: symProp = 'AAPL',
  decimals = 2,
  rows = 16,
  data,
  fetchEndpoint,
  pollMs = 2500,
}: Props) {
  const [symbol, setSymbol] = useState(symProp);
  const [book, setBook] = useState<Book | null>(data ?? null);

  useEffect(() => {
    let dead = false;
    let timer: any;
    async function load() {
      if (!fetchEndpoint) { if (!book) setBook(mockBook()); return; }
      try {
        const q = new URLSearchParams({ symbol });
        const res = await fetch(`${fetchEndpoint}?${q.toString()}`);
        const json = await res.json();
        if (!dead && json && Array.isArray(json.bids) && Array.isArray(json.asks)) {
          setBook({ bids: json.bids, asks: json.asks });
        }
      } catch {
        if (!dead) setBook(mockBook());
      }
    }
    load();
    if (fetchEndpoint) timer = setInterval(load, pollMs);
    return () => { dead = true; if (timer) clearInterval(timer); };
  }, [fetchEndpoint, symbol, pollMs]);

  const trimmed = useMemo(() => {
    const src = book ?? mockBook();
    const bids = [...src.bids].filter(x => x.price > 0 && x.size > 0).sort((a,b)=>b.price-a.price).slice(0, rows);
    const asks = [...src.asks].filter(x => x.price > 0 && x.size > 0).sort((a,b)=>a.price-b.price).slice(0, rows);
    return { bids, asks };
  }, [book, rows]);

  const bestBid = trimmed.bids[0]?.price ?? 0;
  const bestAsk = trimmed.asks[0]?.price ?? 0;
  const mid = bestBid && bestAsk ? (bestBid + bestAsk) / 2 : (bestBid || bestAsk);
  const spread = bestBid && bestAsk ? bestAsk - bestBid : 0;

  const bidsCum = useMemo(() => cumulative(trimmed.bids), [trimmed.bids]);
  const asksCum = useMemo(() => cumulative(trimmed.asks), [trimmed.asks]);

  /* ===== Depth chart (canvas) ===== */
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const W = 920, H = 340;
  const padL = 60, padR = 16, padT = 20, padB = 44;

  type Hover = { x: number; y: number; price: number; side: 'bid'|'ask'; size: number; cum: number } | null;
  const [hover, setHover] = useState<Hover>(null);

  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);

    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xs: number[] = trimmed.bids.map(b => b.price).reverse().concat(trimmed.asks.map(a => a.price));
    if (xs.length === 0) return;
    const xMin = xs[0], xMax = xs[xs.length - 1];
    const yMax = Math.max(
      bidsCum.length ? bidsCum[bidsCum.length - 1].cum : 0,
      asksCum.length ? asksCum[asksCum.length - 1].cum : 0
    );

    const xAt = (p: number) => padL + ((p - xMin) / Math.max(1e-9, (xMax - xMin))) * plotW;
    const yAt = (v: number) => padT + (1 - v / Math.max(1e-9, yMax)) * plotH;

    // grid
    ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let g = 0; g <= 4; g++) {
      const y = padT + (g / 4) * plotH;
      ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
    }
    ctx.stroke();

    // bids area + line
    if (bidsCum.length) {
      const prices = trimmed.bids.map(b => b.price).reverse();
      const cums = bidsCum.map(b => b.cum).reverse();

      ctx.beginPath();
      ctx.moveTo(xAt(prices[0]), yAt(0));
      for (let i = 0; i < prices.length; i++) ctx.lineTo(xAt(prices[i]), yAt(cums[i]));
      ctx.lineTo(xAt(prices[prices.length - 1]), yAt(0));
      ctx.closePath();
      ctx.fillStyle = 'rgba(16,185,129,0.18)'; ctx.fill();

      ctx.beginPath();
      for (let i = 0; i < prices.length; i++) {
        const x = xAt(prices[i]), y = yAt(cums[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#10b981'; ctx.lineWidth = 2; ctx.stroke();
    }

    // asks area + line
    if (asksCum.length) {
      const prices = trimmed.asks.map(a => a.price);
      const cums = asksCum.map(a => a.cum);

      ctx.beginPath();
      ctx.moveTo(xAt(prices[0]), yAt(0));
      for (let i = 0; i < prices.length; i++) ctx.lineTo(xAt(prices[i]), yAt(cums[i]));
      ctx.lineTo(xAt(prices[prices.length - 1]), yAt(0));
      ctx.closePath();
      ctx.fillStyle = 'rgba(239,68,68,0.18)'; ctx.fill();

      ctx.beginPath();
      for (let i = 0; i < prices.length; i++) {
        const x = xAt(prices[i]), y = yAt(cums[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2; ctx.stroke();
    }

    // axes labels
    ctx.font = '11px system-ui'; ctx.fillStyle = '#6b7280';
    const ticks = 6;
    for (let t = 0; t < ticks; t++) {
      const p = xMin + (t / (ticks - 1)) * (xMax - xMin);
      const x = xAt(p);
      ctx.fillText(String(p.toFixed(decimals)), x - 14, padT + plotH + 16);
    }
    for (let g = 0; g <= 4; g++) {
      const v = (g / 4) * Math.max(1, Math.max(
        bidsCum.length ? bidsCum[bidsCum.length - 1].cum : 0,
        asksCum.length ? asksCum[asksCum.length - 1].cum : 0
      ));
      const y = yAt(v);
      ctx.fillText(fmt(v, 0), 8, y + 4);
    }

    // mid line
    if (mid) {
      const xm = xAt(mid);
      ctx.strokeStyle = '#9ca3af';
      ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.moveTo(xm, padT); ctx.lineTo(xm, padT + plotH); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [trimmed, bidsCum, asksCum, decimals, mid]);

  /* ===== mouse hover (fixed typing) ===== */
  function nearestNonNull(levels: Level[], price: number): { idx: number; lev: Level } | null {
    if (!levels.length) return null;
    let bestIdx = 0;
    let bestDist = Math.abs(levels[0].price - price);
    for (let i = 1; i < levels.length; i++) {
      const d = Math.abs(levels[i].price - price);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return { idx: bestIdx, lev: levels[bestIdx] }; // lev is guaranteed
  }

  function onMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;

    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    if (mx < padL || mx > padL + plotW || my < padT || my > padT + plotH) { setHover(null); return; }

    const pricesAll = trimmed.bids.map(b => b.price).reverse().concat(trimmed.asks.map(a => a.price));
    if (pricesAll.length === 0) { setHover(null); return; }

    const xMin = pricesAll[0], xMax = pricesAll[pricesAll.length - 1];
    const xToPrice = (x: number) => xMin + ((x - padL) / plotW) * (xMax - xMin);
    const px = xToPrice(mx);

    const nb = nearestNonNull(trimmed.bids, px); // highest→lowest, but we only need nearest
    const na = nearestNonNull(trimmed.asks, px);

    let pick: { side: 'bid'|'ask'; idx: number; lev: Level } | null = null;
    if (nb && na) {
      const closerIsBid = Math.abs(nb.lev.price - px) < Math.abs(na.lev.price - px);
      pick = closerIsBid ? { side: 'bid', ...nb } : { side: 'ask', ...na };
    } else if (nb) pick = { side: 'bid', ...nb };
    else if (na) pick = { side: 'ask', ...na };
    else { setHover(null); return; }

    // compute cumulative safely
    const cumArr = pick.side === 'bid' ? cumulative(trimmed.bids) : cumulative(trimmed.asks);
    const cum = cumArr[pick.idx]?.cum ?? pick.lev.size;

    setHover({
      x: mx, y: my,
      price: pick.lev.price,
      side: pick.side,
      size: pick.lev.size,
      cum,
    });
  }
  function onLeave() { setHover(null); }

  /* ===== L2 table ===== */
  function L2Table() {
    const bids = cumulative(trimmed.bids);
    const asks = cumulative(trimmed.asks);
    return (
      <div style={S.l2Wrap}>
        <div style={S.sideCard}>
          <div style={S.sideHeader}>Asks</div>
          <div style={{ overflow: 'auto', maxHeight: 300 }}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>Price</th><th style={S.th}>Size</th><th style={S.th}>Cum</th><th style={S.th}>Notional</th>
                </tr>
              </thead>
              <tbody>
                {asks.map((a, i) => (
                  <tr key={`a-${i}`} style={{ background: i === 0 ? '#fff1f2' : '#fff' }}>
                    <td style={{ ...S.td, color: '#991b1b', fontWeight: 700 }}>{a.price.toFixed(decimals)}</td>
                    <td style={S.td}>{fmt(a.size, 2)}</td>
                    <td style={S.td}>{fmt(a.cum, 2)}</td>
                    <td style={S.td}>{fmt(a.price * a.cum, 0)}</td>
                  </tr>
                ))}
                {asks.length === 0 && <tr><td style={S.td} colSpan={4}>-</td></tr>}
              </tbody>
            </table>
          </div>
        </div>

        <div style={S.sideCard}>
          <div style={S.sideHeader}>Bids</div>
          <div style={{ overflow: 'auto', maxHeight: 300 }}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>Price</th><th style={S.th}>Size</th><th style={S.th}>Cum</th><th style={S.th}>Notional</th>
                </tr>
              </thead>
              <tbody>
                {bids.map((b, i) => (
                  <tr key={`b-${i}`} style={{ background: i === 0 ? '#ecfdf5' : '#fff' }}>
                    <td style={{ ...S.td, color: '#065f46', fontWeight: 700 }}>{b.price.toFixed(decimals)}</td>
                    <td style={S.td}>{fmt(b.size, 2)}</td>
                    <td style={S.td}>{fmt(b.cum, 2)}</td>
                    <td style={S.td}>{fmt(b.price * b.cum, 0)}</td>
                  </tr>
                ))}
                {bids.length === 0 && <tr><td style={S.td} colSpan={4}>-</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={S.wrap}>
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>{symbol}</span>
          {mid ? <span style={{ ...S.badge, background: '#eef2ff', color: '#3730a3' }}>Mid: {mid.toFixed(decimals)}</span> : null}
          {!!spread && <span style={{ ...S.badge, background: '#fef3c7', color: '#92400e' }}>Spread: {spread.toFixed(decimals)}</span>}
        </div>
        <div style={S.controls}>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Symbol</span>
            <input value={symbol} onChange={(e)=>setSymbol(e.target.value.toUpperCase())} style={S.input}/>
          </label>
        </div>
      </div>

      <div style={{ position: 'relative', padding: 12 }}>
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          style={{ width: '100%', maxWidth: 1120, border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff' }}
          onMouseMove={onMove}
          onMouseLeave={onLeave}
        />
        {hover && (
          <div
            style={{
              position: 'absolute',
              left: clamp(hover.x + 14, 0, W - 160),
              top: clamp(hover.y + 10, 0, H - 80),
              background: '#111827', color: '#fff',
              padding: '6px 8px', borderRadius: 8, fontSize: 12,
              pointerEvents: 'none', boxShadow: '0 2px 6px rgba(0,0,0,0.25)',
            }}
          >
            <div><b>{hover.side === 'bid' ? 'Bid' : 'Ask'}</b> @ {hover.price.toFixed(decimals)}</div>
            <div>Size: {fmt(hover.size, 2)}</div>
            <div>Cumulative: {fmt(hover.cum, 2)}</div>
          </div>
        )}
      </div>

      <div style={{ padding: 12 }}>
        <L2Table />
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
  input: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 10px', fontSize: 14, outline: 'none', minWidth: 120 },

  l2Wrap: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 },
  sideCard: { border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff', overflow: 'hidden' },
  sideHeader: { padding: '10px 12px', borderBottom: '1px solid #eee', fontWeight: 700, fontSize: 14, color: '#111827' },

  table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0 },
  th: { textAlign: 'right', padding: '8px 10px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #eee', position: 'sticky' as any, top: 0, background: '#fff' },
  td: { textAlign: 'right', padding: '8px 10px', fontSize: 13, color: '#111827', borderBottom: '1px solid #f3f4f6' },
};