'use client';
import React, { useEffect, useMemo, useState } from 'react';

/** ---------- Types ---------- */
type Side = 'C' | 'P';
type OptionRow = {
  symbol: string;     // e.g., "AAPL"
  expiry: string;     // ISO date or "2025-09-26"
  side: Side;         // 'C' or 'P'
  strike: number;
  bid: number;
  ask: number;
  last?: number;
  iv?: number;        // implied vol (0-1)
  delta?: number;
  gamma?: number;
  theta?: number;
  vega?: number;
  oi?: number;        // open interest
  vol?: number;       // volume
};

type Props = {
  title?: string;
  symbol?: string;
  spot?: number;
  expiries?: string[];            // available expiries
  data?: OptionRow[];             // pass pre-fetched chain
  fetchEndpoint?: string;         // optional: GET endpoint returning OptionRow[]
  defaultExpiry?: string;
};

/** ---------- Mock Data (used if none provided) ---------- */
function makeMockChain(sym = 'AAPL', baseSpot = 225, expiry = '2025-09-26'): OptionRow[] {
  const strikes = Array.from({ length: 21 }, (_, i) => Math.round((baseSpot - 20) + i * 2));
  const rows: OptionRow[] = [];
  for (const k of strikes) {
    const dist = Math.abs(k - baseSpot);
    const iv = Math.max(0.15, Math.min(0.65, 0.35 + (dist / 200)));
    const midC = Math.max(0.05, Math.max(baseSpot - k, 0) * 0.5 + dist * 0.02);
    const midP = Math.max(0.05, Math.max(k - baseSpot, 0) * 0.5 + dist * 0.02);
    const mk = (m: number) => [Math.max(0.01, m * 0.95), Math.max(0.02, m * 1.05)];
    const oi = Math.round(800 - dist * 6 + Math.random() * 80);
    const vol = Math.round(150 - dist * 4 + Math.random() * 40);
    const [cbid, cask] = mk(midC);
    const [pbid, pask] = mk(midP);
    rows.push(
      { symbol: sym, expiry, side: 'C', strike: k, bid: +cbid.toFixed(2), ask: +cask.toFixed(2), iv, delta: +(0.5 - dist/1000).toFixed(2), gamma: 0.02, theta: -0.02, vega: 0.10, oi: Math.max(oi,0), vol: Math.max(vol,0) },
      { symbol: sym, expiry, side: 'P', strike: k, bid: +pbid.toFixed(2), ask: +pask.toFixed(2), iv, delta: +(-0.5 + dist/1000).toFixed(2), gamma: 0.02, theta: -0.02, vega: 0.10, oi: Math.max(oi,0), vol: Math.max(vol,0) },
    );
  }
  return rows;
}

/** ---------- Utilities ---------- */
const fmt = (n?: number, d = 2) => (n == null ? '-' : n.toLocaleString(undefined, { maximumFractionDigits: d }));
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const nearestStrike = (strikes: number[], spot: number) =>
  strikes.reduce((p, c) => (Math.abs(c - spot) < Math.abs(p - spot) ? c : p), strikes[0]);

/** ---------- Component ---------- */
export default function OptionChain({
  title = 'Option Chain',
  symbol: symProp = 'AAPL',
  spot: spotProp = 225,
  expiries = ['2025-09-26', '2025-10-31', '2025-12-19'],
  data,
  fetchEndpoint,
  defaultExpiry,
}: Props) {
  const [symbol, setSymbol] = useState(symProp);
  const [spot, setSpot] = useState(spotProp);
  const [expiry, setExpiry] = useState(defaultExpiry || expiries[0]);
  const [rows, setRows] = useState<OptionRow[]>(data || makeMockChain(symProp, spotProp, expiry));
  const [onlyNear, setOnlyNear] = useState(true);     // show strikes around ATM
  const [range, setRange] = useState(10);             // number of strikes around ATM
  const [sortBy, setSortBy] = useState<'strike'|'iv'|'oi'|'vol'>('strike');
  const [desc, setDesc] = useState(false);

  // fetch if endpoint provided
  useEffect(() => {
    let ignore = false;
    if (!fetchEndpoint) return;
    (async () => {
      try {
        const q = new URLSearchParams({ symbol, expiry });
        const res = await fetch(`${fetchEndpoint}?${q.toString()}`);
        const json = await res.json();
        if (!ignore && Array.isArray(json)) setRows(json as OptionRow[]);
      } catch {
        // fallback: mock
        if (!ignore) setRows(makeMockChain(symbol, spot, expiry));
      }
    })();
    return () => { ignore = true; };
  }, [fetchEndpoint, symbol, expiry]);

  // rebuild mock if changing local symbol/expiry/spot (when not using fetch)
  useEffect(() => {
    if (!fetchEndpoint && !data) setRows(makeMockChain(symbol, spot, expiry));
  }, [symbol, expiry, spot, fetchEndpoint, data]);

  const chain = useMemo(() => rows.filter(r => r.symbol === symbol && r.expiry === expiry), [rows, symbol, expiry]);
  const strikes = useMemo(() => Array.from(new Set(chain.map(r => r.strike))).sort((a,b)=>a-b), [chain]);
  const atm = useMemo(() => strikes.length ? nearestStrike(strikes, spot) : undefined, [strikes, spot]);

  const filteredStrikes = useMemo(() => {
    if (!onlyNear || atm == null) return strikes;
    const idx = strikes.indexOf(atm);
    const l = clamp(idx - Math.floor(range/2), 0, Math.max(0, strikes.length-1));
    const r = clamp(l + range - 1, 0, Math.max(0, strikes.length-1));
    return strikes.slice(l, r+1);
  }, [strikes, atm, onlyNear, range]);

  const calls = useMemo(() => chain.filter(r => r.side === 'C' && filteredStrikes.includes(r.strike)), [chain, filteredStrikes]);
  const puts  = useMemo(() => chain.filter(r => r.side === 'P' && filteredStrikes.includes(r.strike)), [chain, filteredStrikes]);

  const sorter = (a: OptionRow, b: OptionRow) => {
    const dir = desc ? -1 : 1;
    const pick = (r: OptionRow) => sortBy === 'strike' ? r.strike :
                                   sortBy === 'iv' ? (r.iv ?? 0) :
                                   sortBy === 'oi' ? (r.oi ?? 0) :
                                   (r.vol ?? 0);
    return (pick(a) > pick(b) ? 1 : pick(a) < pick(b) ? -1 : 0) * dir;
  };

  calls.sort(sorter);
  puts.sort(sorter);

  return (
    <div style={S.wrap}>
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>{symbol}</span>
          <span style={{ ...S.badge, background: '#eef2ff', color: '#3730a3' }}>{expiry}</span>
        </div>

        {/* Controls */}
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
            <span style={S.lbl}>Expiry</span>
            <select value={expiry} onChange={(e)=>setExpiry(e.target.value)} style={S.select}>
              {expiries.map(x => <option key={x} value={x}>{x}</option>)}
            </select>
          </label>

          <label style={S.ctrlCheck}>
            <input type="checkbox" checked={onlyNear} onChange={()=>setOnlyNear(v=>!v)} />
            <span>ATM ± strikes</span>
          </label>
          <label style={S.ctrlItem}>
            <span style={S.lbl}>Range</span>
            <input type="number" min={3} max={strikes.length||50} value={range} onChange={(e)=>setRange(clamp(+e.target.value,3,200))} style={{...S.input, width: 80}} />
          </label>

          <label style={S.ctrlItem}>
            <span style={S.lbl}>Sort</span>
            <select value={`${sortBy}:${desc?'desc':'asc'}`} onChange={(e)=>{
              const [key,dir]=e.target.value.split(':') as any;
              setSortBy(key); setDesc(dir==='desc');
            }} style={S.select}>
              <option value="strike:asc">Strike ↑</option>
              <option value="strike:desc">Strike ↓</option>
              <option value="iv:desc">IV ↓</option>
              <option value="iv:asc">IV ↑</option>
              <option value="oi:desc">OI ↓</option>
              <option value="oi:asc">OI ↑</option>
              <option value="vol:desc">Vol ↓</option>
              <option value="vol:asc">Vol ↑</option>
            </select>
          </label>
        </div>
      </div>

      {/* Tables */}
      <div style={S.tables}>
        <div style={S.card}>
          <div style={S.cardHeader}>Calls</div>
          <Table rows={calls} spot={spot} atm={atm} side="C" />
        </div>
        <div style={S.card}>
          <div style={S.cardHeader}>Puts</div>
          <Table rows={puts} spot={spot} atm={atm} side="P" />
        </div>
      </div>
    </div>
  );
}

/** ---------- Table ---------- */
function Table({ rows, spot, atm, side }: { rows: OptionRow[]; spot: number; atm?: number; side: Side }) {
  return (
    <div style={{ overflow: 'auto' }}>
      <table style={S.table}>
        <thead>
          <tr>
            <th style={S.th}>Strike</th>
            <th style={S.th}>Bid</th>
            <th style={S.th}>Ask</th>
            <th style={S.th}>IV</th>
            <th style={S.th}>Δ</th>
            <th style={S.th}>Γ</th>
            <th style={S.th}>Θ</th>
            <th style={S.th}>V</th>
            <th style={S.th}>OI</th>
            <th style={S.th}>Vol</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isATM = atm != null && r.strike === atm;
            const itm = side === 'C' ? r.strike < spot : r.strike > spot;
            const bg = isATM ? '#fff7ed' : itm ? '#ecfeff' : '#fff';
            const bd = isATM ? '#fde68a' : itm ? '#bae6fd' : '#e5e7eb';
            return (
              <tr key={`${r.side}-${r.strike}`} style={{ background: bg, borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ ...S.td, borderLeft: `3px solid ${bd}`, fontWeight: 600 }}>{fmt(r.strike, 2)}</td>
                <td style={S.td}>{fmt(r.bid)}</td>
                <td style={S.td}>{fmt(r.ask)}</td>
                <td style={S.td}>{r.iv != null ? (r.iv * 100).toFixed(1) + '%' : '-'}</td>
                <td style={S.td}>{fmt(r.delta, 2)}</td>
                <td style={S.td}>{fmt(r.gamma, 3)}</td>
                <td style={S.td}>{fmt(r.theta, 2)}</td>
                <td style={S.td}>{fmt(r.vega, 2)}</td>
                <td style={S.td}>{fmt(r.oi, 0)}</td>
                <td style={S.td}>{fmt(r.vol, 0)}</td>
              </tr>
            );
          })}
          {rows.length === 0 && (
            <tr><td colSpan={10} style={{ ...S.td, textAlign: 'center', color: '#9ca3af' }}>No data</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

/** ---------- Styles ---------- */
const S: Record<string, React.CSSProperties> = {
  wrap: { border: '1px solid #e5e7eb', borderRadius: 16, background: '#fff', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', width: '100%', fontFamily: 'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding: 16, borderBottom: '1px solid #eee' },
  titleRow: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },

  controls: { display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' },
  ctrlItem: { display: 'flex', flexDirection: 'column', gap: 6 },
  ctrlCheck: { display: 'flex', alignItems: 'center', gap: 8, padding: '0 6px' },
  lbl: { fontSize: 12, color: '#6b7280' },
  input: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 10px', minWidth: 120, fontSize: 14, outline: 'none' },
  select: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 8px', fontSize: 14, minWidth: 160, background: '#fff' },

  tables: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, padding: 12 },
  card: { border: '1px solid #e5e7eb', borderRadius: 12, overflow: 'hidden', background: '#fff' },
  cardHeader: { padding: '10px 12px', borderBottom: '1px solid #eee', fontWeight: 700, fontSize: 14, color: '#111827' },

  table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0 },
  th: { textAlign: 'right', padding: '8px 10px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #eee', position: 'sticky' as any, top: 0, background: '#fff' },
  td: { textAlign: 'right', padding: '8px 10px', fontSize: 13, color: '#111827' },
};