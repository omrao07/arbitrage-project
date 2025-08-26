'use client';
import React, { useMemo, useState } from 'react';

/* ---------------- Types ---------------- */
export type Position = {
  id: string;
  symbol: string;          // e.g. "RELIANCE.NS"
  side: 'long' | 'short';
  qty: number;             // signed or unsigned; we'll respect side
  avgPrice: number;        // average entry price
  lastPrice: number;       // current mark
  currency?: string;       // for display only
  venue?: string;          // e.g. "NSE", "NASDAQ"
  sector?: string;         // optional
};

type Props = {
  title?: string;
  currency?: string;       // default currency label
  rows?: Position[];       // pass your positions; falls back to mock
};

/* ---------------- Mock (if none passed) ---------------- */
function mock(): Position[] {
  return [
    { id: '1', symbol: 'RELIANCE.NS', side: 'long',  qty: 250,  avgPrice: 2855.0, lastPrice: 2950.4, currency: 'INR', venue: 'NSE', sector: 'Energy' },
    { id: '2', symbol: 'HDFCBANK.NS', side: 'short', qty: 150,  avgPrice: 1610.0, lastPrice: 1584.2, currency: 'INR', venue: 'NSE', sector: 'Financials' },
    { id: '3', symbol: 'AAPL',        side: 'long',  qty: 200,  avgPrice: 210.5, lastPrice: 225.1,  currency: 'USD', venue: 'NASDAQ', sector: 'Tech' },
    { id: '4', symbol: 'NVDA',        side: 'short', qty: 50,   avgPrice: 121.0, lastPrice: 118.8,  currency: 'USD', venue: 'NASDAQ', sector: 'Tech' },
    { id: '5', symbol: 'BTCUSDT',     side: 'long',  qty: 0.75, avgPrice: 61000, lastPrice: 63550,  currency: 'USDT', venue: 'Binance', sector: 'Crypto' },
  ];
}

/* ---------------- Helpers ---------------- */
const fmt = (n: number, d = 2) => n.toLocaleString(undefined, { maximumFractionDigits: d });
const sign = (x: number) => (x > 0 ? '+' : x < 0 ? '−' : '');
const toCsv = (rows: any[]) =>
  [Object.keys(rows[0] || {}).join(','), ...rows.map(r => Object.values(r).map(v => `"${String(v).replace(/"/g,'""')}"`).join(','))].join('\n');

function computeDerived(p: Position) {
  const qtySigned = p.side === 'short' ? -Math.abs(p.qty) : Math.abs(p.qty);
  const mv = qtySigned * p.lastPrice;                 // market value (signed)
  const cost = qtySigned * p.avgPrice;                // cost basis (signed)
  const pnl = mv - cost;                              // realized on mark
  const ret = cost !== 0 ? pnl / Math.abs(cost) : 0;  // PnL% vs cost (abs)
  return { qtySigned, mv, cost, pnl, ret };
}

/* ---------------- Component ---------------- */
export default function Positions({
  title = 'Positions',
  currency = 'USD',
  rows,
}: Props) {
  const data = rows ?? mock();

  // UI state
  const [query, setQuery] = useState('');
  const [sideFilter, setSideFilter] = useState<'all' | 'long' | 'short'>('all');
  const [venueFilter, setVenueFilter] = useState<string>('all');
  const [sortKey, setSortKey] = useState<'symbol'|'pnl'|'ret'|'mv'|'qty'>('symbol');
  const [desc, setDesc] = useState(false);

  const venues = useMemo(() => ['all', ...Array.from(new Set(data.map(d => d.venue).filter(Boolean))) as string[]], [data]);

  const enriched = useMemo(() => {
    return data.map(p => {
      const d = computeDerived(p);
      return { ...p, ...d };
    });
  }, [data]);

  const filtered = useMemo(() => {
    return enriched.filter(r => {
      const matchQ = query.trim() === '' || r.symbol.toLowerCase().includes(query.trim().toLowerCase()) || (r.sector?.toLowerCase().includes(query.trim().toLowerCase()) ?? false);
      const matchS = sideFilter === 'all' || r.side === sideFilter;
      const matchV = venueFilter === 'all' || r.venue === venueFilter;
      return matchQ && matchS && matchV;
    });
  }, [enriched, query, sideFilter, venueFilter]);

  const sorted = useMemo(() => {
    const copy = [...filtered];
    const dir = desc ? -1 : 1;
    copy.sort((a,b) => {
      const A = sortKey === 'symbol' ? a.symbol : sortKey === 'pnl' ? a.pnl : sortKey === 'ret' ? a.ret : sortKey === 'mv' ? a.mv : a.qtySigned;
      const B = sortKey === 'symbol' ? b.symbol : sortKey === 'pnl' ? b.pnl : sortKey === 'ret' ? b.ret : sortKey === 'mv' ? b.mv : b.qtySigned;
      if (A < B) return -1 * dir;
      if (A > B) return  1 * dir;
      return 0;
    });
    return copy;
  }, [filtered, sortKey, desc]);

  const totals = useMemo(() => {
    let gross = 0, net = 0, pnl = 0, longMv = 0, shortMv = 0;
    for (const r of enriched) {
      pnl += r.pnl;
      net += r.mv;
      if (r.mv >= 0) longMv += r.mv; else shortMv += Math.abs(r.mv);
    }
    gross = longMv + shortMv;
    return { pnl, net, gross, longMv, shortMv };
  }, [enriched]);

  function exportCsv() {
    if (sorted.length === 0) return;
    const csv = toCsv(sorted.map(r => ({
      symbol: r.symbol,
      side: r.side,
      qty: r.qty,
      avgPrice: r.avgPrice,
      lastPrice: r.lastPrice,
      marketValue: r.mv,
      pnl: r.pnl,
      pnlPct: r.ret,
      venue: r.venue ?? '',
      sector: r.sector ?? '',
      currency: r.currency ?? currency,
    })));
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'positions.csv';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <div style={S.wrap}>
      {/* Header / Controls */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>{currency}</span>
        </div>
        <div style={S.controls}>
          <input
            value={query}
            onChange={(e)=>setQuery(e.target.value)}
            placeholder="Search symbol / sector…"
            style={{ ...S.input, minWidth: 220 }}
          />
          <select value={`${sortKey}:${desc?'desc':'asc'}`} onChange={(e)=>{
            const [k,d] = e.target.value.split(':') as any;
            setSortKey(k); setDesc(d==='desc');
          }} style={S.select}>
            <option value="symbol:asc">Symbol ↑</option>
            <option value="symbol:desc">Symbol ↓</option>
            <option value="pnl:desc">PnL ↓</option>
            <option value="pnl:asc">PnL ↑</option>
            <option value="ret:desc">PnL% ↓</option>
            <option value="ret:asc">PnL% ↑</option>
            <option value="mv:desc">Mkt Value ↓</option>
            <option value="mv:asc">Mkt Value ↑</option>
            <option value="qty:desc">Qty ↓</option>
            <option value="qty:asc">Qty ↑</option>
          </select>
          <select value={sideFilter} onChange={(e)=>setSideFilter(e.target.value as any)} style={S.select}>
            <option value="all">All sides</option>
            <option value="long">Long</option>
            <option value="short">Short</option>
          </select>
          <select value={venueFilter} onChange={(e)=>setVenueFilter(e.target.value)} style={S.select}>
            {venues.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
          <button onClick={exportCsv} style={S.btn}>Export CSV</button>
        </div>

        {/* Totals */}
        <div style={S.kpis}>
          <KPI label="Net Exposure" value={`${fmt(totals.net, 0)} ${currency}`} />
          <KPI label="Gross Exposure" value={`${fmt(totals.gross, 0)} ${currency}`} />
          <KPI label="Long MV" value={`${fmt(totals.longMv, 0)} ${currency}`} />
          <KPI label="Short MV" value={`${fmt(totals.shortMv, 0)} ${currency}`} />
          <KPI label="Total PnL" value={`${fmt(totals.pnl, 0)} ${currency}`} accent={totals.pnl >= 0 ? '#10b981' : '#ef4444'} />
        </div>
      </div>

      {/* Table */}
      <div style={S.card}>
        <div style={{ overflow: 'auto' }}>
          <table style={S.table}>
            <thead>
              <tr>
                <th style={{ ...S.th, textAlign: 'left' }}>Symbol</th>
                <th style={S.th}>Side</th>
                <th style={S.th}>Qty</th>
                <th style={S.th}>Avg Px</th>
                <th style={S.th}>Last Px</th>
                <th style={S.th}>Mkt Value</th>
                <th style={S.th}>PnL</th>
                <th style={S.th}>PnL %</th>
                <th style={S.th}>Venue</th>
                <th style={S.th}>Sector</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(r => {
                const pnlColor = r.pnl >= 0 ? '#065f46' : '#991b1b';
                return (
                  <tr key={r.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ ...S.tdLeft, fontWeight: 600 }}>{r.symbol}</td>
                    <td style={S.td}>
                      <span style={{ ...S.side, background: r.side === 'long' ? '#ecfdf5' : '#fef2f2', color: r.side === 'long' ? '#065f46' : '#991b1b', borderColor: r.side === 'long' ? '#a7f3d0' : '#fecaca' }}>
                        {r.side.toUpperCase()}
                      </span>
                    </td>
                    <td style={S.td}>{fmt(r.qtySigned, 4)}</td>
                    <td style={S.td}>{fmt(r.avgPrice, 4)}</td>
                    <td style={S.td}>{fmt(r.lastPrice, 4)}</td>
                    <td style={S.td}>{fmt(r.mv, 2)}</td>
                    <td style={{ ...S.td, fontWeight: 700, color: pnlColor }}>{sign(r.pnl)}{fmt(Math.abs(r.pnl), 2)}</td>
                    <td style={{ ...S.td, color: pnlColor }}>{sign(r.ret)}{fmt(Math.abs(r.ret * 100), 2)}%</td>
                    <td style={S.td}>{r.venue ?? '-'}</td>
                    <td style={S.td}>{r.sector ?? '-'}</td>
                  </tr>
                );
              })}
              {sorted.length === 0 && (
                <tr><td colSpan={10} style={{ ...S.td, textAlign: 'center', color: '#9ca3af' }}>No positions</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ---------------- Small UI atoms ---------------- */
function KPI({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div style={S.kpi}>
      <div style={S.kpiLabel}>{label}</div>
      <div style={{ ...S.kpiValue, color: accent ?? '#111827' }}>{value}</div>
    </div>
  );
}

/* ---------------- Styles ---------------- */
const S: Record<string, React.CSSProperties> = {
  wrap: { border: '1px solid #e5e7eb', borderRadius: 16, background: '#fff', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', width: '100%', fontFamily: 'system-ui,-apple-system,Segoe UI,Roboto,sans-serif' },
  header: { padding: 16, borderBottom: '1px solid #eee' },
  titleRow: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },

  controls: { display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' },
  input: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 10px', fontSize: 14, outline: 'none' },
  select: { height: 36, border: '1px solid #e5e7eb', borderRadius: 10, padding: '0 8px', fontSize: 14, background: '#fff' },
  btn: { height: 36, padding: '0 12px', borderRadius: 10, border: '1px solid transparent', background: '#111', color: '#fff', cursor: 'pointer', fontSize: 13 },

  kpis: { display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: 12, marginTop: 12 },
  kpi: { border: '1px solid #eef2f7', borderRadius: 12, padding: '12px', background: '#fafafa' },
  kpiLabel: { fontSize: 12, color: '#6b7280', marginBottom: 6 },
  kpiValue: { fontSize: 18, fontWeight: 800 },

  card: { borderTop: '1px solid #eee' },

  table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0 },
  th: { textAlign: 'right', padding: '8px 10px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #eee', position: 'sticky' as any, top: 0, background: '#fff' },
  td: { textAlign: 'right', padding: '8px 10px', fontSize: 13, color: '#111827' },
  tdLeft: { textAlign: 'left', padding: '8px 10px', fontSize: 13, color: '#111827' },

  side: { fontSize: 11, padding: '2px 8px', borderRadius: 999, border: '1px solid', display: 'inline-flex' },
};