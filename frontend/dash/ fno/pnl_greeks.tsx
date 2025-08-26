'use client';
import React, { useMemo, useState } from 'react';

/* ---------------- Types ---------------- */
type Position = {
  id: string;
  symbol: string;
  underlier: string;
  qty: number;                 // + long, - short (in option contracts or shares)
  side: 'option' | 'stock';    // stock lines ignore option greeks
  // Market marks
  spot: number;                // underlying spot
  price: number;               // current instrument price (option or stock)
  // Greeks (per 1 option contract; scale by qty)
  delta?: number;              // ∂P/∂S (shares-equivalent), stock=1
  gamma?: number;              // ∂²P/∂S²
  theta?: number;              // ∂P/∂t (per day)
  vega?: number;               // ∂P/∂σ (per 1.00 = 100% vol)
  iv?: number;                 // current implied vol (0–1) for display
  // Optional mini pnl history for sparkline (base currency)
  pnlHistory?: number[];       // last N pnl points
};

type Props = {
  title?: string;
  currency?: string;
  positions?: Position[];
};

/* ---------------- Helpers ---------------- */
const fmt = (n: number, d = 2) => n.toLocaleString(undefined, { maximumFractionDigits: d });
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));

/** Scenario PnL approximation:
 *  dS = S * priceShockPct
 *  dσ = ivShockPct (absolute, so +0.05 = +5 vol points)
 *  dt = daysForward
 *  dP ≈ Δ dS + 0.5 Γ dS² + 𝜈 dσ + Θ dt
 *  For stocks: Δ=1, other greeks=0 (unless supplied).
 */
function scenarioPnL(p: Position, priceShockPct: number, volShockPct: number, daysForward: number): number {
  const dS = p.spot * priceShockPct;
  const delta = p.side === 'stock' ? 1 : (p.delta ?? 0);
  const gamma = p.side === 'stock' ? 0 : (p.gamma ?? 0);
  const theta = p.theta ?? 0;          // per day
  const vega  = p.vega  ?? 0;          // per 1.00 vol (100%)
  const dSigma = volShockPct;          // absolute change in vol (e.g., +0.05 = +5 vol pts)
  const dP = delta * dS + 0.5 * gamma * dS * dS + vega * dSigma + theta * daysForward;
  return dP * p.qty;
}

/* ---------------- Mock (if no props) ---------------- */
function mockPositions(): Position[] {
  return [
    { id: '1', symbol: 'RELIANCE 2950 C', underlier: 'RELIANCE.NS', qty: 2, side: 'option', spot: 2950, price: 82.5, delta: 0.42, gamma: 0.0003, theta: -2.1, vega: 45, iv: 0.26, pnlHistory: [1200, 950, 1020, 800, 1120, 980, 1300] },
    { id: '2', symbol: 'RELIANCE 2900 P', underlier: 'RELIANCE.NS', qty: -1, side: 'option', spot: 2950, price: 64.0, delta: -0.28, gamma: 0.00025, theta: -1.6, vega: 38, iv: 0.27, pnlHistory: [200, 180, 220, 210, 170, 160, 190] },
    { id: '3', symbol: 'AAPL 09/26 230 C', underlier: 'AAPL', qty: 5, side: 'option', spot: 225, price: 6.8, delta: 0.55, gamma: 0.0021, theta: -0.9, vega: 11, iv: 0.34, pnlHistory: [-120, -100, -60, -80, -30, 10, 40] },
    { id: '4', symbol: 'AAPL Stock', underlier: 'AAPL', qty: 200, side: 'stock', spot: 225, price: 225, pnlHistory: [0, 200, 350, 140, 300, 260, 420] },
  ];
}

/* ---------------- Sparkline (inline SVG) ---------------- */
function Spark({ series, stroke = '#111827' }: { series?: number[]; stroke?: string }) {
  if (!series || series.length < 2) return <div style={{ height: 24 }} />;
  const w = 90, h = 24, pad = 2;
  const min = Math.min(...series), max = Math.max(...series);
  const span = max - min || 1;
  const pts = series.map((v, i) => {
    const x = pad + (i / (series.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const trendUp = series[series.length - 1] >= series[0];
  const col = trendUp ? '#10b981' : '#ef4444';
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <polyline fill="none" stroke={col} strokeWidth="2" points={pts} />
    </svg>
  );
}

/* ---------------- Component ---------------- */
export default function PnlGreeks({
  title = 'PnL & Greeks',
  currency = 'INR',
  positions,
}: Props) {
  const rows = positions ?? mockPositions();

  // Shocks (controls)
  const [priceShockPct, setPriceShockPct] = useState(0);   // -0.10 … +0.10
  const [volShockPct, setVolShockPct] = useState(0);       // -0.20 … +0.20 (absolute)
  const [daysForward, setDaysForward] = useState(0);       // 0 … 5

  const totals = useMemo(() => {
    let pnl = 0, d = 0, g = 0, t = 0, v = 0;
    for (const p of rows) {
      pnl += scenarioPnL(p, priceShockPct, volShockPct, daysForward);
      d   += (p.side === 'stock' ? 1 : (p.delta ?? 0)) * p.qty;
      g   += (p.gamma ?? 0) * p.qty;
      t   += (p.theta ?? 0) * p.qty;
      v   += (p.vega  ?? 0) * p.qty;
    }
    return { pnl, d, g, t, v };
  }, [rows, priceShockPct, volShockPct, daysForward]);

  return (
    <div style={S.wrap}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.titleRow}>
          <h2 style={S.title}>{title}</h2>
          <span style={S.badge}>{currency}</span>
        </div>

        {/* Controls */}
        <div style={S.controls}>
          <Control label={`Spot Shock (${(priceShockPct*100).toFixed(1)}%)`}>
            <input
              type="range" min={-10} max={10} step={0.5}
              value={priceShockPct*100}
              onChange={(e)=>setPriceShockPct(clamp(+e.target.value, -10, 10)/100)}
              style={S.slider}
            />
          </Control>
          <Control label={`IV Shock (${(volShockPct*100).toFixed(1)} vol pts)`}>
            <input
              type="range" min={-20} max={20} step={0.5}
              value={volShockPct*100}
              onChange={(e)=>setVolShockPct(clamp(+e.target.value, -20, 20)/100)}
              style={S.slider}
            />
          </Control>
          <Control label={`Days Forward (${daysForward}d)`}>
            <input
              type="range" min={0} max={5} step={1}
              value={daysForward}
              onChange={(e)=>setDaysForward(clamp(+e.target.value, 0, 5))}
              style={S.slider}
            />
          </Control>
        </div>

        {/* Totals row */}
        <div style={S.totalsRow}>
          <KPI label="Scenario PnL" value={`${fmt(totals.pnl, 0)} ${currency}`} accent={totals.pnl >= 0 ? '#10b981' : '#ef4444'} />
          <KPI label="Δ (portfolio)" value={fmt(totals.d, 2)} />
          <KPI label="Γ (portfolio)" value={fmt(totals.g, 4)} />
          <KPI label="Θ /day" value={fmt(totals.t, 2)} />
          <KPI label="𝜈 / 100% vol" value={fmt(totals.v, 2)} />
        </div>
      </div>

      {/* Table */}
      <div style={S.card}>
        <div style={S.cardHeader}>Positions</div>
        <div style={{ overflow: 'auto' }}>
          <table style={S.table}>
            <thead>
              <tr>
                <th style={{ ...S.th, textAlign: 'left' }}>Symbol</th>
                <th style={S.th}>Qty</th>
                <th style={S.th}>Spot</th>
                <th style={S.th}>Price</th>
                <th style={S.th}>IV</th>
                <th style={S.th}>Δ</th>
                <th style={S.th}>Γ</th>
                <th style={S.th}>Θ</th>
                <th style={S.th}>𝜈</th>
                <th style={S.th}>Scenario PnL</th>
                <th style={S.th}>Trend</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => {
                const pnl = scenarioPnL(p, priceShockPct, volShockPct, daysForward);
                const posColor = pnl >= 0 ? '#065f46' : '#991b1b';
                return (
                  <tr key={p.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ ...S.tdLeft, fontWeight: 600 }}>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <span>{p.symbol}</span>
                        <span style={S.underBadge}>{p.underlier}</span>
                      </div>
                    </td>
                    <td style={S.td}>{fmt(p.qty, 0)}</td>
                    <td style={S.td}>{fmt(p.spot, 2)}</td>
                    <td style={S.td}>{fmt(p.price, 2)}</td>
                    <td style={S.td}>{p.iv != null ? (p.iv * 100).toFixed(1) + '%' : '-'}</td>
                    <td style={S.td}>{p.side === 'stock' ? '1.00' : fmt(p.delta ?? 0, 2)}</td>
                    <td style={S.td}>{fmt(p.gamma ?? 0, 4)}</td>
                    <td style={S.td}>{fmt(p.theta ?? 0, 2)}</td>
                    <td style={S.td}>{fmt(p.vega ?? 0, 2)}</td>
                    <td style={{ ...S.td, fontWeight: 700, color: posColor }}>{fmt(pnl, 0)} {currency}</td>
                    <td style={S.td}><Spark series={p.pnlHistory} /></td>
                  </tr>
                );
              })}
              {rows.length === 0 && (
                <tr><td colSpan={11} style={{ ...S.td, textAlign: 'center', color: '#9ca3af' }}>No positions</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ---------------- UI atoms ---------------- */
function Control({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={S.ctrlItem}>
      <span style={S.ctrlLabel}>{label}</span>
      {children}
    </label>
  );
}

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
  wrap: {
    border: '1px solid #e5e7eb',
    borderRadius: 16,
    background: '#fff',
    boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
    width: '100%',
    fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
  },
  header: { padding: 16, borderBottom: '1px solid #eee' },
  titleRow: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 },
  title: { margin: 0, fontSize: 18, fontWeight: 700, color: '#111827' },
  badge: { display: 'inline-flex', alignItems: 'center', borderRadius: 999, padding: '4px 10px', fontSize: 12, fontWeight: 600, background: '#f3f4f6', color: '#111827' },

  controls: { display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center', marginBottom: 12 },
  ctrlItem: { display: 'flex', flexDirection: 'column', gap: 6, minWidth: 220 },
  ctrlLabel: { fontSize: 12, color: '#6b7280' },
  slider: { width: '100%' },

  totalsRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
    gap: 12,
    marginTop: 8,
  },
  kpi: { border: '1px solid #eef2f7', borderRadius: 12, padding: '12px', background: '#fafafa' },
  kpiLabel: { fontSize: 12, color: '#6b7280', marginBottom: 6 },
  kpiValue: { fontSize: 18, fontWeight: 800 },

  card: { borderTop: '1px solid #eee' },
  cardHeader: { padding: '10px 12px', fontWeight: 700, fontSize: 14, color: '#111827' },

  table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0 },
  th: { textAlign: 'right', padding: '8px 10px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #eee', position: 'sticky' as any, top: 0, background: '#fff' },
  td: { textAlign: 'right', padding: '8px 10px', fontSize: 13, color: '#111827' },
  tdLeft: { textAlign: 'left', padding: '8px 10px', fontSize: 13, color: '#111827' },
  underBadge: { fontSize: 11, color: '#6b7280', background: '#f9fafb', border: '1px solid #eef2f7', padding: '2px 6px', borderRadius: 8 },

};