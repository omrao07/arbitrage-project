'use client';
import React, { useEffect, useMemo, useState } from 'react';

/* ============== Types ============== */
type KPI = { label: string; value: string | number };
type PnLPoint = { t: number; v: number };

type Position = {
  symbol: string;
  name?: string;
  assetClass?: 'Equity'|'Future'|'Option'|'FX'|'Crypto'|'Bond'|'ETF'|'Cash'|string;
  sector?: string;
  region?: string;
  qty: number;
  avg: number;           // average price
  last?: number;         // mark
  pnl?: number;          // realized+unrealized total
  uPnl?: number;         // unrealized
  notional?: number;     // |qty|*last*mult
  currency?: string;
  multiplier?: number;   // e.g. futures/option contract size
  greeks?: { delta?: number; gamma?: number; vega?: number; theta?: number };
};

type Order = { id: string; ts: number; symbol: string; side: 'BUY'|'SELL'; qty: number; px?: number; status: string };

type Risk = {
  gross?: number; net?: number;
  var95?: number; var99?: number; es97?: number;
  beta?: number; vol?: number; dd?: number; // drawdown fraction
};

type Patch = {
  kpis?: KPI[];
  pnl?: PnLPoint[];
  positions?: Position[];
  orders?: Order[];
  risk?: Risk;
  cash?: { currency: string; balance: number }[];
  notes?: string;
};

type Props = {
  endpoint?: string;   // e.g. '/api/portfolio/monitor'
  title?: string;
  className?: string;
};

/* ============== Utils ============== */
const fmt = (n: number | undefined, d = 2) => (Number.isFinite(n as number) ? (n as number).toFixed(d) : '');
const fmtTs = (ms: number) => new Date(ms).toLocaleString();
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const by = <T,>(xs: T[], key: (t: T) => string) => {
  const m = new Map<string, T[]>();
  for (const x of xs) { const k = key(x); m.set(k, [...(m.get(k) || []), x]); }
  return m;
};
const clamp = (x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));

/* ============== Component ============== */
export default function PortfolioMonitor({
  endpoint = '/api/portfolio/monitor',
  title = 'Portfolio Monitor',
  className = '',
}: Props) {
  const [kpis, setKpis] = useState<KPI[]>([]);
  const [pnl, setPnl] = useState<PnLPoint[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [risk, setRisk] = useState<Risk | undefined>();
  const [cash, setCash] = useState<{ currency: string; balance: number }[]>([]);
  const [notes, setNotes] = useState('');
  const [error, setError] = useState<string | null>(null);

  // Controls
  const [quoteCcy, setQuoteCcy] = useState('USD');
  const [showGreeks, setShowGreeks] = useState(true);
  const [minNotional, setMinNotional] = useState<number>(0);

  useEffect(() => { load(); }, []);
  async function load() {
    try {
      const res = await fetch(endpoint);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      applyPatch(await res.json());
    } catch (e: any) { setError(e.message); }
  }
  function applyPatch(p: Patch) {
    if (p.kpis) setKpis(p.kpis);
    if (p.pnl) setPnl(p.pnl);
    if (p.positions) setPositions(p.positions);
    if (p.orders) setOrders(p.orders);
    if (p.risk) setRisk(p.risk);
    if (p.cash) setCash(p.cash);
    if (p.notes) setNotes(p.notes);
  }

  // ===== Derived =====
  const visiblePositions = useMemo(() => {
    return positions
      .map(p => ({ ...p, multiplier: p.multiplier ?? 1, notional: p.notional ?? Math.abs(p.qty) * (p.last ?? p.avg) * (p.multiplier ?? 1) }))
      .filter(p => (p.notional ?? 0) >= minNotional)
      .sort((a, b) => (b.notional ?? 0) - (a.notional ?? 0));
  }, [positions, minNotional]);

  const totals = useMemo(() => {
    const gross = sum(visiblePositions.map(p => Math.abs(p.qty) * (p.last ?? p.avg) * (p.multiplier ?? 1)));
    const netQty: Record<string, number> = {};
    for (const p of visiblePositions) netQty[p.symbol] = (netQty[p.symbol] || 0) + p.qty * (p.multiplier ?? 1);
    const net = sum(Object.values(netQty).map(v => v));
    const totalUPnL = sum(visiblePositions.map(p => p.uPnl || 0));
    const totalPnL = sum(visiblePositions.map(p => p.pnl || 0));
    return { gross, net, totalUPnL, totalPnL };
  }, [visiblePositions]);

  const byAsset = useMemo(() => aggShare(visiblePositions, p => p.assetClass || 'Other'), [visiblePositions]);
  const bySector = useMemo(() => aggShare(visiblePositions, p => p.sector || 'Other'), [visiblePositions]);
  const byRegion = useMemo(() => aggShare(visiblePositions, p => p.region || 'Global'), [visiblePositions]);

  // ===== UI =====
  return (
    <div className={`flex flex-col h-full rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <label className="text-[11px] text-neutral-500">Quote</label>
          <select value={quoteCcy} onChange={e=>setQuoteCcy(e.target.value)} className="rounded border px-2 py-1 text-xs">
            <option>USD</option><option>INR</option><option>EUR</option><option>GBP</option><option>JPY</option>
          </select>
          <label className="text-[11px] text-neutral-500 ml-2">Min Notional</label>
          <input type="number" value={minNotional} onChange={e=>setMinNotional(Math.max(0, Number(e.target.value)||0))}
                 className="w-24 rounded border px-2 py-1 text-xs"/>
          <label className="text-[11px] text-neutral-500 ml-2 flex items-center gap-1">
            <input type="checkbox" checked={showGreeks} onChange={e=>setShowGreeks(e.target.checked)} className="accent-indigo-600"/>
            Greeks
          </label>
          <button onClick={load} className="px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white text-xs">Reload</button>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-2 p-3">
        {kpis.map((k, i) => (
          <div key={i} className="rounded border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-2 py-1">
            <div className="text-[11px] text-neutral-500">{k.label}</div>
            <div className="text-sm font-semibold">{String(k.value)}</div>
          </div>
        ))}
        <Kpi label="Gross Notional" value={fmt(totals.gross)} />
        <Kpi label="Net Qty (Σ mult)" value={fmt(totals.net, 0)} />
        <Kpi label="Unrealized PnL" value={fmt(totals.totalUPnL)} tone={signTone(totals.totalUPnL)} />
        <Kpi label="Total PnL" value={fmt(totals.totalPnL)} tone={signTone(totals.totalPnL)} />
        {risk ? <>
          <Kpi label="VaR 95" value={fmt(risk.var95)} />
          <Kpi label="ES 97" value={fmt(risk.es97)} />
        </> : null}
      </div>

      {/* Body */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* Left: PnL + Cash */}
        <div className="space-y-3">
          <Card title="PnL">
            {pnl.length ? <PnLChart series={pnl}/> : <Empty label="No PnL yet"/>}
          </Card>
          <Card title="Cash & Margin">
            {cash.length ? (
              <table className="min-w-full text-xs">
                <thead><tr><th className="text-left px-2 py-1">Currency</th><th className="text-right px-2 py-1">Balance</th></tr></thead>
                <tbody>
                  {cash.map((c,i)=>(
                    <tr key={i} className={i%2?'bg-neutral-50 dark:bg-neutral-950':''}>
                      <td className="px-2 py-1">{c.currency}</td>
                      <td className="px-2 py-1 text-right">{fmt(c.balance)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <Empty label="No cash data"/>}
          </Card>
          <Card title="Risk Snapshot">
            {risk ? (
              <div className="grid grid-cols-2 gap-2 text-sm">
                <Row label="Gross" value={fmt(risk.gross)} />
                <Row label="Net" value={fmt(risk.net)} />
                <Row label="VaR 95" value={fmt(risk.var95)} />
                <Row label="VaR 99" value={fmt(risk.var99)} />
                <Row label="ES 97" value={fmt(risk.es97)} />
                <Row label="Beta" value={fmt(risk.beta)} />
                <Row label="Vol" value={fmt(risk.vol)} />
                <Row label="Drawdown" value={risk.dd!=null ? (risk.dd*100).toFixed(1)+'%' : ''} />
              </div>
            ) : <Empty label="No risk metrics"/>}
          </Card>
        </div>

        {/* Middle: Positions */}
        <div className="xl:col-span-1 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Positions ({visiblePositions.length})</div>
          <div className="overflow-auto">
            <table className="min-w-full text-xs">
              <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800">
                <tr>
                  <Th>Sym</Th><Th>Name</Th><Th>Cls</Th><Th>Qty</Th>
                  <Th>Avg</Th><Th>Last</Th><Th>uPnL</Th><Th>TotPnL</Th>
                  {showGreeks ? <><Th>Δ</Th><Th>Γ</Th><Th>V</Th><Th>Θ</Th></> : null}
                </tr>
              </thead>
              <tbody>
                {visiblePositions.map((p,i)=>(
                  <tr key={p.symbol+'-'+i} className={i%2?'bg-neutral-50 dark:bg-neutral-950':''}>
                    <Td>{p.symbol}</Td>
                    <Td>{p.name || ''}</Td>
                    <Td>{p.assetClass || ''}</Td>
                    <Td> className="text-right"{'>'}{fmt(p.qty,0)}</Td>
                    <Td> className="text-right"{'>'}{fmt(p.avg)}</Td>
                    <Td> className="text-right"{'>'}{fmt(p.last)}</Td>
                    <Td> className={`text-right ${toneClass(p.uPnl)}`}{'>'}{fmt(p.uPnl)}</Td>
                    <Td> className={`text-right ${toneClass(p.pnl)}`}{'>'}{fmt(p.pnl)}</Td>
                    {showGreeks ? <>
                      <Td> className="text-right"{'>'}{fmt(p.greeks?.delta)}</Td>
                      <Td> className="text-right"{'>'}{fmt(p.greeks?.gamma,4)}</Td>
                      <Td> className="text-right"{'>'}{fmt(p.greeks?.vega)}</Td>
                      <Td> className="text-right"{'>'}{fmt(p.greeks?.theta)}</Td>
                    </> : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Exposures + Orders + Notes */}
        <div className="space-y-3">
          <Card title="Exposure — Asset Class">
            <Donut data={byAsset} />
          </Card>
          <Card title="Exposure — Sector">
            <Bar data={bySector} />
          </Card>
          <Card title="Exposure — Region">
            <Bar data={byRegion} />
          </Card>
          <Card title="Recent Orders">
            {orders.length ? (
              <div className="overflow-auto max-h-[24vh]">
                <table className="min-w-full text-xs">
                  <thead><tr><Th>Time</Th><Th>Sym</Th><Th>Side</Th><Th>Qty</Th><Th>Px</Th><Th>Status</Th></tr></thead>
                  <tbody>
                    {orders.slice(0,50).map((o,i)=>(
                      <tr key={o.id||i} className={i%2?'bg-neutral-50 dark:bg-neutral-950':''}>
                        <Td>{fmtTs(o.ts)}</Td><Td>{o.symbol}</Td><Td>{o.side}</Td>
                        <Td> className="text-right"{'>'}{fmt(o.qty,0)}</Td>
                        <Td> className="text-right"{'>'}{fmt(o.px)}</Td>
                        <Td>{o.status}</Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <Empty label="No orders"/>}
          </Card>
        </div>
      </div>

      {/* Footer: Notes + Export */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 p-3 border-t border-neutral-200 dark:border-neutral-800">
        <Card title="Notes">
          <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-700 dark:text-neutral-300">{notes || 'No notes.'}</pre>
          {error && <div className="text-xs text-rose-600 mt-2">{error}</div>}
        </Card>
        <Card title="Export">
          <div className="flex items-center gap-2 text-xs">
            <button onClick={()=>downloadCSV('positions.csv', positionsToCSV(visiblePositions))}
              className="px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Positions CSV</button>
            <button onClick={()=>downloadCSV('orders.csv', ordersToCSV(orders))}
              className="px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Orders CSV</button>
            <button onClick={()=>downloadCSV('pnl.csv', pnlToCSV(pnl))}
              className="px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">PnL CSV</button>
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ============== Subcomponents ============== */
function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
      <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">{title}</div>
      <div className="p-3">{children}</div>
    </div>
  );
}
function Empty({ label }: { label: string }) { return <div className="text-xs text-neutral-500">{label}</div>; }
function Th({ children }:{ children:React.ReactNode }){ return <th className="px-2 py-1 text-left font-semibold">{children}</th>; }
function Td({ children }:{ children:React.ReactNode }){ return <td className="px-2 py-1">{children}</td>; }
function Row({ label, value }:{label:string; value:string}){ return <div className="flex justify-between"><span className="text-[11px] text-neutral-500">{label}</span><span className="text-sm">{value}</span></div>; }
function Kpi({ label, value, tone=0 }:{label:string; value:string; tone?:-1|0|1}) {
  const color = tone>0 ? 'text-emerald-700' : tone<0 ? 'text-rose-700' : 'text-neutral-900 dark:text-neutral-100';
  return (
    <div className="rounded border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-2 py-1">
      <div className="text-[11px] text-neutral-500">{label}</div>
      <div className={`text-sm font-semibold ${color}`}>{value}</div>
    </div>
  );
}
const toneClass = (x?: number) => (x ?? 0) >= 0 ? 'text-emerald-700' : 'text-rose-700';
const signTone = (x?: number) => (x ?? 0) >= 0 ? 1 : -1;

/* ====== Inline charts ====== */
function PnLChart({ series }: { series: PnLPoint[] }) {
  const width = 420, height = 140;
  const ys = series.map(p => p.v);
  const ymin = Math.min(...ys, 0), ymax = Math.max(...ys, 0);
  const span = ymax - ymin || 1;
  const step = width / Math.max(1, series.length - 1);
  const path = series.map((p, i) => {
    const x = i * step; const y = height - ((p.v - ymin) / span) * height;
    return `${i === 0 ? 'M' : 'L'}${x},${y}`;
  }).join(' ');
  const tone = (ys[ys.length - 1] ?? 0) >= (ys[0] ?? 0) ? '#059669' : '#dc2626';
  return <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}><path d={path} fill="none" stroke={tone} strokeWidth={1.5} /></svg>;
}

function Donut({ data }: { data: { key: string; value: number; share: number }[] }) {
  const width = 240, height = 160, r = 60, cx = 120, cy = 80;
  const total = sum(data.map(d => d.value)) || 1;
  let a0 = -Math.PI / 2;
  const arcs = data.map((d, i) => {
    const a1 = a0 + 2 * Math.PI * (d.value / total);
    const arc = arcPath(cx, cy, r, a0, a1);
    a0 = a1;
    return { key: d.key, d: arc, color: color10(i) };
  });
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      {arcs.map(a => <path key={a.key} d={a.d} fill="none" stroke={a.color} strokeWidth={24} />)}
      <g transform={`translate(${cx + r + 30},20)`}>
        {data.map((d, i) => (
          <g key={d.key} transform={`translate(0,${i * 18})`}>
            <rect width="10" height="10" fill={color10(i)} />
            <text x="14" y="10" fontSize="11" fill="#374151">{`${d.key} ${(d.share*100).toFixed(1)}%`}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
function Bar({ data }: { data: { key: string; value: number; share: number }[] }) {
  const width = 360, height = 120, max = Math.max(...data.map(d => d.value), 1);
  const barW = Math.max(12, (width - 20) / Math.max(1, data.length));
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      {data.map((d, i) => {
        const h = (d.value / max) * (height - 28);
        return (
          <g key={d.key} transform={`translate(${10 + i * barW}, ${height - 18 - h})`}>
            <rect width={barW - 6} height={h} fill={color10(i)} />
            <text x={(barW - 6) / 2} y={h + 12} textAnchor="middle" fontSize="10" fill="#374151">{d.key}</text>
          </g>
        );
      })}
    </svg>
  );
}
function arcPath(cx:number, cy:number, r:number, a0:number, a1:number){
  const large = (a1-a0) % (Math.PI*2) > Math.PI ? 1 : 0;
  const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0);
  const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
  return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
}
function color10(i:number){ const c = ['#4f46e5','#10b981','#f59e0b','#ef4444','#06b6d4','#8b5cf6','#64748b','#a855f7','#22c55e','#fb7185']; return c[i % c.length]; }

/* ====== Aggregations & Export ====== */
function aggShare(ps: Position[], key: (p: Position) => string): { key: string; value: number; share: number }[] {
  const groups = by(ps, key);
  const totals: { key: string; value: number }[] = [];
  let grand = 0;
  for (const [k, arr] of groups.entries()) {
    const v = sum(arr.map(p => Math.abs(p.qty) * (p.last ?? p.avg) * (p.multiplier ?? 1)));
    totals.push({ key: k, value: v }); grand += v;
  }
  return totals.sort((a,b)=>b.value-a.value).map(t => ({ ...t, share: grand ? t.value / grand : 0 }));
}
function positionsToCSV(ps: Position[]) {
  const cols = ['symbol','name','assetClass','sector','region','qty','avg','last','uPnl','pnl','notional','currency'];
  const head = cols.join(',');
  const lines = ps.map(p => cols.map(k => {
    const v: any = (p as any)[k];
    return `"${String(v ?? '').replace(/"/g,'""')}"`;
  }).join(','));
  return [head, ...lines].join('\n');
}
function ordersToCSV(os: Order[]) {
  const cols = ['id','ts','symbol','side','qty','px','status'];
  const head = cols.join(',');
  const lines = os.map(o => [o.id, new Date(o.ts).toISOString(), o.symbol, o.side, o.qty, o.px ?? '', o.status]
    .map(x => `"${String(x).replace(/"/g,'""')}"`).join(','));
  return [head, ...lines].join('\n');
}
function pnlToCSV(xs: PnLPoint[]) {
  const head = 't,v';
  const lines = xs.map(p => `${new Date(p.t).toISOString()},${p.v}`);
  return [head, ...lines].join('\n');
}
function downloadCSV(filename: string, text: string) {
  const blob = new Blob([text], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}