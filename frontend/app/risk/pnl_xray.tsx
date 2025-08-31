'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ===================== Types ===================== */
type KPI = { label: string; value: string | number };
type SeriesPoint = { t: number; v: number }; // equity or PnL over time
type BreakdownRow = {
  bucket: string;            // e.g., strategy name / symbol / venue / YYYY-MM-DD
  pnl: number;               // net pnl
  gross?: number;            // before costs
  fees?: number;
  slippage?: number;
  borrow?: number;
  trades?: number;
  winrate?: number;          // 0..1
  sharpe?: number;
  exposure?: number;         // avg exposure
};
type TradeRow = {
  ts: number;
  symbol: string;
  strategy?: string;
  side: 'BUY'|'SELL';
  qty: number;
  price: number;
  fee?: number;
  slip?: number;
  pnl?: number;
  orderId?: string;
  venue?: string;
};
type Patch = {
  kpis?: KPI[];
  series?: SeriesPoint[];      // cumulative equity
  breakdown?: BreakdownRow[];  // table rows
  trade?: TradeRow;
  trades?: TradeRow[];
  notes?: string;
  columns?: string[];          // optional columns override for trades table
};
type Props = {
  endpoint?: string;           // e.g. '/api/pnl/xray'
  title?: string;
  className?: string;
};

/* ===================== Utils ===================== */
const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const clamp = (x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
const fmtMoney = (x:number|undefined, c='USD') => (typeof x==='number' ? (x<0?'-':'') + new Intl.NumberFormat(undefined,{style:'currency',currency:c,maximumFractionDigits:0}).format(Math.abs(x)) : '');
const fmtNum = (x:number|undefined, d=2)=> typeof x==='number'?x.toFixed(d):'';
const pct = (x:number|undefined, d=0)=> typeof x==='number'?`${(x*100).toFixed(d)}%`:'';
const by = <T,>(arr:T[], key:(t:T)=>string)=>Object.values(arr.reduce<Record<string,T[]>>((m, r)=>{const k=key(r); (m[k] ||= []).push(r); return m;}, {}));

/* ===================== Component ===================== */
export default function PnlXray({
  endpoint = '/api/pnl/xray',
  title = 'PnL X-Ray',
  className = '',
}: Props) {
  // filters / controls
  const [from, setFrom] = useState<string>(''); // YYYY-MM-DD
  const [to, setTo] = useState<string>('');
  const [groupBy, setGroupBy] = useState<'strategy'|'symbol'|'venue'|'day'>('strategy');
  const [currency, setCurrency] = useState<'USD'|'EUR'|'GBP'|'INR'>('USD');
  const [net, setNet] = useState<boolean>(true);
  const [query, setQuery] = useState<string>('');

  // data
  const [kpis, setKpis] = useState<KPI[]>([]);
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const [breakdown, setBreakdown] = useState<BreakdownRow[]>([]);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [columns, setColumns] = useState<string[]|null>(null);
  const [notes, setNotes] = useState<string>('');

  // run state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string|null>(null);
  const [log, setLog] = useState<string>('');
  const abortRef = useRef<AbortController|null>(null);
  const logRef = useRef<HTMLDivElement|null>(null);
  useEffect(()=>{ logRef.current?.scrollTo({top:logRef.current.scrollHeight, behavior:'smooth'}); },[log,loading]);

  // derived
  const filteredBreakdown = useMemo(()=>{
    const q = query.trim().toLowerCase();
    let rows = breakdown.map(r=>{
      if (net) return r;
      // if net=false and gross available, show gross and disaggregate costs visually only
      return { ...r, pnl: (r.gross ?? r.pnl) };
    });
    if (q) rows = rows.filter(r => r.bucket.toLowerCase().includes(q));
    // sort desc by pnl magnitude
    rows.sort((a,b)=>(b.pnl ?? 0) - (a.pnl ?? 0));
    return rows;
  },[breakdown, query, net]);

  const topTiles = useMemo(()=> filteredBreakdown.slice(0, 12), [filteredBreakdown]);

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  async function run() {
    setLoading(true);
    setError(null);
    setLog('');
    setKpis([]); setSeries([]); setBreakdown([]); setTrades([]); setColumns(null); setNotes('');

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;
    const payload = {
      id: uid(),
      from: from || null,
      to: to || null,
      groupBy,
      currency,
      net,
    };

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type':'application/json', Accept:'text/event-stream, application/json, text/plain' },
        body: JSON.stringify(payload),
        signal,
      });
      const ctype = res.headers.get('content-type') || '';
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (res.body && (ctype.includes('text/event-stream') || ctype.includes('text/plain'))) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let acc = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream:true });

          if (ctype.includes('text/event-stream')) {
            const lines = (acc + chunk).split('\n'); acc = lines.pop() || '';
            for (const line of lines) {
              if (!line.startsWith('data:')) continue;
              const data = line.slice(5).trim();
              if (!data) continue;
              if (data === '[DONE]') continue;
              setLog(s=>s + data + '\n');
              try {
                applyPatch(JSON.parse(data));
              } catch {
                setNotes(n => (n ? n + '\n' : '') + data);
              }
            }
          } else {
            setLog(s=>s + chunk);
          }
        }
      } else if (ctype.includes('application/json')) {
        applyPatch(await res.json());
      } else {
        setNotes(await res.text());
      }
    } catch(e:any) {
      if (e?.name!=='AbortError') setError(e?.message || 'Request failed');
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function applyPatch(p: Patch) {
    if (Array.isArray(p.kpis)) setKpis(p.kpis);
    if (Array.isArray(p.series)) setSeries(p.series);
    if (Array.isArray(p.breakdown)) setBreakdown(p.breakdown);
    if (Array.isArray(p.trades)) setTrades(prev => [...prev, ...p.trades!]);
    if (p.trade) setTrades(prev => [...prev, p.trade!]);
    if (Array.isArray(p.columns)) setColumns(p.columns);
    if (typeof p.notes === 'string') setNotes(n => (n ? n + '\n' : '') + p.notes);
  }

  function exportCSV() {
    const cols = ['ts','symbol','strategy','side','qty','price','fee','slip','pnl','orderId','venue'];
    const header = cols.join(',');
    const lines = trades.map(t => cols.map(c => {
      const v:any = (t as any)[c];
      if (typeof v === 'number') return v;
      return `"${String(v ?? '').replace(/"/g,'""')}"`;
    }).join(','));
    const csv = [header, ...lines].join('\n');
    const blob = new Blob([csv], { type:'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href=url; a.download='trades.csv'; a.click(); URL.revokeObjectURL(url);
  }

  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button onClick={exportCSV} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export Trades</button>
          {loading ? (
            <button onClick={stop} className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700 hover:bg-rose-50">Stop</button>
          ) : (
            <button onClick={run} className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700">Run</button>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 border-b border-neutral-200 dark:border-neutral-800 grid grid-cols-1 md:grid-cols-7 gap-3">
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">From</label>
          <input type="date" value={from} onChange={e=>setFrom(e.target.value)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm"/>
        </div>
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">To</label>
          <input type="date" value={to} onChange={e=>setTo(e.target.value)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm"/>
        </div>
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">Group By</label>
          <select value={groupBy} onChange={e=>setGroupBy(e.target.value as any)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
            <option value="strategy">Strategy</option>
            <option value="symbol">Symbol</option>
            <option value="venue">Venue</option>
            <option value="day">Day</option>
          </select>
        </div>
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">Currency</label>
          <select value={currency} onChange={e=>setCurrency(e.target.value as any)} className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-2 text-sm">
            <option>USD</option><option>EUR</option><option>GBP</option><option>INR</option>
          </select>
        </div>
        <div className="flex items-end">
          <label className="inline-flex items-center gap-2 text-sm">
            <input type="checkbox" checked={net} onChange={e=>setNet(e.target.checked)} className="accent-indigo-600"/>
            Net (include fees/slip/borrow)
          </label>
        </div>
        <div className="md:col-span-2">
          <label className="block text-[11px] text-neutral-500 mb-1">Search buckets</label>
          <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Filter by strategy/symbol/venue/day" className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 text-sm"/>
        </div>
      </div>

      {/* Body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        <div className="space-y-3">
          <Card title="KPIs">
            {kpis.length ? (
              <div className="grid grid-cols-2 gap-2">
                {kpis.map((k,i)=>(
                  <div key={i} className="rounded border border-neutral-200 dark:border-neutral-800 px-3 py-2">
                    <div className="text-[11px] text-neutral-500">{k.label}</div>
                    <div className="text-sm font-semibold">{String(k.value)}</div>
                  </div>
                ))}
              </div>
            ) : <Empty label="No KPIs yet" />}
          </Card>

          <Card title="Equity Curve">
            {series.length ? <LineChart series={series}/> : <Empty label="No series yet" />}
          </Card>

          <Card title="Top Buckets">
            {topTiles.length ? <Tiles rows={topTiles} currency={currency} net={net}/> : <Empty label="No buckets" />}
          </Card>
        </div>

        {/* Breakdown Table */}
        <div className="lg:col-span-2 min-h-0 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden flex flex-col">
          <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">Attribution ({filteredBreakdown.length})</div>
          <div className="overflow-auto">
            <Table rows={filteredBreakdown} currency={currency} net={net}/>
          </div>
        </div>
      </div>

      {/* Trades + Log */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 border-t border-neutral-200 dark:border-neutral-800">
        <Card title="Trades">
          {trades.length ? <TradesTable rows={trades} columns={columns}/> : <Empty label="No trades yet" />}
        </Card>
        <Card title="Notes">
          <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-700 dark:text-neutral-300">{notes || 'No notes.'}</pre>
        </Card>
        <Card title="Run Log">
          <div ref={logRef} className="max-h-[26vh] overflow-auto">
            {error ? <div className="text-xs text-rose-600">{error}</div> : <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-600 dark:text-neutral-300">{log || 'No streaming output yet.'}</pre>}
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ===================== Subcomponents ===================== */
function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
      <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">{title}</div>
      <div className="p-3">{children}</div>
    </div>
  );
}
function Empty({ label }: { label: string }) {
  return <div className="text-xs text-neutral-500">{label}</div>;
}

/* ---- Tiny Line Chart ---- */
function LineChart({ series, width=360, height=120 }: { series: SeriesPoint[]; width?:number; height?:number }) {
  const ys = series.map(p=>p.v);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const span = (ymax - ymin) || 1;
  const stepX = width / Math.max(1, series.length - 1);
  const path = series.map((p,i)=>{
    const x = i*stepX;
    const y = height - ((p.v - ymin)/span)*height;
    return `${i===0?'M':'L'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
  const tone = (ys[ys.length-1] ?? 0) >= (ys[0] ?? 0) ? '#059669' : '#dc2626';
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      <path d={path} fill="none" stroke={tone} strokeWidth={1.6}/>
    </svg>
  );
}

/* ---- Treemap-like Tiles ---- */
function Tiles({ rows, currency, net }: { rows: BreakdownRow[]; currency:string; net:boolean }) {
  const max = Math.max(...rows.map(r=>Math.abs(r.pnl||0)), 1);
  return (
    <div className="grid grid-cols-2 gap-2">
      {rows.map(r=>{
        const ratio = clamp(Math.abs(r.pnl||0)/max, 0.05, 1);
        const bg = (r.pnl||0) >= 0 ? `rgba(16,185,129,${0.15 + 0.5*ratio})` : `rgba(239,68,68,${0.15 + 0.5*ratio})`;
        return (
          <div key={r.bucket} className="rounded border border-neutral-200 dark:border-neutral-800 p-2" style={{ background:bg }}>
            <div className="text-xs font-semibold truncate">{r.bucket}</div>
            <div className="text-sm">{fmtMoney(r.pnl, currency)}</div>
            {!net && r.gross!=null && (
              <div className="text-[11px] text-neutral-700">gross {fmtMoney(r.gross, currency)}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ---- Breakdown Table ---- */
function Table({ rows, currency, net }: { rows: BreakdownRow[]; currency:string; net:boolean }) {
  const cols = ['bucket','pnl','gross','fees','slippage','borrow','trades','winrate','sharpe','exposure'];
  return (
    <table className="min-w-full text-sm">
      <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800 text-xs">
        <tr>{cols.map(c=>(
          <th key={c} className="px-2 py-2 text-left font-semibold border-b border-neutral-200 dark:border-neutral-700">{c}</th>
        ))}</tr>
      </thead>
      <tbody>
        {rows.map((r,i)=>(
          <tr key={r.bucket} className={i%2 ? 'bg-neutral-50 dark:bg-neutral-950' : 'bg-white dark:bg-neutral-900'}>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.bucket}</td>
            <td className={`px-2 py-2 border-b border-neutral-100 dark:border-neutral-800 ${ (r.pnl||0)>=0 ? 'text-emerald-700' : 'text-rose-700' }`}>{fmtMoney(r.pnl, currency)}</td>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.gross!=null?fmtMoney(r.gross, currency):''}</td>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.fees!=null?fmtMoney(r.fees, currency):''}</td>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.slippage!=null?fmtMoney(r.slippage, currency):''}</td>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.borrow!=null?fmtMoney(r.borrow, currency):''}</td>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.trades ?? ''}</td>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.winrate!=null?pct(r.winrate,0):''}</td>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.sharpe!=null?fmtNum(r.sharpe,2):''}</td>
            <td className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{r.exposure!=null?fmtNum(r.exposure,0):''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ---- Trades Table ---- */
function TradesTable({ rows, columns }: { rows: TradeRow[]; columns?: string[]|null }) {
  const cols = columns && columns.length ? columns : ['ts','symbol','strategy','side','qty','price','fee','slip','pnl','venue','orderId'];
  return (
    <div className="overflow-auto max-h-[26vh]">
      <table className="min-w-full text-sm">
        <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800 text-xs">
          <tr>{cols.map(c=>(
            <th key={c} className="px-2 py-2 text-left font-semibold border-b border-neutral-200 dark:border-neutral-700">{c}</th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, i)=>(
            <tr key={i} className={i%2 ? 'bg-neutral-50 dark:bg-neutral-950' : 'bg-white dark:bg-neutral-900'}>
              {cols.map(c=>{
                const v:any = (r as any)[c];
                let text = '';
                if (c==='ts') text = new Date(r.ts).toLocaleString();
                else if (['fee','slip','pnl'].includes(c)) text = typeof v==='number'? v.toFixed(2):'';
                else if (typeof v==='number' && c!=='qty') text = v.toFixed(2);
                else text = String(v ?? '');
                return <td key={c} className="px-2 py-2 border-b border-neutral-100 dark:border-neutral-800">{text}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}