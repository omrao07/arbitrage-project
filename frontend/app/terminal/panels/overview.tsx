'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ===== Types ===== */
type KPI = { label: string; value: string|number };
type PnLPoint = { t: number; v: number };
type Position = { symbol: string; qty: number; avg: number; pnl: number; uPnl?: number };
type Order = { id: string; ts: number; symbol: string; side: string; qty: number; px?: number; status: string };
type Alert = { ts: number; level: 'info'|'warn'|'risk'; text: string };

type Patch = {
  kpis?: KPI[];
  pnl?: PnLPoint[];
  positions?: Position[];
  orders?: Order[];
  alerts?: Alert[];
  notes?: string;
};

type Props = {
  endpoint?: string;
  title?: string;
  className?: string;
};

/* ===== Utils ===== */
const fmt = (n:number,d=2)=>Number.isFinite(n)?n.toFixed(d):'';
const fmtTs = (ms:number)=> new Date(ms).toLocaleString();

/* ===== Component ===== */
export default function Overview({
  endpoint = '/api/overview',
  title = 'Overview',
  className = '',
}:Props){
  const [kpis,setKpis] = useState<KPI[]>([]);
  const [pnl,setPnl] = useState<PnLPoint[]>([]);
  const [positions,setPositions] = useState<Position[]>([]);
  const [orders,setOrders] = useState<Order[]>([]);
  const [alerts,setAlerts] = useState<Alert[]>([]);
  const [notes,setNotes] = useState('');
  const [error,setError] = useState<string|null>(null);

  useEffect(()=>{ load(); },[]);
  async function load(){
    try{
      const res = await fetch(endpoint);
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      applyPatch(await res.json());
    }catch(e:any){ setError(e.message); }
  }
  function applyPatch(p:Patch){
    if(p.kpis) setKpis(p.kpis);
    if(p.pnl) setPnl(p.pnl);
    if(p.positions) setPositions(p.positions);
    if(p.orders) setOrders(p.orders);
    if(p.alerts) setAlerts(p.alerts);
    if(p.notes) setNotes(p.notes);
  }

  return (
    <div className={`flex flex-col h-full rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      <div className="px-3 py-2 border-b text-sm font-medium">{title}</div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 p-3">
        {kpis.map((k,i)=>(
          <div key={i} className="rounded border px-2 py-1 text-sm">
            <div className="text-[11px] text-neutral-500">{k.label}</div>
            <div className="font-semibold">{String(k.value)}</div>
          </div>
        ))}
      </div>

      {/* pnl chart + positions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 flex-1 overflow-hidden">
        <Card title="PnL">
          {pnl.length? <PnLChart series={pnl}/> : <Empty label="No data"/>}
        </Card>
        <Card title="Positions">
          <div className="overflow-auto max-h-[25vh]">
            <table className="min-w-full text-xs">
              <thead><tr><th>Symbol</th><th>Qty</th><th>Avg</th><th>PnL</th></tr></thead>
              <tbody>
                {positions.map((p,i)=>(
                  <tr key={i}><td>{p.symbol}</td><td>{p.qty}</td><td>{fmt(p.avg)}</td><td className={p.pnl>=0?'text-emerald-600':'text-rose-600'}>{fmt(p.pnl)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Orders">
          <div className="overflow-auto max-h-[25vh]">
            <table className="min-w-full text-xs">
              <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Px</th><th>Status</th></tr></thead>
              <tbody>
                {orders.map((o,i)=>(
                  <tr key={i}>
                    <td>{fmtTs(o.ts)}</td><td>{o.symbol}</td><td>{o.side}</td><td>{o.qty}</td>
                    <td>{fmt(o.px||0)}</td><td>{o.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* alerts + notes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 p-3 border-t">
        <Card title="Alerts">
          <ul className="text-xs space-y-1">
            {alerts.map((a,i)=>(
              <li key={i} className={a.level==='risk'?'text-rose-600':a.level==='warn'?'text-amber-600':'text-neutral-600'}>
                {fmtTs(a.ts)} · {a.text}
              </li>
            ))}
          </ul>
        </Card>
        <Card title="Notes">
          <pre className="text-[11px] whitespace-pre-wrap">{notes||'No notes'}</pre>
          {error && <div className="text-xs text-rose-600">{error}</div>}
        </Card>
      </div>
    </div>
  );
}

/* ===== Subcomponents ===== */
function Card({title,children}:{title:string;children:React.ReactNode}){
  return <div className="rounded border bg-white dark:bg-neutral-900 overflow-hidden">
    <div className="border-b px-2 py-1 text-xs font-medium">{title}</div>
    <div className="p-2">{children}</div>
  </div>;
}
function Empty({label}:{label:string}){return <div className="text-xs text-neutral-500">{label}</div>; }

function PnLChart({series}:{series:PnLPoint[]}){
  const width=360, height=120;
  const ys=series.map(p=>p.v); const ymin=Math.min(...ys), ymax=Math.max(...ys); const span=ymax-ymin||1;
  const step=width/Math.max(1,series.length-1);
  const path=series.map((p,i)=>{const x=i*step;const y=height-((p.v-ymin)/span)*height;return`${i===0?'M':'L'}${x},${y}`;}).join(' ');
  const tone=(ys[ys.length-1]??0)>=(ys[0]??0)?'#059669':'#dc2626';
  return <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
    <path d={path} fill="none" stroke={tone} strokeWidth={1.5}/>
  </svg>;
}