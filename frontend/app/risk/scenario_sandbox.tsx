'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* ========= Types ========= */
type KPI = { label: string; value: string | number };
type Point = { t: number; v: number };
type Attribution = {
  bucket: string;
  pnl: number;
  exposure?: number;
  stress?: number;
};

type Patch = {
  kpis?: KPI[];
  series?: Point[];
  attribution?: Attribution[];
  notes?: string;
};

type Props = {
  endpoint?: string;
  title?: string;
  className?: string;
};

/* ========= Utils ========= */
const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const fmtMoney = (x:number, c='USD') =>
  (x<0?'-':'') + new Intl.NumberFormat(undefined,{style:'currency',currency:c,maximumFractionDigits:0}).format(Math.abs(x));
const fmtPct = (x:number|undefined,d=1)=> typeof x==='number'?`${(x*100).toFixed(d)}%`:'';
const clamp = (x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));

/* ========= Component ========= */
export default function ScenarioSandbox({
  endpoint = '/api/scenario/sandbox',
  title = 'Scenario Sandbox',
  className = '',
}: Props) {
  const [shock, setShock] = useState<'rates'|'fx'|'equity'|'credit'|'vol'>('rates');
  const [magnitude, setMagnitude] = useState<number>(100); // bps or %
  const [horizon, setHorizon] = useState<'1d'|'1w'|'1m'>('1w');
  const [currency, setCurrency] = useState<'USD'|'EUR'|'GBP'|'INR'>('USD');

  const [kpis,setKpis] = useState<KPI[]>([]);
  const [series,setSeries] = useState<Point[]>([]);
  const [attrib,setAttrib] = useState<Attribution[]>([]);
  const [notes,setNotes] = useState('');
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState<string|null>(null);
  const [log,setLog] = useState('');
  const abortRef = useRef<AbortController|null>(null);
  const logRef = useRef<HTMLDivElement|null>(null);
  useEffect(()=>{ logRef.current?.scrollTo({top:logRef.current.scrollHeight}); },[log,loading]);

  const maxShock = useMemo(()=>Math.max(...attrib.map(a=>Math.abs(a.pnl)),1),[attrib]);

  function stop(){
    abortRef.current?.abort();
    abortRef.current=null;
    setLoading(false);
  }
  async function run(){
    setLoading(true); setError(null);
    setKpis([]); setSeries([]); setAttrib([]); setNotes(''); setLog('');
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const {signal} = abortRef.current;
    const payload = { id: uid(), shock, magnitude, horizon, currency };

    try{
      const res = await fetch(endpoint,{
        method:'POST',
        headers:{'Content-Type':'application/json','Accept':'text/event-stream,application/json,text/plain'},
        body:JSON.stringify(payload), signal,
      });
      const ctype = res.headers.get('content-type')||'';
      if(!res.ok) throw new Error(`HTTP ${res.status}`);

      if(res.body && ctype.includes('text/event-stream')){
        const reader=res.body.getReader(); const dec=new TextDecoder('utf-8'); let acc='';
        while(true){
          const {value,done} = await reader.read(); if(done) break;
          const chunk=dec.decode(value,{stream:true});
          const lines=(acc+chunk).split('\n'); acc=lines.pop()||'';
          for(const line of lines){
            if(!line.startsWith('data:')) continue;
            const data=line.slice(5).trim(); if(!data) continue;
            if(data==='[DONE]') continue;
            setLog(s=>s+data+'\n');
            try{ applyPatch(JSON.parse(data)); }catch{ setNotes(n=>(n?n+'\n':'')+data); }
          }
        }
      }else if(ctype.includes('application/json')){
        applyPatch(await res.json());
      }else{
        setNotes(await res.text());
      }
    }catch(e:any){
      if(e?.name!=='AbortError') setError(e?.message||'Request failed');
    }finally{ setLoading(false); abortRef.current=null; }
  }

  function applyPatch(p:Patch){
    if(Array.isArray(p.kpis)) setKpis(p.kpis);
    if(Array.isArray(p.series)) setSeries(p.series);
    if(Array.isArray(p.attribution)) setAttrib(p.attribution);
    if(typeof p.notes==='string') setNotes(n=>(n?n+'\n':'')+p.notes);
  }

  return (
    <div className={`flex flex-col h-full w-full rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex gap-2">
          {loading?(
            <button onClick={stop} className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700">Stop</button>
          ):(
            <button onClick={run} className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white">Run</button>
          )}
        </div>
      </div>

      {/* controls */}
      <div className="p-3 grid grid-cols-2 md:grid-cols-4 gap-3 border-b border-neutral-200 dark:border-neutral-800">
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">Shock</label>
          <select value={shock} onChange={e=>setShock(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="rates">Rates</option>
            <option value="fx">FX</option>
            <option value="equity">Equity</option>
            <option value="credit">Credit Spreads</option>
            <option value="vol">Volatility</option>
          </select>
        </div>
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">Magnitude</label>
          <input type="number" value={magnitude} onChange={e=>setMagnitude(Number(e.target.value))} className="w-full rounded border px-2 py-2 text-sm"/>
        </div>
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">Horizon</label>
          <select value={horizon} onChange={e=>setHorizon(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option value="1d">1D</option><option value="1w">1W</option><option value="1m">1M</option>
          </select>
        </div>
        <div>
          <label className="block text-[11px] text-neutral-500 mb-1">Currency</label>
          <select value={currency} onChange={e=>setCurrency(e.target.value as any)} className="w-full rounded border px-2 py-2 text-sm">
            <option>USD</option><option>EUR</option><option>GBP</option><option>INR</option>
          </select>
        </div>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        <div className="space-y-3">
          <Card title="KPIs">
            {kpis.length?(
              <div className="grid grid-cols-2 gap-2">
                {kpis.map((k,i)=>(
                  <div key={i} className="rounded border px-3 py-2">
                    <div className="text-[11px] text-neutral-500">{k.label}</div>
                    <div className="text-sm font-semibold">{String(k.value)}</div>
                  </div>
                ))}
              </div>
            ):<Empty label="No KPIs"/>}
          </Card>
          <Card title="Equity Impact">
            {series.length? <LineChart series={series}/> : <Empty label="No series"/>}
          </Card>
        </div>
        <div className="lg:col-span-2 min-h-0 rounded border flex flex-col">
          <div className="border-b px-3 py-2 text-xs font-medium">Attribution ({attrib.length})</div>
          <div className="overflow-auto flex-1">
            <table className="min-w-full text-sm">
              <thead className="sticky top-0 bg-neutral-100 text-xs">
                <tr><th className="px-2 py-2">Bucket</th><th>PnL</th><th>Exposure</th><th>Stress</th></tr>
              </thead>
              <tbody>
                {attrib.map((r,i)=>(
                  <tr key={i} className={i%2?'bg-neutral-50':''}>
                    <td className="px-2 py-1">{r.bucket}</td>
                    <td className={(r.pnl||0)>=0?'text-emerald-700':'text-rose-700'}>{fmtMoney(r.pnl,currency)}</td>
                    <td>{r.exposure!=null?fmtMoney(r.exposure,currency):''}</td>
                    <td>{r.stress!=null?fmtPct(r.stress):''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* notes + log */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 p-3 border-t">
        <Card title="Notes"><pre className="text-[11px] whitespace-pre-wrap">{notes||'No notes.'}</pre></Card>
        <Card title="Run Log"><div ref={logRef} className="max-h-[20vh] overflow-auto"><pre className="text-[11px] whitespace-pre-wrap">{error?error:log||'No output.'}</pre></div></Card>
      </div>
    </div>
  );
}

/* ========= Subcomponents ========= */
function Card({title,children}:{title:string;children:React.ReactNode}){
  return <div className="rounded border bg-white dark:bg-neutral-900 overflow-hidden">
    <div className="border-b px-3 py-2 text-xs font-medium">{title}</div>
    <div className="p-3">{children}</div>
  </div>;
}
function Empty({label}:{label:string}){ return <div className="text-xs text-neutral-500">{label}</div>; }

function LineChart({series,width=360,height=120}:{series:Point[];width?:number;height?:number}){
  const ys=series.map(p=>p.v); const ymin=Math.min(...ys), ymax=Math.max(...ys); const span=ymax-ymin||1;
  const step=width/Math.max(1,series.length-1);
  const path=series.map((p,i)=>{const x=i*step; const y=height-((p.v-ymin)/span)*height; return `${i===0?'M':'L'}${x.toFixed(2)},${y.toFixed(2)}`;}).join(' ');
  const tone=(ys[ys.length-1]??0) >= (ys[0]??0)?'#059669':'#dc2626';
  return <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}><path d={path} fill="none" stroke={tone} strokeWidth={1.5}/></svg>;
}