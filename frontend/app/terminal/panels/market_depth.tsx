'use client';
import React, { useEffect, useMemo, useRef, useState } from 'react';

/* =============== Types =============== */
type Side = 'bid'|'ask';
type L2 = { price:number; size:number };          // one level
type Snapshot = { bids:L2[]; asks:L2[]; last?:number; ts?:number };
type Delta = { side:Side; price:number; size:number }; // size=0 means remove
type Patch = {
  snapshot?: Snapshot;
  deltas?: Delta[];
  trade?: { price:number; size:number; ts:number };
  symbol?: string;
  notes?: string;
};
type Props = {
  endpoint?: string;              // e.g. '/api/market/depth'
  title?: string;
  defaultSymbol?: string;
  className?: string;
};

/* =============== Utils =============== */
const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
const fmtPx = (x:number, d=2) => Number.isFinite(x) ? x.toFixed(d) : '';
const fmtSz = (x:number) => Number.isFinite(x) ? (x>=1000 ? (x/1000).toFixed(1)+'k' : x.toFixed(2)) : '';
const clamp = (x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
function download(filename:string, text:string, mime='text/plain'){
  const blob = new Blob([text], { type:mime }); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
}

/* =============== Component =============== */
export default function MarketDepth({
  endpoint = '/api/market/depth',
  title = 'Market Depth',
  defaultSymbol = 'AAPL',
  className = '',
}: Props){
  // controls
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [agg, setAgg] = useState<number>(0.01);        // price aggregation step
  const [levels, setLevels] = useState<number>(15);    // rows per side
  const [throttleMs, setThrottleMs] = useState<number>(80); // render throttle
  const [priceDp, setPriceDp] = useState<number>(2);

  // state
  const [bids, setBids] = useState<L2[]>([]);
  const [asks, setAsks] = useState<L2[]>([]);
  const [last, setLast] = useState<number|undefined>(undefined);
  const [ts, setTs] = useState<number|undefined>(undefined);
  const [notes, setNotes] = useState<string>('');
  const [error, setError] = useState<string|null>(null);
  const [loading, setLoading] = useState(false);
  const [log, setLog] = useState<string>('');

  const abortRef = useRef<AbortController|null>(null);
  const queueRef = useRef<Delta[]>([]);
  const lastFlushRef = useRef<number>(0);

  // derived
  const mid = useMemo(()=> {
    const bb = bids[0]?.price, aa = asks[0]?.price;
    if (!Number.isFinite(bb) || !Number.isFinite(aa)) return undefined;
    return (bb + aa) / 2;
  }, [bids, asks]);
  const spread = useMemo(()=> {
    const bb = bids[0]?.price, aa = asks[0]?.price;
    if (!Number.isFinite(bb) || !Number.isFinite(aa)) return undefined;
    return aa - bb;
  }, [bids, asks]);

  const visBids = bids.slice(0, levels);
  const visAsks = asks.slice(0, levels);
  const maxSize = Math.max(...visBids.map(x=>x.size), ...visAsks.map(x=>x.size), 1);

  const imbalance = useMemo(()=>{
    const b = visBids.reduce((s,x)=>s+x.size,0);
    const a = visAsks.reduce((s,x)=>s+x.size,0);
    const t = b+a || 1;
    return (b - a) / t; // -1..+1
  }, [visBids, visAsks]);

  const vwap5 = useMemo(()=> vwapN([...visBids, ...visAsks], 5), [visBids, visAsks]);
  const vwap10= useMemo(()=> vwapN([...visBids, ...visAsks],10), [visBids, visAsks]);
  const vwap20= useMemo(()=> vwapN([...visBids, ...visAsks],20), [visBids, visAsks]);

  // throttled animation flush
  useEffect(()=>{
    const t = setInterval(()=> flushQueue(), Math.max(16, throttleMs));
    return ()=>clearInterval(t);
  }, [throttleMs]);

  function resetBook(s: Snapshot){
    setBids(sortBids(aggregate(s.bids || [], agg)));
    setAsks(sortAsks(aggregate(s.asks || [], agg)));
    if (Number.isFinite(s.last!)) setLast(s.last);
    if (s.ts) setTs(s.ts);
  }

  function applyDelta(d: Delta){
    if (d.side === 'bid'){
      setBids(prev => upsertLevel(prev, d.price, d.size, agg, 'bid'));
    } else {
      setAsks(prev => upsertLevel(prev, d.price, d.size, agg, 'ask'));
    }
  }

  function flushQueue(){
    if (!queueRef.current.length) return;
    const now = performance.now();
    if (now - lastFlushRef.current < throttleMs*0.8) return;
    const batch = queueRef.current.splice(0, queueRef.current.length);
    // apply batched deltas
    let nb = bids, na = asks;
    for (const d of batch){
      if (d.side==='bid'){
        nb = upsertLevel(nb, d.price, d.size, agg, 'bid');
      } else {
        na = upsertLevel(na, d.price, d.size, agg, 'ask');
      }
    }
    setBids(sortBids(nb));
    setAsks(sortAsks(na));
    lastFlushRef.current = now;
  }

  async function run(){
    setLoading(true); setError(null); setLog(''); setNotes('');
    setBids([]); setAsks([]); setLast(undefined); setTs(undefined);
    queueRef.current = [];
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;
    const payload = { id: uid(), symbol, agg };

    try{
      const res = await fetch(endpoint, {
        method:'POST',
        headers:{ 'Content-Type':'application/json', Accept:'text/event-stream, application/json, text/plain' },
        body: JSON.stringify(payload),
        signal
      });
      const ctype = res.headers.get('content-type') || '';
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      if (res.body && ctype.includes('text/event-stream')){
        const reader = res.body.getReader(); const dec = new TextDecoder('utf-8'); let acc='';
        while(true){
          const { value, done } = await reader.read(); if (done) break;
          const chunk = dec.decode(value, { stream:true });
          const lines = (acc + chunk).split('\n'); acc = lines.pop() || '';
          for (const line of lines){
            if (!line.startsWith('data:')) continue;
            const data = line.slice(5).trim(); if (!data) continue;
            if (data === '[DONE]') continue;
            setLog(s=>s+data+'\n');
            try {
              const p:Patch = JSON.parse(data);
              if (p.symbol) setSymbol(p.symbol);
              if (p.snapshot) resetBook(p.snapshot);
              if (Array.isArray(p.deltas)) queueRef.current.push(...p.deltas);
              if (p.trade){ setLast(p.trade.price); setTs(p.trade.ts); }
              if (p.notes) setNotes(n=>(n?n+'\n':'')+p.notes);
            } catch {
              setNotes(n=>(n?n+'\n':'')+data);
            }
          }
        }
      } else if (ctype.includes('application/json')){
        const p:Patch = await res.json();
        if (p.snapshot) resetBook(p.snapshot);
        if (Array.isArray(p.deltas)) { queueRef.current.push(...p.deltas); flushQueue(); }
        if (p.notes) setNotes(p.notes);
      } else {
        setNotes(await res.text());
      }
    } catch(e:any){
      if (e?.name!=='AbortError') setError(e?.message || 'Request failed');
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function stop(){ abortRef.current?.abort(); abortRef.current=null; setLoading(false); }

  function exportCSV(){
    const cols = ['side','price','size'];
    const lines = [
      ...bids.map(b=>['bid',b.price,b.size].join(',')),
      ...asks.map(a=>['ask',a.price,a.size].join(',')),
    ];
    download(`${symbol}_depth.csv`, [cols.join(','), ...lines].join('\n'), 'text/csv');
  }

  // when aggregation step changes, re-aggregate existing book
  useEffect(()=>{
    setBids(prev => sortBids(aggregate(prev, agg)));
    setAsks(prev => sortAsks(aggregate(prev, agg)));
  }, [agg]);

  // auto DPs from tick size (simple heuristic)
  useEffect(()=>{ setPriceDp(agg>=1 ? 2 : Math.min(6, Math.max(2, (''+agg).split('.')[1]?.length || 2))); }, [agg]);

  /* =============== UI =============== */
  return (
    <div className={`flex h-full w-full flex-col rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-950 ${className}`}>
      {/* header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-200 dark:border-neutral-800">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex items-center gap-2">
          <button onClick={exportCSV} className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-900">Export CSV</button>
          {loading
            ? <button onClick={stop} className="text-xs px-3 py-1 rounded border border-rose-400 text-rose-700 hover:bg-rose-50">Stop</button>
            : <button onClick={run} className="text-xs px-3 py-1 rounded border border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700">Run</button>}
        </div>
      </div>

      {/* controls */}
      <div className="p-3 grid grid-cols-1 md:grid-cols-7 gap-3 border-b border-neutral-200 dark:border-neutral-800">
        <Field label="Symbol">
          <input value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())} className="w-full rounded border px-3 py-2 text-sm"/>
        </Field>
        <Field label="Aggregation (price step)">
          <input type="number" step="any" value={agg} onChange={e=>setAgg(Math.max(0.000001, Number(e.target.value)||agg))}
                 className="w-full rounded border px-2 py-2 text-sm"/>
        </Field>
        <Field label="Visible levels">
          <input type="number" value={levels} onChange={e=>setLevels(clamp(Number(e.target.value)||levels, 5, 50))}
                 className="w-full rounded border px-2 py-2 text-sm"/>
        </Field>
        <Field label="Throttle (ms)">
          <input type="number" value={throttleMs} onChange={e=>setThrottleMs(Math.max(16, Number(e.target.value)||throttleMs))}
                 className="w-full rounded border px-2 py-2 text-sm"/>
        </Field>
        <Kpi label="Best Bid" value={fmtPx(bids[0]?.price ?? NaN, priceDp)} />
        <Kpi label="Best Ask" value={fmtPx(asks[0]?.price ?? NaN, priceDp)} />
        <Kpi label="Spread" value={spread!=null ? fmtPx(spread, priceDp) : ''}/>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0 flex-1 overflow-hidden">
        {/* Bids */}
        <Card title={`Bids (${visBids.length})`}>
          <DepthTable side="bid" rows={visBids} maxSize={maxSize} priceDp={priceDp}/>
        </Card>

        {/* Asks */}
        <Card title={`Asks (${visAsks.length})`}>
          <DepthTable side="ask" rows={visAsks} maxSize={maxSize} priceDp={priceDp}/>
        </Card>

        {/* Right: curves + stats */}
        <div className="space-y-3">
          <Card title="Cumulative Depth">
            <DepthCurve bids={visBids} asks={visAsks} />
          </Card>
          <Card title="Order Flow">
            <div className="grid grid-cols-3 gap-2">
              <Stat label="Imbalance" value={`${(imbalance*100).toFixed(0)}%`} tone={imbalance>=0?1:-1}/>
              <Stat label="VWAP 5" value={fmtPx(vwap5 ?? NaN, priceDp)} />
              <Stat label="VWAP 10" value={fmtPx(vwap10 ?? NaN, priceDp)} />
              <Stat label="VWAP 20" value={fmtPx(vwap20 ?? NaN, priceDp)} />
              <Stat label="Last" value={last!=null?fmtPx(last, priceDp):''} />
              <Stat label="Time" value={ts? new Date(ts).toLocaleTimeString(): ''} />
            </div>
          </Card>
          <Card title="Notes">
            <pre className="text-[11px] whitespace-pre-wrap break-words text-neutral-700 dark:text-neutral-300">{notes || 'No notes.'}</pre>
            {error && <div className="text-xs text-rose-600 mt-2">{error}</div>}
          </Card>
        </div>
      </div>
    </div>
  );
}

/* =============== Subcomponents =============== */
function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 overflow-hidden">
      <div className="border-b border-neutral-200 dark:border-neutral-800 px-3 py-2 text-xs font-medium">{title}</div>
      <div className="p-3">{children}</div>
    </div>
  );
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="block text-[11px] text-neutral-500 mb-1">{label}</label>{children}</div>;
}
function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 px-3 py-2">
      <div className="text-[11px] text-neutral-500">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}
function Stat({ label, value, tone=0 }:{ label:string; value:string; tone?:-1|0|1 }){
  const color = tone>0 ? 'text-emerald-700' : tone<0 ? 'text-rose-700' : 'text-neutral-900 dark:text-neutral-100';
  return <div className="rounded border border-neutral-200 dark:border-neutral-800 px-3 py-2">
    <div className="text-[11px] text-neutral-500">{label}</div>
    <div className={`text-sm font-semibold ${color}`}>{value}</div>
  </div>;
}

function DepthTable({ side, rows, maxSize, priceDp }:{ side:Side; rows:L2[]; maxSize:number; priceDp:number }){
  const isBid = side==='bid';
  return (
    <div className="overflow-auto max-h-[52vh]">
      <table className="min-w-full text-sm">
        <thead className="sticky top-0 bg-neutral-100 dark:bg-neutral-800 text-xs">
          <tr>
            {isBid ? (<>
              <Th>Size</Th><Th>Price</Th><Th>Bar</Th>
            </>):(<>
              <Th>Bar</Th><Th>Price</Th><Th>Size</Th>
            </>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r,i)=>{
            const ratio = clamp(r.size / (maxSize||1), 0, 1);
            const bg = side==='bid' ? `rgba(16,185,129,${0.15 + 0.6*ratio})` : `rgba(239,68,68,${0.15 + 0.6*ratio})`;
            return (
              <tr key={`${side}-${r.price}-${i}`} className={i%2 ? 'bg-neutral-50 dark:bg-neutral-950' : 'bg-white dark:bg-neutral-900'}>
                {isBid ? (
                  <>
                    <Td>{fmtSz(r.size)}</Td>
                    <Td className="font-semibold">{fmtPx(r.price, priceDp)}</Td>
                    <Td>
                      <div className="h-3 rounded" style={{ width: `${ratio*100}%`, background:bg }} />
                    </Td>
                  </>
                ) : (
                  <>
                    <Td>
                      <div className="ml-auto h-3 rounded" style={{ width: `${ratio*100}%`, background:bg }} />
                    </Td>
                    <Td className="font-semibold">{fmtPx(r.price, priceDp)}</Td>
                    <Td className="text-right">{fmtSz(r.size)}</Td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
function Th({ children }:{ children:React.ReactNode }){ return <th className="px-2 py-2 text-left font-semibold border-b border-neutral-200 dark:border-neutral-700">{children}</th>; }
function Td({ children, className='' }:{ children:React.ReactNode; className?:string }){ return <td className={`px-2 py-2 border-b border-neutral-100 dark:border-neutral-800 ${className}`}>{children}</td>; }

function DepthCurve({ bids, asks }:{ bids:L2[]; asks:L2[] }){
  const width = 360, height = 120;
  // cumulative depth from mid outward (approx)
  const bb = bids.slice().reverse(); // far->near
  const aa = asks.slice();           // near->far
  const cumB = cum(bb.map(x=>x.size));
  const cumA = cum(aa.map(x=>x.size));
  const max = Math.max(cumB[cumB.length-1]||1, cumA[cumA.length-1]||1);
  const stepB = width * 0.45 / Math.max(1, bb.length-1);
  const stepA = width * 0.45 / Math.max(1, aa.length-1);

  const pathB = bb.map((x, i)=>`L${(width*0.5 - i*stepB).toFixed(1)},${(height - (cumB[i]/max)*height).toFixed(1)}`).join(' ').replace(/^L/,'M');
  const pathA = aa.map((x, i)=>`L${(width*0.5 + i*stepA).toFixed(1)},${(height - (cumA[i]/max)*height).toFixed(1)}`).join(' ').replace(/^L/,'M');

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      <path d={pathB} fill="none" stroke="#10b981" strokeWidth={1.6}/>
      <path d={pathA} fill="none" stroke="#ef4444" strokeWidth={1.6}/>
      <line x1={width/2} y1={0} x2={width/2} y2={height} stroke="#9ca3af" strokeDasharray="4 4"/>
    </svg>
  );
}

/* =============== Book helpers =============== */
function sortBids(arr:L2[]):L2[]{ return arr.slice().sort((a,b)=> b.price - a.price); }
function sortAsks(arr:L2[]):L2[]{ return arr.slice().sort((a,b)=> a.price - b.price); }

function bucket(p:number, step:number){
  if (step <= 0) return p;
  return Math.round(p / step) * step;
}
function aggregate(levels:L2[], step:number):L2[]{
  if (!levels.length) return levels;
  const map = new Map<number, number>();
  for (const {price, size} of levels){
    const k = bucket(price, step);
    map.set(k, (map.get(k)||0) + size);
  }
  return Array.from(map.entries()).map(([price,size])=>({price, size}));
}
function upsertLevel(prev:L2[], price:number, size:number, step:number, side:Side):L2[]{
  const k = bucket(price, step);
  const map = new Map<number, number>(prev.map(l=>[l.price, l.size]));
  if (size <= 0) { map.delete(k); }
  else { map.set(k, size); }
  const arr = Array.from(map.entries()).map(([price,size])=>({price,size}));
  return side==='bid' ? sortBids(arr) : sortAsks(arr);
}
function cum(xs:number[]):number[]{ const out:number[]=[]; let s=0; for (const x of xs){ s+=x; out.push(s); } return out; }
function vwapN(levels:L2[], n:number):number|undefined{
  if (!levels.length) return undefined;
  const arr = levels.slice(0, n);
  const num = arr.reduce((s,l)=>s + l.price*l.size, 0);
  const den = arr.reduce((s,l)=>s + l.size, 0);
  return den>0 ? num/den : undefined;
}